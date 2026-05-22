import argparse
import copy
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split

from PlantFusionNet import PlantFusionNet
from PlantMultimodalDataset import PlantMultimodalDataset


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_metrics(targets, preds, num_classes):
    targets = np.asarray(targets, dtype=np.int64)
    preds = np.asarray(preds, dtype=np.int64)
    accuracy = float((targets == preds).mean()) if len(targets) else 0.0

    per_class_recall = []
    per_class_f1 = []
    for class_idx in range(num_classes):
        tp = int(np.sum((targets == class_idx) & (preds == class_idx)))
        fp = int(np.sum((targets != class_idx) & (preds == class_idx)))
        fn = int(np.sum((targets == class_idx) & (preds != class_idx)))
        support = int(np.sum(targets == class_idx))

        recall = tp / support if support else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        per_class_recall.append(recall)
        per_class_f1.append(f1)

    return {
        "accuracy": accuracy,
        "balanced_accuracy": float(np.mean(per_class_recall)) if per_class_recall else 0.0,
        "macro_f1": float(np.mean(per_class_f1)) if per_class_f1 else 0.0,
    }


def run_epoch(model, dataloader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)

    running_loss = 0.0
    targets = []
    preds = []

    for volt, imp, label in dataloader:
        volt = volt.to(device)
        imp = imp.to(device)
        label = label.to(device)

        if training:
            optimizer.zero_grad()

        probs, _, _, _ = model(volt, imp)
        loss = criterion(torch.log(probs + 1e-8), label)

        if training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        running_loss += loss.item()
        targets.extend(label.detach().cpu().tolist())
        preds.extend(torch.argmax(probs, dim=1).detach().cpu().tolist())

    metrics = compute_metrics(targets, preds, model.model_config["num_classes"])
    metrics["loss"] = running_loss / max(1, len(dataloader))
    return metrics


def build_checkpoint(model, dataset, args, best_metrics):
    preprocessing_summary = {}
    if args.preprocessing_summary and os.path.exists(args.preprocessing_summary):
        with open(args.preprocessing_summary, "r", encoding="utf-8") as handle:
            preprocessing_summary = json.load(handle)

    return {
        "model_state": copy.deepcopy(model.state_dict()),
        "label_to_index": dataset.label_to_index,
        "index_to_label": dataset.index_to_label,
        "imp_mean": dataset.imp_mean,
        "imp_std": dataset.imp_std,
        "voltage_mean": dataset.voltage_mean,
        "voltage_std": dataset.voltage_std,
        "window_size": args.window_size,
        "train_stride": args.stride,
        "stream_stride": args.stream_stride,
        "sample_rates": dataset.sample_rates,
        "preprocessing_summary": preprocessing_summary,
        "model_config": model.model_config,
        "training_args": vars(args),
        "best_metrics": best_metrics,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="训练植物多模态状态识别模型。")
    parser.add_argument("--data-root", default="dataset_real_condition_filtered_v2", help="训练数据目录")
    parser.add_argument("--output", default="model/plant_fusion_best.pt", help="最佳模型输出路径")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--window-size", type=int, default=250)
    parser.add_argument("--stride", type=int, default=50)
    parser.add_argument("--stream-stride", type=int, default=50)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--preprocessing-summary",
        default="analysis_outputs/preprocessing_summary.json",
        help="可选的预处理摘要 JSON，存在时会写入 checkpoint",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    dataset = PlantMultimodalDataset(
        root_dir=args.data_root,
        window_size=args.window_size,
        stride=args.stride,
    )
    print(f"样本数: {len(dataset)} | 类别映射: {dataset.label_to_index}")

    val_size = max(1, int(len(dataset) * args.val_ratio))
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(args.seed)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = PlantFusionNet(
        num_classes=len(dataset.label_to_index),
        voltage_channels=1,
        impedance_dim=1,
        dropout=args.dropout,
    ).to(device)

    criterion = nn.NLLLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    best_score = -float("inf")
    best_checkpoint = None

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer=optimizer)
        with torch.no_grad():
            val_metrics = run_epoch(model, val_loader, criterion, device, optimizer=None)
        scheduler.step()

        score = val_metrics["macro_f1"] + val_metrics["accuracy"]
        if score > best_score:
            best_score = score
            best_checkpoint = build_checkpoint(
                model,
                dataset,
                args,
                {
                    "epoch": float(epoch),
                    "train_loss": train_metrics["loss"],
                    "train_accuracy": train_metrics["accuracy"],
                    "train_balanced_accuracy": train_metrics["balanced_accuracy"],
                    "train_macro_f1": train_metrics["macro_f1"],
                    "val_loss": val_metrics["loss"],
                    "val_accuracy": val_metrics["accuracy"],
                    "val_balanced_accuracy": val_metrics["balanced_accuracy"],
                    "val_macro_f1": val_metrics["macro_f1"],
                },
            )
            torch.save(best_checkpoint, args.output)
            marker = " *"
        else:
            marker = ""

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['accuracy']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f} "
            f"val_f1={val_metrics['macro_f1']:.4f}{marker}"
        )

    if best_checkpoint is None:
        raise RuntimeError("训练未生成任何可保存模型。")

    last_output = args.output.replace(".pt", "_last.pt")
    last_checkpoint = build_checkpoint(
        model,
        dataset,
        args,
        best_checkpoint["best_metrics"],
    )
    torch.save(last_checkpoint, last_output)

    print(f"最佳模型已保存到: {args.output}")
    print(f"最后一轮模型已保存到: {last_output}")


if __name__ == "__main__":
    main()
