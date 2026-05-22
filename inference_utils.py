import json
from collections import Counter, deque
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from PlantFusionNet import PlantFusionNet


@dataclass
class LoadedPlantModel:
    model: PlantFusionNet
    device: torch.device
    label_to_index: dict
    index_to_label: dict
    voltage_mean: float
    voltage_std: float
    imp_mean: float
    imp_std: float
    window_size: int
    train_stride: int
    stream_stride: int
    raw_checkpoint: dict


def _default_model_config(label_to_index):
    return {
        "num_classes": len(label_to_index),
        "voltage_channels": 1,
        "impedance_dim": 1,
        "dropout": 0.15,
    }


def load_plant_model(weights_path, device=None):
    resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = torch.load(weights_path, map_location=resolved_device, weights_only=False)
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]
        label_to_index = checkpoint.get("label_to_index", {"light": 0, "normal": 1, "touch": 2})
        model_config = checkpoint.get("model_config", _default_model_config(label_to_index))
    else:
        state_dict = checkpoint
        label_to_index = {"light": 0, "normal": 1, "touch": 2}
        model_config = _default_model_config(label_to_index)
        checkpoint = {"model_state": state_dict, "label_to_index": label_to_index}

    model = PlantFusionNet(**model_config).to(resolved_device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    index_to_label = checkpoint.get("index_to_label") or {
        idx: label for label, idx in label_to_index.items()
    }

    return LoadedPlantModel(
        model=model,
        device=resolved_device,
        label_to_index=label_to_index,
        index_to_label={int(idx): str(label) for idx, label in index_to_label.items()},
        voltage_mean=float(checkpoint.get("voltage_mean", 0.0)),
        voltage_std=float(checkpoint.get("voltage_std", 1.0)) or 1.0,
        imp_mean=float(checkpoint.get("imp_mean", 0.0)),
        imp_std=float(checkpoint.get("imp_std", 1.0)) or 1.0,
        window_size=int(checkpoint.get("window_size", 250)),
        train_stride=int(checkpoint.get("train_stride", 50)),
        stream_stride=int(checkpoint.get("stream_stride", 50)),
        raw_checkpoint=checkpoint,
    )


def find_signal_columns(df):
    voltage_candidates = [col for col in df.columns if "电压" in col]
    impedance_candidates = [col for col in df.columns if "阻抗" in col or "幅值" in col]
    if not voltage_candidates or not impedance_candidates:
        raise ValueError("CSV 中未找到可用的电压列或阻抗列。")
    return voltage_candidates[0], impedance_candidates[0]


def normalize_window(bundle, voltage_window, impedance_scalar):
    voltage = np.asarray(voltage_window, dtype=np.float32)
    voltage = (voltage - bundle.voltage_mean) / (bundle.voltage_std + 1e-8)
    impedance = (float(impedance_scalar) - bundle.imp_mean) / (bundle.imp_std + 1e-8)

    volt_tensor = torch.tensor(voltage, dtype=torch.float32, device=bundle.device).unsqueeze(0).unsqueeze(0)
    imp_tensor = torch.tensor([[impedance]], dtype=torch.float32, device=bundle.device)
    return volt_tensor, imp_tensor


def predict_window(bundle, voltage_window, impedance_scalar):
    volt_tensor, imp_tensor = normalize_window(bundle, voltage_window, impedance_scalar)
    with torch.no_grad():
        probs, fast, slow, alpha = bundle.model(volt_tensor, imp_tensor)

    probs = probs.squeeze(0).cpu().numpy()
    alpha = alpha.squeeze(0).cpu().numpy()
    pred_idx = int(np.argmax(probs))
    label = bundle.index_to_label[pred_idx]

    probabilities = {
        bundle.index_to_label[idx].upper(): round(float(prob), 4)
        for idx, prob in enumerate(probs)
    }

    return {
        "label": label.upper(),
        "label_index": pred_idx,
        "confidence": round(float(probs[pred_idx]), 4),
        "probabilities": probabilities,
        "fast_response": round(float(alpha[pred_idx]), 4),
        "slow_response": round(float(1.0 - alpha[pred_idx]), 4),
        "fast_branch_score": round(float(fast[0, pred_idx].item()), 4),
        "slow_branch_score": round(float(slow[0, pred_idx].item()), 4),
    }


def iter_windows(df, window_size, stride):
    voltage_col, impedance_col = find_signal_columns(df)
    voltages = df[voltage_col].to_numpy(dtype=np.float32)
    impedances = df[impedance_col].to_numpy(dtype=np.float32)
    if len(voltages) < window_size:
        return

    for start in range(0, len(voltages) - window_size + 1, stride):
        end = start + window_size
        yield {
            "start": start,
            "end": end,
            "voltage_window": voltages[start:end],
            "impedance_scalar": float(np.mean(impedances[start:end])),
        }


def predict_file(bundle, csv_path, stride=None, vote_smoothing=5, include_windows=False):
    stride = stride or bundle.stream_stride
    df = pd.read_csv(csv_path)
    smoother = deque(maxlen=max(1, vote_smoothing))
    window_results = []
    votes = Counter()

    for window in iter_windows(df, bundle.window_size, stride):
        result = predict_window(bundle, window["voltage_window"], window["impedance_scalar"])
        smoother.append(result["label"])
        smoothed_label = Counter(smoother).most_common(1)[0][0]
        votes[smoothed_label] += 1
        result.update({"start": window["start"], "end": window["end"], "smoothed_label": smoothed_label})
        window_results.append(result)

    if not window_results:
        raise ValueError("文件长度不足以生成完整推理窗口。")

    dominant_label = votes.most_common(1)[0][0]
    representative = max(window_results, key=lambda item: item["confidence"])

    probability_keys = list(window_results[0]["probabilities"].keys())
    mean_probabilities = {}
    for key in probability_keys:
        mean_probabilities[key] = round(
            float(np.mean([item["probabilities"][key] for item in window_results])), 4
        )

    summary = {
        "file": csv_path,
        "window_count": len(window_results),
        "dominant_label": dominant_label,
        "vote_distribution": dict(votes),
        "mean_probabilities": mean_probabilities,
        "representative_window": representative,
    }
    if include_windows:
        summary["window_results"] = window_results
    return summary


def to_pretty_json(payload):
    return json.dumps(payload, ensure_ascii=False, indent=2)
