import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


DEFAULT_CLASS_ORDER = ("light", "normal", "touch", "stress")


class PlantMultimodalDataset(Dataset):
    def __init__(
        self,
        root_dir,
        window_size=250,
        stride=50,
        label_to_index=None,
        voltage_mean=None,
        voltage_std=None,
        imp_mean=None,
        imp_std=None,
        return_meta=False,
    ):
        self.root_dir = root_dir
        self.window_size = int(window_size)
        self.stride = int(stride)
        self.return_meta = return_meta
        self.label_to_index = label_to_index or self._discover_labels(root_dir)
        self.index_to_label = {idx: label for label, idx in self.label_to_index.items()}
        self.samples = []
        self.sample_rates = {}

        stats = self._compute_global_stats(root_dir)
        self.voltage_mean = float(stats["voltage_mean"] if voltage_mean is None else voltage_mean)
        self.voltage_std = float(stats["voltage_std"] if voltage_std is None else voltage_std)
        self.imp_mean = float(stats["imp_mean"] if imp_mean is None else imp_mean)
        self.imp_std = float(stats["imp_std"] if imp_std is None else imp_std)

        self._load_samples()

    def _discover_labels(self, root_dir):
        present_dirs = {
            name for name in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, name))
        }
        ordered = [label for label in DEFAULT_CLASS_ORDER if label in present_dirs]
        ordered.extend(sorted(present_dirs - set(ordered)))
        return {label: idx for idx, label in enumerate(ordered)}

    def _find_signal_columns(self, df):
        voltage_candidates = [col for col in df.columns if "电压" in col]
        impedance_candidates = [col for col in df.columns if "阻抗" in col or "幅值" in col]
        return (
            voltage_candidates[0] if voltage_candidates else None,
            impedance_candidates[0] if impedance_candidates else None,
        )

    def _estimate_sample_rate(self, df):
        if "时间(s)" not in df.columns:
            return None
        diffs = df["时间(s)"].diff().dropna().to_numpy(dtype=np.float64)
        diffs = diffs[diffs > 0]
        if diffs.size == 0:
            return None
        return float(1.0 / np.median(diffs))

    def _compute_global_stats(self, root_dir):
        voltage_values = []
        impedance_values = []

        for label in self.label_to_index:
            class_dir = os.path.join(root_dir, label)
            if not os.path.isdir(class_dir):
                continue
            for file_name in sorted(os.listdir(class_dir)):
                if not file_name.endswith(".csv"):
                    continue
                file_path = os.path.join(class_dir, file_name)
                df = pd.read_csv(file_path)
                voltage_col, impedance_col = self._find_signal_columns(df)
                if not voltage_col or not impedance_col:
                    continue
                voltage_values.append(df[voltage_col].to_numpy(dtype=np.float32))
                impedance_values.append(df[impedance_col].to_numpy(dtype=np.float32))

        if not voltage_values or not impedance_values:
            raise RuntimeError(f"未能在 {root_dir} 中找到可用的电压/阻抗 CSV 数据。")

        all_voltages = np.concatenate(voltage_values)
        all_impedances = np.concatenate(impedance_values)

        return {
            "voltage_mean": float(np.mean(all_voltages)),
            "voltage_std": float(np.std(all_voltages) + 1e-8),
            "imp_mean": float(np.mean(all_impedances)),
            "imp_std": float(np.std(all_impedances) + 1e-8),
        }

    def _load_samples(self):
        for label_name, label_idx in self.label_to_index.items():
            class_dir = os.path.join(self.root_dir, label_name)
            if not os.path.isdir(class_dir):
                continue

            for file_name in sorted(os.listdir(class_dir)):
                if not file_name.endswith(".csv"):
                    continue
                file_path = os.path.join(class_dir, file_name)
                df = pd.read_csv(file_path)
                voltage_col, impedance_col = self._find_signal_columns(df)
                if not voltage_col or not impedance_col:
                    continue

                voltages = df[voltage_col].to_numpy(dtype=np.float32)
                impedances = df[impedance_col].to_numpy(dtype=np.float32)
                sample_rate = self._estimate_sample_rate(df)
                if sample_rate is not None:
                    self.sample_rates[f"{label_name}/{file_name}"] = sample_rate

                if len(voltages) < self.window_size:
                    continue

                for start in range(0, len(voltages) - self.window_size + 1, self.stride):
                    end = start + self.window_size
                    self.samples.append(
                        {
                            "volt": voltages[start:end],
                            "imp": float(np.mean(impedances[start:end])),
                            "label": label_idx,
                            "file_name": file_name,
                            "class_name": label_name,
                            "start": start,
                            "end": end,
                        }
                    )

        if not self.samples:
            raise RuntimeError(f"在 {self.root_dir} 中没有生成任何有效样本。")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        volt = (sample["volt"] - self.voltage_mean) / self.voltage_std
        imp = (sample["imp"] - self.imp_mean) / self.imp_std

        volt_tensor = torch.tensor(volt, dtype=torch.float32).unsqueeze(0)
        imp_tensor = torch.tensor([imp], dtype=torch.float32)
        label_tensor = torch.tensor(sample["label"], dtype=torch.long)

        if not self.return_meta:
            return volt_tensor, imp_tensor, label_tensor

        meta = {
            "file_name": sample["file_name"],
            "class_name": sample["class_name"],
            "start": sample["start"],
            "end": sample["end"],
        }
        return volt_tensor, imp_tensor, label_tensor, meta
