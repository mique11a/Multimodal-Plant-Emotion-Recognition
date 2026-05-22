import argparse
import json
import math
import os
from collections import Counter

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.signal import butter, filtfilt, find_peaks, iirnotch, savgol_filter
from tqdm import tqdm


STATE_PROFILES = {
    "active": {"pole_radius": 0.95, "ensemble_size": 8, "noise_coeff": 0.20, "wavelet": "sym8"},
    "inactive": {"pole_radius": 0.90, "ensemble_size": 6, "noise_coeff": 0.10, "wavelet": "db4"},
    "abnormal": {"pole_radius": 0.99, "ensemble_size": 10, "noise_coeff": 0.25, "wavelet": "sym12"},
}
WAVELET_ORDER = {"db4": 4, "sym8": 8, "sym12": 12}
WAVELET_LEVELS = ("db4", "sym8", "sym12")


def infer_state(data):
    centered = data - np.mean(data)
    std = float(np.std(centered))
    diff_std = float(np.std(np.diff(centered))) if len(centered) > 2 else 0.0
    span = float(np.ptp(centered))
    if diff_std > 0.12 or span > 8.0 * (std + 1e-8):
        return "abnormal"
    if std < 0.01:
        return "inactive"
    return "active"


def sanitize_for_json(value):
    if isinstance(value, dict):
        return {str(k): sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_for_json(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def find_voltage_column(df):
    cols = [col for col in df.columns if "电压" in col]
    return cols[0] if cols else None


class PlantSignalFilter:
    def __init__(self, fs=250.0, window_size=250, overlap=125):
        self.fs = float(fs)
        self.nyq = self.fs / 2.0
        self.window_size = int(window_size)
        self.overlap = int(overlap)
        self.hop_size = max(1, self.window_size - self.overlap)
        self.beta = 0.9
        self.mu_alpha = 0.08
        self.xi = 1e-6
        self.cost_weights = (0.45, 0.40, 0.15)
        self.max_feedback_iters = 3
        self.max_imfs = 5
        self.max_sift_iters = 8
        self.residual_eps = 1e-4
        self.alpha_init = 1.05
        self.alpha_bounds = (0.6, 1.8)

    def analog_prefilter(self, data):
        high_b, high_a = butter(2, 0.1 / self.nyq, btype="highpass")
        low_b, low_a = butter(4, 100.0 / self.nyq, btype="lowpass")
        return filtfilt(low_b, low_a, filtfilt(high_b, high_a, data))

    def adaptive_notch_filter(self, data, pole_radius, harmonics=(50.0, 100.0)):
        filtered = data.copy()
        q = float(np.clip(1.0 / (2.0 * max(0.005, 1.0 - pole_radius)), 5.0, 80.0))
        for freq in harmonics:
            if freq >= self.nyq:
                continue
            b, a = iirnotch(freq / self.nyq, q)
            filtered = filtfilt(b, a, filtered)
        return filtered

    def _build_envelope(self, signal, peak_idx):
        idx = np.concatenate(([0], peak_idx, [len(signal) - 1]))
        vals = signal[idx]
        idx, uniq = np.unique(idx, return_index=True)
        vals = vals[uniq]
        if len(idx) < 3:
            return np.interp(np.arange(len(signal)), idx, vals)
        return CubicSpline(idx, vals, bc_type="natural")(np.arange(len(signal)))

    def _extract_first_imf(self, signal):
        imf = signal.copy()
        for _ in range(self.max_sift_iters):
            max_idx, _ = find_peaks(imf)
            min_idx, _ = find_peaks(-imf)
            if len(max_idx) + len(min_idx) < 4:
                break
            mean_env = 0.5 * (
                self._build_envelope(imf, max_idx) + self._build_envelope(imf, min_idx)
            )
            updated = imf - mean_env
            zero_crossings = np.sum(updated[:-1] * updated[1:] < 0)
            extrema_count = len(max_idx) + len(min_idx)
            imf = updated
            if abs(extrema_count - zero_crossings) <= 1 and np.mean(np.abs(mean_env)) < 0.1 * (np.std(imf) + 1e-8):
                break
        return imf

    def iceemdan_decompose(self, signal, ensemble_size, noise_coeff):
        residual = signal.copy()
        imfs = []
        for _ in range(self.max_imfs):
            res_std = np.std(residual) + 1e-8
            ensemble = []
            for _ in range(max(2, ensemble_size)):
                noisy = residual + noise_coeff * np.random.normal(0.0, res_std, len(signal))
                ensemble.append(self._extract_first_imf(noisy))
            imf = np.mean(ensemble, axis=0)
            new_residual = residual - imf
            sd = np.sum((residual - new_residual) ** 2) / (np.sum(residual ** 2) + 1e-8)
            imfs.append(imf)
            residual = new_residual
            extrema_count = len(find_peaks(residual)[0]) + len(find_peaks(-residual)[0])
            if sd < self.residual_eps or extrema_count < 4:
                break
        return imfs, residual

    def permutation_entropy(self, signal, order=3, delay=1):
        if len(signal) < order * delay + 1:
            return 0.0
        patterns = {}
        total = 0
        for start in range(len(signal) - delay * (order - 1)):
            pattern = tuple(np.argsort(signal[start:start + order * delay:delay]))
            patterns[pattern] = patterns.get(pattern, 0) + 1
            total += 1
        probs = np.array(list(patterns.values()), dtype=np.float64) / max(1, total)
        return float(-np.sum(probs * np.log(probs + 1e-12)) / np.log(math.factorial(order)))

    def wavelet_threshold_denoise(self, signal, wavelet_name, alpha_scale, imf_index, levels=3):
        order = WAVELET_ORDER[wavelet_name]
        approx, details = signal.copy(), []
        for level in range(levels):
            window = max(5, min(order + 2 * level + 3, len(signal) - (1 - len(signal) % 2)))
            if window % 2 == 0:
                window = max(5, window - 1)
            smooth = savgol_filter(approx, window_length=window, polyorder=min(3, window - 2), mode="interp")
            details.append(approx - smooth)
            approx = smooth

        denoised_details = []
        scale = max(np.log(imf_index + 1.0), 1e-6)
        for detail in details:
            sigma = np.median(np.abs(detail)) / 0.6745 + 1e-8
            threshold = alpha_scale * sigma * np.sqrt(2.0 * np.log(len(signal) + 1.0)) / scale
            denoised_details.append(np.sign(detail) * np.maximum(np.abs(detail) - threshold, 0.0))
        return approx + np.sum(denoised_details, axis=0)

    def quality_metrics(self, raw_signal, filtered_signal):
        if np.std(raw_signal) < 1e-8 or np.std(filtered_signal) < 1e-8:
            corr = 0.0
        else:
            corr = float(np.corrcoef(raw_signal, filtered_signal)[0, 1])
        noise = raw_signal - filtered_signal
        gsnr = 10.0 * np.log10((np.sum(raw_signal ** 2) + 1e-8) / (np.sum(noise ** 2) + 1e-8))
        nrmse = np.sqrt(np.mean(noise ** 2)) / (np.ptp(raw_signal) + 1e-8)
        return corr, float(gsnr), float(nrmse)

    def shift_wavelet(self, current, step):
        idx = WAVELET_LEVELS.index(current)
        return WAVELET_LEVELS[min(len(WAVELET_LEVELS) - 1, max(0, idx + step))]

    def _process_window(self, raw_window):
        state = infer_state(raw_window)
        params = STATE_PROFILES[state].copy()
        alpha = self.alpha_init
        prev_alpha = 1.0
        prev_cost = None
        r_opt = None
        working = self.adaptive_notch_filter(raw_window, params["pole_radius"])
        best_signal = working.copy()
        best_cost = -np.inf
        best_summary = {"state": state, **params, "corr": None, "gsnr": None, "nrmse": None, "cost": None, "alpha": alpha}

        for _ in range(self.max_feedback_iters):
            imfs, residual = self.iceemdan_decompose(working, params["ensemble_size"], params["noise_coeff"])
            reconstructed = residual.copy()
            for idx, imf in enumerate(imfs, start=1):
                rho = abs(np.corrcoef(working, imf)[0, 1]) if np.std(imf) > 1e-8 else 0.0
                if self.permutation_entropy(imf) > 0.6 and rho < 0.1:
                    continue
                reconstructed += self.wavelet_threshold_denoise(imf, params["wavelet"], alpha, idx)

            corr, gsnr, nrmse = self.quality_metrics(working, reconstructed)
            cost = self.cost_weights[0] * gsnr + self.cost_weights[1] * corr - self.cost_weights[2] * nrmse
            if cost > best_cost:
                best_cost = cost
                best_signal = reconstructed.copy()
                best_summary = {
                    "state": state,
                    "pole_radius": float(params["pole_radius"]),
                    "ensemble_size": int(params["ensemble_size"]),
                    "noise_coeff": float(params["noise_coeff"]),
                    "wavelet": params["wavelet"],
                    "corr": float(corr),
                    "gsnr": float(gsnr),
                    "nrmse": float(nrmse),
                    "cost": float(cost),
                    "alpha": float(alpha),
                }

            if prev_cost is not None:
                delta = alpha - prev_alpha
                direction = np.sign(delta) if abs(delta) > 1e-8 else 1.0
                next_alpha = alpha + self.mu_alpha * direction * (cost - prev_cost) / (abs(delta) + self.xi)
                prev_alpha, alpha = alpha, float(np.clip(next_alpha, *self.alpha_bounds))
            prev_cost = cost
            r_opt = corr if r_opt is None else self.beta * r_opt + (1.0 - self.beta) * corr

            if corr < r_opt and nrmse > 0.18:
                params["pole_radius"] = min(0.995, params["pole_radius"] + 0.01)
                params["ensemble_size"] = max(4, params["ensemble_size"] - 1)
                params["noise_coeff"] = max(0.05, params["noise_coeff"] * 0.9)
                params["wavelet"] = self.shift_wavelet(params["wavelet"], -1)
            elif corr > r_opt and nrmse < 0.08:
                params["pole_radius"] = max(0.88, params["pole_radius"] - 0.01)
                params["ensemble_size"] = min(12, params["ensemble_size"] + 1)
                params["noise_coeff"] = min(0.35, params["noise_coeff"] * 1.08)
                params["wavelet"] = self.shift_wavelet(params["wavelet"], 1)
            working = self.adaptive_notch_filter(raw_window, params["pole_radius"])

        best_summary["r_opt"] = r_opt
        return best_signal, best_summary

    def _overlap_add(self, data):
        if len(data) <= self.window_size:
            signal, summary = self._process_window(data)
            return signal, [summary]

        output = np.zeros(len(data), dtype=np.float64)
        weights = np.zeros(len(data), dtype=np.float64)
        taper = np.hanning(self.window_size)
        taper[0] = taper[-1] = 1e-6
        summaries = []

        starts = list(range(0, len(data) - self.window_size + 1, self.hop_size))
        if (len(data) - self.window_size) % self.hop_size != 0:
            starts.append(len(data) - self.window_size)

        for start in starts:
            end = start + self.window_size
            filtered, summary = self._process_window(data[start:end])
            output[start:end] += filtered * taper
            weights[start:end] += taper
            summaries.append(summary)
        return output / np.where(weights < 1e-8, 1.0, weights), summaries

    def process(self, data, return_summary=False):
        data = np.asarray(data, dtype=np.float64)
        if len(data) < 8:
            filtered = (data - np.mean(data)) / (np.std(data) + 1e-8)
            summary = {"signal_length": int(len(data)), "window_count": 0, "short_signal": True}
            return (filtered, summary) if return_summary else filtered

        filtered, window_summaries = self._overlap_add(self.analog_prefilter(data))
        mean_val = float(np.mean(filtered))
        std_val = float(np.std(filtered) + 1e-8)
        normalized = (filtered - mean_val) / std_val

        def mean_of(name):
            values = [item[name] for item in window_summaries if item.get(name) is not None and math.isfinite(float(item[name]))]
            return float(np.mean(values)) if values else None

        wavelets = [item["wavelet"] for item in window_summaries]
        summary = {
            "signal_length": int(len(data)),
            "window_count": int(len(window_summaries)),
            "short_signal": False,
            "state_distribution": dict(Counter(item["state"] for item in window_summaries)),
            "avg_pole_radius": mean_of("pole_radius"),
            "avg_ensemble_size": mean_of("ensemble_size"),
            "avg_noise_coeff": mean_of("noise_coeff"),
            "dominant_wavelet": Counter(wavelets).most_common(1)[0][0] if wavelets else None,
            "avg_corr": mean_of("corr"),
            "avg_gsnr": mean_of("gsnr"),
            "avg_nrmse": mean_of("nrmse"),
            "avg_cost": mean_of("cost"),
            "avg_alpha": mean_of("alpha"),
            "avg_r_opt": mean_of("r_opt"),
            "normalization_mean": mean_val,
            "normalization_std": std_val,
        }
        return (normalized, summary) if return_summary else normalized


def batch_process_dataset(input_root, output_root, summary_path=None):
    processor = PlantSignalFilter(fs=250.0)
    summary_records = {}
    skipped_files = {}
    os.makedirs(output_root, exist_ok=True)

    categories = sorted(name for name in os.listdir(input_root) if os.path.isdir(os.path.join(input_root, name)))
    for category in categories:
        src_dir = os.path.join(input_root, category)
        dst_dir = os.path.join(output_root, category)
        os.makedirs(dst_dir, exist_ok=True)
        files = sorted(name for name in os.listdir(src_dir) if name.endswith(".csv"))
        print(f"正在处理类别: {category}")

        for file_name in tqdm(files, desc=f"Filtering {category}"):
            file_key = f"{category}/{file_name}"
            src_path = os.path.join(src_dir, file_name)
            dst_path = os.path.join(dst_dir, file_name)
            try:
                df = pd.read_csv(src_path)
                voltage_col = find_voltage_column(df)
                if voltage_col is None:
                    skipped_files[file_key] = "missing voltage column"
                    continue
                filtered_voltage, process_summary = processor.process(
                    df[voltage_col].to_numpy(dtype=np.float64),
                    return_summary=True,
                )
                df[voltage_col] = filtered_voltage
                df.to_csv(dst_path, index=False)
                summary_records[file_key] = sanitize_for_json(
                    {
                        "input_file": file_key,
                        "output_file": os.path.relpath(dst_path, output_root).replace(os.sep, "/"),
                        "voltage_column": voltage_col,
                        "sample_count": int(len(df)),
                        "processor_fs": processor.fs,
                        "window_size": processor.window_size,
                        "overlap": processor.overlap,
                        "hop_size": processor.hop_size,
                        "analog_highpass_hz": 0.1,
                        "analog_lowpass_hz": 100.0,
                        "adaptive_notch_harmonics_hz": [50.0, 100.0],
                        **process_summary,
                    }
                )
            except Exception as exc:
                skipped_files[file_key] = str(exc)
                print(f"跳过文件 {file_name}: {exc}")

    if summary_path:
        summary_dir = os.path.dirname(summary_path)
        if summary_dir:
            os.makedirs(summary_dir, exist_ok=True)
        payload = {
            "__meta__": sanitize_for_json(
                {
                    "input_root": input_root,
                    "output_root": output_root,
                    "processed_file_count": len(summary_records),
                    "skipped_file_count": len(skipped_files),
                    "skipped_files": skipped_files,
                }
            )
        }
        payload.update(summary_records)
        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        print(f"预处理摘要已写入: {summary_path}")
    return summary_records


def parse_args():
    parser = argparse.ArgumentParser(description="批量清洗植物电压信号并输出标准化 CSV。")
    parser.add_argument("--input", default="dataset_real_condition", help="输入数据目录")
    parser.add_argument("--output", default="dataset_real_condition_filtered_v2", help="输出数据目录")
    parser.add_argument("--summary-json", default="analysis_outputs/preprocessing_summary.json", help="预处理摘要 JSON 输出路径")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    batch_process_dataset(args.input, args.output, summary_path=args.summary_json)
    print(f"已完成滤波与标准化，输出目录: {args.output}")
