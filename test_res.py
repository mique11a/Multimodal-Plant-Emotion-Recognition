import glob
import os
from collections import Counter, deque

import pandas as pd
import requests

from inference_utils import find_signal_columns, iter_windows, load_plant_model, predict_file


API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/predict")
HEALTH_URL = os.getenv("HEALTH_URL", "http://127.0.0.1:8000/health")

INFERENCE_MODE = os.getenv("INFERENCE_MODE", "local").lower()
EXPECTED_TYPE = os.getenv("EXPECTED_TYPE", "TOUCH").upper()
TARGET_DIR = os.getenv(
    "TARGET_DIR",
    os.path.join("dataset_raw_test_fullfix_filtered", EXPECTED_TYPE.lower()),
)
LOCAL_WEIGHTS = os.getenv("LOCAL_WEIGHTS", os.path.join("model", "plant_fusion_best.pt"))
LOCAL_DEVICE = os.getenv("LOCAL_DEVICE", None)
VOTE_SMOOTHING = int(os.getenv("VOTE_SMOOTHING", "5"))


def check_health():
    try:
        response = requests.get(HEALTH_URL, timeout=5)
        return response.status_code == 200 and response.json().get("status") == "ok"
    except Exception:
        return False


def process_single_file_cloud(csv_file_path, window_size=250, step_size=50):
    print(f"\n📄 正在分析: {os.path.basename(csv_file_path)}")
    df = pd.read_csv(csv_file_path)
    voltage_col, impedance_col = find_signal_columns(df)
    smoother = deque(maxlen=max(1, VOTE_SMOOTHING))
    votes = Counter()

    for window in iter_windows(df, window_size=window_size, stride=step_size):
        payload = {
            "voltage": window["voltage_window"].tolist(),
            "impedance": float(window["impedance_scalar"]),
        }
        response = requests.post(API_URL, json=payload, timeout=5)
        response.raise_for_status()
        label = response.json()["label"].upper()
        smoother.append(label)
        votes[Counter(smoother).most_common(1)[0][0]] += 1

    if not votes:
        print("⚠️ 文件长度不足，未形成任何有效窗口。")
        return None

    dominant = votes.most_common(1)[0][0]
    icon = "✅" if dominant == EXPECTED_TYPE else "❌"
    print(f"{icon} 文件主导状态判决: 【{dominant}】 (细节分布: {dict(votes)})")
    return dominant


def process_single_file_local(bundle, csv_file_path):
    print(f"\n📄 正在分析: {os.path.basename(csv_file_path)}")
    summary = predict_file(bundle, csv_file_path, stride=bundle.stream_stride, vote_smoothing=VOTE_SMOOTHING)
    dominant = summary["dominant_label"].upper()
    icon = "✅" if dominant == EXPECTED_TYPE else "❌"
    print(f"{icon} 文件主导状态判决: 【{dominant}】 (细节分布: {summary['vote_distribution']})")
    return dominant


def batch_inference_with_accuracy(directory_path, expected_type):
    print(f"🚀 开始批量推理目录: {directory_path}")
    print(f"🎯 本批次期望标签设为: 【{expected_type}】")
    print(f"⚙️ 推理模式: {INFERENCE_MODE.upper()}")

    bundle = None
    if INFERENCE_MODE == "local":
        bundle = load_plant_model(LOCAL_WEIGHTS, device=LOCAL_DEVICE)
        print(f"🧠 本地权重: {LOCAL_WEIGHTS}")
    else:
        if not check_health():
            print("🚨 无法连接到推理服务。")
            return
        print(f"☁️ 服务地址: {API_URL}")

    csv_files = glob.glob(os.path.join(directory_path, "*.csv"))
    if not csv_files:
        print("⚠️ 目录下没有找到 CSV 文件。")
        return

    total_valid_files = 0
    correct_predictions = 0

    print(f"📁 共找到 {len(csv_files)} 个文件，准备开始压测...\n" + "=" * 50)
    for file_path in sorted(csv_files):
        if INFERENCE_MODE == "local":
            predicted_state = process_single_file_local(bundle, file_path)
        else:
            predicted_state = process_single_file_cloud(
                file_path,
                window_size=bundle.window_size if bundle else 250,
                step_size=bundle.stream_stride if bundle else 50,
            )

        if predicted_state is None:
            continue
        total_valid_files += 1
        if predicted_state == expected_type:
            correct_predictions += 1

    print("\n" + "=" * 50)
    print("🎉 批量推理压测完成！")
    if total_valid_files == 0:
        print("⚠️ 未完成任何有效文件的推理。")
        return

    accuracy = correct_predictions / total_valid_files * 100.0
    print(f"📊 测试样本总数: {total_valid_files}")
    print(f"🎯 准确预测数:   {correct_predictions}")
    print(f"🏆 综合准确率:   {accuracy:.2f}%")


if __name__ == "__main__":
    batch_inference_with_accuracy(TARGET_DIR, EXPECTED_TYPE)
