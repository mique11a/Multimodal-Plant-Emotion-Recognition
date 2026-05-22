import argparse

from inference_utils import load_plant_model, predict_file, to_pretty_json


def parse_args():
    parser = argparse.ArgumentParser(description="对单个 CSV 文件执行植物状态推理。")
    parser.add_argument("--weights", default="model/plant_fusion_best.pt", help="模型 checkpoint 路径")
    parser.add_argument(
        "--csv",
        default="analysis_outputs/local_smoke/raw_test_touch_000.csv",
        help="待推理的 CSV 文件路径",
    )
    parser.add_argument("--device", default=None, help="cpu 或 cuda")
    parser.add_argument("--stride", type=int, default=None, help="滑窗步长，默认使用 checkpoint 中的 stream_stride")
    parser.add_argument("--vote-smoothing", type=int, default=5, help="多数投票平滑窗口大小")
    parser.add_argument("--include-windows", action="store_true", help="输出全部窗口级结果")
    return parser.parse_args()


def main():
    args = parse_args()
    bundle = load_plant_model(args.weights, device=args.device)
    summary = predict_file(
        bundle,
        args.csv,
        stride=args.stride,
        vote_smoothing=args.vote_smoothing,
        include_windows=args.include_windows,
    )
    print(to_pretty_json(summary))


if __name__ == "__main__":
    main()
