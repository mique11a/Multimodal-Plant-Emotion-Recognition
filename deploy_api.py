import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from inference_utils import load_plant_model, predict_window


WEIGHTS_PATH = os.getenv("PLANT_MODEL_WEIGHTS", "model/plant_fusion_best.pt")
DEVICE = os.getenv("PLANT_MODEL_DEVICE", None)
BUNDLE = load_plant_model(WEIGHTS_PATH, device=DEVICE)


class PlantInferenceHandler(BaseHTTPRequestHandler):
    server_version = "PlantConditionHTTP/1.0"

    def _send_json(self, payload, status=200):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path != "/health":
            self._send_json({"error": "not found"}, status=404)
            return
        self._send_json(
            {
                "status": "ok",
                "weights": WEIGHTS_PATH,
                "window_size": BUNDLE.window_size,
                "labels": list(BUNDLE.label_to_index.keys()),
            }
        )

    def do_POST(self):
        if self.path != "/predict":
            self._send_json({"error": "not found"}, status=404)
            return

        length = int(self.headers.get("Content-Length", "0"))
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            voltage = payload["voltage"]
            impedance = payload["impedance"]
        except Exception as exc:
            self._send_json({"error": f"invalid request: {exc}"}, status=400)
            return

        if len(voltage) != BUNDLE.window_size:
            self._send_json(
                {
                    "error": f"voltage 长度必须为 {BUNDLE.window_size}，当前为 {len(voltage)}。"
                },
                status=400,
            )
            return

        result = predict_window(BUNDLE, voltage, impedance)
        self._send_json(result)

    def log_message(self, format, *args):
        return


def parse_args():
    parser = argparse.ArgumentParser(description="启动植物状态识别 HTTP 推理服务。")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


def main():
    args = parse_args()
    server = ThreadingHTTPServer((args.host, args.port), PlantInferenceHandler)
    print(f"Serving on http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
