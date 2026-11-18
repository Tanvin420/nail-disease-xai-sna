"""Stage 12 - on-device (smartphone) latency benchmark.

Serves phone_bench/index.html and the ONNX models from runs/onnx/ over the local network. Open the printed URL on
the phone (same Wi-Fi), enter the phone model, choose models and press "Run". Inference runs on the phone CPU with
ONNX Runtime Web (WebAssembly, single thread, SIMD). Results are POSTed back and saved to results/bench/phone/.

    python scripts/s12_phone_server.py [--port 8765]
"""
import argparse
import json
import socket
import sys
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nailrev.config import ROOT, load_config  # noqa: E402


def lan_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    cfg = load_config()
    onnx_dir = cfg.path("runs") / "onnx"
    out = cfg.path("results") / "bench" / "phone"
    out.mkdir(parents=True, exist_ok=True)
    page = ROOT / "phone_bench" / "index.html"
    if not (onnx_dir / "models.json").exists():
        sys.exit("run scripts/s11_bench_cpu_onnx.py first (it exports the ONNX models)")

    class H(SimpleHTTPRequestHandler):
        def _send(self, path: Path, ctype: str):
            data = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                return self._send(page, "text/html; charset=utf-8")
            if self.path == "/models.json":
                return self._send(onnx_dir / "models.json", "application/json")
            if self.path.startswith("/onnx/"):
                f = onnx_dir / Path(self.path[len("/onnx/"):]).name
                if f.exists() and f.suffix == ".onnx":
                    return self._send(f, "application/octet-stream")
            self.send_error(404)

        def do_POST(self):
            if self.path != "/result":
                return self.send_error(404)
            body = self.rfile.read(int(self.headers["Content-Length"]))
            rec = json.loads(body)
            name = f"{time.strftime('%Y%m%d_%H%M%S')}_{rec.get('model', 'x')}_{rec.get('variant', '')}.json"
            (out / name).write_text(json.dumps(rec, indent=2))
            print("saved", name, f"median {rec.get('median_ms', 0):.1f} ms", flush=True)
            self.send_response(200)
            self.end_headers()

    print(f"Open on the phone:  http://{lan_ip()}:{args.port}/   (Ctrl+C to stop)")
    ThreadingHTTPServer(("0.0.0.0", args.port), H).serve_forever()


if __name__ == "__main__":
    main()
