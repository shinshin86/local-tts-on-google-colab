"""Serve the local sample library as a listening grid on localhost.

    python3 tools/serve_samples.py            # http://127.0.0.1:8000
    python3 tools/serve_samples.py --port 8123 --open

Reads the runs saved by tools/fetch_samples.py under `samples/` (git-ignored)
and needs no network access or Colab. Binds to 127.0.0.1 only.
"""

from __future__ import annotations

import argparse
import functools
import http.server
import json
import webbrowser
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TOOLS_DIR.parent
VIEWER_HTML = TOOLS_DIR / "sample_viewer" / "index.html"
TEXTS_FILE = TOOLS_DIR / "sample_texts.json"
DEFAULT_LIBRARY = REPO_ROOT / "samples"


def build_manifest(library: Path) -> dict:
    texts = json.loads(TEXTS_FILE.read_text(encoding="utf-8"))["texts"]
    runs = []
    for meta_path in sorted(library.glob("*/*/meta.json")):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        run_dir = meta_path.parent
        rel = run_dir.relative_to(library).as_posix()
        items = {}
        for r in meta.get("results", []):
            item = {k: r.get(k) for k in ("elapsed_sec", "duration_sec", "error")}
            if r.get("file") and (run_dir / r["file"]).is_file():
                item["src"] = f"{rel}/{r['file']}"
            items[r.get("id")] = item
        runs.append(
            {
                "path": rel,
                "engine": meta.get("engine") or run_dir.parent.name,
                "voice": meta.get("voice", "default"),
                "instructions": meta.get("instructions", ""),
                "created_at": meta.get("created_at", ""),
                "note": meta.get("note", ""),
                "items": items,
            }
        )
    return {"texts": texts, "runs": runs}


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, library: Path, **kwargs):
        self.library = library
        super().__init__(*args, directory=str(library), **kwargs)

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            return self._send(VIEWER_HTML.read_bytes(), "text/html; charset=utf-8")
        if path == "/manifest.json":
            body = json.dumps(build_manifest(self.library), ensure_ascii=False).encode("utf-8")
            return self._send(body, "application/json; charset=utf-8")
        return super().do_GET()

    def _send(self, body: bytes, content_type: str):
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):  # keep the terminal quiet for audio range requests
        pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--library", type=Path, default=DEFAULT_LIBRARY, help="Library directory (default: samples/)")
    parser.add_argument("--open", action="store_true", help="Open the viewer in the default browser")
    args = parser.parse_args()

    library = args.library.resolve()
    library.mkdir(parents=True, exist_ok=True)
    handler = functools.partial(Handler, library=library)
    server = http.server.ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"Serving {library} at {url} (Ctrl+C to stop)")
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
