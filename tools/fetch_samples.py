"""Download listening samples from a running engine into the local library.

Run this on your own machine while the Colab cell is up. It sends the fixed
lines in tools/sample_texts.json to the engine's public
`/v1/audio/speech` endpoint and stores the returned audio under `samples/`
(git-ignored), so you can listen again later without starting Colab:

    python3 tools/fetch_samples.py --url https://xxxx.trycloudflare.com
    python3 tools/fetch_samples.py --url https://xxxx.trycloudflare.com --lang ja --voice clone

Nothing here runs an engine locally; it only makes HTTP requests. Browse the
library with `python3 tools/serve_samples.py`.
"""

from __future__ import annotations

import argparse
import datetime as dt
import io
import json
import re
import sys
import time
import urllib.error
import urllib.request
import wave
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TEXTS_FILE = Path(__file__).resolve().parent / "sample_texts.json"
DEFAULT_OUT = REPO_ROOT / "samples"
USER_AGENT = "local-tts-on-google-colab/fetch_samples"


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._") or "unknown"


def http_get_json(url: str, timeout: float) -> dict | None:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, ValueError, OSError):
        return None
    return data if isinstance(data, dict) else None


def request_speech(
    base_url: str, text: str, voice: str | None, instructions: str | None, timeout: float
) -> tuple[bytes, dict[str, str]]:
    body: dict[str, object] = {"input": text, "response_format": "wav"}
    if voice:
        body["voice"] = voice
    if instructions:
        body["instructions"] = instructions
    req = urllib.request.Request(
        f"{base_url}/v1/audio/speech",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "User-Agent": USER_AGENT},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read(), {k.lower(): v for k, v in resp.headers.items()}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc


def wav_duration(data: bytes) -> float | None:
    try:
        with wave.open(io.BytesIO(data)) as w:
            return round(w.getnframes() / float(w.getframerate()), 3)
    except (wave.Error, EOFError, ZeroDivisionError):
        return None


def load_texts(lang: str, ids: list[str] | None) -> list[dict]:
    texts = json.loads(TEXTS_FILE.read_text(encoding="utf-8"))["texts"]
    if lang != "all":
        texts = [t for t in texts if t["lang"] == lang]
    if ids:
        wanted = set(ids)
        unknown = wanted - {t["id"] for t in texts}
        if unknown:
            sys.exit(f"Unknown text id(s) for --lang {lang}: {', '.join(sorted(unknown))}")
        texts = [t for t in texts if t["id"] in wanted]
    return texts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--url", required=True, help="Public base URL, e.g. https://xxxx.trycloudflare.com")
    parser.add_argument("--engine", help="Engine name. Defaults to the `engine` field reported by GET /")
    parser.add_argument("--voice", help="OpenAI `voice` value. Omitted = the engine's default voice")
    parser.add_argument("--instructions", help="OpenAI `instructions` value (e.g. a voice-design caption) for engines that support it")
    parser.add_argument("--lang", choices=["ja", "en", "all"], default="all", help="Which lines to generate (default: all)")
    parser.add_argument("--ids", nargs="+", help="Only these text ids (e.g. ja-01 en-04)")
    parser.add_argument("--note", default="", help="Free-form note stored with this run (shown in the viewer)")
    parser.add_argument("--timeout", type=float, default=300.0, help="Per-request timeout in seconds (default: 300)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Library directory (default: samples/)")
    args = parser.parse_args()

    base_url = args.url.rstrip("/")
    texts = load_texts(args.lang, args.ids)
    if not texts:
        sys.exit("No lines selected.")

    server_info = http_get_json(f"{base_url}/", timeout=30)
    engine = args.engine or (server_info or {}).get("engine")
    if not engine:
        sys.exit("Could not determine the engine name from GET /. Pass --engine explicitly.")

    started = dt.datetime.now()
    voice_label = args.voice or "default"
    run_id = f"{started:%Y%m%d-%H%M%S}_{safe_name(voice_label)}"
    if args.instructions:
        run_id += "_instructions"
    run_dir = args.out / safe_name(engine) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Engine: {engine} / voice: {voice_label} / {len(texts)} line(s)")
    if args.instructions:
        print(f"Instructions: {args.instructions}")
    print(f"Saving to: {run_dir}")

    results = []
    for t in texts:
        print(f"  [{t['id']}] ", end="", flush=True)
        entry: dict[str, object] = {"id": t["id"], "lang": t["lang"], "text": t["text"]}
        t0 = time.monotonic()
        try:
            audio, headers = request_speech(base_url, t["text"], args.voice, args.instructions, args.timeout)
        except Exception as exc:  # keep going so one bad line doesn't lose the rest
            entry["error"] = str(exc)
            print(f"failed: {exc}")
            results.append(entry)
            continue
        elapsed = round(time.monotonic() - t0, 2)
        if not audio.startswith(b"RIFF"):
            entry["error"] = f"Response is not WAV (content-type: {headers.get('content-type')})"
            print(f"failed: {entry['error']}")
            results.append(entry)
            continue
        filename = f"{t['id']}.wav"
        (run_dir / filename).write_bytes(audio)
        entry.update(
            file=filename,
            elapsed_sec=elapsed,
            duration_sec=wav_duration(audio),
            response_model=headers.get("x-openai-model"),
            response_voice=headers.get("x-openai-voice"),
        )
        print(f"ok ({elapsed}s, audio {entry['duration_sec']}s)")
        results.append(entry)

    meta = {
        "engine": engine,
        "voice": voice_label,
        "instructions": args.instructions or "",
        "created_at": started.isoformat(timespec="seconds"),
        "note": args.note,
        "server_info": server_info,
        "results": results,
    }
    # Don't keep the public URL: trycloudflare hostnames are short-lived and not useful later.
    (run_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    ok = sum(1 for r in results if "file" in r)
    if ok == 0:
        (run_dir / "meta.json").unlink()
        run_dir.rmdir()
        print("Nothing was saved; removed the empty run directory.")
        return 1
    print(f"Done: {ok}/{len(results)} saved.")
    return 0 if ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
