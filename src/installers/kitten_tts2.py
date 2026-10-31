from __future__ import annotations

import os
from pathlib import Path

from src.config import Settings
from src.runtime import ensure_venv, popen, run, uv_pip_install, write_text


KITTEN_TTS2_PACKAGE = "kittenml==0.9.7"
KITTEN_TTS2_MODEL_ID = "KittenML/kitten-tts-2"
KITTEN_TTS2_MODEL_REF = "c93cb7c271baa51d9ae8204d5277988906f235a2"


def _download_model(python_bin: Path, settings: Settings, model_dir: Path) -> None:
    """Fetch only the files the selected weight / decoder variant needs.

    The Hub repo also carries a llama.cpp bundle and every LM packing; the
    kittenml helper narrows the download to what config.json points at.
    """
    model_id = settings.kitten_tts2_hf_model
    revision = KITTEN_TTS2_MODEL_REF if model_id == KITTEN_TTS2_MODEL_ID else None
    marker = model_dir / ".download-complete"
    marker_value = (
        f"{model_id}@{revision or 'default'}"
        f"#{settings.kitten_tts2_weights}#{settings.kitten_tts2_decoder}\n"
    )
    if marker.exists() and marker.read_text(encoding="utf-8") == marker_value:
        return
    code = (
        "import json\n"
        "from huggingface_hub import hf_hub_download, snapshot_download\n"
        "from kittenml.get_model import _kitten2_files\n"
        f"cfg = json.load(open(hf_hub_download({model_id!r}, 'config.json', revision={revision!r})))\n"
        f"files = _kitten2_files(cfg, decoder={settings.kitten_tts2_decoder!r}, "
        f"weights={settings.kitten_tts2_weights!r})\n"
        f"snapshot_download(repo_id={model_id!r}, revision={revision!r}, "
        f"local_dir={str(model_dir)!r}, allow_patterns=files)\n"
    )
    run([str(python_bin), "-c", code])
    marker.write_text(marker_value, encoding="utf-8")


def install(settings: Settings) -> dict:
    engine_dir = settings.engines_dir / "kitten-tts-2-openai"
    engine_dir.mkdir(parents=True, exist_ok=True)
    python_bin = ensure_venv(engine_dir)
    uv_pip_install(
        python_bin,
        [KITTEN_TTS2_PACKAGE, "fastapi", "uvicorn", "soundfile"],
    )

    model_dir = engine_dir / "models" / settings.kitten_tts2_hf_model.replace("/", "--")
    _download_model(python_bin, settings, model_dir)

    write_text(engine_dir / "app.py", settings.read_repo_text("src/apps/kitten_tts2_app.py"))
    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "OPENAI_MODEL_ID": settings.openai_model_id or "kitten-tts-2",
        "KITTEN_TTS2_MODEL_DIR": str(model_dir),
        "KITTEN_TTS2_WEIGHTS": settings.kitten_tts2_weights,
        "KITTEN_TTS2_DECODER": settings.kitten_tts2_decoder,
        "KITTEN_TTS2_DEFAULT_VOICE": settings.kitten_tts2_default_voice,
        "KITTEN_TTS2_PROMPT_WAV": settings.kitten_tts2_prompt_wav,
        "KITTEN_TTS2_PROMPT_TEXT": settings.kitten_tts2_prompt_text,
        "KITTEN_TTS2_PRESET": settings.kitten_tts2_preset,
        "KITTEN_TTS2_SEED": str(settings.kitten_tts2_seed),
    }
    proc = popen(
        [
            str(engine_dir / ".venv" / "bin" / "uvicorn"),
            "app:app",
            "--host",
            "0.0.0.0",
            "--port",
            str(settings.app_port),
            "--log-level",
            "info",
        ],
        cwd=str(engine_dir),
        env=env,
        log_path=settings.log_dir / "kitten-tts-2-uvicorn.log",
    )
    return {
        "proc": proc,
        "app_dir": engine_dir,
        "log_path": settings.log_dir / "kitten-tts-2-uvicorn.log",
    }
