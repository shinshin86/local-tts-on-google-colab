from __future__ import annotations

import os

from src.config import Settings
from src.runtime import ensure_venv, popen, uv_pip_install, write_text


def install(settings: Settings) -> dict:
    engine_dir = settings.engines_dir / "audio8-tts"
    engine_dir.mkdir(parents=True, exist_ok=True)
    python_bin = ensure_venv(engine_dir)
    uv_pip_install(
        python_bin,
        [
            "fastapi",
            "uvicorn",
            "torch>=2.5.0",
            "torchaudio>=2.5.0",
            "transformers>=4.57.0,<5",
            "numpy>=1.26",
            "soundfile>=0.12",
            "safetensors>=0.4",
        ],
    )
    write_text(engine_dir / "app.py", settings.read_repo_text("src/apps/audio8_tts_app.py"))
    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "OPENAI_MODEL_ID": settings.openai_model_id or "audio8-tts",
        "AUDIO8_HF_MODEL": settings.audio8_hf_model,
        "AUDIO8_PROMPT_WAV": settings.audio8_prompt_wav,
        "AUDIO8_PROMPT_TEXT": settings.audio8_prompt_text,
        "AUDIO8_DEFAULT_VOICE": settings.audio8_default_voice,
        "AUDIO8_DEVICE": settings.audio8_device,
        "AUDIO8_DTYPE": settings.audio8_dtype,
        "AUDIO8_MAX_NEW_TOKENS": str(settings.audio8_max_new_tokens),
        "AUDIO8_TEMPERATURE": str(settings.audio8_temperature),
        "AUDIO8_TOP_P": str(settings.audio8_top_p),
        "AUDIO8_TOP_K": str(settings.audio8_top_k),
    }
    log_path = settings.log_dir / "audio8-tts-uvicorn.log"
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
        log_path=log_path,
    )
    return {"proc": proc, "app_dir": engine_dir, "log_path": log_path}
