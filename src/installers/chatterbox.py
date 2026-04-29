from __future__ import annotations

import os

from src.config import Settings
from src.runtime import ensure_venv, popen, uv_pip_install, write_text


def install(settings: Settings) -> dict:
    engine_dir = settings.engines_dir / "chatterbox"
    engine_dir.mkdir(parents=True, exist_ok=True)

    python_bin = ensure_venv(engine_dir)
    # chatterbox-tts pulls in torch, torchaudio, transformers, librosa, etc.
    uv_pip_install(
        python_bin,
        [
            "fastapi",
            "uvicorn",
            "soundfile",
            "numpy",
            "chatterbox-tts",
        ],
    )

    write_text(engine_dir / "app.py", settings.read_repo_text("src/apps/chatterbox_app.py"))

    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "OPENAI_MODEL_ID": settings.openai_model_id or "chatterbox",
        "CHATTERBOX_LANGUAGE": settings.chatterbox_language,
        "CHATTERBOX_PROMPT_WAV": settings.chatterbox_prompt_wav,
        "CHATTERBOX_DEFAULT_VOICE": settings.chatterbox_default_voice,
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
        log_path=settings.log_dir / "chatterbox-uvicorn.log",
    )
    return {
        "proc": proc,
        "app_dir": engine_dir,
        "log_path": settings.log_dir / "chatterbox-uvicorn.log",
    }
