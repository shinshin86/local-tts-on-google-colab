from __future__ import annotations

import os

from src.config import Settings
from src.runtime import ensure_venv, popen, uv_pip_install, write_text


def install(settings: Settings) -> dict:
    engine_dir = settings.engines_dir / "outetts"
    engine_dir.mkdir(parents=True, exist_ok=True)

    python_bin = ensure_venv(engine_dir)
    uv_pip_install(
        python_bin,
        [
            "fastapi",
            "uvicorn",
            "soundfile",
            "numpy",
            "outetts",
        ],
    )

    write_text(engine_dir / "app.py", settings.read_repo_text("src/apps/outetts_app.py"))

    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "OPENAI_MODEL_ID": settings.openai_model_id or "outetts",
        "OUTETTS_MODEL_SIZE": settings.outetts_model_size,
        "OUTETTS_BACKEND": settings.outetts_backend,
        "OUTETTS_DEFAULT_SPEAKER": settings.outetts_default_speaker,
        "OUTETTS_PROMPT_WAV": settings.outetts_prompt_wav,
        "OUTETTS_PROMPT_TEXT": settings.outetts_prompt_text,
        "OUTETTS_DEFAULT_VOICE": settings.outetts_default_voice,
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
        log_path=settings.log_dir / "outetts-uvicorn.log",
    )
    return {
        "proc": proc,
        "app_dir": engine_dir,
        "log_path": settings.log_dir / "outetts-uvicorn.log",
    }
