from __future__ import annotations

import os

from src.config import Settings
from src.runtime import ensure_venv, popen, uv_pip_install, write_text


def install(settings: Settings) -> dict:
    engine_dir = settings.engines_dir / "pocket-tts"
    engine_dir.mkdir(parents=True, exist_ok=True)

    python_bin = ensure_venv(engine_dir)
    uv_pip_install(
        python_bin,
        [
            "fastapi",
            "uvicorn",
            "soundfile",
            "numpy",
            "huggingface_hub",
            "pocket-tts",
        ],
    )

    write_text(engine_dir / "app.py", settings.read_repo_text("src/apps/pocket_tts_app.py"))

    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "OPENAI_MODEL_ID": settings.openai_model_id or "pocket-tts",
        "POCKET_LANGUAGE": settings.pocket_language,
        "POCKET_DEFAULT_SPEAKER": settings.pocket_default_speaker,
        "POCKET_PROMPT_WAV": settings.pocket_prompt_wav,
        "POCKET_DEFAULT_VOICE": settings.pocket_default_voice,
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
        log_path=settings.log_dir / "pocket-tts-uvicorn.log",
    )
    return {
        "proc": proc,
        "app_dir": engine_dir,
        "log_path": settings.log_dir / "pocket-tts-uvicorn.log",
    }
