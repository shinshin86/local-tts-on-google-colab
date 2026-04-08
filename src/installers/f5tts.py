from __future__ import annotations

import os

from src.config import Settings
from src.runtime import ensure_venv, popen, uv_pip_install, write_text


def install(settings: Settings) -> dict:
    engine_dir = settings.engines_dir / "f5-tts"
    engine_dir.mkdir(parents=True, exist_ok=True)
    python_bin = ensure_venv(engine_dir)
    uv_pip_install(
        python_bin,
        [
            "fastapi",
            "uvicorn",
            "f5-tts",
            "soundfile",
            "numpy",
        ],
    )
    write_text(engine_dir / "app.py", settings.read_repo_text("src/apps/f5tts_app.py"))
    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "OPENAI_MODEL_ID": settings.openai_model_id or "f5-tts",
        "F5TTS_MODEL": settings.f5tts_model,
        "F5TTS_CKPT_FILE": settings.f5tts_ckpt_file,
        "F5TTS_VOCAB_FILE": settings.f5tts_vocab_file,
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
        log_path=settings.log_dir / "f5-tts-uvicorn.log",
    )
    return {"proc": proc, "app_dir": engine_dir, "log_path": settings.log_dir / "f5-tts-uvicorn.log"}
