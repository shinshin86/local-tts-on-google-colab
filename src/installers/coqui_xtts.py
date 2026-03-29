from __future__ import annotations

import os

from src.config import Settings
from src.runtime import ensure_venv, popen, uv_pip_install, write_text


def install(settings: Settings) -> dict:
    engine_dir = settings.engines_dir / "coqui-xtts"
    engine_dir.mkdir(parents=True, exist_ok=True)
    python_bin = ensure_venv(engine_dir)
    uv_pip_install(
        python_bin,
        [
            "fastapi",
            "uvicorn",
            "coqui-tts",
            "transformers>=4.46.0,<5.0.0",
            "torch",
            "torchaudio",
        ],
    )
    write_text(engine_dir / "app.py", settings.read_repo_text("src/apps/coqui_xtts_app.py"))
    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "OPENAI_MODEL_ID": settings.openai_model_id or "xtts_v2",
        "XTTS_LANGUAGE": settings.xtts_language,
        "XTTS_SPEAKER_WAV": settings.xtts_speaker_wav,
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
        log_path=settings.log_dir / "coqui-xtts-uvicorn.log",
    )
    return {"proc": proc, "app_dir": engine_dir, "log_path": settings.log_dir / "coqui-xtts-uvicorn.log"}
