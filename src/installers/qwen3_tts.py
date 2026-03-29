from __future__ import annotations

import os

from src.config import Settings
from src.runtime import ensure_venv, popen, uv_pip_install, write_text


def install(settings: Settings) -> dict:
    engine_dir = settings.engines_dir / "qwen3-tts"
    engine_dir.mkdir(parents=True, exist_ok=True)
    python_bin = ensure_venv(engine_dir)
    uv_pip_install(
        python_bin,
        [
            "fastapi",
            "uvicorn",
            "qwen-tts",
            "soundfile",
            "numpy",
        ],
    )
    write_text(engine_dir / "app.py", settings.read_repo_text("src/apps/qwen3_tts_app.py"))
    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "MPLBACKEND": "agg",
        "OPENAI_MODEL_ID": settings.openai_model_id or "qwen3-tts",
        "QWEN3_HF_MODEL": settings.qwen3_hf_model,
        "QWEN3_LANGUAGE": settings.qwen3_language,
        "QWEN3_DEFAULT_SPEAKER": settings.qwen3_default_speaker,
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
        log_path=settings.log_dir / "qwen3-tts-uvicorn.log",
    )
    return {"proc": proc, "app_dir": engine_dir, "log_path": settings.log_dir / "qwen3-tts-uvicorn.log"}
