from __future__ import annotations

import os

from src.config import Settings
from src.runtime import ensure_venv, popen, run, uv_pip_install, write_text


def install(settings: Settings) -> dict:
    engine_dir = settings.engines_dir / "tiny-tts-openai"
    engine_dir.mkdir(parents=True, exist_ok=True)
    python_bin = ensure_venv(engine_dir)
    uv_pip_install(
        python_bin,
        ["fastapi", "uvicorn", "tiny-tts", "soundfile"],
    )
    # Download NLTK data required by g2p-en (dependency of tiny-tts)
    run(
        [str(python_bin), "-c", "import nltk; nltk.download('averaged_perceptron_tagger_eng')"],
        cwd=str(engine_dir),
    )
    write_text(engine_dir / "app.py", settings.read_repo_text("src/apps/tiny_tts_app.py"))
    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "OPENAI_MODEL_ID": settings.openai_model_id or "tiny-tts",
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
        log_path=settings.log_dir / "tiny-tts-uvicorn.log",
    )
    return {"proc": proc, "app_dir": engine_dir, "log_path": settings.log_dir / "tiny-tts-uvicorn.log"}
