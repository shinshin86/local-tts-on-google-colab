from __future__ import annotations

import os

from src.config import Settings
from src.runtime import ensure_venv, popen, uv_pip_install, write_text


def install(settings: Settings) -> dict:
    engine_dir = settings.engines_dir / "bark"
    engine_dir.mkdir(parents=True, exist_ok=True)

    python_bin = ensure_venv(engine_dir)
    uv_pip_install(
        python_bin,
        [
            "fastapi",
            "uvicorn",
            "soundfile",
            "numpy",
            # Upstream README documents `pip install git+https://github.com/suno-ai/bark.git`
            # but the package is also published on PyPI as `bark`.
            "git+https://github.com/suno-ai/bark.git",
        ],
    )

    write_text(engine_dir / "app.py", settings.read_repo_text("src/apps/bark_app.py"))

    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "OPENAI_MODEL_ID": settings.openai_model_id or "bark",
        "BARK_DEFAULT_VOICE": settings.bark_default_voice,
        "BARK_USE_SMALL_MODELS": "1" if settings.bark_use_small_models else "0",
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
        log_path=settings.log_dir / "bark-uvicorn.log",
    )
    return {
        "proc": proc,
        "app_dir": engine_dir,
        "log_path": settings.log_dir / "bark-uvicorn.log",
    }
