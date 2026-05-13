from __future__ import annotations

import os

from src.config import Settings
from src.runtime import ensure_venv, popen, uv_pip_install, write_text


def install(settings: Settings) -> dict:
    engine_dir = settings.engines_dir / "supertonic-openai"
    engine_dir.mkdir(parents=True, exist_ok=True)
    python_bin = ensure_venv(engine_dir)
    # Upstream supertonic-py pins numpy<2.0 (see requirements.txt).
    uv_pip_install(
        python_bin,
        [
            "fastapi",
            "uvicorn",
            "supertonic>=0.1.0",
            "soundfile",
            "numpy<2.0",
        ],
    )
    write_text(engine_dir / "app.py", settings.read_repo_text("src/apps/supertonic_app.py"))
    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "OPENAI_MODEL_ID": settings.openai_model_id or "supertonic-3",
        "SUPERTONIC_MODEL": settings.supertonic_model,
        "SUPERTONIC_DEFAULT_VOICE": settings.supertonic_default_voice,
        "SUPERTONIC_DEFAULT_LANG": settings.supertonic_default_lang,
        "SUPERTONIC_TOTAL_STEPS": str(settings.supertonic_total_steps),
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
        log_path=settings.log_dir / "supertonic-uvicorn.log",
    )
    return {"proc": proc, "app_dir": engine_dir, "log_path": settings.log_dir / "supertonic-uvicorn.log"}
