from __future__ import annotations

import os

from src.config import Settings
from src.runtime import ensure_venv, popen, uv_pip_install, write_text


def install(settings: Settings) -> dict:
    engine_dir = settings.engines_dir / "phonon"
    engine_dir.mkdir(parents=True, exist_ok=True)
    python_bin = ensure_venv(engine_dir)
    uv_pip_install(python_bin, ["fastapi", "uvicorn", "soundfile", "ptts==0.7.1"])

    write_text(engine_dir / "app.py", settings.read_repo_text("src/apps/phonon_app.py"))
    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "OPENAI_MODEL_ID": settings.openai_model_id or "phonon",
        "PHONON_TEMPERATURE": str(settings.phonon_temperature),
    }
    log_path = settings.log_dir / "phonon-uvicorn.log"
    proc = popen(
        [
            str(python_bin), "-m", "uvicorn", "app:app",
            "--host", "0.0.0.0", "--port", str(settings.app_port), "--log-level", "info",
        ],
        cwd=str(engine_dir),
        env=env,
        log_path=log_path,
    )
    return {"proc": proc, "app_dir": engine_dir, "log_path": log_path}
