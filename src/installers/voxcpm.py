from __future__ import annotations

import os

from src.config import Settings
from src.runtime import ensure_venv, popen, uv_pip_install, write_text


def install(settings: Settings) -> dict:
    engine_dir = settings.engines_dir / "voxcpm"
    engine_dir.mkdir(parents=True, exist_ok=True)
    python_bin = ensure_venv(engine_dir)
    uv_pip_install(
        python_bin,
        [
            "fastapi",
            "uvicorn",
            "voxcpm",
            "soundfile",
            "numpy",
        ],
    )
    write_text(engine_dir / "app.py", settings.read_repo_text("src/apps/voxcpm_app.py"))
    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "OPENAI_MODEL_ID": settings.openai_model_id or "voxcpm2",
        "VOXCPM_HF_MODEL": settings.voxcpm_hf_model,
        "VOXCPM_CFG_VALUE": str(settings.voxcpm_cfg_value),
        "VOXCPM_INFERENCE_TIMESTEPS": str(settings.voxcpm_inference_timesteps),
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
        log_path=settings.log_dir / "voxcpm-uvicorn.log",
    )
    return {"proc": proc, "app_dir": engine_dir, "log_path": settings.log_dir / "voxcpm-uvicorn.log"}
