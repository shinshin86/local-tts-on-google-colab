from __future__ import annotations

import os
from pathlib import Path

from src.config import Settings
from src.runtime import ensure_venv, popen, run, uv_pip_install, write_text


DRAMABOX_REPO_URL = "https://github.com/resemble-ai/DramaBox.git"


def install(settings: Settings) -> dict:
    engine_dir = settings.engines_dir / "dramabox"
    engine_dir.mkdir(parents=True, exist_ok=True)

    repo_dir = engine_dir / "DramaBox"
    if not repo_dir.exists():
        run(["git", "clone", DRAMABOX_REPO_URL, str(repo_dir)])

    python_bin = ensure_venv(engine_dir)
    uv_pip_install(python_bin, ["-r", str(repo_dir / "requirements.txt")])
    uv_pip_install(python_bin, ["fastapi", "uvicorn", "soundfile"])

    write_text(engine_dir / "app.py", settings.read_repo_text("src/apps/dramabox_app.py"))

    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "OPENAI_MODEL_ID": settings.openai_model_id or "dramabox",
        "DRAMABOX_REPO_DIR": str(repo_dir),
        "DRAMABOX_HF_MODEL": settings.dramabox_hf_model,
        "DRAMABOX_GEMMA_REPO": settings.dramabox_gemma_repo,
        "DRAMABOX_DEFAULT_VOICE": settings.dramabox_default_voice,
        "DRAMABOX_DEFAULT_REF_VOICE": settings.dramabox_default_ref_voice,
        "DRAMABOX_PROMPT_WAV": settings.dramabox_prompt_wav,
        "DRAMABOX_DTYPE": settings.dramabox_dtype,
        "DRAMABOX_CFG_SCALE": str(settings.dramabox_cfg_scale),
        "DRAMABOX_STG_SCALE": str(settings.dramabox_stg_scale),
        "DRAMABOX_DURATION_MULTIPLIER": str(settings.dramabox_duration_multiplier),
        "DRAMABOX_SEED": str(settings.dramabox_seed),
        "DRAMABOX_COMPILE": "1" if settings.dramabox_compile else "0",
        "DRAMABOX_BNB_4BIT": "1" if settings.dramabox_bnb_4bit else "0",
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
        log_path=settings.log_dir / "dramabox-uvicorn.log",
    )
    return {
        "proc": proc,
        "app_dir": engine_dir,
        "log_path": settings.log_dir / "dramabox-uvicorn.log",
    }
