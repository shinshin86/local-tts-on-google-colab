from __future__ import annotations

import os
from pathlib import Path

from src.config import Settings
from src.runtime import ensure_venv, popen, run, uv_pip_install, write_text


VIBEVOICE_REPO_URL = "https://github.com/microsoft/VibeVoice.git"
VIBEVOICE_REPO_REF = "94da20d98b2fa7688e9cbfaf7692ddb4954f7600"
VIBEVOICE_MODEL_ID = "microsoft/VibeVoice-Realtime-0.5B"
VIBEVOICE_MODEL_REF = "6bce5f06044837fe6d2c5d7a71a84f0416bd57e4"


def _ensure_repo(repo_dir: Path) -> None:
    if not repo_dir.exists():
        run(["git", "clone", VIBEVOICE_REPO_URL, str(repo_dir)])
    run(["git", "fetch", "origin", VIBEVOICE_REPO_REF], cwd=str(repo_dir))
    run(["git", "checkout", "--detach", VIBEVOICE_REPO_REF], cwd=str(repo_dir))


def install(settings: Settings) -> dict:
    engine_dir = settings.engines_dir / "vibevoice-realtime"
    engine_dir.mkdir(parents=True, exist_ok=True)

    repo_dir = engine_dir / "VibeVoice"
    _ensure_repo(repo_dir)

    python_bin = ensure_venv(engine_dir)
    uv_pip_install(python_bin, ["-e", f"{repo_dir}[streamingtts]"])
    uv_pip_install(python_bin, ["fastapi", "uvicorn", "soundfile", "numpy"])
    write_text(engine_dir / "app.py", settings.read_repo_text("src/apps/vibevoice_app.py"))

    model_revision = VIBEVOICE_MODEL_REF if settings.vibevoice_hf_model == VIBEVOICE_MODEL_ID else ""
    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "OPENAI_MODEL_ID": settings.openai_model_id or "vibevoice-realtime",
        "VIBEVOICE_HF_MODEL": settings.vibevoice_hf_model,
        "VIBEVOICE_HF_REVISION": model_revision,
        "VIBEVOICE_VOICES_DIR": str(repo_dir / "demo" / "voices" / "streaming_model"),
        "VIBEVOICE_DEFAULT_SPEAKER": settings.vibevoice_default_speaker,
        "VIBEVOICE_DDPM_STEPS": str(settings.vibevoice_ddpm_steps),
        "VIBEVOICE_CFG_SCALE": str(settings.vibevoice_cfg_scale),
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
        log_path=settings.log_dir / "vibevoice-realtime-uvicorn.log",
    )
    return {
        "proc": proc,
        "app_dir": engine_dir,
        "log_path": settings.log_dir / "vibevoice-realtime-uvicorn.log",
    }
