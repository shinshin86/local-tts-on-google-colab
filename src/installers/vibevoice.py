from __future__ import annotations

import os

from src.config import Settings
from src.runtime import ensure_git_clone, ensure_venv, popen, uv_pip_install, write_text


VIBEVOICE_REPO_URL = "https://github.com/microsoft/VibeVoice.git"


def install(settings: Settings) -> dict:
    engine_dir = settings.engines_dir / "vibevoice"
    engine_dir.mkdir(parents=True, exist_ok=True)

    repo_dir = engine_dir / "VibeVoice"
    ensure_git_clone(VIBEVOICE_REPO_URL, repo_dir)

    python_bin = ensure_venv(engine_dir)
    # The vibevoice package is installed editable from the cloned repo so the
    # bundled `demo/voices/*.wav` reference samples land alongside the code.
    uv_pip_install(python_bin, ["-e", str(repo_dir)])
    uv_pip_install(python_bin, ["fastapi", "uvicorn", "soundfile", "numpy"])

    write_text(engine_dir / "app.py", settings.read_repo_text("src/apps/vibevoice_app.py"))

    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "OPENAI_MODEL_ID": settings.openai_model_id or "vibevoice",
        "VIBEVOICE_HF_MODEL": settings.vibevoice_hf_model,
        "VIBEVOICE_VOICES_DIR": str(repo_dir / "demo" / "voices"),
        "VIBEVOICE_DEFAULT_SPEAKER": settings.vibevoice_default_speaker,
        "VIBEVOICE_PROMPT_WAV": settings.vibevoice_prompt_wav,
        "VIBEVOICE_DEFAULT_VOICE": settings.vibevoice_default_voice,
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
        log_path=settings.log_dir / "vibevoice-uvicorn.log",
    )
    return {
        "proc": proc,
        "app_dir": engine_dir,
        "log_path": settings.log_dir / "vibevoice-uvicorn.log",
    }
