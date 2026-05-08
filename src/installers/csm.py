from __future__ import annotations

import os
from pathlib import Path

from src.config import Settings
from src.runtime import popen, run, uv_pip_install, write_text


CSM_REPO_URL = "https://github.com/SesameAILabs/csm.git"


def _ensure_py311_venv(engine_dir: Path) -> Path:
    venv_dir = engine_dir / ".venv"
    if not venv_dir.exists():
        # Upstream requirements pin torch==2.4.0 / torchtune==0.4.0 / torchao==0.9.0,
        # which only ship cp39-cp312 wheels. Default Colab Python 3.12 is fine, but
        # uv's default may auto-fetch 3.13 if the runtime advertises it. Pin 3.11
        # for the widest wheel coverage across torch's ecosystem.
        run(["uv", "venv", "--python", "3.11", str(venv_dir)])
    return venv_dir / "bin" / "python"


def install(settings: Settings) -> dict:
    engine_dir = settings.engines_dir / "csm"
    engine_dir.mkdir(parents=True, exist_ok=True)

    repo_dir = engine_dir / "csm"
    if not repo_dir.exists():
        run(["git", "clone", CSM_REPO_URL, str(repo_dir)])

    python_bin = _ensure_py311_venv(engine_dir)

    uv_pip_install(python_bin, ["-r", str(repo_dir / "requirements.txt")])
    uv_pip_install(python_bin, ["fastapi", "uvicorn", "soundfile"])

    write_text(engine_dir / "app.py", settings.read_repo_text("src/apps/csm_app.py"))

    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        # Upstream README requires this env to disable torch.compile for stability on
        # CUDA 12.4 / 12.6 (default Colab toolchain).
        "NO_TORCH_COMPILE": "1",
        "PYTHONPATH": str(repo_dir),
        "OPENAI_MODEL_ID": settings.openai_model_id or "csm-1b",
        "CSM_REPO_DIR": str(repo_dir),
        "CSM_HF_MODEL": settings.csm_hf_model,
        "CSM_LLAMA_MODEL": settings.csm_llama_model,
        "CSM_DEFAULT_VOICE": settings.csm_default_voice,
        "CSM_DEFAULT_SPEAKER": str(settings.csm_default_speaker),
        "CSM_MAX_AUDIO_LENGTH_MS": str(settings.csm_max_audio_length_ms),
        "CSM_TEMPERATURE": str(settings.csm_temperature),
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
        log_path=settings.log_dir / "csm-uvicorn.log",
    )
    return {
        "proc": proc,
        "app_dir": engine_dir,
        "log_path": settings.log_dir / "csm-uvicorn.log",
    }
