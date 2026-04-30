from __future__ import annotations

import os

from src.config import Settings
from src.runtime import ensure_git_clone, ensure_venv, popen, uv_pip_install, write_text


DIA_REPO_URL = "https://github.com/nari-labs/dia.git"


def install(settings: Settings) -> dict:
    engine_dir = settings.engines_dir / "dia"
    engine_dir.mkdir(parents=True, exist_ok=True)

    repo_dir = engine_dir / "dia-repo"
    ensure_git_clone(DIA_REPO_URL, repo_dir)

    python_bin = ensure_venv(engine_dir)
    uv_pip_install(python_bin, ["-e", str(repo_dir)])
    uv_pip_install(python_bin, ["fastapi", "uvicorn", "soundfile", "numpy"])

    write_text(engine_dir / "app.py", settings.read_repo_text("src/apps/dia_app.py"))

    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "OPENAI_MODEL_ID": settings.openai_model_id or "dia",
        "DIA_HF_MODEL": settings.dia_hf_model,
        "DIA_COMPUTE_DTYPE": settings.dia_compute_dtype,
        "DIA_PROMPT_WAV": settings.dia_prompt_wav,
        "DIA_PROMPT_TEXT": settings.dia_prompt_text,
        "DIA_DEFAULT_VOICE": settings.dia_default_voice,
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
        log_path=settings.log_dir / "dia-uvicorn.log",
    )
    return {
        "proc": proc,
        "app_dir": engine_dir,
        "log_path": settings.log_dir / "dia-uvicorn.log",
    }
