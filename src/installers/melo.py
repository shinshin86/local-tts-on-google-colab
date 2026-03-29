from __future__ import annotations

import os

from src.config import Settings
from src.runtime import ensure_git_clone, ensure_venv, popen, run, uv_pip_install, write_text


def install(settings: Settings) -> dict:
    repo_dir = settings.engines_dir / "MeloTTS"
    ensure_git_clone("https://github.com/myshell-ai/MeloTTS.git", repo_dir)
    python_bin = ensure_venv(repo_dir)
    uv_pip_install(
        python_bin,
        ["-e", ".", "fastapi", "uvicorn", "huggingface_hub"],
        cwd=str(repo_dir),
    )
    run([str(python_bin), "-m", "unidic", "download"], cwd=str(repo_dir))
    write_text(repo_dir / "openai_wrapper_app.py", settings.read_repo_text("src/apps/melo_app.py"))
    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "OPENAI_MODEL_ID": settings.openai_model_id or f"melotts-{settings.melo_language.lower()}",
        "MELO_LANGUAGE": settings.melo_language,
        "MELO_DEFAULT_VOICE": settings.melo_default_voice,
        "MELO_DEVICE": "auto",
    }
    proc = popen(
        [
            str(repo_dir / ".venv" / "bin" / "uvicorn"),
            "openai_wrapper_app:app",
            "--host",
            "0.0.0.0",
            "--port",
            str(settings.app_port),
            "--log-level",
            "info",
        ],
        cwd=str(repo_dir),
        env=env,
        log_path=settings.log_dir / "melo-uvicorn.log",
    )
    return {"proc": proc, "app_dir": repo_dir, "log_path": settings.log_dir / "melo-uvicorn.log"}
