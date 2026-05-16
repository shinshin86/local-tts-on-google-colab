from __future__ import annotations

import os

from src.config import Settings
from src.runtime import ensure_git_clone, popen, run, uv_pip_install, write_text


def install(settings: Settings) -> dict:
    repo_dir = settings.engines_dir / "Irodori-TTS"
    ensure_git_clone("https://github.com/Aratako/Irodori-TTS", repo_dir)
    write_text(repo_dir / "app.py", settings.read_repo_text("src/apps/irodori_app.py"))
    run(["uv", "sync"], cwd=str(repo_dir))
    python_bin = repo_dir / ".venv" / "bin" / "python"
    uv_pip_install(
        python_bin,
        ["fastapi", "uvicorn", "huggingface_hub"],
        cwd=str(repo_dir),
    )
    uv_pip_install(
        python_bin,
        ["git+https://github.com/facebookresearch/dacvae.git"],
        cwd=str(repo_dir),
    )
    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "IRODORI_HF_CHECKPOINT": settings.irodori_hf_checkpoint,
        "IRODORI_CODEC_REPO": settings.irodori_codec_repo,
        "IRODORI_MODEL_PRECISION": settings.irodori_model_precision,
        "IRODORI_CODEC_PRECISION": settings.irodori_codec_precision,
        "OPENAI_MODEL_ID": settings.openai_model_id or settings.irodori_hf_checkpoint,
    }
    proc = popen(
        [
            str(repo_dir / ".venv" / "bin" / "uvicorn"),
            "app:app",
            "--host",
            "0.0.0.0",
            "--port",
            str(settings.app_port),
            "--log-level",
            "debug",
            "--access-log",
        ],
        cwd=str(repo_dir),
        env=env,
        log_path=settings.log_dir / "irodori-uvicorn.log",
    )
    return {"proc": proc, "app_dir": repo_dir, "log_path": settings.log_dir / "irodori-uvicorn.log"}
