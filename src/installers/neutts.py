from __future__ import annotations

import os

from src.config import Settings
from src.runtime import ensure_git_clone, ensure_venv, popen, uv_pip_install, write_text


def install(settings: Settings) -> dict:
    engine_dir = settings.engines_dir / "neutts"
    repo_dir = engine_dir / "neutts-repo"
    engine_dir.mkdir(parents=True, exist_ok=True)
    # Clone the upstream NeuTTS repo only to obtain the bundled reference samples
    # (dave/jo: EN, mateo: ES, greta: DE, juliette: FR). The package itself is
    # installed via pip below.
    ensure_git_clone("https://github.com/neuphonic/neutts.git", repo_dir)

    python_bin = ensure_venv(engine_dir)
    uv_pip_install(
        python_bin,
        [
            "fastapi",
            "uvicorn",
            "soundfile",
            "numpy",
            "neutts",
        ],
    )

    write_text(engine_dir / "app.py", settings.read_repo_text("src/apps/neutts_app.py"))

    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "OPENAI_MODEL_ID": settings.openai_model_id or "neutts",
        "NEUTTS_BACKBONE_REPO": settings.neutts_backbone_repo,
        "NEUTTS_CODEC_REPO": settings.neutts_codec_repo,
        "NEUTTS_DEFAULT_VOICE": settings.neutts_default_voice,
        "NEUTTS_SAMPLES_DIR": str(repo_dir / "samples"),
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
        log_path=settings.log_dir / "neutts-uvicorn.log",
    )
    return {
        "proc": proc,
        "app_dir": engine_dir,
        "log_path": settings.log_dir / "neutts-uvicorn.log",
    }
