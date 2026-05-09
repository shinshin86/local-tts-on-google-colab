from __future__ import annotations

import os
from pathlib import Path

from src.config import Settings
from src.runtime import popen, run, uv_pip_install, write_text


def _ensure_py311_venv(engine_dir: Path) -> Path:
    venv_dir = engine_dir / ".venv"
    if not venv_dir.exists():
        # The pip-published `styletts2` (sidharthrajaram/StyleTTS2) was last released
        # 2024-01 and pins to older transformers/torch. A dedicated venv keeps these
        # legacy pins from polluting other engines.
        run(["uv", "venv", "--python", "3.11", str(venv_dir)])
    return venv_dir / "bin" / "python"


def install(settings: Settings) -> dict:
    engine_dir = settings.engines_dir / "styletts2"
    engine_dir.mkdir(parents=True, exist_ok=True)

    python_bin = _ensure_py311_venv(engine_dir)

    # Use sidharthrajaram's MIT-licensed pip package, which substitutes gruut (MIT)
    # for the upstream phonemizer (GPL-3.0) and auto-downloads the LibriTTS
    # checkpoint from Hugging Face.
    uv_pip_install(
        python_bin,
        [
            "fastapi",
            "uvicorn",
            "soundfile",
            "numpy",
            "styletts2",
        ],
    )

    write_text(engine_dir / "app.py", settings.read_repo_text("src/apps/styletts2_app.py"))

    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "OPENAI_MODEL_ID": settings.openai_model_id or "styletts2",
        "STYLETTS2_DEFAULT_VOICE": settings.styletts2_default_voice,
        "STYLETTS2_PROMPT_WAV": settings.styletts2_prompt_wav,
        "STYLETTS2_ALPHA": str(settings.styletts2_alpha),
        "STYLETTS2_BETA": str(settings.styletts2_beta),
        "STYLETTS2_DIFFUSION_STEPS": str(settings.styletts2_diffusion_steps),
        "STYLETTS2_EMBEDDING_SCALE": str(settings.styletts2_embedding_scale),
        # Colab sets MPLBACKEND to its inline backend, which isn't installed in
        # this venv. styletts2.utils imports matplotlib at module load time, so
        # force a headless backend before that import happens.
        "MPLBACKEND": "Agg",
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
        log_path=settings.log_dir / "styletts2-uvicorn.log",
    )
    return {
        "proc": proc,
        "app_dir": engine_dir,
        "log_path": settings.log_dir / "styletts2-uvicorn.log",
    }
