from __future__ import annotations

import os

from src.config import Settings
from src.runtime import ensure_venv, popen, uv_pip_install, write_text


def install(settings: Settings) -> dict:
    engine_dir = settings.engines_dir / "kyutai-tts"
    engine_dir.mkdir(parents=True, exist_ok=True)

    python_bin = ensure_venv(engine_dir)
    uv_pip_install(
        python_bin,
        [
            "fastapi",
            "uvicorn",
            "soundfile",
            "numpy",
            "huggingface_hub",
            "moshi>=0.2.11",
            "sphn",
        ],
    )

    write_text(engine_dir / "app.py", settings.read_repo_text("src/apps/kyutai_tts_app.py"))

    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "OPENAI_MODEL_ID": settings.openai_model_id or "kyutai-tts",
        "KYUTAI_HF_REPO": settings.kyutai_hf_repo,
        "KYUTAI_VOICE_REPO": settings.kyutai_voice_repo,
        "KYUTAI_VOICE": settings.kyutai_voice,
        "KYUTAI_PROMPT_WAV": settings.kyutai_prompt_wav,
        "KYUTAI_DEFAULT_VOICE": settings.kyutai_default_voice,
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
        log_path=settings.log_dir / "kyutai-tts-uvicorn.log",
    )
    return {
        "proc": proc,
        "app_dir": engine_dir,
        "log_path": settings.log_dir / "kyutai-tts-uvicorn.log",
    }
