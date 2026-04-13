from __future__ import annotations

import os

from src.config import Settings
from src.runtime import ensure_git_clone, ensure_venv, popen, run, uv_pip_install, write_text


def install(settings: Settings) -> dict:
    engine_dir = settings.engines_dir / "moss-tts-nano"
    repo_dir = engine_dir / "MOSS-TTS-Nano"
    engine_dir.mkdir(parents=True, exist_ok=True)
    ensure_git_clone("https://github.com/OpenMOSS/MOSS-TTS-Nano", repo_dir)

    python_bin = ensure_venv(engine_dir)

    uv_pip_install(
        python_bin,
        [
            "fastapi",
            "uvicorn",
            "soundfile",
            "numpy",
            "transformers",
            "torch",
            "torchaudio",
            "huggingface_hub",
            "librosa",
            "einops",
        ],
    )
    # Install the MOSS-TTS-Nano package itself. Use --no-deps to avoid the pynini /
    # WeTextProcessing chain which does not build cleanly under Colab's uv venv.
    uv_pip_install(
        python_bin,
        ["--no-deps", "-e", str(repo_dir)],
    )

    write_text(engine_dir / "app.py", settings.read_repo_text("src/apps/moss_tts_nano_app.py"))

    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "OPENAI_MODEL_ID": settings.openai_model_id or "moss-tts-nano",
        "MOSS_TTS_NANO_HF_MODEL": settings.moss_tts_nano_hf_model,
        "MOSS_TTS_NANO_MODE": settings.moss_tts_nano_mode,
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
        log_path=settings.log_dir / "moss-tts-nano-uvicorn.log",
    )
    return {
        "proc": proc,
        "app_dir": engine_dir,
        "log_path": settings.log_dir / "moss-tts-nano-uvicorn.log",
    }
