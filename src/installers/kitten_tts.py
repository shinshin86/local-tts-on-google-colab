from __future__ import annotations

import os

from src.config import Settings
from src.runtime import ensure_venv, popen, uv_pip_install, write_text


KITTEN_TTS_WHEEL = (
    "https://github.com/KittenML/KittenTTS/releases/download/0.8.1/"
    "kittentts-0.8.1-py3-none-any.whl"
)


def install(settings: Settings) -> dict:
    engine_dir = settings.engines_dir / "kitten-tts-openai"
    engine_dir.mkdir(parents=True, exist_ok=True)
    python_bin = ensure_venv(engine_dir)
    uv_pip_install(
        python_bin,
        [KITTEN_TTS_WHEEL, "fastapi", "uvicorn", "soundfile"],
    )
    write_text(engine_dir / "app.py", settings.read_repo_text("src/apps/kitten_tts_app.py"))
    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        # KittenTTS is intentionally served through its CPU ONNX backend even
        # when the surrounding Colab runtime has a GPU attached.
        "CUDA_VISIBLE_DEVICES": "",
        "OPENAI_MODEL_ID": settings.openai_model_id or "kitten-tts",
        "KITTEN_TTS_HF_MODEL": settings.kitten_tts_hf_model,
        "KITTEN_TTS_DEFAULT_VOICE": settings.kitten_tts_default_voice,
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
        log_path=settings.log_dir / "kitten-tts-uvicorn.log",
    )
    return {
        "proc": proc,
        "app_dir": engine_dir,
        "log_path": settings.log_dir / "kitten-tts-uvicorn.log",
    }
