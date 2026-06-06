from __future__ import annotations

import os
from pathlib import Path

from src.config import Settings
from src.runtime import popen, run, uv_pip_install, write_text


def _ensure_py312_venv(engine_dir: Path) -> Path:
    venv_dir = engine_dir / ".venv"
    if not venv_dir.exists():
        # liquid-audio declares requires-python >=3.12 and pins torch>=2.8.0
        # (cp312 wheels). 3.12 matches Colab's system interpreter and the recent
        # engines (dots.tts / Higgs v3).
        run(["uv", "venv", "--python", "3.12", str(venv_dir)])
    return venv_dir / "bin" / "python"


def install(settings: Settings) -> dict:
    engine_dir = settings.engines_dir / "lfm2-audio-jp"
    engine_dir.mkdir(parents=True, exist_ok=True)

    python_bin = _ensure_py312_venv(engine_dir)

    # liquid-audio ships on PyPI as a pure-Python package (src layout). It pulls
    # torch>=2.8 / transformers / librosa / sentencepiece / accelerate. flash-attn
    # is optional — the model falls back to torch SDPA when it is absent, so we do
    # NOT install it (avoids a from-source CUDA build on Colab).
    uv_pip_install(python_bin, ["liquid-audio"])
    # fastapi/uvicorn are not liquid-audio deps; soundfile comes via librosa but
    # pin it explicitly so the in-process wrapper never breaks.
    uv_pip_install(python_bin, ["fastapi", "uvicorn", "soundfile"])

    write_text(
        engine_dir / "app.py",
        settings.read_repo_text("src/apps/lfm2_audio_jp_app.py"),
    )

    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "HF_HUB_ETAG_TIMEOUT": "60",
        "HF_HUB_DOWNLOAD_TIMEOUT": "60",
        "OPENAI_MODEL_ID": settings.openai_model_id or "lfm2-audio-jp",
        "LFM2_AUDIO_JP_HF_MODEL": settings.lfm2_audio_jp_hf_model,
        "LFM2_AUDIO_JP_SYSTEM_PROMPT": settings.lfm2_audio_jp_system_prompt,
        "LFM2_AUDIO_JP_MAX_NEW_TOKENS": str(settings.lfm2_audio_jp_max_new_tokens),
        "LFM2_AUDIO_JP_AUDIO_TEMPERATURE": str(settings.lfm2_audio_jp_audio_temperature),
        "LFM2_AUDIO_JP_AUDIO_TOP_K": str(settings.lfm2_audio_jp_audio_top_k),
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
        log_path=settings.log_dir / "lfm2-audio-jp-uvicorn.log",
    )
    return {
        "proc": proc,
        "app_dir": engine_dir,
        "log_path": settings.log_dir / "lfm2-audio-jp-uvicorn.log",
    }
