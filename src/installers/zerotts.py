from __future__ import annotations

import os

from src.config import Settings
from src.runtime import ensure_venv, popen, uv_pip_install, write_text


def install(settings: Settings) -> dict:
    engine_dir = settings.engines_dir / "zerotts"
    engine_dir.mkdir(parents=True, exist_ok=True)
    python_bin = ensure_venv(engine_dir)
    uv_pip_install(python_bin, ["fastapi", "uvicorn", "zerotts"])
    write_text(engine_dir / "app.py", settings.read_repo_text("src/apps/zerotts_app.py"))
    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "OPENAI_MODEL_ID": settings.openai_model_id or "zerotts",
        "ZEROTTS_HF_MODEL": settings.zerotts_hf_model,
        "ZEROTTS_DEFAULT_VOICE": settings.zerotts_default_voice,
        "ZEROTTS_CFG_SCALE": str(settings.zerotts_cfg_scale),
        "ZEROTTS_AUDIO_TEMPERATURE": str(settings.zerotts_audio_temperature),
        "ZEROTTS_AUDIO_TOPK": str(settings.zerotts_audio_topk),
        "ZEROTTS_AUDIO_TOPP": str(settings.zerotts_audio_topp),
        "ZEROTTS_AUDIO_REPETITION_PENALTY": str(settings.zerotts_audio_repetition_penalty),
    }
    log_path = settings.log_dir / "zerotts-uvicorn.log"
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
        log_path=log_path,
    )
    return {"proc": proc, "app_dir": engine_dir, "log_path": log_path}
