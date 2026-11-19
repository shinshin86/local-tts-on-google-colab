from __future__ import annotations

import os
from pathlib import Path

from src.config import Settings
from src.runtime import ensure_venv, popen, run, uv_pip_install, write_text


MODEL_ID = "Zyphra/ZONOS3-Mini-hf"
# Pin the snapshot because Transformers loads Python code from it.
MODEL_REVISION = "e39d30df560125d9c434acd5cb0e83e4c2d64e0d"


def install(settings: Settings) -> dict:
    engine_dir = settings.engines_dir / "zonos3-mini"
    engine_dir.mkdir(parents=True, exist_ok=True)
    python_bin = ensure_venv(engine_dir)

    uv_pip_install(
        python_bin,
        ["--index-url", "https://download.pytorch.org/whl/cu128", "torch==2.9.1+cu128"],
    )
    uv_pip_install(
        python_bin,
        ["fastapi", "uvicorn", "soundfile", "transformers>=5,<6", "huggingface_hub", "zyphra-textnorm"],
    )

    # Download all runtime assets before the launcher's HTTP readiness timer.
    model_dir = engine_dir / "models" / MODEL_REVISION
    run(
        [
            str(python_bin),
            "-c",
            "import sys; from huggingface_hub import snapshot_download; "
            "snapshot_download(sys.argv[1], revision=sys.argv[2], local_dir=sys.argv[3])",
            MODEL_ID,
            MODEL_REVISION,
            str(model_dir),
        ]
    )
    write_text(engine_dir / "app.py", settings.read_repo_text("src/apps/zonos3_mini_app.py"))

    prompt_wav = settings.zonos3_mini_prompt_wav
    if prompt_wav:
        prompt_wav = str(Path(prompt_wav).expanduser().resolve())

    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "OPENAI_MODEL_ID": settings.openai_model_id or "zonos3-mini",
        "ZONOS3_MINI_MODEL_DIR": str(model_dir),
        "ZONOS3_MINI_LANGUAGE": settings.zonos3_mini_language,
        "ZONOS3_MINI_PROMPT_WAV": prompt_wav,
        "ZONOS3_MINI_DEFAULT_VOICE": settings.zonos3_mini_default_voice,
        "ZONOS3_MINI_SPEAKING_RATE": str(settings.zonos3_mini_speaking_rate),
        "ZONOS3_MINI_SPEAKER_CFG_SCALE": str(settings.zonos3_mini_speaker_cfg_scale),
    }
    log_path = settings.log_dir / "zonos3-mini-uvicorn.log"
    proc = popen(
        [
            str(python_bin), "-m", "uvicorn", "app:app",
            "--host", "0.0.0.0", "--port", str(settings.app_port), "--log-level", "info",
        ],
        cwd=str(engine_dir),
        env=env,
        log_path=log_path,
    )
    return {"proc": proc, "app_dir": engine_dir, "log_path": log_path}
