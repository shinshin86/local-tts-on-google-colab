from __future__ import annotations

import os
from pathlib import Path

from src.config import Settings
from src.runtime import popen, run, uv_pip_install, write_text


HIGGS_REPO_URL = "https://github.com/boson-ai/higgs-audio.git"


def _ensure_py310_venv(engine_dir: Path) -> Path:
    venv_dir = engine_dir / ".venv"
    if not venv_dir.exists():
        # Upstream README documents Python 3.10 as the supported environment
        # (NVIDIA pytorch:25.02-py3 container). Pin 3.10 for tested wheel coverage.
        run(["uv", "venv", "--python", "3.10", str(venv_dir)])
    return venv_dir / "bin" / "python"


def install(settings: Settings) -> dict:
    engine_dir = settings.engines_dir / "higgs-audio"
    engine_dir.mkdir(parents=True, exist_ok=True)

    repo_dir = engine_dir / "higgs-audio"
    if not repo_dir.exists():
        run(["git", "clone", HIGGS_REPO_URL, str(repo_dir)])

    python_bin = _ensure_py310_venv(engine_dir)

    uv_pip_install(python_bin, ["-r", str(repo_dir / "requirements.txt")])
    # `pip install -e .` exposes the boson_multimodal package on the venv.
    uv_pip_install(python_bin, ["-e", str(repo_dir)])
    uv_pip_install(python_bin, ["fastapi", "uvicorn", "soundfile"])

    write_text(engine_dir / "app.py", settings.read_repo_text("src/apps/higgs_audio_app.py"))

    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "OPENAI_MODEL_ID": settings.openai_model_id or "higgs-audio-v2",
        "HIGGS_REPO_DIR": str(repo_dir),
        "HIGGS_HF_MODEL": settings.higgs_hf_model,
        "HIGGS_HF_TOKENIZER": settings.higgs_hf_tokenizer,
        "HIGGS_DEFAULT_VOICE": settings.higgs_default_voice,
        "HIGGS_DEFAULT_REF_VOICE": settings.higgs_default_ref_voice,
        "HIGGS_PROMPT_WAV": settings.higgs_prompt_wav,
        "HIGGS_PROMPT_TEXT": settings.higgs_prompt_text,
        "HIGGS_MAX_NEW_TOKENS": str(settings.higgs_max_new_tokens),
        "HIGGS_TEMPERATURE": str(settings.higgs_temperature),
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
        log_path=settings.log_dir / "higgs-audio-uvicorn.log",
    )
    return {
        "proc": proc,
        "app_dir": engine_dir,
        "log_path": settings.log_dir / "higgs-audio-uvicorn.log",
    }
