from __future__ import annotations

import os
from pathlib import Path

from src.config import Settings
from src.runtime import ensure_git_clone, popen, run, uv_pip_install, write_text


MOSS_REPO_URL = "https://github.com/OpenMOSS/MOSS-TTS.git"
CU128_INDEX = "https://download.pytorch.org/whl/cu128"


def _ensure_py312_venv(engine_dir: Path) -> Path:
    venv_dir = engine_dir / ".venv"
    if not venv_dir.exists():
        # Upstream MOSS-TTS pins torch==2.9.1+cu128 / transformers==5.0.0 via
        # the [torch-runtime] extra and targets Python 3.12 in its README.
        run(["uv", "venv", "--python", "3.12", str(venv_dir)])
    return venv_dir / "bin" / "python"


def install(settings: Settings) -> dict:
    engine_dir = settings.engines_dir / "moss-tts-v1-5"
    repo_dir = engine_dir / "MOSS-TTS"
    engine_dir.mkdir(parents=True, exist_ok=True)
    ensure_git_clone(MOSS_REPO_URL, repo_dir)

    python_bin = _ensure_py312_venv(engine_dir)

    # Install the upstream `torch-runtime` extra. This pulls torch==2.9.1+cu128,
    # torchaudio, torchcodec, transformers==5.0.0 from the cu128 wheel index.
    uv_pip_install(
        python_bin,
        ["--extra-index-url", CU128_INDEX, "-e", f"{repo_dir}[torch-runtime]"],
    )
    uv_pip_install(python_bin, ["fastapi", "uvicorn", "soundfile"])

    write_text(engine_dir / "app.py", settings.read_repo_text("src/apps/moss_tts_v1_5_app.py"))

    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "OPENAI_MODEL_ID": settings.openai_model_id or "moss-tts-v1.5",
        "MOSS_TTS_V1_5_HF_MODEL": settings.moss_tts_v1_5_hf_model,
        "MOSS_TTS_V1_5_LANGUAGE": settings.moss_tts_v1_5_language,
        "MOSS_TTS_V1_5_PROMPT_WAV": settings.moss_tts_v1_5_prompt_wav,
        "MOSS_TTS_V1_5_ATTN_IMPL": settings.moss_tts_v1_5_attn_impl,
        "MOSS_TTS_V1_5_MAX_NEW_TOKENS": str(settings.moss_tts_v1_5_max_new_tokens),
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
        log_path=settings.log_dir / "moss-tts-v1-5-uvicorn.log",
    )
    return {
        "proc": proc,
        "app_dir": engine_dir,
        "log_path": settings.log_dir / "moss-tts-v1-5-uvicorn.log",
    }
