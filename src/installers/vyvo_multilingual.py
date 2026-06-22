from __future__ import annotations

import os
from pathlib import Path

from src.config import Settings
from src.runtime import popen, run, uv_pip_install, write_text


CU128_INDEX = "https://download.pytorch.org/whl/cu128"


def _ensure_py312_venv(engine_dir: Path) -> Path:
    venv_dir = engine_dir / ".venv"
    if not venv_dir.exists():
        # Vyvo ships no upstream package — inference is the plain transformers
        # snippet from the model card (Qwen3 backbone + kyutai/mimi codec). 3.12
        # matches Colab's system interpreter and the recent transformers engines.
        run(["uv", "venv", "--python", "3.12", str(venv_dir)])
    return venv_dir / "bin" / "python"


def install(settings: Settings) -> dict:
    engine_dir = settings.engines_dir / "vyvo-multilingual"
    engine_dir.mkdir(parents=True, exist_ok=True)

    python_bin = _ensure_py312_venv(engine_dir)

    # The model card uses the new `dtype=` kwarg and the Qwen3 + MimiModel
    # classes, so we need a recent transformers. Reuse the MOSS-proven stack
    # (torch==2.9.1+cu128 / transformers==5.0.0), which is verified on Colab L4.
    # `--index-strategy unsafe-best-match` lets uv fall back to pypi for packages
    # the pytorch wheel index only partially mirrors.
    uv_pip_install(
        python_bin,
        [
            "--index-strategy",
            "unsafe-best-match",
            "--extra-index-url",
            CU128_INDEX,
            "torch==2.9.1",
            "torchaudio==2.9.1",
            "transformers==5.0.0",
            "accelerate",
            "soundfile",
            "fastapi",
            "uvicorn",
        ],
    )

    write_text(
        engine_dir / "app.py",
        settings.read_repo_text("src/apps/vyvo_multilingual_app.py"),
    )

    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "HF_HUB_ETAG_TIMEOUT": "60",
        "HF_HUB_DOWNLOAD_TIMEOUT": "60",
        "OPENAI_MODEL_ID": settings.openai_model_id or "vyvo-multilingual",
        "VYVO_HF_MODEL": settings.vyvo_hf_model,
        "VYVO_MIMI_REPO": settings.vyvo_mimi_repo,
        "VYVO_DEFAULT_VOICE": settings.vyvo_default_voice,
        "VYVO_PROMPT_WAV": settings.vyvo_prompt_wav,
        "VYVO_PROMPT_TEXT": settings.vyvo_prompt_text,
        "VYVO_TEMPERATURE": str(settings.vyvo_temperature),
        "VYVO_TOP_P": str(settings.vyvo_top_p),
        "VYVO_REPETITION_PENALTY": str(settings.vyvo_repetition_penalty),
        "VYVO_MAX_NEW_TOKENS": str(settings.vyvo_max_new_tokens),
        "VYVO_MIN_NEW_TOKENS": str(settings.vyvo_min_new_tokens),
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
        log_path=settings.log_dir / "vyvo-multilingual-uvicorn.log",
    )
    return {
        "proc": proc,
        "app_dir": engine_dir,
        "log_path": settings.log_dir / "vyvo-multilingual-uvicorn.log",
    }
