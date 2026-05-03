from __future__ import annotations

import os
from pathlib import Path

from src.config import Settings
from src.runtime import popen, run, uv_pip_install, write_text


def _ensure_py312_venv(engine_dir: Path) -> Path:
    venv_dir = engine_dir / ".venv"
    if not venv_dir.exists():
        # vllm==0.7.3 transitively depends on xgrammar==0.1.11, which has wheels only
        # for cp39-cp312. Pin the venv to Python 3.12 so uv does not auto-fetch 3.13.
        run(["uv", "venv", "--python", "3.12", str(venv_dir)])
    return venv_dir / "bin" / "python"


def install(settings: Settings) -> dict:
    engine_dir = settings.engines_dir / "orpheus-tts"
    engine_dir.mkdir(parents=True, exist_ok=True)

    python_bin = _ensure_py312_venv(engine_dir)
    # orpheus-speech bundles vLLM; the upstream README pins vllm==0.7.3 because newer 0.7.x
    # had a March-18 regression that breaks Orpheus' streaming generator.
    uv_pip_install(
        python_bin,
        [
            "fastapi",
            "uvicorn",
            "numpy",
            "vllm==0.7.3",
            "orpheus-speech",
        ],
    )

    write_text(engine_dir / "app.py", settings.read_repo_text("src/apps/orpheus_app.py"))

    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "OPENAI_MODEL_ID": settings.openai_model_id or "orpheus-tts",
        "ORPHEUS_HF_MODEL": settings.orpheus_hf_model,
        "ORPHEUS_DEFAULT_VOICE": settings.orpheus_default_voice,
        "ORPHEUS_MAX_MODEL_LEN": str(settings.orpheus_max_model_len),
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
        log_path=settings.log_dir / "orpheus-tts-uvicorn.log",
    )
    return {
        "proc": proc,
        "app_dir": engine_dir,
        "log_path": settings.log_dir / "orpheus-tts-uvicorn.log",
    }
