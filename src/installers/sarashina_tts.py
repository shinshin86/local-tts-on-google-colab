from __future__ import annotations

import os

from src.config import Settings
from src.runtime import ensure_git_clone, ensure_venv, popen, uv_pip_install, write_text


SARASHINA_REPO_URL = "https://github.com/sbintuitions/sarashina2.2-tts.git"


def install(settings: Settings) -> dict:
    engine_dir = settings.engines_dir / "sarashina-tts"
    engine_dir.mkdir(parents=True, exist_ok=True)

    repo_dir = engine_dir / "sarashina2.2-tts"
    ensure_git_clone(SARASHINA_REPO_URL, repo_dir)

    python_bin = ensure_venv(engine_dir)

    # Install the upstream package as editable. pyproject pins core deps
    # (torch<=2.9, transformers, librosa, s3tokenizer, diffusers, silentcipher git, ...).
    install_spec = f"-e {repo_dir}"
    if settings.sarashina_use_vllm:
        install_spec = f"-e {repo_dir}[vllm]"
    uv_pip_install(python_bin, [install_spec])
    uv_pip_install(python_bin, ["fastapi", "uvicorn"])

    write_text(engine_dir / "app.py", settings.read_repo_text("src/apps/sarashina_tts_app.py"))

    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "OPENAI_MODEL_ID": settings.openai_model_id or "sarashina-tts",
        "SARASHINA_HF_MODEL": settings.sarashina_hf_model,
        "SARASHINA_USE_VLLM": "1" if settings.sarashina_use_vllm else "0",
        "SARASHINA_PROMPT_WAV": settings.sarashina_prompt_wav,
        "SARASHINA_PROMPT_TEXT": settings.sarashina_prompt_text,
        "SARASHINA_DEFAULT_VOICE": settings.sarashina_default_voice,
        "SARASHINA_MODEL_DIR": str(engine_dir / "pretrained_models"),
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
        log_path=settings.log_dir / "sarashina-tts-uvicorn.log",
    )
    return {
        "proc": proc,
        "app_dir": engine_dir,
        "log_path": settings.log_dir / "sarashina-tts-uvicorn.log",
    }
