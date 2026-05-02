from __future__ import annotations

import os
from pathlib import Path

from src.config import Settings
from src.runtime import popen, run, uv_pip_install, write_text


COSYVOICE_REPO_URL = "https://github.com/FunAudioLLM/CosyVoice.git"


def _ensure_py310_venv(engine_dir: Path) -> Path:
    venv_dir = engine_dir / ".venv"
    if not venv_dir.exists():
        # Upstream requirements pin torch==2.3.1 / openai-whisper==20231117 etc.
        # which do not resolve under Colab's default Python 3.12. Force 3.10.
        run(["uv", "venv", "--python", "3.10", str(venv_dir)])
    return venv_dir / "bin" / "python"


def install(settings: Settings) -> dict:
    engine_dir = settings.engines_dir / "cosyvoice2"
    engine_dir.mkdir(parents=True, exist_ok=True)

    repo_dir = engine_dir / "CosyVoice"
    if not repo_dir.exists():
        run(["git", "clone", "--recursive", COSYVOICE_REPO_URL, str(repo_dir)])
    else:
        run(["git", "submodule", "update", "--init", "--recursive"], cwd=str(repo_dir))

    run(["apt-get", "install", "-y", "-qq", "sox", "libsox-dev"], check=False)

    python_bin = _ensure_py310_venv(engine_dir)

    # Upstream pins are heavy (torch==2.3.1, deepspeed==0.15.1, onnxruntime-gpu==1.18.0,
    # openai-whisper==20231117, etc.). Install as-is to match upstream expectations.
    uv_pip_install(python_bin, ["-r", str(repo_dir / "requirements.txt")])
    uv_pip_install(python_bin, ["fastapi", "uvicorn", "soundfile"])

    model_dir = repo_dir / "pretrained_models" / "CosyVoice2-0.5B"
    if not model_dir.exists():
        run(
            [
                str(python_bin),
                "-c",
                (
                    "from huggingface_hub import snapshot_download; "
                    f"snapshot_download('{settings.cosyvoice_hf_model}', local_dir='{model_dir}')"
                ),
            ]
        )

    write_text(engine_dir / "app.py", settings.read_repo_text("src/apps/cosyvoice2_app.py"))

    matcha_dir = repo_dir / "third_party" / "Matcha-TTS"
    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "PYTHONPATH": f"{repo_dir}:{matcha_dir}",
        "OPENAI_MODEL_ID": settings.openai_model_id or "cosyvoice2",
        "COSYVOICE_REPO_DIR": str(repo_dir),
        "COSYVOICE_MODEL_DIR": str(model_dir),
        "COSYVOICE_PROMPT_WAV": settings.cosyvoice_prompt_wav,
        "COSYVOICE_PROMPT_TEXT": settings.cosyvoice_prompt_text,
        "COSYVOICE_DEFAULT_VOICE": settings.cosyvoice_default_voice,
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
        log_path=settings.log_dir / "cosyvoice2-uvicorn.log",
    )
    return {
        "proc": proc,
        "app_dir": engine_dir,
        "log_path": settings.log_dir / "cosyvoice2-uvicorn.log",
    }
