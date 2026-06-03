from __future__ import annotations

import os
from pathlib import Path

from src.config import Settings
from src.runtime import ensure_git_clone, popen, run, uv_pip_install, write_text


MISOTTS_REPO_URL = "https://github.com/MisoLabsAI/MisoTTS.git"


def _ensure_py311_venv(engine_dir: Path) -> Path:
    venv_dir = engine_dir / ".venv"
    if not venv_dir.exists():
        # MisoTTS is a Sesame-CSM fork and reuses the same pinned stack
        # (torch==2.4.0 / torchtune==0.4.0 / torchao==0.9.0, cp39-cp312 wheels
        # only) with requires-python >=3.10,<3.13. Pin 3.11 for the widest wheel
        # coverage, matching the CSM engine.
        run(["uv", "venv", "--python", "3.11", str(venv_dir)])
    return venv_dir / "bin" / "python"


def install(settings: Settings) -> dict:
    engine_dir = settings.engines_dir / "misotts"
    repo_dir = engine_dir / "MisoTTS"
    engine_dir.mkdir(parents=True, exist_ok=True)
    ensure_git_clone(MISOTTS_REPO_URL, repo_dir)

    python_bin = _ensure_py311_venv(engine_dir)

    # requirements.txt pulls the full CSM-lineage stack, including the
    # silentcipher git dependency that powers the watermark applied inside
    # Generator.generate(). bitsandbytes is gated to Linux there, which matches
    # Colab.
    uv_pip_install(python_bin, ["-r", str(repo_dir / "requirements.txt")])
    uv_pip_install(python_bin, ["fastapi", "uvicorn", "soundfile"])

    write_text(engine_dir / "app.py", settings.read_repo_text("src/apps/misotts_app.py"))

    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        # Inherited from CSM/the upstream run script: disable torch.compile for
        # stability on the default Colab CUDA toolchain.
        "NO_TORCH_COMPILE": "1",
        "HF_HUB_ETAG_TIMEOUT": "60",
        "HF_HUB_DOWNLOAD_TIMEOUT": "60",
        "PYTHONPATH": str(repo_dir),
        "OPENAI_MODEL_ID": settings.openai_model_id or "miso-tts-8b",
        "MISOTTS_REPO_DIR": str(repo_dir),
        "MISOTTS_HF_MODEL": settings.misotts_hf_model,
        "MISOTTS_DEFAULT_VOICE": settings.misotts_default_voice,
        "MISOTTS_DEFAULT_SPEAKER": str(settings.misotts_default_speaker),
        "MISOTTS_PROMPT_WAV": settings.misotts_prompt_wav,
        "MISOTTS_PROMPT_TEXT": settings.misotts_prompt_text,
        "MISOTTS_MAX_AUDIO_LENGTH_MS": str(settings.misotts_max_audio_length_ms),
        "MISOTTS_TEMPERATURE": str(settings.misotts_temperature),
        "MISOTTS_TOPK": str(settings.misotts_topk),
        "MISOTTS_TOKENIZER_REPO": settings.misotts_tokenizer_repo,
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
        log_path=settings.log_dir / "misotts-uvicorn.log",
    )
    return {
        "proc": proc,
        "app_dir": engine_dir,
        "log_path": settings.log_dir / "misotts-uvicorn.log",
    }
