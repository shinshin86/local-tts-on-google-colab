from __future__ import annotations

import os
from pathlib import Path

from src.config import Settings
from src.runtime import popen, run, uv_pip_install, write_text


COSYVOICE_REPO_URL = "https://github.com/FunAudioLLM/CosyVoice.git"
COSYVOICE_REPO_REF = "074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc"
COSYVOICE3_MODEL_REF = "29e01c4e8d000f4bcd70751be16fa94bf3d85a18"


def _ensure_py310_venv(engine_dir: Path) -> Path:
    venv_dir = engine_dir / ".venv"
    if not venv_dir.exists():
        run(["uv", "venv", "--python", "3.10", str(venv_dir)])
    return venv_dir / "bin" / "python"


def _ensure_repo(repo_dir: Path) -> None:
    if not repo_dir.exists():
        run(["git", "clone", "--recursive", COSYVOICE_REPO_URL, str(repo_dir)])
    run(["git", "fetch", "origin", COSYVOICE_REPO_REF], cwd=str(repo_dir))
    run(["git", "checkout", "--detach", COSYVOICE_REPO_REF], cwd=str(repo_dir))
    run(["git", "submodule", "update", "--init", "--recursive"], cwd=str(repo_dir))


def install(settings: Settings) -> dict:
    engine_dir = settings.engines_dir / "cosyvoice3"
    engine_dir.mkdir(parents=True, exist_ok=True)

    repo_dir = engine_dir / "CosyVoice"
    _ensure_repo(repo_dir)
    run(["apt-get", "install", "-y", "-qq", "sox", "libsox-dev"], check=False)

    python_bin = _ensure_py310_venv(engine_dir)
    # CosyVoice currently pins torch 2.3.1 and a legacy openai-whisper build.
    # The latter imports pkg_resources during setup, so reuse setuptools<70.
    uv_pip_install(python_bin, ["setuptools<70", "wheel"])
    run(
        [
            "uv", "pip", "install",
            "--python", str(python_bin),
            "--index-strategy", "unsafe-best-match",
            "--no-build-isolation-package", "openai-whisper",
            "-r", str(repo_dir / "requirements.txt"),
        ]
    )
    uv_pip_install(python_bin, ["fastapi", "uvicorn", "soundfile"])

    model_dir = repo_dir / "pretrained_models" / "Fun-CosyVoice3-0.5B-2512"
    if not model_dir.exists():
        run(
            [
                str(python_bin),
                "-c",
                (
                    "from huggingface_hub import snapshot_download; "
                    f"snapshot_download({settings.cosyvoice3_hf_model!r}, "
                    f"revision={COSYVOICE3_MODEL_REF!r}, local_dir={str(model_dir)!r})"
                ),
            ]
        )

    write_text(engine_dir / "app.py", settings.read_repo_text("src/apps/cosyvoice3_app.py"))

    matcha_dir = repo_dir / "third_party" / "Matcha-TTS"
    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "PYTHONPATH": f"{repo_dir}:{matcha_dir}",
        "OPENAI_MODEL_ID": settings.openai_model_id or "cosyvoice3",
        "COSYVOICE3_REPO_DIR": str(repo_dir),
        "COSYVOICE3_MODEL_DIR": str(model_dir),
        "COSYVOICE3_PROMPT_WAV": settings.cosyvoice3_prompt_wav,
        "COSYVOICE3_PROMPT_TEXT": settings.cosyvoice3_prompt_text,
        "COSYVOICE3_INSTRUCT": settings.cosyvoice3_instruct,
        "COSYVOICE3_DEFAULT_VOICE": settings.cosyvoice3_default_voice,
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
        log_path=settings.log_dir / "cosyvoice3-uvicorn.log",
    )
    return {
        "proc": proc,
        "app_dir": engine_dir,
        "log_path": settings.log_dir / "cosyvoice3-uvicorn.log",
    }
