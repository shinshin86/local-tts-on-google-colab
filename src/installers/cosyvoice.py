from __future__ import annotations

import os

from src.config import Settings
from src.runtime import ensure_git_clone, popen, run, tail_log, wait_http, write_text


COSYVOICE_REPO_URL = "https://github.com/FunAudioLLM/CosyVoice.git"
COSYVOICE_BACKEND_PORT = 50000

def install(settings: Settings) -> dict:
    engine_dir = settings.engines_dir / "cosyvoice"
    engine_dir.mkdir(parents=True, exist_ok=True)

    # Clone CosyVoice repo
    repo_dir = engine_dir / "CosyVoice"
    ensure_git_clone(COSYVOICE_REPO_URL, repo_dir)
    run(["git", "submodule", "update", "--init", "--recursive"], cwd=str(repo_dir))

    # Install system dependencies (python3.10-venv needed for full python3.10)
    run(["apt-get", "install", "-y", "-qq", "sox", "libsox-dev",
         "python3.10-venv", "python3.10-dev"], check=False)

    # Create Python 3.10 venv with pip (openai-whisper needs Python <=3.10)
    venv_dir = engine_dir / ".venv"
    if not venv_dir.exists():
        run(["/usr/bin/python3.10", "-m", "venv", str(venv_dir)])
    python_bin = venv_dir / "bin" / "python"

    # Install CosyVoice requirements
    run([str(python_bin), "-m", "pip", "install", "-q", "--upgrade", "pip", "setuptools"])
    run([str(python_bin), "-m", "pip", "install", "-q",
         "-r", str(repo_dir / "requirements.txt")])
    run([str(python_bin), "-m", "pip", "install", "-q",
         "fastapi", "uvicorn", "requests", "soundfile", "numpy"])

    # Start CosyVoice backend server
    backend_env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "PYTHONPATH": f"{repo_dir}:{repo_dir / 'third_party' / 'Matcha-TTS'}",
    }
    backend_proc = popen(
        [
            str(python_bin),
            str(repo_dir / "runtime" / "python" / "fastapi" / "server.py"),
            "--port",
            str(COSYVOICE_BACKEND_PORT),
            "--model_dir",
            settings.cosyvoice_model_dir,
        ],
        cwd=str(repo_dir),
        env=backend_env,
        log_path=settings.log_dir / "cosyvoice-backend.log",
    )

    if not wait_http(f"http://127.0.0.1:{COSYVOICE_BACKEND_PORT}/inference_cross_lingual", timeout=300):
        tail_log(settings.log_dir / "cosyvoice-backend.log")
        raise RuntimeError("CosyVoice backend did not become ready.")

    # Start OpenAI-compatible proxy
    write_text(engine_dir / "app.py", settings.read_repo_text("src/apps/cosyvoice_app.py"))
    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "OPENAI_MODEL_ID": settings.openai_model_id or "cosyvoice2",
        "BACKEND_URL": f"http://127.0.0.1:{COSYVOICE_BACKEND_PORT}",
    }
    proc = popen(
        [
            str(python_bin), "-m", "uvicorn",
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
        log_path=settings.log_dir / "cosyvoice-proxy-uvicorn.log",
    )
    return {
        "proc": proc,
        "backend_proc": backend_proc,
        "app_dir": engine_dir,
        "log_path": settings.log_dir / "cosyvoice-proxy-uvicorn.log",
        "backend_log_path": settings.log_dir / "cosyvoice-backend.log",
    }
