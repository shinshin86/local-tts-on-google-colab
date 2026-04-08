from __future__ import annotations

import os

from src.config import Settings
from src.runtime import ensure_git_clone, ensure_venv, popen, run, tail_log, uv_pip_install, wait_http, write_text


FISH_SPEECH_REPO_URL = "https://github.com/fishaudio/fish-speech.git"
FISH_SPEECH_BACKEND_PORT = 8080


def install(settings: Settings) -> dict:
    engine_dir = settings.engines_dir / "fish-speech"
    engine_dir.mkdir(parents=True, exist_ok=True)

    # Clone Fish Speech repo
    repo_dir = engine_dir / "fish-speech"
    ensure_git_clone(FISH_SPEECH_REPO_URL, repo_dir)

    # Install system dependencies
    run(["apt-get", "install", "-y", "-qq", "portaudio19-dev", "libsox-dev", "ffmpeg"], check=False)

    # Create venv and install fish-speech from source
    python_bin = ensure_venv(engine_dir)
    uv_pip_install(python_bin, ["-e", f"{repo_dir}[cu126]"])
    uv_pip_install(python_bin, ["fastapi", "uvicorn", "requests"])

    # Download model checkpoint
    checkpoint_dir = repo_dir / "checkpoints" / "s2-pro"
    if not checkpoint_dir.exists():
        run(
            [
                str(python_bin),
                "-m",
                "huggingface_hub.cli",
                "download",
                settings.fish_speech_model,
                "--local-dir",
                str(checkpoint_dir),
            ],
        )

    # Start Fish Speech backend API server
    backend_env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
    }
    backend_proc = popen(
        [
            str(python_bin),
            str(repo_dir / "tools" / "api_server.py"),
            "--llama-checkpoint-path",
            str(checkpoint_dir),
            "--decoder-checkpoint-path",
            str(checkpoint_dir / "codec.pth"),
            "--listen",
            f"0.0.0.0:{FISH_SPEECH_BACKEND_PORT}",
        ],
        cwd=str(repo_dir),
        env=backend_env,
        log_path=settings.log_dir / "fish-speech-backend.log",
    )

    if not wait_http(f"http://127.0.0.1:{FISH_SPEECH_BACKEND_PORT}/v1/health", timeout=300):
        tail_log(settings.log_dir / "fish-speech-backend.log")
        raise RuntimeError("Fish Speech backend did not become ready.")

    # Start OpenAI-compatible proxy
    write_text(engine_dir / "app.py", settings.read_repo_text("src/apps/fish_speech_app.py"))
    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "OPENAI_MODEL_ID": settings.openai_model_id or "fish-speech",
        "BACKEND_URL": f"http://127.0.0.1:{FISH_SPEECH_BACKEND_PORT}",
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
        log_path=settings.log_dir / "fish-speech-proxy-uvicorn.log",
    )
    return {
        "proc": proc,
        "backend_proc": backend_proc,
        "app_dir": engine_dir,
        "log_path": settings.log_dir / "fish-speech-proxy-uvicorn.log",
        "backend_log_path": settings.log_dir / "fish-speech-backend.log",
    }
