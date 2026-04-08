from __future__ import annotations

import os
import tempfile

from src.config import Settings
from src.runtime import (
    ensure_git_clone,
    ensure_venv,
    popen,
    run,
    tail_log,
    uv_pip_install,
    wait_http,
    write_text,
)


COSYVOICE_REPO_URL = "https://github.com/FunAudioLLM/CosyVoice.git"
COSYVOICE_BACKEND_PORT = 50000

# Packages excluded from CosyVoice requirements.txt due to build issues
# on Python 3.12+ or because they are not needed for TTS inference:
# - openai-whisper: only used for reference audio transcription
# - onnxruntime-gpu/onnxruntime: inference uses PyTorch, not ONNX
# - grpcio/grpcio-tools: v1.57.0 lacks 3.12 wheels; only for gRPC serving
# - tensorboard: pulls in incompatible grpcio; not needed for inference
# - tensorrt-*: optional optimization; not needed for basic inference
EXCLUDED_PACKAGES = {
    "openai-whisper",
    "onnxruntime-gpu", "onnxruntime",
    "grpcio", "grpcio-tools",
    "tensorboard",
    "tensorrt-cu12", "tensorrt-cu12-bindings", "tensorrt-cu12-libs",
}


def _filter_requirements(requirements_path):
    """Filter out problematic packages from requirements.txt."""
    filtered = tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8",
    )
    with open(requirements_path, encoding="utf-8") as f:
        for line in f:
            pkg_name = line.split("==")[0].split(">=")[0].split("<=")[0].strip().lower()
            if pkg_name in EXCLUDED_PACKAGES:
                continue
            filtered.write(line)
    filtered.close()
    return filtered.name


def install(settings: Settings) -> dict:
    engine_dir = settings.engines_dir / "cosyvoice"
    engine_dir.mkdir(parents=True, exist_ok=True)

    # Clone CosyVoice repo
    repo_dir = engine_dir / "CosyVoice"
    ensure_git_clone(COSYVOICE_REPO_URL, repo_dir)
    run(["git", "submodule", "update", "--init", "--recursive"], cwd=str(repo_dir))

    # Install system dependencies
    run(["apt-get", "install", "-y", "-qq", "sox", "libsox-dev"], check=False)

    # Create Python 3.12 venv (CosyVoice pins torch==2.3.1 etc. which
    # lack Python 3.13 wheels). openai-whisper is excluded because it has
    # build issues on Python 3.12+. This means automatic transcription of
    # reference audio for zero-shot voice cloning is not available.
    venv_dir = engine_dir / ".venv"
    if not venv_dir.exists():
        run(["uv", "venv", "--python", "3.12", str(venv_dir)])
    python_bin = venv_dir / "bin" / "python"
    filtered_req = _filter_requirements(repo_dir / "requirements.txt")
    uv_pip_install(
        python_bin,
        ["--index-strategy", "unsafe-best-match", "-r", filtered_req],
    )
    os.unlink(filtered_req)
    uv_pip_install(
        python_bin,
        ["fastapi", "uvicorn", "requests", "soundfile", "numpy"],
    )

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
        log_path=settings.log_dir / "cosyvoice-proxy-uvicorn.log",
    )
    return {
        "proc": proc,
        "backend_proc": backend_proc,
        "app_dir": engine_dir,
        "log_path": settings.log_dir / "cosyvoice-proxy-uvicorn.log",
        "backend_log_path": settings.log_dir / "cosyvoice-backend.log",
    }
