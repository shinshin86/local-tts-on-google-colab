from __future__ import annotations

import os

from src.config import Settings
from src.runtime import ensure_venv, popen, tail_log, uv_pip_install, wait_http, write_text


def install(settings: Settings) -> dict:
    engine_dir = settings.engines_dir / "voxtral-openai"
    engine_dir.mkdir(parents=True, exist_ok=True)
    python_bin = ensure_venv(engine_dir)
    uv_pip_install(
        python_bin,
        [
            "fastapi",
            "uvicorn",
            "httpx",
            "vllm>=0.18.0",
            "vllm-omni @ git+https://github.com/vllm-project/vllm-omni.git",
        ],
    )
    # Start vLLM backend server
    backend_env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
    }
    backend_proc = popen(
        [
            str(engine_dir / ".venv" / "bin" / "vllm"),
            "serve",
            settings.voxtral_hf_model,
            "--host",
            "127.0.0.1",
            "--port",
            str(settings.voxtral_backend_port),
        ],
        cwd=str(engine_dir),
        env=backend_env,
        log_path=settings.log_dir / "voxtral-backend.log",
    )
    # Wait for vLLM backend to be ready (model loading can take a while)
    if not wait_http(f"http://127.0.0.1:{settings.voxtral_backend_port}/v1/models", timeout=600):
        tail_log(settings.log_dir / "voxtral-backend.log")
        raise RuntimeError("Voxtral vLLM backend did not become ready.")
    # Deploy OpenAI-compatible proxy
    write_text(engine_dir / "app.py", settings.read_repo_text("src/apps/voxtral_app.py"))
    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "OPENAI_MODEL_ID": settings.openai_model_id or "voxtral-tts",
        "VLLM_BACKEND_URL": f"http://127.0.0.1:{settings.voxtral_backend_port}",
        "VOXTRAL_HF_MODEL": settings.voxtral_hf_model,
        "VOXTRAL_DEFAULT_VOICE": settings.voxtral_default_voice,
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
        log_path=settings.log_dir / "voxtral-proxy-uvicorn.log",
    )
    return {
        "proc": proc,
        "backend_proc": backend_proc,
        "app_dir": engine_dir,
        "log_path": settings.log_dir / "voxtral-proxy-uvicorn.log",
        "backend_log_path": settings.log_dir / "voxtral-backend.log",
    }
