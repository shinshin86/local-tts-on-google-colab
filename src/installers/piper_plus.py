from __future__ import annotations

import os

from src.config import Settings
from src.runtime import ensure_venv, popen, run, tail_log, uv_pip_install, wait_http, write_text


def install(settings: Settings) -> dict:
    engine_dir = settings.engines_dir / "piper-plus-openai"
    engine_dir.mkdir(parents=True, exist_ok=True)
    python_bin = ensure_venv(engine_dir)
    uv_pip_install(
        python_bin,
        ["fastapi", "uvicorn", "requests", "flask", "piper-tts-plus"],
    )
    run(
        [
            str(python_bin), "-m", "piper",
            "--download-model", settings.piper_plus_model,
        ],
        cwd=str(engine_dir),
    )
    backend_proc = popen(
        [
            str(python_bin),
            "-m",
            "piper.http_server",
            "-m",
            settings.piper_plus_model,
            "--data-dir",
            str(engine_dir),
            "--host",
            "127.0.0.1",
            "--port",
            str(settings.piper_backend_port),
        ],
        cwd=str(engine_dir),
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
        log_path=settings.log_dir / "piper-plus-backend.log",
    )
    if not wait_http(f"http://127.0.0.1:{settings.piper_backend_port}/voices", timeout=120):
        tail_log(settings.log_dir / "piper-plus-backend.log")
        raise RuntimeError("Piper-Plus backend did not become ready.")
    write_text(engine_dir / "app.py", settings.read_repo_text("src/apps/piper_proxy_app.py"))
    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "BACKEND_URL": f"http://127.0.0.1:{settings.piper_backend_port}",
        "OPENAI_MODEL_ID": settings.openai_model_id or settings.piper_plus_model,
        "PROXY_DEFAULT_VOICE": settings.piper_plus_model,
        "PROXY_SPEAKER_ID": "-1",
        "PROXY_ENGINE": "Piper-Plus",
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
        log_path=settings.log_dir / "piper-plus-proxy-uvicorn.log",
    )
    return {
        "proc": proc,
        "backend_proc": backend_proc,
        "app_dir": engine_dir,
        "log_path": settings.log_dir / "piper-plus-proxy-uvicorn.log",
        "backend_log_path": settings.log_dir / "piper-plus-backend.log",
    }
