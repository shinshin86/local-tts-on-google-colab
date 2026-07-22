from __future__ import annotations

import os

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


UPSTREAM_REPO = "https://github.com/shinshin86/sine-wave-tts.git"


def install(settings: Settings) -> dict:
    app_dir = settings.engines_dir / "sine-wave-tts-openai"
    upstream_dir = app_dir / "upstream"
    app_dir.mkdir(parents=True, exist_ok=True)

    ensure_git_clone(UPSTREAM_REPO, upstream_dir)
    run(["git", "fetch", "--tags", "--prune"], cwd=str(upstream_dir))
    run(["git", "checkout", "--detach", settings.sine_wave_tts_ref], cwd=str(upstream_dir))
    run(["npm", "ci"], cwd=str(upstream_dir))
    run(["npm", "run", "build"], cwd=str(upstream_dir))

    backend_log_path = settings.log_dir / "sine-wave-tts-backend.log"
    backend_proc = popen(
        ["node", "dist/src/server/cli.js"],
        cwd=str(upstream_dir),
        env={
            **os.environ,
            "HOST": "127.0.0.1",
            "PORT": str(settings.sine_wave_tts_backend_port),
        },
        log_path=backend_log_path,
    )
    if not wait_http(
        f"http://127.0.0.1:{settings.sine_wave_tts_backend_port}/v1/health",
        timeout=120,
    ):
        tail_log(backend_log_path)
        raise RuntimeError("Sine-Wave-TTS backend did not become ready.")

    python_bin = ensure_venv(app_dir)
    uv_pip_install(python_bin, ["fastapi", "uvicorn", "requests"])
    write_text(app_dir / "app.py", settings.read_repo_text("src/apps/sine_wave_tts_app.py"))

    wrapper_log_path = settings.log_dir / "sine-wave-tts-uvicorn.log"
    proc = popen(
        [
            str(app_dir / ".venv" / "bin" / "uvicorn"),
            "app:app",
            "--host",
            "0.0.0.0",
            "--port",
            str(settings.app_port),
            "--log-level",
            "info",
        ],
        cwd=str(app_dir),
        env={
            **os.environ,
            "PYTHONUNBUFFERED": "1",
            "BACKEND_URL": f"http://127.0.0.1:{settings.sine_wave_tts_backend_port}",
            "OPENAI_MODEL_ID": settings.openai_model_id or "sine-wave-tts",
            "SINE_WAVE_TTS_DEFAULT_SPEAKER": settings.sine_wave_tts_default_speaker,
            "SINE_WAVE_TTS_DEFAULT_EMOTION": settings.sine_wave_tts_default_emotion,
        },
        log_path=wrapper_log_path,
    )
    return {
        "proc": proc,
        "backend_proc": backend_proc,
        "app_dir": app_dir,
        "log_path": wrapper_log_path,
        "backend_log_path": backend_log_path,
    }
