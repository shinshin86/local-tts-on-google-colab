from __future__ import annotations

import os

from src.config import Settings
from src.runtime import ensure_venv, popen, run, uv_pip_install, write_text


def install(settings: Settings) -> dict:
    engine_dir = settings.engines_dir / "kokoro-openai"
    engine_dir.mkdir(parents=True, exist_ok=True)
    python_bin = ensure_venv(engine_dir)
    run(["apt-get", "-qq", "-y", "install", "espeak-ng"])
    uv_pip_install(
        python_bin,
        ["fastapi", "uvicorn", "kokoro>=0.9.2", "soundfile", "numpy", "misaki[ja]", "misaki[zh]", "misaki[en]", "unidic"],
    )
    run([str(python_bin), "-m", "unidic", "download"], cwd=str(engine_dir))
    write_text(engine_dir / "app.py", settings.read_repo_text("src/apps/kokoro_app.py"))
    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "OPENAI_MODEL_ID": settings.openai_model_id or "kokoro-82m",
        "KOKORO_DEFAULT_VOICE": settings.kokoro_default_voice,
        "KOKORO_DEFAULT_LANG_CODE": settings.kokoro_default_lang_code,
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
        log_path=settings.log_dir / "kokoro-uvicorn.log",
    )
    return {"proc": proc, "app_dir": engine_dir, "log_path": settings.log_dir / "kokoro-uvicorn.log"}
