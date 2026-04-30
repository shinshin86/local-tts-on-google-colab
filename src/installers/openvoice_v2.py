from __future__ import annotations

import os

from src.config import Settings
from src.runtime import ensure_git_clone, ensure_venv, popen, run, uv_pip_install, write_text


OPENVOICE_REPO_URL = "https://github.com/myshell-ai/OpenVoice.git"
MELO_REPO_URL = "https://github.com/myshell-ai/MeloTTS.git"
OPENVOICE_V2_CKPT_URL = (
    "https://myshell-public-repo-host.s3.amazonaws.com/openvoice/checkpoints_v2_0417.zip"
)


def install(settings: Settings) -> dict:
    engine_dir = settings.engines_dir / "openvoice-v2"
    engine_dir.mkdir(parents=True, exist_ok=True)

    repo_dir = engine_dir / "OpenVoice"
    ensure_git_clone(OPENVOICE_REPO_URL, repo_dir)

    python_bin = ensure_venv(engine_dir)
    # OpenVoice itself.
    uv_pip_install(python_bin, ["-e", str(repo_dir)])
    # MeloTTS provides the base multilingual TTS that V2 layers tone color over.
    # NOTE: this dependency is what makes the standalone MeloTTS engine "Not
    # working" today on Colab (Rust toolchain required for tokenizers); the
    # same risk applies here.
    uv_pip_install(python_bin, [f"git+{MELO_REPO_URL}"])
    run([str(python_bin), "-m", "unidic", "download"], check=False)
    uv_pip_install(python_bin, ["fastapi", "uvicorn", "soundfile", "numpy"])

    # Download and unzip OpenVoice V2 checkpoints (~600MB). Idempotent.
    ckpt_zip = engine_dir / "checkpoints_v2.zip"
    ckpt_dir = engine_dir / "checkpoints_v2"
    if not ckpt_dir.exists():
        run(["wget", "-q", "-O", str(ckpt_zip), OPENVOICE_V2_CKPT_URL])
        run(["unzip", "-q", "-o", str(ckpt_zip), "-d", str(engine_dir)])

    write_text(engine_dir / "app.py", settings.read_repo_text("src/apps/openvoice_v2_app.py"))

    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "OPENAI_MODEL_ID": settings.openai_model_id or "openvoice-v2",
        "OPENVOICE_LANGUAGE": settings.openvoice_language,
        "OPENVOICE_CKPT_DIR": str(ckpt_dir),
        "OPENVOICE_REPO_DIR": str(repo_dir),
        "OPENVOICE_PROMPT_WAV": settings.openvoice_prompt_wav,
        "OPENVOICE_DEFAULT_VOICE": settings.openvoice_default_voice,
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
        log_path=settings.log_dir / "openvoice-v2-uvicorn.log",
    )
    return {
        "proc": proc,
        "app_dir": engine_dir,
        "log_path": settings.log_dir / "openvoice-v2-uvicorn.log",
    }
