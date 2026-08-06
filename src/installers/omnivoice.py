from __future__ import annotations

import os
from pathlib import Path

from src.config import Settings
from src.runtime import popen, run, uv_pip_install, write_text


OMNIVOICE_REPO_URL = "https://github.com/k2-fsa/OmniVoice.git"
OMNIVOICE_REPO_REF = "38e992bc60f85548faeb77e8fa70158ba71deb30"
OMNIVOICE_MODEL_ID = "k2-fsa/OmniVoice"
OMNIVOICE_MODEL_REF = "c5fdb5ccb189668d56333f77ba2629f4cd7535f4"


def _ensure_repo(repo_dir: Path) -> None:
    if not repo_dir.exists():
        run(["git", "clone", OMNIVOICE_REPO_URL, str(repo_dir)])
    run(["git", "fetch", "origin", OMNIVOICE_REPO_REF], cwd=str(repo_dir))
    run(["git", "checkout", "--detach", OMNIVOICE_REPO_REF], cwd=str(repo_dir))


def _ensure_py312_venv(engine_dir: Path) -> Path:
    venv_dir = engine_dir / ".venv"
    if not venv_dir.exists():
        run(["uv", "venv", "--python", "3.12", str(venv_dir)])
    return venv_dir / "bin" / "python"


def install(settings: Settings) -> dict:
    engine_dir = settings.engines_dir / "omnivoice"
    engine_dir.mkdir(parents=True, exist_ok=True)

    repo_dir = engine_dir / "OmniVoice"
    _ensure_repo(repo_dir)

    python_bin = _ensure_py312_venv(engine_dir)
    run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(python_bin),
            "--index-url",
            "https://download.pytorch.org/whl/cu128",
            "torch==2.8.0+cu128",
            "torchaudio==2.8.0+cu128",
        ]
    )
    uv_pip_install(python_bin, ["-e", str(repo_dir)])
    uv_pip_install(python_bin, ["fastapi", "uvicorn", "soundfile"])

    model_dir = engine_dir / "models" / "OmniVoice"
    model_revision = (
        OMNIVOICE_MODEL_REF if settings.omnivoice_hf_model == OMNIVOICE_MODEL_ID else None
    )
    if not (model_dir / "model.safetensors").exists():
        run(
            [
                str(python_bin),
                "-c",
                (
                    "from huggingface_hub import snapshot_download; "
                    f"snapshot_download({settings.omnivoice_hf_model!r}, "
                    f"revision={model_revision!r}, local_dir={str(model_dir)!r})"
                ),
            ]
        )

    write_text(engine_dir / "app.py", settings.read_repo_text("src/apps/omnivoice_app.py"))

    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "OPENAI_MODEL_ID": settings.openai_model_id or "omnivoice",
        "OMNIVOICE_HF_MODEL": settings.omnivoice_hf_model,
        "OMNIVOICE_MODEL_DIR": str(model_dir),
        "OMNIVOICE_LANGUAGE": settings.omnivoice_language,
        "OMNIVOICE_DEFAULT_VOICE": settings.omnivoice_default_voice,
        "OMNIVOICE_PROMPT_WAV": settings.omnivoice_prompt_wav,
        "OMNIVOICE_PROMPT_TEXT": settings.omnivoice_prompt_text,
        "OMNIVOICE_INSTRUCT": settings.omnivoice_instruct,
        "OMNIVOICE_NUM_STEPS": str(settings.omnivoice_num_steps),
        "OMNIVOICE_GUIDANCE_SCALE": str(settings.omnivoice_guidance_scale),
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
        log_path=settings.log_dir / "omnivoice-uvicorn.log",
    )
    return {
        "proc": proc,
        "app_dir": engine_dir,
        "log_path": settings.log_dir / "omnivoice-uvicorn.log",
    }
