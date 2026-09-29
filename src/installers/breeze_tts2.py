from __future__ import annotations

import os
from pathlib import Path

from src.config import Settings
from src.runtime import popen, run, tail_log, uv_pip_install, wait_http, write_text


BREEZE_REPO_URL = "https://github.com/breezeblue-ai/breeze-tts.git"
BREEZE_REPO_REF = "008f769016b0a24711becd7a4925030bc93f608c"
BREEZE_MODEL_ID = "BreezeBlue/Breeze-TTS-2"
BREEZE_MODEL_REF = "3e28c5151381a722f1d8661b4118c298caa77aa4"


def _ensure_repo(repo_dir: Path) -> None:
    if not repo_dir.exists():
        run(["git", "clone", BREEZE_REPO_URL, str(repo_dir)])
    run(["git", "fetch", "origin", BREEZE_REPO_REF], cwd=str(repo_dir))
    run(["git", "checkout", "--detach", BREEZE_REPO_REF], cwd=str(repo_dir))


def _ensure_py312_venv(engine_dir: Path) -> Path:
    venv_dir = engine_dir / ".venv"
    if not venv_dir.exists():
        run(["uv", "venv", "--python", "3.12", str(venv_dir)])
    return venv_dir / "bin" / "python"


def _download_model(python_bin: Path, model_id: str, model_dir: Path) -> None:
    revision = BREEZE_MODEL_REF if model_id == BREEZE_MODEL_ID else None
    marker = model_dir / ".download-complete"
    marker_value = f"{model_id}@{revision or 'default'}\n"
    if marker.exists() and marker.read_text(encoding="utf-8") == marker_value:
        return
    code = (
        "from huggingface_hub import snapshot_download; "
        f"snapshot_download(repo_id={model_id!r}, revision={revision!r}, "
        f"local_dir={str(model_dir)!r})"
    )
    run([str(python_bin), "-c", code])
    marker.write_text(marker_value, encoding="utf-8")


def install(settings: Settings) -> dict:
    engine_dir = settings.engines_dir / "breeze-tts-2"
    engine_dir.mkdir(parents=True, exist_ok=True)

    repo_dir = engine_dir / "breeze-tts"
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
            "torch==2.9.1",
            "torchaudio==2.9.1",
        ]
    )
    uv_pip_install(python_bin, ["-r", str(repo_dir / "requirements.txt")])
    uv_pip_install(python_bin, ["requests", "huggingface-hub"])

    model_dir = engine_dir / "models" / settings.breeze_tts2_hf_model.replace("/", "--")
    _download_model(python_bin, settings.breeze_tts2_hf_model, model_dir)

    backend_env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
    }
    backend_proc = popen(
        [
            str(python_bin),
            "-m",
            "breeze_infer.api",
            str(model_dir),
            "--host",
            "127.0.0.1",
            "--port",
            str(settings.breeze_tts2_backend_port),
        ],
        cwd=str(repo_dir),
        env=backend_env,
        log_path=settings.log_dir / "breeze-tts2-backend.log",
    )
    backend_url = f"http://127.0.0.1:{settings.breeze_tts2_backend_port}"
    if not wait_http(f"{backend_url}/health", timeout=900):
        tail_log(settings.log_dir / "breeze-tts2-backend.log")
        raise RuntimeError("Breeze TTS 2 backend did not become ready.")

    write_text(engine_dir / "app.py", settings.read_repo_text("src/apps/breeze_tts2_app.py"))
    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "OPENAI_MODEL_ID": settings.openai_model_id or "breeze-tts-2",
        "BREEZE_TTS2_BACKEND_URL": backend_url,
        "BREEZE_TTS2_PROMPT_WAV": settings.breeze_tts2_prompt_wav,
        "BREEZE_TTS2_PROMPT_TEXT": settings.breeze_tts2_prompt_text,
        "BREEZE_TTS2_DEFAULT_VOICE": settings.breeze_tts2_default_voice,
        "BREEZE_TTS2_DEFAULT_INSTRUCTIONS": settings.breeze_tts2_default_instructions,
        "BREEZE_TTS2_SEED": str(settings.breeze_tts2_seed),
        "BREEZE_TTS2_INSTRUCTION_CFG_SCALE": str(
            settings.breeze_tts2_instruction_cfg_scale
        ),
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
        log_path=settings.log_dir / "breeze-tts2-uvicorn.log",
    )
    return {
        "proc": proc,
        "backend_proc": backend_proc,
        "app_dir": engine_dir,
        "log_path": settings.log_dir / "breeze-tts2-uvicorn.log",
        "backend_log_path": settings.log_dir / "breeze-tts2-backend.log",
    }
