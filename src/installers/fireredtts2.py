from __future__ import annotations

import os
from pathlib import Path

from src.config import Settings
from src.runtime import popen, run, uv_pip_install, write_text


FIREREDTTS2_REPO_URL = "https://github.com/FireRedTeam/FireRedTTS2.git"
FIREREDTTS2_REPO_REF = "404f3f61d25bb4804859b588a6a734bf8468090c"
FIREREDTTS2_MODEL_ID = "FireRedTeam/FireRedTTS2"
FIREREDTTS2_MODEL_REF = "4af3f5cc4963373b86b52d750220d4de85261f05"


def _ensure_repo(repo_dir: Path) -> None:
    if not repo_dir.exists():
        run(["git", "clone", FIREREDTTS2_REPO_URL, str(repo_dir)])
    run(["git", "fetch", "origin", FIREREDTTS2_REPO_REF], cwd=str(repo_dir))
    run(["git", "checkout", "--detach", FIREREDTTS2_REPO_REF], cwd=str(repo_dir))


def _ensure_py311_venv(engine_dir: Path) -> Path:
    venv_dir = engine_dir / ".venv"
    if not venv_dir.exists():
        run(["uv", "venv", "--python", "3.11", str(venv_dir)])
    return venv_dir / "bin" / "python"


def install(settings: Settings) -> dict:
    engine_dir = settings.engines_dir / "fireredtts2"
    engine_dir.mkdir(parents=True, exist_ok=True)

    repo_dir = engine_dir / "FireRedTTS2"
    _ensure_repo(repo_dir)

    python_bin = _ensure_py311_venv(engine_dir)
    run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(python_bin),
            "--index-url",
            "https://download.pytorch.org/whl/cu126",
            "torch==2.7.1+cu126",
            "torchaudio==2.7.1+cu126",
        ]
    )
    uv_pip_install(python_bin, ["-e", str(repo_dir)])
    uv_pip_install(
        python_bin,
        [
            "torchao==0.10.0",
            "torchtune",
            "transformers",
            "einops",
            "librosa",
            "accelerate",
            "soundfile",
            "fastapi",
            "uvicorn",
        ],
    )

    model_dir = engine_dir / "models" / "FireRedTTS2"
    selected_llm = (
        "llm_posttrain.pt"
        if settings.fireredtts2_generation_mode == "dialogue"
        else "llm_pretrain.pt"
    )
    model_revision = (
        FIREREDTTS2_MODEL_REF
        if settings.fireredtts2_hf_model == FIREREDTTS2_MODEL_ID
        else None
    )
    if not (model_dir / selected_llm).exists():
        run(
            [
                str(python_bin),
                "-c",
                (
                    "from huggingface_hub import snapshot_download; "
                    f"snapshot_download({settings.fireredtts2_hf_model!r}, "
                    f"revision={model_revision!r}, local_dir={str(model_dir)!r}, "
                    f"allow_patterns=['config_*.json','codec.pt',{selected_llm!r},'Qwen2.5-1.5B/*'])"
                ),
            ]
        )

    write_text(engine_dir / "app.py", settings.read_repo_text("src/apps/fireredtts2_app.py"))

    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "OPENAI_MODEL_ID": settings.openai_model_id or "fireredtts2",
        "FIREREDTTS2_HF_MODEL": settings.fireredtts2_hf_model,
        "FIREREDTTS2_MODEL_DIR": str(model_dir),
        "FIREREDTTS2_GENERATION_MODE": settings.fireredtts2_generation_mode,
        "FIREREDTTS2_DEFAULT_VOICE": settings.fireredtts2_default_voice,
        "FIREREDTTS2_PROMPT_WAV": settings.fireredtts2_prompt_wav,
        "FIREREDTTS2_PROMPT_TEXT": settings.fireredtts2_prompt_text,
        "FIREREDTTS2_TEMPERATURE": str(settings.fireredtts2_temperature),
        "FIREREDTTS2_TOPK": str(settings.fireredtts2_topk),
        "FIREREDTTS2_USE_BF16": "1" if settings.fireredtts2_use_bf16 else "0",
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
        log_path=settings.log_dir / "fireredtts2-uvicorn.log",
    )
    return {
        "proc": proc,
        "app_dir": engine_dir,
        "log_path": settings.log_dir / "fireredtts2-uvicorn.log",
    }
