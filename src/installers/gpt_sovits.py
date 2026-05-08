from __future__ import annotations

import os
from pathlib import Path

from src.config import Settings
from src.runtime import popen, run, uv_pip_install, write_text


GPT_SOVITS_REPO_URL = "https://github.com/RVC-Boss/GPT-SoVITS.git"
PRETRAINED_HF_REPO = "lj1995/GPT-SoVITS"


def _ensure_py311_venv(engine_dir: Path) -> Path:
    venv_dir = engine_dir / ".venv"
    if not venv_dir.exists():
        # Upstream README documents Python 3.10 / 3.11 as the tested environments.
        # 3.11 gives us the widest wheel coverage for torch>=2.5.1 + faiss + numba.
        run(["uv", "venv", "--python", "3.11", str(venv_dir)])
    return venv_dir / "bin" / "python"


def install(settings: Settings) -> dict:
    engine_dir = settings.engines_dir / "gpt-sovits"
    engine_dir.mkdir(parents=True, exist_ok=True)

    repo_dir = engine_dir / "GPT-SoVITS"
    if not repo_dir.exists():
        run(["git", "clone", GPT_SOVITS_REPO_URL, str(repo_dir)])

    run(["apt-get", "install", "-y", "-qq", "ffmpeg", "libsox-dev"], check=False)

    python_bin = _ensure_py311_venv(engine_dir)

    # Upstream split: requirements.txt (resolvable) + extra-req.txt (--no-deps for
    # version-pinned ML libs that would otherwise downgrade torch).
    uv_pip_install(python_bin, ["-r", str(repo_dir / "requirements.txt")])
    run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(python_bin),
            "--no-deps",
            "-r",
            str(repo_dir / "extra-req.txt"),
        ],
        check=False,
    )
    uv_pip_install(python_bin, ["fastapi", "uvicorn", "soundfile", "huggingface_hub"])

    # Selectively download just the v2 weights (~1.2 GB) plus the BERT/HuBERT
    # base models. Pulling the entire 5.3 GB lj1995/GPT-SoVITS repo is wasteful
    # for the v2 default we ship.
    pretrained_dir = repo_dir / "GPT_SoVITS" / "pretrained_models"
    pretrained_dir.mkdir(parents=True, exist_ok=True)

    snapshot_script = (
        "from huggingface_hub import snapshot_download; "
        f"snapshot_download('{PRETRAINED_HF_REPO}', local_dir='{pretrained_dir}', "
        "allow_patterns=["
        "'chinese-hubert-base/*', "
        "'chinese-roberta-wwm-ext-large/*', "
        "'gsv-v2final-pretrained/*'"
        "])"
    )
    run([str(python_bin), "-c", snapshot_script])

    write_text(engine_dir / "app.py", settings.read_repo_text("src/apps/gpt_sovits_app.py"))

    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "PYTHONPATH": f"{repo_dir}:{repo_dir / 'GPT_SoVITS'}",
        "OPENAI_MODEL_ID": settings.openai_model_id or "gpt-sovits-v2",
        "GPT_SOVITS_REPO_DIR": str(repo_dir),
        "GPT_SOVITS_VERSION": settings.gpt_sovits_version,
        "GPT_SOVITS_DEFAULT_VOICE": settings.gpt_sovits_default_voice,
        "GPT_SOVITS_PROMPT_WAV": settings.gpt_sovits_prompt_wav,
        "GPT_SOVITS_PROMPT_TEXT": settings.gpt_sovits_prompt_text,
        "GPT_SOVITS_PROMPT_LANG": settings.gpt_sovits_prompt_lang,
        "GPT_SOVITS_TARGET_LANG": settings.gpt_sovits_target_lang,
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
        log_path=settings.log_dir / "gpt-sovits-uvicorn.log",
    )
    return {
        "proc": proc,
        "app_dir": engine_dir,
        "log_path": settings.log_dir / "gpt-sovits-uvicorn.log",
    }
