from __future__ import annotations

import os
from pathlib import Path

from src.config import Settings
from src.runtime import popen, run, uv_pip_install, write_text


AMPHION_REPO_URL = "https://github.com/open-mmlab/Amphion.git"


def _ensure_py310_venv(engine_dir: Path) -> Path:
    venv_dir = engine_dir / ".venv"
    if not venv_dir.exists():
        # Upstream pins torch==2.0.1 / transformers==4.41.2 / numpy==1.26.0 etc.
        # which only ship cp39-cp311 wheels. Pin Python 3.10 for the widest
        # compatibility with these legacy version constraints.
        run(["uv", "venv", "--python", "3.10", str(venv_dir)])
    return venv_dir / "bin" / "python"


def _sparse_checkout_amphion(repo_dir: Path) -> None:
    if (repo_dir / "models" / "tts" / "maskgct").exists():
        print(f"reuse: {repo_dir}")
        return
    repo_dir.mkdir(parents=True, exist_ok=True)
    # The full Amphion repo is huge; sparse-checkout only the MaskGCT module
    # plus the shared codec / utils it depends on (per upstream README).
    run(
        [
            "git",
            "clone",
            "--no-checkout",
            "--filter=blob:none",
            AMPHION_REPO_URL,
            str(repo_dir),
        ]
    )
    run(["git", "sparse-checkout", "init", "--cone"], cwd=str(repo_dir))
    run(
        [
            "git",
            "sparse-checkout",
            "set",
            "models/tts/maskgct",
            "models/codec",
            "utils",
        ],
        cwd=str(repo_dir),
    )
    run(["git", "checkout", "main"], cwd=str(repo_dir))


def install(settings: Settings) -> dict:
    engine_dir = settings.engines_dir / "maskgct"
    engine_dir.mkdir(parents=True, exist_ok=True)

    repo_dir = engine_dir / "Amphion"
    _sparse_checkout_amphion(repo_dir)

    run(["apt-get", "install", "-y", "-qq", "espeak-ng"], check=False)

    python_bin = _ensure_py310_venv(engine_dir)

    # `openai-whisper==20231117` (transitive) has a legacy setup.py that imports
    # `pkg_resources`, which modern setuptools (>=80) no longer ships. Pre-install
    # setuptools<70 + wheel and disable build isolation for whisper so it picks up
    # the older setuptools (same workaround used by the CosyVoice2 installer).
    uv_pip_install(python_bin, ["setuptools<70", "wheel"])
    run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(python_bin),
            "--no-build-isolation-package",
            "openai-whisper",
            "-r",
            str(repo_dir / "models" / "tts" / "maskgct" / "requirements.txt"),
        ]
    )
    uv_pip_install(python_bin, ["fastapi", "uvicorn", "soundfile"])

    # Upstream requirements pin torch==2.0.1 but leave torchaudio unpinned, so
    # uv resolves the latest torchaudio (2.11.x, CUDA 13). Without the matching
    # torchaudio (2.0.2 for torch 2.0.1), import fails with libcudart.so.13.
    pin_torchaudio = (
        "import subprocess, sys, torch;"
        " v = torch.__version__.split('+')[0];"
        " major_minor = '.'.join(v.split('.')[:2]);"
        " subprocess.check_call(["
        "'uv', 'pip', 'install', '--python', sys.executable,"
        " f'torchaudio=={major_minor}.*'"
        "])"
    )
    run([str(python_bin), "-c", pin_torchaudio])

    write_text(engine_dir / "app.py", settings.read_repo_text("src/apps/maskgct_app.py"))

    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "PYTHONPATH": str(repo_dir),
        "OPENAI_MODEL_ID": settings.openai_model_id or "maskgct",
        "MASKGCT_REPO_DIR": str(repo_dir),
        "MASKGCT_DEFAULT_VOICE": settings.maskgct_default_voice,
        "MASKGCT_PROMPT_WAV": settings.maskgct_prompt_wav,
        "MASKGCT_PROMPT_TEXT": settings.maskgct_prompt_text,
        "MASKGCT_PROMPT_LANG": settings.maskgct_prompt_lang,
        "MASKGCT_TARGET_LANG": settings.maskgct_target_lang,
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
        log_path=settings.log_dir / "maskgct-uvicorn.log",
    )
    return {
        "proc": proc,
        "app_dir": engine_dir,
        "log_path": settings.log_dir / "maskgct-uvicorn.log",
    }
