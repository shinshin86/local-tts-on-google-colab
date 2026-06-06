from __future__ import annotations

import os
from pathlib import Path

from src.config import Settings
from src.runtime import ensure_git_clone, popen, run, uv_pip_install, write_text


DOTS_TTS_REPO_URL = "https://github.com/rednote-hilab/dots.tts.git"


def _ensure_py312_venv(engine_dir: Path) -> Path:
    venv_dir = engine_dir / ".venv"
    if not venv_dir.exists():
        # dots.tts declares requires-python >=3.10,<3.13 and pins torch>=2.8.0
        # (cp312 wheels available). 3.12 matches Colab's system interpreter and
        # the recent engines (Higgs v3 / MOSS v1.5). The one fragile dep,
        # WeTextProcessing -> pynini==2.1.6, ships a cp312 manylinux_2_28 wheel,
        # so nothing builds OpenFst from source here.
        run(["uv", "venv", "--python", "3.12", str(venv_dir)])
    return venv_dir / "bin" / "python"


def install(settings: Settings) -> dict:
    engine_dir = settings.engines_dir / "dots-tts"
    repo_dir = engine_dir / "dots.tts"
    engine_dir.mkdir(parents=True, exist_ok=True)
    ensure_git_clone(DOTS_TTS_REPO_URL, repo_dir)

    python_bin = _ensure_py312_venv(engine_dir)

    # Editable install with the upstream recommended constraints (pins torch==2.8.0
    # etc.). The package depends on gradio (which pulls fastapi/uvicorn), but pin
    # the wrapper's own deps explicitly so the in-process server never breaks if
    # upstream drops one. soundfile/librosa are already runtime deps.
    constraints = repo_dir / "constraints" / "recommended.txt"
    uv_pip_install(
        python_bin,
        ["-e", ".", "-c", str(constraints)],
        cwd=str(repo_dir),
    )
    uv_pip_install(python_bin, ["fastapi", "uvicorn", "soundfile"])

    write_text(engine_dir / "app.py", settings.read_repo_text("src/apps/dots_tts_app.py"))

    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "HF_HUB_ETAG_TIMEOUT": "60",
        "HF_HUB_DOWNLOAD_TIMEOUT": "60",
        # Keep the package importable for `from dots_tts...` even though `-e .`
        # already places it on the path; harmless and explicit.
        "PYTHONPATH": str(repo_dir / "src"),
        "OPENAI_MODEL_ID": settings.openai_model_id or "dots-tts",
        "DOTS_TTS_HF_MODEL": settings.dots_tts_hf_model,
        "DOTS_TTS_DEFAULT_VOICE": settings.dots_tts_default_voice,
        "DOTS_TTS_PROMPT_WAV": settings.dots_tts_prompt_wav,
        "DOTS_TTS_PROMPT_TEXT": settings.dots_tts_prompt_text,
        "DOTS_TTS_LANGUAGE": settings.dots_tts_language,
        "DOTS_TTS_PRECISION": settings.dots_tts_precision,
        "DOTS_TTS_NUM_STEPS": str(settings.dots_tts_num_steps),
        "DOTS_TTS_GUIDANCE_SCALE": str(settings.dots_tts_guidance_scale),
        "DOTS_TTS_SPEAKER_SCALE": str(settings.dots_tts_speaker_scale),
        "DOTS_TTS_MAX_GENERATE_LENGTH": str(settings.dots_tts_max_generate_length),
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
        log_path=settings.log_dir / "dots-tts-uvicorn.log",
    )
    return {
        "proc": proc,
        "app_dir": engine_dir,
        "log_path": settings.log_dir / "dots-tts-uvicorn.log",
    }
