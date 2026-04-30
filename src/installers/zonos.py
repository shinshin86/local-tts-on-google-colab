from __future__ import annotations

import os

from src.config import Settings
from src.runtime import ensure_git_clone, ensure_venv, popen, run, uv_pip_install, write_text


ZONOS_REPO_URL = "https://github.com/Zyphra/Zonos.git"


def install(settings: Settings) -> dict:
    engine_dir = settings.engines_dir / "zonos"
    engine_dir.mkdir(parents=True, exist_ok=True)

    # Zonos requires espeak-ng for phonemization at the system level.
    run(["bash", "-lc", "apt-get update && apt-get install -y espeak-ng"], check=False)

    repo_dir = engine_dir / "Zonos"
    ensure_git_clone(ZONOS_REPO_URL, repo_dir)

    python_bin = ensure_venv(engine_dir)
    # Install the upstream Zonos package (transformer backbone). The hybrid backbone
    # additionally needs mamba-ssm + causal_conv1d, which require a 30xx-series GPU
    # or newer; we stick with the transformer to keep T4 / L4 compatibility.
    uv_pip_install(python_bin, ["-e", str(repo_dir)])
    # Pin torch / torchaudio to 2.6 -- newer torchaudio (2.11) requires the
    # separate `torchcodec` package for `torchaudio.load`, and Zonos does not
    # depend on either explicitly, so uv resolves to whatever happens to be
    # latest. 2.6 ships its own audio backend and avoids the torchcodec hop.
    uv_pip_install(python_bin, ["torch==2.6.0", "torchaudio==2.6.0"])
    uv_pip_install(python_bin, ["fastapi", "uvicorn", "soundfile", "numpy"])

    write_text(engine_dir / "app.py", settings.read_repo_text("src/apps/zonos_app.py"))

    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "OPENAI_MODEL_ID": settings.openai_model_id or "zonos",
        "ZONOS_HF_MODEL": settings.zonos_hf_model,
        "ZONOS_LANGUAGE": settings.zonos_language,
        "ZONOS_PROMPT_WAV": settings.zonos_prompt_wav,
        "ZONOS_DEFAULT_PROMPT_WAV": str(repo_dir / "assets" / "exampleaudio.mp3"),
        "ZONOS_DEFAULT_VOICE": settings.zonos_default_voice,
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
        log_path=settings.log_dir / "zonos-uvicorn.log",
    )
    return {
        "proc": proc,
        "app_dir": engine_dir,
        "log_path": settings.log_dir / "zonos-uvicorn.log",
    }
