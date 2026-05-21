from __future__ import annotations

import os

from src.config import Settings
from src.runtime import ensure_git_clone, popen, run, uv_pip_install, write_text


def install(settings: Settings) -> dict:
    # Irodori-TTS-Lite is a monkey-patch over upstream irodori_tts.inference_runtime,
    # so we still need the upstream Irodori-TTS repo for the InferenceRuntime / SamplingRequest
    # surface; Lite's patch() rewires from_key to load int4 safetensors.
    repo_dir = settings.engines_dir / "Irodori-TTS-Lite"
    ensure_git_clone("https://github.com/Aratako/Irodori-TTS", repo_dir)
    write_text(repo_dir / "app.py", settings.read_repo_text("src/apps/irodori_lite_app.py"))
    run(["uv", "sync"], cwd=str(repo_dir))
    python_bin = repo_dir / ".venv" / "bin" / "python"
    uv_pip_install(
        python_bin,
        ["fastapi", "uvicorn", "huggingface_hub", "pyopenjtalk"],
        cwd=str(repo_dir),
    )
    uv_pip_install(
        python_bin,
        ["git+https://github.com/facebookresearch/dacvae.git"],
        cwd=str(repo_dir),
    )
    uv_pip_install(
        python_bin,
        ["git+https://github.com/kizuna-intelligence/Irodori-TTS-Lite.git"],
        cwd=str(repo_dir),
    )
    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "IRODORI_LITE_HF_CHECKPOINT": settings.irodori_lite_hf_checkpoint,
        "IRODORI_LITE_CHECKPOINT_FILE": settings.irodori_lite_checkpoint_file,
        "IRODORI_LITE_CODEC_REPO": settings.irodori_lite_codec_repo,
        "IRODORI_LITE_CODEC_INT4": "1" if settings.irodori_lite_codec_int4 else "0",
        "OPENAI_MODEL_ID": settings.openai_model_id or settings.irodori_lite_hf_checkpoint,
    }
    proc = popen(
        [
            str(repo_dir / ".venv" / "bin" / "uvicorn"),
            "app:app",
            "--host",
            "0.0.0.0",
            "--port",
            str(settings.app_port),
            "--log-level",
            "debug",
            "--access-log",
        ],
        cwd=str(repo_dir),
        env=env,
        log_path=settings.log_dir / "irodori-lite-uvicorn.log",
    )
    return {
        "proc": proc,
        "app_dir": repo_dir,
        "log_path": settings.log_dir / "irodori-lite-uvicorn.log",
    }
