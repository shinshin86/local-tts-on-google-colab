from __future__ import annotations

import os

from src.config import Settings
from src.runtime import ensure_git_clone, popen, run, uv_pip_install, write_text


def install_runtime(
    settings: Settings,
    *,
    engine_dir_name: str,
    app_source: str,
    checkpoint: str,
    codec_repo: str,
    model_precision: str,
    codec_precision: str,
    engine_name: str,
    log_filename: str,
    num_steps: int | None = 40,
    prompt_wav: str = "",
    default_voice: str = "default",
) -> dict:
    repo_dir = settings.engines_dir / engine_dir_name
    ensure_git_clone("https://github.com/Aratako/Irodori-TTS", repo_dir)
    if app_source != "src/apps/irodori_app.py":
        write_text(
            repo_dir / "irodori_app_shared.py",
            settings.read_repo_text("src/apps/irodori_app.py"),
        )
    write_text(repo_dir / "app.py", settings.read_repo_text(app_source))
    # Follow upstream's supported NVIDIA environment so uv resolves CUDA wheels.
    run(["uv", "sync", "--extra", "cu128"], cwd=str(repo_dir))
    python_bin = repo_dir / ".venv" / "bin" / "python"
    uv_pip_install(
        python_bin,
        ["fastapi", "uvicorn", "huggingface_hub"],
        cwd=str(repo_dir),
    )
    uv_pip_install(
        python_bin,
        ["git+https://github.com/facebookresearch/dacvae.git"],
        cwd=str(repo_dir),
    )
    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "IRODORI_HF_CHECKPOINT": checkpoint,
        "IRODORI_CODEC_REPO": codec_repo,
        "IRODORI_MODEL_PRECISION": model_precision,
        "IRODORI_CODEC_PRECISION": codec_precision,
        "IRODORI_ENGINE_NAME": engine_name,
        "IRODORI_NUM_STEPS": "" if num_steps is None else str(num_steps),
        "IRODORI_PROMPT_WAV": prompt_wav,
        "IRODORI_DEFAULT_VOICE": default_voice,
        "OPENAI_MODEL_ID": settings.openai_model_id or checkpoint,
    }
    log_path = settings.log_dir / log_filename
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
        log_path=log_path,
    )
    return {"proc": proc, "app_dir": repo_dir, "log_path": log_path}


def install(settings: Settings) -> dict:
    return install_runtime(
        settings,
        engine_dir_name="Irodori-TTS",
        app_source="src/apps/irodori_app.py",
        checkpoint=settings.irodori_hf_checkpoint,
        codec_repo=settings.irodori_codec_repo,
        model_precision=settings.irodori_model_precision,
        codec_precision=settings.irodori_codec_precision,
        engine_name="Irodori-TTS",
        log_filename="irodori-uvicorn.log",
        num_steps=40,
        prompt_wav=settings.irodori_prompt_wav,
        default_voice=settings.irodori_default_voice,
    )
