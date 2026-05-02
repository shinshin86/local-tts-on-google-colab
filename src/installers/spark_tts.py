from __future__ import annotations

import os

from src.config import Settings
from src.runtime import ensure_venv, popen, run, uv_pip_install, write_text


SPARK_REPO_URL = "https://github.com/SparkAudio/Spark-TTS.git"


def install(settings: Settings) -> dict:
    engine_dir = settings.engines_dir / "spark-tts"
    engine_dir.mkdir(parents=True, exist_ok=True)

    repo_dir = engine_dir / "Spark-TTS"
    if not repo_dir.exists():
        run(["git", "clone", SPARK_REPO_URL, str(repo_dir)])

    python_bin = ensure_venv(engine_dir)

    uv_pip_install(python_bin, ["-r", str(repo_dir / "requirements.txt")])
    uv_pip_install(python_bin, ["fastapi", "uvicorn", "huggingface_hub"])

    model_dir = repo_dir / "pretrained_models" / "Spark-TTS-0.5B"
    if not model_dir.exists():
        run(
            [
                str(python_bin),
                "-c",
                (
                    "from huggingface_hub import snapshot_download; "
                    f"snapshot_download('{settings.spark_hf_model}', local_dir='{model_dir}')"
                ),
            ]
        )

    write_text(engine_dir / "app.py", settings.read_repo_text("src/apps/spark_tts_app.py"))

    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "PYTHONPATH": str(repo_dir),
        "OPENAI_MODEL_ID": settings.openai_model_id or "spark-tts",
        "SPARK_REPO_DIR": str(repo_dir),
        "SPARK_MODEL_DIR": str(model_dir),
        "SPARK_DEFAULT_VOICE": settings.spark_default_voice,
        "SPARK_DEFAULT_GENDER": settings.spark_default_gender,
        "SPARK_DEFAULT_PITCH": settings.spark_default_pitch,
        "SPARK_DEFAULT_SPEED": settings.spark_default_speed,
        "SPARK_PROMPT_WAV": settings.spark_prompt_wav,
        "SPARK_PROMPT_TEXT": settings.spark_prompt_text,
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
        log_path=settings.log_dir / "spark-tts-uvicorn.log",
    )
    return {
        "proc": proc,
        "app_dir": engine_dir,
        "log_path": settings.log_dir / "spark-tts-uvicorn.log",
    }
