from __future__ import annotations

import os

from src.config import Settings
from src.runtime import ensure_venv, popen, run, uv_pip_install, write_text


def install(settings: Settings) -> dict:
    engine_dir = settings.engines_dir / "kokoro-onnx-openai"
    engine_dir.mkdir(parents=True, exist_ok=True)
    python_bin = ensure_venv(engine_dir)
    # espeak-ng backs misaki's G2P (English OOV fallback + es/fr/hi/it/pt).
    run(["apt-get", "-qq", "-y", "install", "espeak-ng"])
    uv_pip_install(
        python_bin,
        [
            "fastapi",
            "uvicorn",
            "onnxruntime-gpu",
            "huggingface_hub",
            "soundfile",
            "numpy",
            "misaki[en]",
            "misaki[ja]",
            "misaki[zh]",
            "unidic",
        ],
    )
    # JAG2P (Japanese) needs the full UniDic dictionary, not just unidic-lite.
    run([str(python_bin), "-m", "unidic", "download"], cwd=str(engine_dir))
    write_text(engine_dir / "app.py", settings.read_repo_text("src/apps/kokoro_onnx_app.py"))
    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "OPENAI_MODEL_ID": settings.openai_model_id or "kokoro-82m-onnx",
        "KOKORO_ONNX_HF_MODEL": settings.kokoro_onnx_hf_model,
        "KOKORO_ONNX_DEFAULT_VOICE": settings.kokoro_onnx_default_voice,
        "KOKORO_ONNX_DEFAULT_LANG_CODE": settings.kokoro_onnx_default_lang_code,
        "KOKORO_ONNX_PROVIDER": settings.kokoro_onnx_provider,
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
        log_path=settings.log_dir / "kokoro-onnx-uvicorn.log",
    )
    return {
        "proc": proc,
        "app_dir": engine_dir,
        "log_path": settings.log_dir / "kokoro-onnx-uvicorn.log",
    }
