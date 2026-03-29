from __future__ import annotations

import os

from src.config import Settings
from src.runtime import ensure_venv, popen, uv_pip_install, write_text


def _find_model_paths(engine_dir):
    """Find the downloaded onnx and config paths in engine_dir."""
    onnx_files = list(engine_dir.glob("*.onnx"))
    if not onnx_files:
        return None, None
    onnx_path = str(onnx_files[0])
    config_path = ""
    json_candidate = engine_dir / (onnx_files[0].name + ".json")
    if json_candidate.exists():
        config_path = str(json_candidate)
    else:
        plain_config = engine_dir / "config.json"
        if plain_config.exists():
            config_path = str(plain_config)
    return onnx_path, config_path


def install(settings: Settings) -> dict:
    engine_dir = settings.engines_dir / "piper-plus-openai"
    engine_dir.mkdir(parents=True, exist_ok=True)
    python_bin = ensure_venv(engine_dir)
    uv_pip_install(
        python_bin,
        ["fastapi", "uvicorn", "piper-tts-plus"],
    )
    # Download NLTK data required by g2p-en
    from src.runtime import run
    run(
        [str(python_bin), "-c", "import nltk; nltk.download('averaged_perceptron_tagger_eng')"],
        cwd=str(engine_dir),
    )
    # Download model using piper CLI (files go to cwd = engine_dir)
    run(
        [str(python_bin), "-m", "piper", "--download-model", settings.piper_plus_model],
        cwd=str(engine_dir),
    )
    onnx_path, config_path = _find_model_paths(engine_dir)
    if not onnx_path:
        raise RuntimeError(f"No .onnx model found after downloading {settings.piper_plus_model}")

    write_text(engine_dir / "app.py", settings.read_repo_text("src/apps/piper_plus_app.py"))
    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "OPENAI_MODEL_ID": settings.openai_model_id or settings.piper_plus_model,
        "PIPER_PLUS_ONNX": onnx_path,
        "PIPER_PLUS_CONFIG": config_path or "",
        "PIPER_PLUS_DEFAULT_LANGUAGE": "ja",
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
        log_path=settings.log_dir / "piper-plus-uvicorn.log",
    )
    return {"proc": proc, "app_dir": engine_dir, "log_path": settings.log_dir / "piper-plus-uvicorn.log"}
