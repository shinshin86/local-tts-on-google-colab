from __future__ import annotations

import os

from src.config import Settings
from src.runtime import ensure_venv, popen, run, uv_pip_install, write_text


MODEL_ID = "n4ze3m/tiny-voca"
MODEL_REF = "1e73262226dbf4ef02e9df277e81aaa1db5ce16c"
TINY_VOCA = (
    "tiny-voca @ git+https://github.com/n4ze3m/tiny-voca.git"
    "@0eea2be213b53ec0b1949c169b80e611544718e1"
)
DIZIONE = (
    "dizione @ git+https://github.com/n4ze3m/dizione.git"
    "@8cd889e4de7be6bd1dd2e6891344a0729fa14dd4"
)


def install(settings: Settings) -> dict:
    engine_dir = settings.engines_dir / "tiny-voca"
    engine_dir.mkdir(parents=True, exist_ok=True)
    python_bin = ensure_venv(engine_dir)
    uv_pip_install(
        python_bin,
        ["--index-url", "https://download.pytorch.org/whl/cpu", "torch==2.9.1+cpu"],
    )
    uv_pip_install(
        python_bin, [TINY_VOCA, DIZIONE, "fastapi", "uvicorn", "soundfile", "huggingface-hub"],
    )

    model_dir = engine_dir / "models" / MODEL_REF
    run(
        [
            str(python_bin), "-c",
            "import sys; from huggingface_hub import snapshot_download; "
            "snapshot_download(sys.argv[1], revision=sys.argv[2], local_dir=sys.argv[3], "
            "allow_patterns=['tiny-voca.pt', 'phonemizer/*', 'LICENSE'])",
            MODEL_ID, MODEL_REF, str(model_dir),
        ]
    )

    write_text(engine_dir / "app.py", settings.read_repo_text("src/apps/tiny_voca_app.py"))
    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "OPENAI_MODEL_ID": settings.openai_model_id or "tiny-voca",
        "TINY_VOCA_MODEL_PATH": str(model_dir / "tiny-voca.pt"),
        "TINY_VOCA_FRONTEND": "own",
        "DIZIONE_DATA": str(model_dir / "phonemizer"),
        "TINY_VOCA_DEFAULT_VOICE": settings.tiny_voca_default_voice,
    }
    log_path = settings.log_dir / "tiny-voca-uvicorn.log"
    proc = popen(
        [
            str(python_bin), "-m", "uvicorn", "app:app",
            "--host", "0.0.0.0", "--port", str(settings.app_port), "--log-level", "info",
        ],
        cwd=str(engine_dir),
        env=env,
        log_path=log_path,
    )
    return {"proc": proc, "app_dir": engine_dir, "log_path": log_path}
