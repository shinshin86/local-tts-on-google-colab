from __future__ import annotations

import os
from pathlib import Path

from src.config import Settings
from src.runtime import ensure_git_clone, popen, run, tail_log, uv_pip_install, wait_http, write_text


SGLANG_OMNI_REPO_URL = "https://github.com/sgl-project/sglang-omni.git"
# Pin a known-good commit. sglang-omni has no PyPI release and its main branch
# moves fast; pinning keeps the heavy dependency resolution reproducible.
SGLANG_OMNI_REF = "01400e9504599fed11561b4fef82ad8a983abe0c"


def _ensure_py312_venv(engine_dir: Path) -> Path:
    venv_dir = engine_dir / ".venv"
    if not venv_dir.exists():
        # sglang-omni declares requires-python >=3.10; its pinned stack
        # (torch==2.9.1 cp312 / sgl-kernel==0.3.21 cp310-abi3) installs cleanly
        # on 3.12, which is also Colab's system interpreter. Pin 3.12 to match
        # the verified PoC environment.
        run(["uv", "venv", "--python", "3.12", str(venv_dir)])
    return venv_dir / "bin" / "python"


def install(settings: Settings) -> dict:
    engine_dir = settings.engines_dir / "higgs-audio-v3"
    engine_dir.mkdir(parents=True, exist_ok=True)

    repo_dir = engine_dir / "sglang-omni"
    ensure_git_clone(SGLANG_OMNI_REPO_URL, repo_dir)
    run(["git", "-C", str(repo_dir), "checkout", SGLANG_OMNI_REF], check=False)

    python_bin = _ensure_py312_venv(engine_dir)

    # Higgs v3 ships only as weights served by SGLang-Omni (no PyPI package, and
    # the boson-ai/higgs-audio GitHub repo is for v2 only). `pip install -e .`
    # FAILS here: descript-audiotools pins protobuf<3.20 while grpcio 1.75.1
    # needs >=6.31.1. uv resolves it via the `[tool.uv] override-dependencies`
    # declared in sglang-omni's pyproject.toml, but only when invoked from the
    # repo root, so cwd must be repo_dir.
    uv_pip_install(python_bin, ["-e", "."], cwd=str(repo_dir))
    # The proxy below runs inside this venv; fastapi/uvicorn/httpx are already
    # transitive deps, but pin them explicitly so the wrapper never breaks if
    # upstream drops one.
    uv_pip_install(python_bin, ["fastapi", "uvicorn", "httpx"])

    # Start the SGLang-Omni backend, which natively exposes an OpenAI-compatible
    # /v1/audio/speech. Weights (~9.3GB) download on first launch (ungated, no
    # HF_TOKEN). Running in this isolated venv avoids Colab's preinstalled
    # TensorFlow, whose bundled protobuf otherwise double-registers descriptors
    # and crashes sgl-omni at startup.
    backend_env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "HF_HUB_ETAG_TIMEOUT": "60",
        "HF_HUB_DOWNLOAD_TIMEOUT": "60",
    }
    backend_proc = popen(
        [
            str(engine_dir / ".venv" / "bin" / "sgl-omni"),
            "serve",
            "--model-path",
            settings.higgs_v3_hf_model,
            "--host",
            "127.0.0.1",
            "--port",
            str(settings.higgs_v3_backend_port),
            "--log-level",
            "info",
        ],
        cwd=str(engine_dir),
        env=backend_env,
        log_path=settings.log_dir / "higgs-audio-v3-backend.log",
    )
    # Startup is slow: download + weight load + torch.compile / CUDA-graph
    # capture took ~10-12 min on an L4 in testing. Allow a generous window.
    if not wait_http(
        f"http://127.0.0.1:{settings.higgs_v3_backend_port}/health", timeout=1200
    ):
        tail_log(settings.log_dir / "higgs-audio-v3-backend.log")
        raise RuntimeError("Higgs Audio v3 (SGLang-Omni) backend did not become ready.")

    write_text(engine_dir / "app.py", settings.read_repo_text("src/apps/higgs_v3_app.py"))

    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "OPENAI_MODEL_ID": settings.openai_model_id or "higgs-audio-v3",
        "HIGGS_V3_BACKEND_URL": f"http://127.0.0.1:{settings.higgs_v3_backend_port}",
        "HIGGS_V3_DEFAULT_VOICE": settings.higgs_v3_default_voice,
        "HIGGS_V3_PROMPT_WAV": settings.higgs_v3_prompt_wav,
        "HIGGS_V3_PROMPT_TEXT": settings.higgs_v3_prompt_text,
        "HIGGS_V3_TEMPERATURE": str(settings.higgs_v3_temperature),
        "HIGGS_V3_TOP_K": str(settings.higgs_v3_top_k),
        "HIGGS_V3_MAX_NEW_TOKENS": str(settings.higgs_v3_max_new_tokens),
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
        log_path=settings.log_dir / "higgs-audio-v3-proxy-uvicorn.log",
    )
    return {
        "proc": proc,
        "backend_proc": backend_proc,
        "app_dir": engine_dir,
        "log_path": settings.log_dir / "higgs-audio-v3-proxy-uvicorn.log",
        "backend_log_path": settings.log_dir / "higgs-audio-v3-backend.log",
    }
