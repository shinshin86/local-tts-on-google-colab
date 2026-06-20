from __future__ import annotations

import os

from src.config import Settings
from src.runtime import (
    ensure_git_clone,
    ensure_venv,
    popen,
    run,
    tail_log,
    uv_pip_install,
    wait_http,
    write_text,
)


MIOTTS_REPO_URL = "https://github.com/Aratako/MioTTS-Inference.git"


def install(settings: Settings) -> dict:
    engine_dir = settings.engines_dir / "miotts"
    engine_dir.mkdir(parents=True, exist_ok=True)

    repo_dir = engine_dir / "MioTTS-Inference"
    ensure_git_clone(MIOTTS_REPO_URL, repo_dir)

    # MioTTS ships a uv project (pyproject + uv.lock, requires Python >=3.12)
    # pulling transformers + miocodec (built from git). `uv sync` materializes
    # the project's own .venv; we launch run_server.py from it via `uv run`.
    run(["uv", "sync"], cwd=str(repo_dir))

    # --- LLM backend (llama-cpp-python OpenAI-compatible server) -------------
    # MioTTS' LLM client hits /v1/chat/completions and relies on the chat
    # template embedded in the GGUF, so we host the GGUF with llama.cpp rather
    # than re-templating ourselves. No prebuilt Linux CUDA `llama-server` binary
    # exists, so we use the prebuilt llama-cpp-python CUDA wheel instead.
    llm_dir = engine_dir / "llm"
    llm_dir.mkdir(parents=True, exist_ok=True)
    llm_python = ensure_venv_py312(llm_dir)
    cuda_index = f"https://abetlen.github.io/llama-cpp-python/whl/{settings.miotts_llama_cuda}"
    run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(llm_python),
            "--extra-index-url",
            cuda_index,
            "--index-strategy",
            "unsafe-best-match",
            # Force the CUDA wheel from the index above (PyPI only ships an sdist
            # that would build a CPU-only extension).
            "--only-binary",
            "llama-cpp-python",
            "llama-cpp-python[server]",
            "huggingface_hub",
            "hf_transfer",
        ]
    )

    # Pre-download the GGUF so failures surface here (not on the first request).
    download_env = {**os.environ, "HF_HUB_ENABLE_HF_TRANSFER": "1"}
    completed = run(
        [
            str(llm_python),
            "-c",
            (
                "from huggingface_hub import hf_hub_download;"
                f"print(hf_hub_download({settings.miotts_gguf_repo!r}, {settings.miotts_gguf_file!r}))"
            ),
        ],
        env=download_env,
        capture_output=True,
    )
    gguf_path = completed.stdout.strip().splitlines()[-1].strip()
    print(f"GGUF: {gguf_path}")

    llm_port = settings.miotts_llm_port
    llm_proc = popen(
        [
            str(llm_python),
            "-m",
            "llama_cpp.server",
            "--model",
            gguf_path,
            "--n_gpu_layers",
            "-1",
            "--n_ctx",
            str(settings.miotts_n_ctx),
            "--host",
            "127.0.0.1",
            "--port",
            str(llm_port),
        ],
        cwd=str(llm_dir),
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
        log_path=settings.log_dir / "miotts-llm.log",
    )
    if not wait_http(f"http://127.0.0.1:{llm_port}/v1/models", timeout=600):
        tail_log(settings.log_dir / "miotts-llm.log")
        raise RuntimeError("MioTTS llama-cpp-python LLM server did not become ready.")

    # --- Synthesis backend (run_server.py) ----------------------------------
    backend_port = settings.miotts_backend_port
    backend_env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
    }
    backend_proc = popen(
        [
            "uv",
            "run",
            "python",
            "run_server.py",
            "--host",
            "127.0.0.1",
            "--port",
            str(backend_port),
            "--llm-base-url",
            f"http://127.0.0.1:{llm_port}/v1",
            "--codec-model",
            settings.miotts_codec_model,
            "--device",
            "cuda",
            "--presets-dir",
            "presets",
        ],
        cwd=str(repo_dir),
        env=backend_env,
        log_path=settings.log_dir / "miotts-backend.log",
    )
    # /v1/presets is only served once the codec service has finished loading,
    # so it doubles as a readiness probe (model/codec download on first run).
    if not wait_http(f"http://127.0.0.1:{backend_port}/v1/presets", timeout=900):
        tail_log(settings.log_dir / "miotts-backend.log")
        raise RuntimeError("MioTTS synthesis backend did not become ready.")

    # --- OpenAI-compatible proxy (our app.py) -------------------------------
    python_bin = ensure_venv(engine_dir)
    uv_pip_install(python_bin, ["fastapi", "uvicorn", "requests"])
    write_text(engine_dir / "app.py", settings.read_repo_text("src/apps/miotts_app.py"))

    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "OPENAI_MODEL_ID": settings.openai_model_id or "miotts",
        "MIOTTS_BACKEND_URL": f"http://127.0.0.1:{backend_port}",
        "MIOTTS_DEFAULT_VOICE": settings.miotts_default_voice,
        "MIOTTS_DEFAULT_PRESET": settings.miotts_default_preset,
        "MIOTTS_PROMPT_WAV": settings.miotts_prompt_wav,
        "MIOTTS_TEMPERATURE": str(settings.miotts_temperature),
        "MIOTTS_TOP_P": str(settings.miotts_top_p),
        "MIOTTS_REPETITION_PENALTY": str(settings.miotts_repetition_penalty),
        "MIOTTS_MAX_TOKENS": str(settings.miotts_max_tokens),
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
        log_path=settings.log_dir / "miotts-uvicorn.log",
    )
    return {
        "proc": proc,
        "backend_proc": backend_proc,
        "llm_proc": llm_proc,
        "app_dir": engine_dir,
        "log_path": settings.log_dir / "miotts-uvicorn.log",
        "backend_log_path": settings.log_dir / "miotts-backend.log",
        "llm_log_path": settings.log_dir / "miotts-llm.log",
    }


def ensure_venv_py312(engine_dir):
    """Create a dedicated Python 3.12 venv (matches the cp312 CUDA wheels)."""
    venv_dir = engine_dir / ".venv"
    if not venv_dir.exists():
        run(["uv", "venv", "--python", "3.12", str(venv_dir)])
    return venv_dir / "bin" / "python"
