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


ZONOS2_REPO_URL = "https://github.com/Zyphra/Zonos2.git"


def install(settings: Settings) -> dict:
    engine_dir = settings.engines_dir / "zonos2"
    engine_dir.mkdir(parents=True, exist_ok=True)

    repo_dir = engine_dir / "Zonos2"
    ensure_git_clone(ZONOS2_REPO_URL, repo_dir)

    # ZONOS2 ships a uv project (pyproject + uv.lock) whose backbone relies on
    # flashinfer / sgl_kernel / cutlass GPU kernels (sm_80+). `uv sync` builds
    # the project's own .venv from the lockfile; we launch the bundled
    # Mini-SGLang server from it via `uv run`.
    run(["uv", "sync"], cwd=str(repo_dir))

    default_voices_dir = repo_dir / "default_voices"
    backend_port = settings.zonos2_backend_port
    backend_env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
    }
    backend_proc = popen(
        [
            "uv",
            "run",
            "python",
            "-m",
            "zonos2",
            "--model-path",
            settings.zonos2_hf_model,
            "--tts-default-voices-dir",
            str(default_voices_dir),
            "--host",
            "127.0.0.1",
            "--port",
            str(backend_port),
        ],
        cwd=str(repo_dir),
        env=backend_env,
        log_path=settings.log_dir / "zonos2-backend.log",
    )
    # The frontend binds the HTTP port shortly after launch; the Mini-SGLang
    # worker processes keep loading the model (and downloading weights on the
    # first run) in the background, so the first /tts/generate call blocks
    # until they are ready. We only wait for the port to come up here.
    if not wait_http(f"http://127.0.0.1:{backend_port}/v1", timeout=600):
        tail_log(settings.log_dir / "zonos2-backend.log")
        raise RuntimeError("ZONOS2 Mini-SGLang backend did not become ready.")

    # Resolve the default reference voice to an absolute path. A bare filename
    # is looked up inside the cloned repo's default_voices/ directory.
    default_ref = settings.zonos2_default_ref
    if default_ref and not os.path.isabs(default_ref):
        default_ref = str(default_voices_dir / default_ref)

    # The OpenAI-compatible proxy runs in its own lightweight venv so the
    # uv-synced project environment stays untouched.
    python_bin = ensure_venv(engine_dir)
    uv_pip_install(python_bin, ["fastapi", "uvicorn", "requests", "soundfile", "numpy"])
    write_text(engine_dir / "app.py", settings.read_repo_text("src/apps/zonos2_app.py"))

    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "OPENAI_MODEL_ID": settings.openai_model_id or "zonos2",
        "ZONOS2_BACKEND_URL": f"http://127.0.0.1:{backend_port}",
        "ZONOS2_LANGUAGE": settings.zonos2_language,
        "ZONOS2_DEFAULT_REF": default_ref,
        "ZONOS2_PROMPT_WAV": settings.zonos2_prompt_wav,
        "ZONOS2_DEFAULT_VOICE": settings.zonos2_default_voice,
        "ZONOS2_ACCURATE_MODE": "1" if settings.zonos2_accurate_mode else "0",
        "ZONOS2_SEED": str(settings.zonos2_seed) if settings.zonos2_seed >= 0 else "",
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
        log_path=settings.log_dir / "zonos2-uvicorn.log",
    )
    return {
        "proc": proc,
        "backend_proc": backend_proc,
        "app_dir": engine_dir,
        "log_path": settings.log_dir / "zonos2-uvicorn.log",
        "backend_log_path": settings.log_dir / "zonos2-backend.log",
    }
