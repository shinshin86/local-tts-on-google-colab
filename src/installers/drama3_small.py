from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

from src.config import Settings
from src.runtime import ensure_venv, popen, run, tail_log, uv_pip_install, wait_http, write_text


# drama3-small needs audio.cpp at this exact commit plus the patch shipped in the model repo.
# The commit is not on any branch of JarodMica/audio.cpp, so it is fetched as a tarball by SHA.
AUDIOCPP_TARBALL_URL = "https://codeload.github.com/JarodMica/audio.cpp/tar.gz/{commit}"
HF_RESOLVE_URL = "https://huggingface.co/{repo}/resolve/{revision}/{filename}"
GGUF_FILENAME = "drama3-small-IQ3_XXS.gguf"
PATCH_FILENAME = "drama3-small-audiocpp.patch"
MODEL_ID = "drama3-small"
CLONE_PRESET = "clone"


def _cuda_arch() -> str:
    """Return the GPU's compute capability as CMake expects it (e.g. "89" on an L4)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip().splitlines()[0]
    except (OSError, subprocess.CalledProcessError, IndexError) as exc:
        raise RuntimeError("drama3-small needs an NVIDIA GPU runtime (nvidia-smi was not usable).") from exc
    return out.strip().replace(".", "")


def _build_audiocpp(settings: Settings, src_dir: Path) -> Path:
    server_bin = src_dir / "build" / "linux-cuda-release" / "bin" / "audiocpp_server"
    stamp = src_dir / ".drama3-build-stamp"
    expected = f"{settings.drama3_audiocpp_commit}:{settings.drama3_hf_revision}"
    if server_bin.exists() and stamp.exists() and stamp.read_text().strip() == expected:
        print(f"drama3-small: reusing the audio.cpp build at {server_bin}")
        return server_bin

    if src_dir.exists():
        shutil.rmtree(src_dir)
    src_dir.mkdir(parents=True)
    tarball = src_dir.parent / "audio.cpp.tar.gz"
    patch = src_dir.parent / PATCH_FILENAME
    run(["curl", "-fsSL", "-o", str(tarball), AUDIOCPP_TARBALL_URL.format(commit=settings.drama3_audiocpp_commit)])
    run(["tar", "-xzf", str(tarball), "-C", str(src_dir), "--strip-components=1"])
    run(
        [
            "curl",
            "-fsSL",
            "-o",
            str(patch),
            HF_RESOLVE_URL.format(
                repo=settings.drama3_hf_repo, revision=settings.drama3_hf_revision, filename=PATCH_FILENAME
            ),
        ]
    )
    run(["git", "apply", str(patch)], cwd=str(src_dir))
    print("drama3-small: building audio.cpp with CUDA (this takes a while on the first run)...")
    run(
        [
            "bash",
            "scripts/build_linux.sh",
            "--cuda",
            "on",
            "--cuda-arch",
            settings.drama3_cuda_arch or _cuda_arch(),
            # Without "--model-set custom" the script ignores --models and builds every model family.
            "--model-set",
            "custom",
            "--models",
            "fish_drama3",
            "--target",
            "audiocpp_server",
        ],
        cwd=str(src_dir),
    )
    if not server_bin.exists():
        raise RuntimeError(f"audio.cpp build did not produce {server_bin}.")
    stamp.write_text(expected + "\n")
    return server_bin


def _download_model(settings: Settings, model_dir: Path) -> Path:
    python_bin = ensure_venv(model_dir.parent)
    uv_pip_install(python_bin, ["huggingface_hub"])
    run(
        [
            str(python_bin),
            "-c",
            "import sys; from huggingface_hub import snapshot_download; "
            "snapshot_download(sys.argv[1], revision=sys.argv[2], local_dir=sys.argv[3], "
            "allow_patterns=[sys.argv[4], 'silence_codes.json', 'LICENSE', 'NOTICE'])",
            settings.drama3_hf_repo,
            settings.drama3_hf_revision,
            str(model_dir),
            GGUF_FILENAME,
        ]
    )
    gguf = model_dir / GGUF_FILENAME
    if not gguf.exists():
        raise RuntimeError(f"{GGUF_FILENAME} was not downloaded into {model_dir}.")
    return gguf


def install(settings: Settings) -> dict:
    app_dir = settings.engines_dir / "drama3-small"
    model_dir = app_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    server_bin = _build_audiocpp(settings, app_dir / "audio.cpp")
    gguf = _download_model(settings, model_dir)

    clone_enabled = bool(settings.drama3_prompt_wav)
    if clone_enabled:
        if not Path(settings.drama3_prompt_wav).is_file():
            raise RuntimeError(f"--drama3-prompt-wav does not exist: {settings.drama3_prompt_wav}")
        if not settings.drama3_prompt_text.strip():
            # audio.cpp rejects a Drama 3 reference without its transcript.
            raise RuntimeError("--drama3-prompt-wav needs --drama3-prompt-text (the transcript of the reference).")

    model_entry: dict = {
        "id": MODEL_ID,
        "family": "fish_drama3",
        "path": str(gguf),
        # "tts" accepts requests with or without a reference; "clon" refuses requests without one.
        "task": "tts",
        "mode": "offline",
        "lazy": False,
        "session_options": {
            "fish_drama3.cpu_expert_layers": "0",
            "fish_drama3.context_length": str(settings.drama3_context_length),
        },
        "default_request_options": {
            "temperature": settings.drama3_temperature,
            "top_p": settings.drama3_top_p,
            "top_k": settings.drama3_top_k,
            "max_tokens": settings.drama3_max_tokens,
        },
    }
    if clone_enabled:
        model_entry["voice_presets"] = {
            CLONE_PRESET: {
                "voice_ref": str(Path(settings.drama3_prompt_wav).resolve()),
                "reference_text": settings.drama3_prompt_text.strip(),
            }
        }

    config = {
        "host": "127.0.0.1",
        "port": settings.drama3_backend_port,
        "backend": "cuda",
        "device": 0,
        "threads": 1,
        "lazy_load": False,
        "models": [model_entry],
    }
    config_path = app_dir / "server.json"
    write_text(config_path, json.dumps(config, ensure_ascii=False, indent=2) + "\n")

    backend_env = {**os.environ}
    if settings.drama3_max_pause_frames > 0:
        # Pause fix from the drama3-small patch: stops silent runaways at tags / sentence boundaries.
        backend_env["AUDIOCPP_DRAMA3_MAX_PAUSE_FRAMES"] = str(settings.drama3_max_pause_frames)
        backend_env["AUDIOCPP_DRAMA3_SILENCE_CODES"] = str(model_dir / "silence_codes.json")

    backend_log_path = settings.log_dir / "drama3-small-backend.log"
    backend_proc = popen(
        [str(server_bin), "--config", str(config_path)],
        cwd=str(server_bin.parent.parent.parent.parent),
        env=backend_env,
        log_path=backend_log_path,
    )
    if not wait_http(f"http://127.0.0.1:{settings.drama3_backend_port}/v1/models", timeout=1200):
        tail_log(backend_log_path)
        raise RuntimeError("audio.cpp server for drama3-small did not become ready.")
    tail_log(backend_log_path, lines=30)

    python_bin = ensure_venv(app_dir)
    uv_pip_install(python_bin, ["fastapi", "uvicorn", "requests"])
    write_text(app_dir / "app.py", settings.read_repo_text("src/apps/drama3_small_app.py"))

    wrapper_log_path = settings.log_dir / "drama3-small-uvicorn.log"
    proc = popen(
        [
            str(app_dir / ".venv" / "bin" / "uvicorn"),
            "app:app",
            "--host",
            "0.0.0.0",
            "--port",
            str(settings.app_port),
            "--log-level",
            "info",
        ],
        cwd=str(app_dir),
        env={
            **os.environ,
            "PYTHONUNBUFFERED": "1",
            "BACKEND_URL": f"http://127.0.0.1:{settings.drama3_backend_port}",
            "BACKEND_MODEL_ID": MODEL_ID,
            "OPENAI_MODEL_ID": settings.openai_model_id or MODEL_ID,
            "DRAMA3_DEFAULT_VOICE": settings.drama3_default_voice,
            "DRAMA3_CLONE_ENABLED": "1" if clone_enabled else "0",
            "DRAMA3_INSTRUCTIONS": settings.drama3_instructions,
        },
        log_path=wrapper_log_path,
    )
    return {
        "proc": proc,
        "backend_proc": backend_proc,
        "app_dir": app_dir,
        "log_path": wrapper_log_path,
        "backend_log_path": backend_log_path,
    }
