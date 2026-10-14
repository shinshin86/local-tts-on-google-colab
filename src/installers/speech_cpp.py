from __future__ import annotations

import glob
import json
import os
import shutil
from pathlib import Path

from src.config import Settings
from src.runtime import ensure_venv, popen, run, tail_log, uv_pip_install, wait_http, write_text


INSTALL_SCRIPT_URL = "https://raw.githubusercontent.com/nyosegawa/speech.cpp/v{version}/install.sh"
QWEN3_MODELS = {"qwen3-tts-0.6b", "qwen3-tts-1.7b"}
IRODORI_MODELS = {"irodori-tts-mf", "irodori-tts"}
ICD_DIRS = ["/etc/vulkan/icd.d", "/usr/share/vulkan/icd.d"]


def _find_nvidia_vulkan_library() -> str:
    """Return the path of NVIDIA's Vulkan ICD library, or "" when the driver ships none.

    The EGL library comes first: on Colab's headless L4, libGLX_nvidia.so.0 cannot create a Vulkan
    instance ("Could not get 'vkCreateInstance'"), while libEGL_nvidia.so.0 exposes the GPU.
    """
    candidates = [
        *glob.glob("/usr/lib64-nvidia/libEGL_nvidia.so.0"),
        *glob.glob("/usr/lib/x86_64-linux-gnu/libEGL_nvidia.so.0"),
        *glob.glob("/usr/lib64-nvidia/libGLX_nvidia.so.0"),
        *glob.glob("/usr/lib/x86_64-linux-gnu/libGLX_nvidia.so.0"),
    ]
    return candidates[0] if candidates else ""


def ensure_vulkan_runtime() -> None:
    """Install the Vulkan loader and register NVIDIA's ICD so speech.cpp's Vulkan build sees the GPU.

    Colab mounts the NVIDIA driver libraries but ships neither libvulkan1 nor an ICD manifest. Without
    the loader the installer picks the CPU build; with the loader but no ICD the Vulkan build runs on
    the CPU, so every failure here degrades to CPU rather than stopping the launch.
    """
    if not shutil.which("nvidia-smi"):
        print("speech.cpp: nvidia-smi not found; using the CPU build.")
        return
    run(["bash", "-lc", "apt-get -qq update && apt-get -qq install -y libvulkan1 >/dev/null"], check=False)
    if any(glob.glob(f"{d}/nvidia_icd*.json") for d in ICD_DIRS):
        return
    library = _find_nvidia_vulkan_library()
    if not library:
        print("speech.cpp: NVIDIA Vulkan ICD library not found; the Vulkan build will fall back to the CPU.")
        return
    icd_dir = Path(ICD_DIRS[0])
    icd_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"file_format_version": "1.0.0", "ICD": {"library_path": library, "api_version": "1.3.0"}}
    write_text(icd_dir / "nvidia_icd.json", json.dumps(manifest, indent=2) + "\n")
    print(f"speech.cpp: registered NVIDIA Vulkan ICD -> {library}")


def install(settings: Settings) -> dict:
    model = settings.speech_cpp_model
    if model not in QWEN3_MODELS | IRODORI_MODELS:
        raise RuntimeError(
            f"Unsupported speech.cpp model: {model}. Choose one of {sorted(QWEN3_MODELS | IRODORI_MODELS)}."
        )

    app_dir = settings.engines_dir / "speech-cpp-openai"
    model_dir = app_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    if settings.speech_cpp_use_vulkan:
        ensure_vulkan_runtime()
    run(
        [
            "bash",
            "-lc",
            f"curl -fsSL {INSTALL_SCRIPT_URL.format(version=settings.speech_cpp_version)}"
            f" | sh -s -- --version {settings.speech_cpp_version} --no-modify-path",
        ]
    )
    speech_bin = Path.home() / ".local" / "bin" / "speech"
    if not speech_bin.exists():
        raise RuntimeError(f"speech.cpp installer did not create {speech_bin}.")

    speech_env = {
        **os.environ,
        "SPEECH_MODEL_DIR": str(model_dir),
        "SPEECH_NO_UPDATE_CHECK": "1",
    }
    run([str(speech_bin), "--version"], env=speech_env, check=False)
    run([str(speech_bin), "devices"], env=speech_env, check=False)
    run([str(speech_bin), "pull", model], env=speech_env)

    serve_cmd = [
        str(speech_bin),
        "serve",
        model,
        "--host",
        "127.0.0.1",
        "--port",
        str(settings.speech_cpp_backend_port),
        "--device",
        settings.speech_cpp_device,
    ]
    if settings.speech_cpp_language:
        serve_cmd += ["--language", settings.speech_cpp_language]
    clone_enabled = False
    if model in IRODORI_MODELS:
        if settings.speech_cpp_steps > 0:
            serve_cmd += ["--steps", str(settings.speech_cpp_steps)]
        if settings.speech_cpp_prompt_wav:
            serve_cmd += ["--add-voice", f"clone={settings.speech_cpp_prompt_wav}"]
            clone_enabled = True
    elif settings.speech_cpp_prompt_wav:
        print("speech.cpp: Qwen3-TTS CustomVoice cannot clone; --speech-cpp-prompt-wav is ignored.")

    backend_log_path = settings.log_dir / "speech-cpp-backend.log"
    backend_proc = popen(serve_cmd, cwd=str(app_dir), env=speech_env, log_path=backend_log_path)
    if not wait_http(f"http://127.0.0.1:{settings.speech_cpp_backend_port}/health", timeout=600):
        tail_log(backend_log_path)
        raise RuntimeError("speech.cpp server did not become ready.")
    tail_log(backend_log_path, lines=30)

    python_bin = ensure_venv(app_dir)
    uv_pip_install(python_bin, ["fastapi", "uvicorn", "requests"])
    write_text(app_dir / "app.py", settings.read_repo_text("src/apps/speech_cpp_app.py"))

    wrapper_log_path = settings.log_dir / "speech-cpp-uvicorn.log"
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
            "BACKEND_URL": f"http://127.0.0.1:{settings.speech_cpp_backend_port}",
            "OPENAI_MODEL_ID": settings.openai_model_id or model,
            "SPEECH_CPP_MODEL": model,
            "SPEECH_CPP_DEFAULT_VOICE": settings.speech_cpp_default_voice,
            "SPEECH_CPP_QWEN3_SPEAKER": settings.speech_cpp_qwen3_speaker,
            "SPEECH_CPP_CLONE_ENABLED": "1" if clone_enabled else "0",
            "SPEECH_CPP_INSTRUCTIONS": settings.speech_cpp_instructions,
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
