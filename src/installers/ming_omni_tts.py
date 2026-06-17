from __future__ import annotations

import os
from pathlib import Path

from src.config import Settings
from src.runtime import ensure_git_clone, popen, run, uv_pip_install, write_text


MING_OMNI_TTS_REPO_URL = "https://github.com/inclusionAI/Ming-omni-tts.git"

# Prebuilt FlashAttention wheel matching the pinned stack (torch 2.6 / cu12 /
# cp310 / cxx11abiFALSE). Upstream's requirements.txt leaves flash_attn
# commented out and tells you to download a wheel manually, because building it
# from source on a CPU-only install step takes 10+ minutes and easily OOMs.
FLASH_ATTN_WHEEL = (
    "https://github.com/Dao-AILab/flash-attention/releases/download/"
    "v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-"
    "cp310-cp310-linux_x86_64.whl"
)


def _ensure_py310_venv(engine_dir: Path) -> Path:
    venv_dir = engine_dir / ".venv"
    if not venv_dir.exists():
        # Upstream pins torch==2.6.0 and ships a cp310 FlashAttention wheel, so
        # build the venv on Python 3.10 (uv fetches the interpreter) rather than
        # Colab's system 3.12. The MoE backbone needs grouped_gemm, which is
        # compiled from source against this exact torch at install time.
        run(["uv", "venv", "--python", "3.10", str(venv_dir)])
    return venv_dir / "bin" / "python"


def install(settings: Settings) -> dict:
    engine_dir = settings.engines_dir / "ming-omni-tts"
    engine_dir.mkdir(parents=True, exist_ok=True)

    repo_dir = engine_dir / "Ming-omni-tts"
    ensure_git_clone(MING_OMNI_TTS_REPO_URL, repo_dir)

    python_bin = _ensure_py310_venv(engine_dir)

    # Install torch first so grouped_gemm (the MoE GEMM kernel) and any other
    # source build can see it. grouped_gemm compiles CUDA against the GPU it
    # finds at build time; on A100 that is sm_80. Pin the arch list explicitly
    # so the build never depends on whatever GPU happens to be visible.
    os.environ["TORCH_CUDA_ARCH_LIST"] = os.environ.get("TORCH_CUDA_ARCH_LIST", "8.0;9.0")
    uv_pip_install(
        python_bin,
        ["torch==2.6.0", "torchaudio==2.6.0", "torchvision==0.21.0"],
    )
    # The rest of the pinned stack (transformers==4.52.4, diffusers, grouped_gemm,
    # x_transformers, torchdiffeq, torchtune, torchao, decord, hyperpyyaml, ...).
    # flash_attn is left commented out in requirements.txt; we install a matching
    # prebuilt wheel below instead of building it from source.
    uv_pip_install(
        python_bin,
        ["-r", str(repo_dir / "requirements.txt")],
        cwd=str(repo_dir),
    )
    uv_pip_install(python_bin, [FLASH_ATTN_WHEEL])
    # onnxruntime drives the campplus.onnx speaker-embedding extractor (needed
    # for voice='clone'); it is not in upstream's requirements.txt. fastapi /
    # uvicorn / soundfile back the in-process OpenAI-compatible server.
    uv_pip_install(python_bin, ["onnxruntime", "fastapi", "uvicorn", "soundfile"])

    # The app imports the repo's top-level modules (modeling_bailingmm,
    # spkemb_extractor) and, for the MoE checkpoint, loads the tokenizer from
    # "." — the repo root carries tokenizer_config.json / tokenization_bailing.py.
    # So the app file lives in repo_dir and uvicorn runs with cwd=repo_dir.
    write_text(repo_dir / "app.py", settings.read_repo_text("src/apps/ming_omni_tts_app.py"))

    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "HF_HUB_ETAG_TIMEOUT": "60",
        "HF_HUB_DOWNLOAD_TIMEOUT": "60",
        "PYTHONPATH": str(repo_dir),
        "OPENAI_MODEL_ID": settings.openai_model_id or "ming-omni-tts",
        "MING_OMNI_TTS_HF_MODEL": settings.ming_omni_tts_hf_model,
        "MING_OMNI_TTS_DEFAULT_VOICE": settings.ming_omni_tts_default_voice,
        "MING_OMNI_TTS_PROMPT_WAV": settings.ming_omni_tts_prompt_wav,
        "MING_OMNI_TTS_PROMPT_TEXT": settings.ming_omni_tts_prompt_text,
        "MING_OMNI_TTS_GEN_PROMPT": settings.ming_omni_tts_gen_prompt,
        "MING_OMNI_TTS_MAX_DECODE_STEPS": str(settings.ming_omni_tts_max_decode_steps),
        "MING_OMNI_TTS_CFG": str(settings.ming_omni_tts_cfg),
        "MING_OMNI_TTS_SIGMA": str(settings.ming_omni_tts_sigma),
        "MING_OMNI_TTS_TEMPERATURE": str(settings.ming_omni_tts_temperature),
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
        cwd=str(repo_dir),
        env=env,
        log_path=settings.log_dir / "ming-omni-tts-uvicorn.log",
    )
    return {
        "proc": proc,
        "app_dir": repo_dir,
        "log_path": settings.log_dir / "ming-omni-tts-uvicorn.log",
    }
