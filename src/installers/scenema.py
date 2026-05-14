from __future__ import annotations

import os
from pathlib import Path

from src.config import Settings
from src.runtime import ensure_git_clone, ensure_venv, popen, uv_pip_install, write_text


SCENEMA_REPO_URL = "https://github.com/ScenemaAI/scenema-audio.git"
SEEDVC_REPO_URL = "https://github.com/Plachtaa/seed-vc.git"
MELBAND_NODE_REPO_URL = "https://github.com/kijai/ComfyUI-MelBandRoFormer.git"

# Pinned in upstream Dockerfile to keep ltx-core / ltx-pipelines compatible
# with the AudioEngine API Scenema imports.
LTX2_PIN = "41d924371612b692c0fd1e4d9d94c3dfb3c02cb3"


def install(settings: Settings) -> dict:
    engine_dir = settings.engines_dir / "scenema"
    engine_dir.mkdir(parents=True, exist_ok=True)

    scenema_dir = engine_dir / "scenema-audio"
    seedvc_dir = engine_dir / "seed-vc"
    melband_dir = engine_dir / "melband_roformer_node"
    models_dir = engine_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    ensure_git_clone(SCENEMA_REPO_URL, scenema_dir)
    ensure_git_clone(SEEDVC_REPO_URL, seedvc_dir)
    ensure_git_clone(MELBAND_NODE_REPO_URL, melband_dir)

    python_bin = ensure_venv(engine_dir)

    # 1) Torch (cu128 wheels, matches scenema-audio Dockerfile pin).
    uv_pip_install(
        python_bin,
        [
            "--index-url",
            "https://download.pytorch.org/whl/cu128",
            "torch==2.7.1",
            "torchaudio==2.7.1",
        ],
    )

    # 2) Core ML deps + LTX-2 packages (pinned subdirectory installs).
    uv_pip_install(
        python_bin,
        [
            "numpy==2.2.6",
            "transformers==4.57.6",
            "accelerate==1.13.0",
            "safetensors==0.7.0",
            "sentencepiece==0.2.1",
            f"ltx-core @ git+https://github.com/Lightricks/LTX-2.git@{LTX2_PIN}#subdirectory=packages/ltx-core",
            f"ltx-pipelines @ git+https://github.com/Lightricks/LTX-2.git@{LTX2_PIN}#subdirectory=packages/ltx-pipelines",
        ],
    )

    # 3) SeedVC / MelBandRoFormer architecture deps.
    uv_pip_install(
        python_bin,
        [
            "scipy==1.13.1",
            "librosa==0.10.2",
            "huggingface-hub==0.36.2",
            "munch==4.0.0",
            "einops==0.8.0",
            "descript-audio-codec==1.0.0",
            "pydub==0.25.1",
            "soundfile==0.12.1",
            "hydra-core==1.3.2",
            "pyyaml==6.0.3",
            "python-dotenv==1.2.2",
            "diffusers==0.37.1",
            "onnxruntime==1.25.0",
            "funasr==1.3.1",
            "rotary-embedding-torch==0.8.9",
            "beartype==0.22.9",
        ],
    )

    # 4) Server-side deps + Kokoro phoneme chunker + faster-whisper validator.
    uv_pip_install(
        python_bin,
        [
            "fastapi==0.136.1",
            "uvicorn[standard]==0.46.0",
            "httpx==0.28.1",
            "psutil==7.2.2",
            "bitsandbytes==0.49.2",
            "kokoro==0.9.4",
            "faster-whisper==1.2.1",
            "ctranslate2==4.7.1",
        ],
    )

    # 5) MelBandRoFormer small fp16 weight (~436MB). Bake here so the
    #    server's lifespan doesn't have to re-download on every cold start.
    melband_weight = models_dir / "MelBandRoformer_fp16.safetensors"
    if not melband_weight.exists():
        from src.runtime import run

        run(
            [
                "wget",
                "-q",
                "-O",
                str(melband_weight),
                "https://huggingface.co/Kijai/MelBandRoFormer_comfy/resolve/main/MelBandRoformer_fp16.safetensors",
            ]
        )

    write_text(engine_dir / "app.py", settings.read_repo_text("src/apps/scenema_app.py"))

    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "OPENAI_MODEL_ID": settings.openai_model_id or "scenema",
        "SCENEMA_REPO_DIR": str(scenema_dir),
        "SEEDVC_PATH": str(seedvc_dir),
        "MELBAND_NODE_PATH": str(melband_dir),
        "MELBAND_MODEL_PATH": str(melband_weight),
        "MODEL_DIR": str(models_dir),
        "HF_HUB_CACHE": str(models_dir / "hf_cache"),
        # Default 24GB profile (INT8 audio transformer + NF4 Gemma) — fits
        # comfortably on Colab A100 40GB.
        "AUDIO_CKPT": str(models_dir / "scenema-audio-transformer-int8.safetensors"),
        "VAE_ENCODER_CKPT": str(models_dir / "scenema-audio-vae-encoder.safetensors"),
        "PIPELINE_CKPT": str(models_dir / "scenema-audio-pipeline.safetensors"),
        "GEMMA_ROOT": str(models_dir / "gemma-3-12b-it"),
        "GEMMA_QUANTIZE": settings.scenema_gemma_quantize,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "SCENEMA_DEFAULT_VOICE": settings.scenema_default_voice,
        "SCENEMA_PROMPT_WAV": settings.scenema_prompt_wav,
        "SCENEMA_DEFAULT_GENDER": settings.scenema_default_gender,
        "SCENEMA_SEED": str(settings.scenema_seed),
        "SCENEMA_PACE": str(settings.scenema_pace),
        "SCENEMA_VALIDATE": "1" if settings.scenema_validate else "0",
        "SCENEMA_MIN_MATCH_RATIO": str(settings.scenema_min_match_ratio),
        "SCENEMA_SKIP_VC": "1" if settings.scenema_skip_vc else "0",
        "SCENEMA_VC_STEPS": str(settings.scenema_vc_steps),
        "SCENEMA_VC_CFG_RATE": str(settings.scenema_vc_cfg_rate),
        "SCENEMA_BACKGROUND_SFX": "1" if settings.scenema_background_sfx else "0",
        "SCENEMA_APP_PORT": str(settings.app_port),
    }
    hf_token = os.environ.get("HF_TOKEN", "")
    if hf_token:
        env["HF_TOKEN"] = hf_token

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
        log_path=settings.log_dir / "scenema-uvicorn.log",
    )
    return {
        "proc": proc,
        "app_dir": engine_dir,
        "log_path": settings.log_dir / "scenema-uvicorn.log",
    }
