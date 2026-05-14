from __future__ import annotations

import logging
import os
import sys
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from xml.sax.saxutils import escape

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

logger = logging.getLogger("uvicorn.error")

OPENAI_MODEL_ID = os.environ.get("OPENAI_MODEL_ID", "scenema")
SCENEMA_REPO_DIR = os.environ.get("SCENEMA_REPO_DIR", "")
SEEDVC_PATH = os.environ.get("SEEDVC_PATH", "")
MELBAND_NODE_PATH = os.environ.get("MELBAND_NODE_PATH", "")
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/content/scenema-models"))
PROMPT_WAV = os.environ.get("SCENEMA_PROMPT_WAV", "")
DEFAULT_VOICE = os.environ.get("SCENEMA_DEFAULT_VOICE", "default")
DEFAULT_GENDER = os.environ.get("SCENEMA_DEFAULT_GENDER", "male")
SEED = int(os.environ.get("SCENEMA_SEED", "-1"))
PACE = float(os.environ.get("SCENEMA_PACE", "1.5"))
VALIDATE = os.environ.get("SCENEMA_VALIDATE", "1") == "1"
MIN_MATCH_RATIO = float(os.environ.get("SCENEMA_MIN_MATCH_RATIO", "0.90"))
SKIP_VC = os.environ.get("SCENEMA_SKIP_VC", "0") == "1"
VC_STEPS = int(os.environ.get("SCENEMA_VC_STEPS", "25"))
VC_CFG_RATE = float(os.environ.get("SCENEMA_VC_CFG_RATE", "0.5"))
BACKGROUND_SFX = os.environ.get("SCENEMA_BACKGROUND_SFX", "0") == "1"
APP_PORT = int(os.environ.get("SCENEMA_APP_PORT", "8000"))

# Upstream layout: scenema-audio uses PYTHONPATH=/app/src + sibling clones for
# seed-vc and the MelBandRoFormer node. Mirror that here.
if SCENEMA_REPO_DIR:
    for sub in ("src",):
        path = str(Path(SCENEMA_REPO_DIR) / sub)
        if path not in sys.path:
            sys.path.insert(0, path)
for extra in (SEEDVC_PATH, MELBAND_NODE_PATH):
    if extra and extra not in sys.path:
        sys.path.insert(0, extra)


# Free-form voice description presets. Users can also pass an arbitrary
# description directly as the `voice` parameter — if it doesn't match a
# preset name, it is used as the voice description verbatim.
SCENEMA_VOICE_PRESETS: dict[str, dict[str, str]] = {
    "default": {
        "description": "A warm, clear male narrator with a slight British accent. Measured, thoughtful pacing.",
        "gender": "male",
    },
    "warm_male": {
        "description": "Warm middle-aged male narrator. Deep but gentle tone, unhurried pacing, slight rasp.",
        "gender": "male",
    },
    "smoky_female": {
        "description": "Smoky low-register female voice, intimate confessional tone, slight breathiness.",
        "gender": "female",
    },
    "child_girl": {
        "description": "Bright six-year-old girl, breathless and excited, slight lisp on S sounds.",
        "gender": "female",
    },
    "elderly_male": {
        "description": "Weathered elderly male storyteller, deep baritone, slow deliberate pacing, nostalgic warmth.",
        "gender": "male",
    },
    "elderly_female": {
        "description": "Soft alto, woman in her seventies, unhurried, warm and grandmotherly.",
        "gender": "female",
    },
}


_processor = None  # type: ignore[var-annotated]
_ProcessJob = None  # type: ignore[var-annotated]


def _download_models() -> None:
    """Fetch missing checkpoints. Ported from scenema-audio src/server.py."""
    from huggingface_hub import hf_hub_download, snapshot_download

    HF_REPO = "ScenemaAI/scenema-audio"
    GEMMA_REPO = "google/gemma-3-12b-it"
    SEEDVC_REPO = "Plachta/Seed-VC"
    BIGVGAN_REPO = "nvidia/bigvgan_v2_22khz_80band_256x"
    WHISPER_REPO = "openai/whisper-small"

    token = os.environ.get("HF_TOKEN") or None
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    audio_ckpt = Path(
        os.environ.get(
            "AUDIO_CKPT",
            str(MODEL_DIR / "scenema-audio-transformer-int8.safetensors"),
        )
    )
    if not audio_ckpt.exists():
        logger.info("Scenema: downloading audio transformer (INT8, ~4.9 GB)...")
        hf_hub_download(
            HF_REPO,
            "scenema-audio-transformer-int8.safetensors",
            local_dir=str(audio_ckpt.parent),
            token=token,
        )

    pipeline_ckpt = Path(
        os.environ.get(
            "PIPELINE_CKPT",
            str(MODEL_DIR / "scenema-audio-pipeline.safetensors"),
        )
    )
    if not pipeline_ckpt.exists():
        logger.info("Scenema: downloading pipeline checkpoint (~7.1 GB)...")
        hf_hub_download(
            HF_REPO,
            "scenema-audio-pipeline.safetensors",
            local_dir=str(pipeline_ckpt.parent),
            token=token,
        )

    vae_ckpt = Path(
        os.environ.get(
            "VAE_ENCODER_CKPT",
            str(MODEL_DIR / "scenema-audio-vae-encoder.safetensors"),
        )
    )
    if not vae_ckpt.exists():
        logger.info("Scenema: downloading VAE encoder (~42 MB)...")
        hf_hub_download(
            HF_REPO,
            "scenema-audio-vae-encoder.safetensors",
            local_dir=str(vae_ckpt.parent),
            token=token,
        )

    gemma_root = Path(os.environ.get("GEMMA_ROOT", str(MODEL_DIR / "gemma-3-12b-it")))
    if not gemma_root.exists() or not any(gemma_root.glob("*.safetensors")):
        logger.info(
            "Scenema: downloading Gemma 3 12B IT (~24 GB, gated — needs HF_TOKEN)..."
        )
        snapshot_download(
            GEMMA_REPO,
            local_dir=str(gemma_root),
            ignore_patterns=["*.gguf"],
            token=token,
        )

    seedvc_cache = Path(SEEDVC_PATH) / "checkpoints" if SEEDVC_PATH else None
    if seedvc_cache is not None and (
        not seedvc_cache.exists() or not any(seedvc_cache.glob("*.pth"))
    ):
        logger.info("Scenema: downloading SeedVC checkpoints (~1.6 GB)...")
        hf_cache = seedvc_cache / "hf_cache"
        hf_cache.mkdir(parents=True, exist_ok=True)
        os.environ["HF_HUB_CACHE"] = str(hf_cache)
        hf_hub_download(
            SEEDVC_REPO,
            "DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth",
            local_dir=str(seedvc_cache),
            token=token,
        )
        hf_hub_download(
            SEEDVC_REPO,
            "config_dit_mel_seed_uvit_whisper_small_wavenet.yml",
            local_dir=str(seedvc_cache),
            token=token,
        )
        snapshot_download(BIGVGAN_REPO, local_dir=str(hf_cache / "bigvgan"))
        snapshot_download(WHISPER_REPO, local_dir=str(hf_cache / "whisper-small"))


def _resolve_voice_description(voice: str) -> tuple[str, str]:
    """Return (voice_description, gender) for a `voice` parameter.

    Preset names map to a baked description. Anything else is treated as a
    free-form Scenema voice description and used verbatim.
    """
    preset = SCENEMA_VOICE_PRESETS.get(voice)
    if preset is not None:
        return preset["description"], preset["gender"]
    return voice, DEFAULT_GENDER


def _wrap_as_speak(text: str, description: str, gender: str) -> str:
    """Wrap plain text in a <speak> envelope. Pass through XML unchanged."""
    stripped = text.lstrip()
    if stripped.startswith("<speak"):
        return text
    safe_desc = escape(description, {'"': "&quot;"})
    safe_gender = escape(gender, {'"': "&quot;"})
    body = escape(text)
    return f'<speak voice="{safe_desc}" gender="{safe_gender}">{body}</speak>'


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _processor, _ProcessJob

    _download_models()

    # Lazy imports — these pull in torch / bitsandbytes / Gemma weights.
    from audio_core.processor import AudioProcessor  # type: ignore[import-not-found]
    from common.handlers.base import ProcessJob  # type: ignore[import-not-found]

    _ProcessJob = ProcessJob
    _processor = AudioProcessor()
    _processor.startup()
    logger.info("Scenema: AudioProcessor ready")

    try:
        yield
    finally:
        try:
            _processor.shutdown()
        except Exception:
            logger.exception("Scenema: shutdown failed")


app = FastAPI(title="Scenema Audio OpenAI Compatible TTS", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[],
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["x-openai-model", "x-openai-voice"],
)


class AudioSpeechRequest(BaseModel):
    model: str = OPENAI_MODEL_ID
    input: str
    voice: str | None = None
    response_format: str = "wav"
    speed: float = 1.0


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception while serving request")
    return JSONResponse(
        status_code=500,
        content={"error": type(exc).__name__, "detail": str(exc)},
    )


@app.get("/")
def root():
    return {
        "ok": True,
        "engine": "Scenema",
        "model": OPENAI_MODEL_ID,
        "default_voice": DEFAULT_VOICE,
        "presets": sorted(SCENEMA_VOICE_PRESETS.keys()),
        "clone_available": bool(PROMPT_WAV),
    }


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [{"id": OPENAI_MODEL_ID, "object": "model", "owned_by": "scenema-ai"}],
    }


@app.get("/v1/voices")
def list_voices():
    voices: list[dict] = []
    for name, preset in SCENEMA_VOICE_PRESETS.items():
        voices.append(
            {
                "id": name,
                "object": "voice",
                "gender": preset["gender"],
                "description": preset["description"],
            }
        )
    if PROMPT_WAV:
        voices.append({"id": "clone", "object": "voice", "ref": PROMPT_WAV})
    return {"object": "list", "data": voices}


@app.get("/scenema-prompt-wav")
def serve_prompt_wav():
    """Serve the configured prompt wav so AudioProcessor can fetch it via httpx.

    AudioProcessor downloads `reference_voice_url` over HTTP — file:// won't
    work. We host the local prompt wav here and pass our own URL through.
    """
    if not PROMPT_WAV or not Path(PROMPT_WAV).exists():
        raise HTTPException(status_code=404, detail="No prompt wav configured.")
    return FileResponse(PROMPT_WAV, media_type="audio/wav")


@app.post("/v1/audio/speech")
async def audio_speech(payload: AudioSpeechRequest):
    if payload.response_format.lower() != "wav":
        raise HTTPException(status_code=400, detail="This wrapper currently supports only wav.")
    if _processor is None or _ProcessJob is None:
        raise HTTPException(status_code=503, detail="Scenema processor not ready yet.")

    voice = payload.voice or DEFAULT_VOICE

    reference_voice_url: str | None = None
    if voice == "clone":
        if not PROMPT_WAV:
            raise HTTPException(
                status_code=400,
                detail="voice='clone' requires --scenema-prompt-wav at startup.",
            )
        # Serve the wav off our own port so AudioProcessor can httpx.get it.
        reference_voice_url = f"http://127.0.0.1:{APP_PORT}/scenema-prompt-wav"
        description, gender = _resolve_voice_description(DEFAULT_VOICE)
    else:
        description, gender = _resolve_voice_description(voice)

    prompt_xml = _wrap_as_speak(payload.input, description, gender)

    job = _ProcessJob(
        job_id=str(uuid.uuid4()),
        input={
            "prompt": prompt_xml,
            "mode": "generate",
            "reference_voice_url": reference_voice_url,
            "background_sfx": BACKGROUND_SFX,
            "validate": VALIDATE,
            "seed": SEED,
            "pace": PACE,
            "min_match_ratio": MIN_MATCH_RATIO,
            "skip_vc": SKIP_VC,
            "vc_steps": VC_STEPS,
            "vc_cfg_rate": VC_CFG_RATE,
        },
    )

    result = await _processor.process(job)

    if not result.success or result.output is None or not result.output.data:
        detail = result.error or (
            result.output.error if result.output else "generation failed"
        )
        raise HTTPException(status_code=500, detail=detail)

    audio_bytes = result.output.data
    return Response(
        content=audio_bytes,
        media_type="audio/wav",
        headers={
            "Content-Length": str(len(audio_bytes)),
            "x-openai-model": payload.model,
            "x-openai-voice": voice,
        },
    )
