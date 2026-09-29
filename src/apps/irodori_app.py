from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from irodori_tts.inference_runtime import (
    InferenceRuntime,
    RuntimeKey,
    SamplingRequest,
    default_runtime_device,
    download_hf_checkpoint,
    resolve_cfg_scales,
    save_wav,
)

logger = logging.getLogger("uvicorn.error")

# V1: checkpoint="Aratako/Irodori-TTS-500M", codec_repo="facebook/dacvae-watermarked"
# V2/V3/V4 remain supported by the current upstream runtime.
HF_CHECKPOINT = os.environ.get("IRODORI_HF_CHECKPOINT", "Aratako/Irodori-TTS-v4.1-Small")
MODEL_DEVICE = os.environ.get("IRODORI_MODEL_DEVICE", default_runtime_device())
CODEC_DEVICE = os.environ.get("IRODORI_CODEC_DEVICE", default_runtime_device())
MODEL_PRECISION = os.environ.get("IRODORI_MODEL_PRECISION", "fp32")
CODEC_PRECISION = os.environ.get("IRODORI_CODEC_PRECISION", "fp32")
CODEC_REPO = os.environ.get("IRODORI_CODEC_REPO", "Aratako/Semantic-DACVAE-Japanese-32dim")
OPENAI_MODEL_ID = os.environ.get("OPENAI_MODEL_ID", HF_CHECKPOINT)
ENGINE_NAME = os.environ.get("IRODORI_ENGINE_NAME", "Irodori-TTS")
DEFAULT_VOICE_ONLY = os.environ.get("IRODORI_DEFAULT_VOICE_ONLY", "0") == "1"
PROMPT_WAV = os.environ.get("IRODORI_PROMPT_WAV", "")
DEFAULT_VOICE = os.environ.get("IRODORI_DEFAULT_VOICE", "default")
DEFAULT_INSTRUCTIONS = os.environ.get("IRODORI_DEFAULT_INSTRUCTIONS", "")
INSTRUCTIONS_ENABLED = os.environ.get("IRODORI_INSTRUCTIONS_ENABLED", "1") == "1"
PROMPT_FLAG = os.environ.get("IRODORI_PROMPT_FLAG", "--irodori-prompt-wav")
_num_steps = os.environ.get("IRODORI_NUM_STEPS", "40").strip()
NUM_STEPS = int(_num_steps) if _num_steps else None

# v3/v4/v4.1 upstream ship SilentCipher; it is initialized unconditionally inside InferenceRuntime
# and applied automatically when the watermarker reports ready=True. There is no public
# kill-switch and that is intentional — per the model release the watermark must remain.

app = FastAPI(title=f"{ENGINE_NAME} OpenAI Compatible TTS")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[],
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["x-openai-model", "x-openai-voice", "x-irodori-mode"],
)


class AudioSpeechRequest(BaseModel):
    model: str = OPENAI_MODEL_ID
    input: str
    voice: str | None = None
    instructions: str | None = None
    response_format: str = "wav"
    speed: float = Field(default=1.0, ge=0.25, le=4.0)


_runtime = None


def clone_is_configured() -> bool:
    return bool(PROMPT_WAV)


def resolve_instructions(value: str | None) -> str:
    # An explicit empty string suppresses a configured startup default.
    instructions = DEFAULT_INSTRUCTIONS.strip() if value is None else value.strip()
    if instructions and not INSTRUCTIONS_ENABLED:
        raise HTTPException(
            status_code=400,
            detail=f"{ENGINE_NAME} does not expose caption or instructions control.",
        )
    return instructions


def resolve_mode(voice: str, instructions: str) -> str:
    if voice == "clone":
        return "direction" if instructions else "clone"
    return "design" if instructions else "plain"


def get_runtime():
    global _runtime
    if _runtime is None:
        checkpoint_path = download_hf_checkpoint(HF_CHECKPOINT)
        _runtime = InferenceRuntime.from_key(
            RuntimeKey(
                checkpoint=checkpoint_path,
                model_device=MODEL_DEVICE,
                codec_repo=CODEC_REPO,
                model_precision=MODEL_PRECISION,
                codec_device=CODEC_DEVICE,
                codec_precision=CODEC_PRECISION,
                compile_model=False,
                compile_dynamic=False,
            )
        )
    return _runtime


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
        "engine": ENGINE_NAME,
        "model": OPENAI_MODEL_ID,
        "default_voice": DEFAULT_VOICE,
        "default_instructions_configured": bool(DEFAULT_INSTRUCTIONS),
    }


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": OPENAI_MODEL_ID,
                "object": "model",
                "owned_by": "local",
            }
        ],
    }


@app.get("/v1/voices")
def list_voices():
    voices = ["default"]
    if not DEFAULT_VOICE_ONLY and clone_is_configured():
        voices.append("clone")
    return {
        "object": "list",
        "data": [
            {
                "id": voice,
                "object": "voice",
                "description": (
                    "Plain TTS, or Voice Design when instructions are provided."
                    if voice == "default"
                    else "Voice Clone, or style-controlled cloning when instructions are provided."
                ),
            }
            for voice in voices
        ],
    }


@app.post("/v1/audio/speech")
async def audio_speech(payload: AudioSpeechRequest):
    if payload.response_format.lower() != "wav":
        raise HTTPException(status_code=400, detail="This wrapper currently supports only wav.")
    if not payload.input.strip():
        raise HTTPException(status_code=400, detail="input must not be empty.")

    voice = payload.voice or DEFAULT_VOICE
    instructions = resolve_instructions(payload.instructions)
    if DEFAULT_VOICE_ONLY and voice != "default":
        raise HTTPException(
            status_code=400,
            detail="This model supports only voice='default'.",
        )
    if not DEFAULT_VOICE_ONLY and voice not in {"default", "clone"}:
        raise HTTPException(status_code=400, detail="voice must be 'default' or 'clone'.")
    if voice == "clone" and not clone_is_configured():
        raise HTTPException(
            status_code=400,
            detail=f"voice='clone' requires {PROMPT_FLAG}.",
        )

    ref_path = Path(PROMPT_WAV).expanduser() if voice == "clone" else None
    if ref_path is not None and not ref_path.is_file():
        raise HTTPException(
            status_code=400,
            detail="The configured Irodori reference audio does not exist.",
        )

    runtime = get_runtime()
    use_speaker_condition = bool(
        ref_path is not None and runtime.model_cfg.use_speaker_condition_resolved
    )
    if ref_path is not None and not use_speaker_condition:
        raise HTTPException(
            status_code=400,
            detail="The selected Irodori checkpoint does not support reference-audio conditioning.",
        )
    if instructions and not runtime.model_cfg.use_caption_condition:
        raise HTTPException(
            status_code=400,
            detail="The selected Irodori checkpoint does not support Voice Design instructions.",
        )
    cfg_scale_text, _cfg_scale_caption, cfg_scale_speaker, _ = resolve_cfg_scales(
        cfg_guidance_mode="independent",
        cfg_scale_text=3.0,
        cfg_scale_caption=3.0,
        cfg_scale_speaker=5.0,
        cfg_scale=None,
        use_caption_condition=bool(instructions),
        use_speaker_condition=use_speaker_condition,
    )
    mode = resolve_mode(voice, instructions)

    # Use checkpoint metadata instead of its versioned repo name. v3/v4/v4.1 expose a
    # Duration Predictor; legacy checkpoints fall back to their fixed 30-second slot.
    seconds = None if runtime.model_cfg.use_duration_predictor else 30.0
    result = runtime.synthesize(
        SamplingRequest(
            text=payload.input,
            caption=instructions or None,
            ref_wav=None if ref_path is None else str(ref_path),
            ref_latent=None,
            no_ref=ref_path is None,
            ref_normalize_db=-16.0,
            ref_ensure_max=True,
            num_candidates=1,
            decode_mode="sequential",
            seconds=seconds,
            duration_scale=1.0 / payload.speed,
            max_ref_seconds=None,
            max_text_len=None,
            max_caption_len=None,
            num_steps=NUM_STEPS,
            cfg_scale_text=cfg_scale_text,
            cfg_scale_caption=_cfg_scale_caption,
            cfg_scale_speaker=cfg_scale_speaker,
            cfg_guidance_mode="independent",
            cfg_scale=None,
            cfg_min_t=0.5,
            cfg_max_t=1.0,
            truncation_factor=None,
            rescale_k=None,
            rescale_sigma=None,
            context_kv_cache=True,
            speaker_kv_scale=None,
            speaker_kv_min_t=None,
            speaker_kv_max_layers=None,
            seed=None,
            trim_tail=True,
            tail_window_size=20,
            tail_std_threshold=0.05,
            tail_mean_threshold=0.1,
        ),
        log_fn=None,
    )

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        save_wav(tmp_path, result.audio, result.sample_rate)
        audio_bytes = Path(tmp_path).read_bytes()
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return Response(
        content=audio_bytes,
        media_type="audio/wav",
        headers={
            "Content-Length": str(len(audio_bytes)),
            "x-openai-model": payload.model,
            "x-openai-voice": voice,
            "x-irodori-mode": mode,
        },
    )
