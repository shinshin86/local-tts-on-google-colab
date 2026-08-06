from __future__ import annotations

import io
import logging
import os
from typing import Any

import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from omnivoice import OmniVoice, OmniVoiceGenerationConfig
from pydantic import BaseModel

logger = logging.getLogger("uvicorn.error")

OPENAI_MODEL_ID = os.environ.get("OPENAI_MODEL_ID", "omnivoice")
OMNIVOICE_HF_MODEL = os.environ.get("OMNIVOICE_HF_MODEL", "k2-fsa/OmniVoice")
OMNIVOICE_MODEL_DIR = os.environ.get("OMNIVOICE_MODEL_DIR", OMNIVOICE_HF_MODEL)
OMNIVOICE_LANGUAGE = os.environ.get("OMNIVOICE_LANGUAGE", "ja")
OMNIVOICE_DEFAULT_VOICE = os.environ.get("OMNIVOICE_DEFAULT_VOICE", "auto")
OMNIVOICE_PROMPT_WAV = os.environ.get("OMNIVOICE_PROMPT_WAV", "")
OMNIVOICE_PROMPT_TEXT = os.environ.get("OMNIVOICE_PROMPT_TEXT", "")
OMNIVOICE_INSTRUCT = os.environ.get("OMNIVOICE_INSTRUCT", "")
OMNIVOICE_NUM_STEPS = int(os.environ.get("OMNIVOICE_NUM_STEPS", "32"))
OMNIVOICE_GUIDANCE_SCALE = float(os.environ.get("OMNIVOICE_GUIDANCE_SCALE", "2.0"))

app = FastAPI(title="OmniVoice OpenAI Compatible TTS")
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


_model: Any | None = None


def get_model() -> Any:
    global _model
    if _model is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        logger.info("Loading OmniVoice from %s on %s", OMNIVOICE_MODEL_DIR, device)
        _model = OmniVoice.from_pretrained(
            OMNIVOICE_MODEL_DIR,
            device_map=device,
            dtype=dtype,
            attn_implementation="sdpa",
            load_asr=False,
        )
        _model.eval()
        logger.info("OmniVoice ready (sample_rate=%s)", _model.sampling_rate)
    return _model


def _available_voices() -> list[str]:
    voices = ["auto"]
    if OMNIVOICE_INSTRUCT:
        voices.append("design")
    if OMNIVOICE_PROMPT_WAV and OMNIVOICE_PROMPT_TEXT:
        voices.append("clone")
    return voices


def _resolve_voice(voice: str | None) -> str:
    requested = OMNIVOICE_DEFAULT_VOICE if not voice or voice == "default" else voice
    if requested not in _available_voices():
        raise HTTPException(
            status_code=400,
            detail=f"Unknown or unavailable voice '{requested}'. Available: {', '.join(_available_voices())}",
        )
    return requested


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception while serving request")
    return JSONResponse(status_code=500, content={"error": type(exc).__name__, "detail": str(exc)})


@app.get("/")
def root():
    return {
        "ok": True,
        "engine": "OmniVoice",
        "model": OPENAI_MODEL_ID,
        "hf_model": OMNIVOICE_HF_MODEL,
        "language": OMNIVOICE_LANGUAGE,
        "default_voice": OMNIVOICE_DEFAULT_VOICE,
        "voices": _available_voices(),
    }


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [{"id": OPENAI_MODEL_ID, "object": "model", "owned_by": "k2-fsa"}],
    }


@app.get("/v1/voices")
def list_voices():
    descriptions = {
        "auto": "The model selects a voice automatically.",
        "design": "Uses the startup voice-design instruction.",
        "clone": "Uses the configured reference audio and transcript.",
    }
    return {
        "object": "list",
        "data": [
            {"id": voice, "object": "voice", "description": descriptions[voice]}
            for voice in _available_voices()
        ],
    }


@app.post("/v1/audio/speech")
async def audio_speech(payload: AudioSpeechRequest):
    if payload.response_format.lower() != "wav":
        raise HTTPException(status_code=400, detail="This wrapper currently supports only wav.")
    if not 0.5 <= payload.speed <= 1.5:
        raise HTTPException(status_code=400, detail="speed must be between 0.5 and 1.5.")

    voice = _resolve_voice(payload.voice)
    model = get_model()
    kwargs: dict[str, Any] = {
        "text": payload.input,
        "language": OMNIVOICE_LANGUAGE or None,
        "speed": payload.speed,
        "generation_config": OmniVoiceGenerationConfig(
            num_step=OMNIVOICE_NUM_STEPS,
            guidance_scale=OMNIVOICE_GUIDANCE_SCALE,
        ),
    }
    if voice == "design":
        kwargs["instruct"] = OMNIVOICE_INSTRUCT
    elif voice == "clone":
        kwargs["ref_audio"] = OMNIVOICE_PROMPT_WAV
        kwargs["ref_text"] = OMNIVOICE_PROMPT_TEXT

    with torch.inference_mode():
        audios = model.generate(**kwargs)
    if not audios:
        raise HTTPException(status_code=500, detail="No audio was generated.")

    buf = io.BytesIO()
    sf.write(buf, audios[0], model.sampling_rate, format="WAV", subtype="PCM_16")
    audio_bytes = buf.getvalue()
    return Response(
        content=audio_bytes,
        media_type="audio/wav",
        headers={
            "Content-Length": str(len(audio_bytes)),
            "x-openai-model": payload.model,
            "x-openai-voice": voice,
        },
    )
