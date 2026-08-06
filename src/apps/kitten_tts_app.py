from __future__ import annotations

import io
import logging
import os

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from kittentts import KittenTTS
from pydantic import BaseModel

logger = logging.getLogger("uvicorn.error")

OPENAI_MODEL_ID = os.environ.get("OPENAI_MODEL_ID", "kitten-tts")
HF_MODEL = os.environ.get("KITTEN_TTS_HF_MODEL", "KittenML/kitten-tts-mini-0.8")
DEFAULT_VOICE = os.environ.get("KITTEN_TTS_DEFAULT_VOICE", "Jasper")
SAMPLE_RATE = 24000
VOICES = ["Bella", "Jasper", "Luna", "Bruno", "Rosie", "Hugo", "Kiki", "Leo"]

app = FastAPI(title="KittenTTS OpenAI Compatible TTS")

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


_tts: KittenTTS | None = None


def get_tts() -> KittenTTS:
    global _tts
    if _tts is None:
        _tts = KittenTTS(HF_MODEL, backend="cpu")
    return _tts


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception while serving request")
    return JSONResponse(
        status_code=500,
        content={"error": type(exc).__name__, "detail": str(exc)},
    )


@app.get("/")
def root():
    return {"ok": True, "engine": "KittenTTS", "model": OPENAI_MODEL_ID}


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [{"id": OPENAI_MODEL_ID, "object": "model", "owned_by": "local"}],
    }


@app.get("/v1/voices")
def list_voices():
    return {
        "object": "list",
        "data": [
            {"id": "default", "object": "voice"},
            *[{"id": voice, "object": "voice"} for voice in VOICES],
        ],
    }


@app.post("/v1/audio/speech")
async def audio_speech(payload: AudioSpeechRequest):
    if payload.response_format.lower() != "wav":
        raise HTTPException(status_code=400, detail="This wrapper currently supports only wav.")
    if not payload.input.strip():
        raise HTTPException(status_code=400, detail="input must not be empty.")
    if not 0.25 <= payload.speed <= 4.0:
        raise HTTPException(status_code=400, detail="speed must be between 0.25 and 4.0.")

    voice = payload.voice or DEFAULT_VOICE
    if voice == "default":
        voice = DEFAULT_VOICE
    if voice not in VOICES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown voice '{voice}'. Available voices: {', '.join(VOICES)}",
        )

    audio = np.asarray(
        get_tts().generate(payload.input, voice=voice, speed=float(payload.speed)),
        dtype=np.float32,
    ).squeeze()
    if audio.size == 0:
        raise HTTPException(status_code=500, detail="No audio was generated.")

    buffer = io.BytesIO()
    sf.write(buffer, audio, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    audio_bytes = buffer.getvalue()
    return Response(
        content=audio_bytes,
        media_type="audio/wav",
        headers={
            "Content-Length": str(len(audio_bytes)),
            "x-openai-model": payload.model,
            "x-openai-voice": voice,
        },
    )
