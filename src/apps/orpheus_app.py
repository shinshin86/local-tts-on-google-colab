from __future__ import annotations

import io
import logging
import os
import wave

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from orpheus_tts import OrpheusModel
from pydantic import BaseModel

logger = logging.getLogger("uvicorn.error")

OPENAI_MODEL_ID = os.environ.get("OPENAI_MODEL_ID", "orpheus-tts")
HF_MODEL = os.environ.get("ORPHEUS_HF_MODEL", "canopylabs/orpheus-tts-0.1-finetune-prod")
DEFAULT_VOICE = os.environ.get("ORPHEUS_DEFAULT_VOICE", "tara")
MAX_MODEL_LEN = int(os.environ.get("ORPHEUS_MAX_MODEL_LEN", "2048"))

VOICES = ["tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"]
SAMPLE_RATE = 24000

app = FastAPI(title="Orpheus-TTS OpenAI Compatible TTS")

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


_model: OrpheusModel | None = None


def get_model() -> OrpheusModel:
    global _model
    if _model is None:
        logger.info("Loading Orpheus model %s (max_model_len=%d)", HF_MODEL, MAX_MODEL_LEN)
        _model = OrpheusModel(model_name=HF_MODEL, max_model_len=MAX_MODEL_LEN)
        logger.info("Orpheus model loaded.")
    return _model


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
        "engine": "Orpheus-TTS",
        "model": OPENAI_MODEL_ID,
        "hf_model": HF_MODEL,
        "default_voice": DEFAULT_VOICE,
    }


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [{"id": OPENAI_MODEL_ID, "object": "model", "owned_by": "canopylabs"}],
    }


@app.get("/v1/voices")
def list_voices():
    return {
        "object": "list",
        "data": [{"id": v, "object": "voice"} for v in VOICES],
    }


@app.post("/v1/audio/speech")
async def audio_speech(payload: AudioSpeechRequest):
    if payload.response_format.lower() != "wav":
        raise HTTPException(status_code=400, detail="This wrapper currently supports only wav.")

    voice = payload.voice or DEFAULT_VOICE
    if voice not in VOICES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown voice: {voice}. Available: {VOICES}",
        )

    model = get_model()
    syn_tokens = model.generate_speech(prompt=payload.input, voice=voice)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        for chunk in syn_tokens:
            wf.writeframes(chunk)
    audio_bytes = buf.getvalue()
    if len(audio_bytes) <= 44:
        raise HTTPException(status_code=500, detail="No audio was generated.")

    return Response(
        content=audio_bytes,
        media_type="audio/wav",
        headers={
            "Content-Length": str(len(audio_bytes)),
            "x-openai-model": payload.model,
            "x-openai-voice": voice,
        },
    )
