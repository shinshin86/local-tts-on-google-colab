from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from melo.api import TTS
from pydantic import BaseModel

logger = logging.getLogger("uvicorn.error")

OPENAI_MODEL_ID = os.environ.get("OPENAI_MODEL_ID", "melotts")
DEFAULT_LANGUAGE = os.environ.get("MELO_LANGUAGE", "JP")
DEFAULT_VOICE = os.environ.get("MELO_DEFAULT_VOICE", "JP")
DEVICE = os.environ.get("MELO_DEVICE", "auto")

app = FastAPI(title="MeloTTS OpenAI Compatible TTS")

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
    language: str | None = None


_models = {}


def get_model(language: str) -> TTS:
    model = _models.get(language)
    if model is None:
        model = TTS(language=language, device=DEVICE)
        _models[language] = model
    return model


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception while serving request")
    return JSONResponse(
        status_code=500,
        content={"error": type(exc).__name__, "detail": str(exc)},
    )


@app.get("/")
def root():
    return {"ok": True, "engine": "MeloTTS", "model": OPENAI_MODEL_ID}


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [{"id": OPENAI_MODEL_ID, "object": "model", "owned_by": "local"}],
    }


@app.get("/v1/voices")
def list_voices(language: str | None = None):
    target_language = language or DEFAULT_LANGUAGE
    model = get_model(target_language)
    voices = list(model.hps.data.spk2id.keys())
    return {
        "object": "list",
        "data": [{"id": voice, "object": "voice"} for voice in voices],
    }


@app.post("/v1/audio/speech")
async def audio_speech(payload: AudioSpeechRequest):
    if payload.response_format.lower() != "wav":
        raise HTTPException(status_code=400, detail="This wrapper currently supports only wav.")

    target_language = payload.language or DEFAULT_LANGUAGE
    model = get_model(target_language)
    speaker_ids = model.hps.data.spk2id
    voice = payload.voice or DEFAULT_VOICE or next(iter(speaker_ids))
    if voice not in speaker_ids:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown voice '{voice}'. Available: {', '.join(speaker_ids.keys())}",
        )

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        model.tts_to_file(payload.input, speaker_ids[voice], tmp_path, speed=float(payload.speed))
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
        },
    )
