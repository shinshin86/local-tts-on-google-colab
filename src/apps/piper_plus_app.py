from __future__ import annotations

import io
import logging
import os
import wave

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from piper import PiperVoice
from pydantic import BaseModel

logger = logging.getLogger("uvicorn.error")

OPENAI_MODEL_ID = os.environ.get("OPENAI_MODEL_ID", "piper-plus")
PIPER_PLUS_ONNX = os.environ.get("PIPER_PLUS_ONNX", "")
PIPER_PLUS_CONFIG = os.environ.get("PIPER_PLUS_CONFIG", "")
PIPER_PLUS_DEFAULT_LANGUAGE = os.environ.get("PIPER_PLUS_DEFAULT_LANGUAGE", "ja")

# piper-plus multilingual language_id mapping
LANGUAGE_ID_MAP = {"ja": 0, "en": 1, "zh": 2, "es": 3, "fr": 4, "pt": 5}

app = FastAPI(title="Piper-Plus OpenAI Compatible TTS")

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


_voice: PiperVoice | None = None


def get_voice() -> PiperVoice:
    global _voice
    if _voice is None:
        config_path = PIPER_PLUS_CONFIG if PIPER_PLUS_CONFIG else None
        _voice = PiperVoice.load(PIPER_PLUS_ONNX, config_path=config_path)
    return _voice


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception while serving request")
    return JSONResponse(
        status_code=500,
        content={"error": type(exc).__name__, "detail": str(exc)},
    )


@app.get("/")
def root():
    return {"ok": True, "engine": "Piper-Plus", "model": OPENAI_MODEL_ID}


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [{"id": OPENAI_MODEL_ID, "object": "model", "owned_by": "local"}],
    }


@app.get("/v1/voices")
def list_voices():
    voice = get_voice()
    lang_map = getattr(voice.config, "language_id_map", None) or {}
    data = [{"id": lang, "object": "voice"} for lang in sorted(lang_map.keys())]
    return {"object": "list", "data": data}


@app.post("/v1/audio/speech")
async def audio_speech(payload: AudioSpeechRequest):
    if payload.response_format.lower() != "wav":
        raise HTTPException(status_code=400, detail="This wrapper currently supports only wav.")

    voice = get_voice()
    language = payload.language or PIPER_PLUS_DEFAULT_LANGUAGE
    language_id = LANGUAGE_ID_MAP.get(language)
    length_scale = 1.0 / max(float(payload.speed), 0.25)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav_file:
        voice.synthesize(
            payload.input,
            wav_file,
            length_scale=length_scale,
            language_id=language_id,
        )

    audio_bytes = buf.getvalue()
    if len(audio_bytes) <= 44:  # WAV header only = no audio
        raise HTTPException(status_code=500, detail="No audio was generated.")

    return Response(
        content=audio_bytes,
        media_type="audio/wav",
        headers={
            "Content-Length": str(len(audio_bytes)),
            "x-openai-model": payload.model,
            "x-openai-voice": language,
        },
    )
