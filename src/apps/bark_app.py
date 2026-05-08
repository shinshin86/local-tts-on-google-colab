from __future__ import annotations

import io
import logging
import os

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# `SUNO_USE_SMALL_MODELS` is read by the upstream `bark` package at import time,
# so it must be set before importing.
if os.environ.get("BARK_USE_SMALL_MODELS", "0") == "1":
    os.environ["SUNO_USE_SMALL_MODELS"] = "True"

from bark import SAMPLE_RATE, generate_audio, preload_models  # noqa: E402

logger = logging.getLogger("uvicorn.error")

OPENAI_MODEL_ID = os.environ.get("OPENAI_MODEL_ID", "bark")
DEFAULT_VOICE = os.environ.get("BARK_DEFAULT_VOICE", "v2/en_speaker_6")

# Bark's official speaker library covers 13 languages with 10 speakers each.
# Listing a curated set keeps `/v1/voices` readable; users can pass any
# `v2/<lang>_speaker_<n>` string via the request payload.
VOICE_PRESETS = [
    "v2/en_speaker_0",
    "v2/en_speaker_6",
    "v2/en_speaker_9",
    "v2/ja_speaker_0",
    "v2/ja_speaker_6",
    "v2/ja_speaker_9",
    "v2/zh_speaker_0",
    "v2/zh_speaker_6",
    "v2/de_speaker_0",
    "v2/es_speaker_0",
    "v2/fr_speaker_0",
    "v2/hi_speaker_0",
    "v2/it_speaker_0",
    "v2/ko_speaker_0",
    "v2/pt_speaker_0",
    "v2/ru_speaker_0",
]

app = FastAPI(title="Bark OpenAI Compatible TTS")

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


_models_loaded = False


def ensure_models_loaded() -> None:
    global _models_loaded
    if not _models_loaded:
        logger.info("Preloading Bark models (small=%s)", os.environ.get("SUNO_USE_SMALL_MODELS", "False"))
        preload_models()
        _models_loaded = True


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
        "engine": "Bark",
        "model": OPENAI_MODEL_ID,
        "default_voice": DEFAULT_VOICE,
        "small_models": os.environ.get("SUNO_USE_SMALL_MODELS", "False"),
    }


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [{"id": OPENAI_MODEL_ID, "object": "model", "owned_by": "suno-ai"}],
    }


@app.get("/v1/voices")
def list_voices():
    return {
        "object": "list",
        "data": [{"id": voice, "object": "voice"} for voice in VOICE_PRESETS],
    }


@app.post("/v1/audio/speech")
async def audio_speech(payload: AudioSpeechRequest):
    if payload.response_format.lower() != "wav":
        raise HTTPException(status_code=400, detail="This wrapper currently supports only wav.")

    voice = payload.voice or DEFAULT_VOICE
    ensure_models_loaded()

    audio_array = generate_audio(payload.input, history_prompt=voice)
    if audio_array is None or len(audio_array) == 0:
        raise HTTPException(status_code=500, detail="No audio was generated.")

    audio_array = np.asarray(audio_array, dtype=np.float32)

    buf = io.BytesIO()
    sf.write(buf, audio_array, SAMPLE_RATE, format="WAV", subtype="PCM_16")
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
