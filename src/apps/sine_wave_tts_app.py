from __future__ import annotations

import logging
import os

import requests
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logger = logging.getLogger("uvicorn.error")

BACKEND_URL = os.environ.get("BACKEND_URL", "http://127.0.0.1:5006").rstrip("/")
OPENAI_MODEL_ID = os.environ.get("OPENAI_MODEL_ID", "sine-wave-tts")
DEFAULT_SPEAKER = os.environ.get("SINE_WAVE_TTS_DEFAULT_SPEAKER", "default")
DEFAULT_EMOTION = os.environ.get("SINE_WAVE_TTS_DEFAULT_EMOTION", "neutral")
REQUEST_TIMEOUT_SECONDS = 120

app = FastAPI(title="Sine-Wave-TTS OpenAI Compatible TTS")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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


def backend_get(path: str) -> requests.Response:
    try:
        return requests.get(f"{BACKEND_URL}{path}", timeout=10)
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Sine-Wave-TTS backend unavailable: {exc}") from exc


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception while serving request")
    return JSONResponse(
        status_code=500,
        content={"error": type(exc).__name__, "detail": str(exc)},
    )


@app.get("/")
def root():
    health = backend_get("/v1/health")
    if not health.ok:
        raise HTTPException(status_code=502, detail="Sine-Wave-TTS backend health check failed.")
    return {
        "ok": True,
        "engine": "Sine-Wave-TTS",
        "model": OPENAI_MODEL_ID,
        "backend": health.json(),
    }


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [{"id": OPENAI_MODEL_ID, "object": "model", "owned_by": "local"}],
    }


@app.get("/v1/voices")
def list_voices():
    speakers_response = backend_get("/v1/speakers")
    emotions_response = backend_get("/v1/emotions")
    if not speakers_response.ok or not emotions_response.ok:
        raise HTTPException(status_code=502, detail="Failed to read Sine-Wave-TTS presets.")

    speakers = [item["name"] for item in speakers_response.json()]
    emotions = [item["name"] for item in emotions_response.json()]
    return {
        "object": "list",
        "data": [
            {
                "id": f"{speaker}:{emotion}",
                "object": "voice",
                "speaker": speaker,
                "emotion": emotion,
            }
            for speaker in speakers
            for emotion in emotions
        ],
    }


@app.post("/v1/audio/speech")
def audio_speech(payload: AudioSpeechRequest):
    voice = payload.voice or f"{DEFAULT_SPEAKER}:{DEFAULT_EMOTION}"
    try:
        backend_response = requests.post(
            f"{BACKEND_URL}/v1/audio/speech",
            json={
                "model": payload.model,
                "input": payload.input,
                "voice": voice,
                "response_format": payload.response_format,
                "speed": payload.speed,
            },
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Sine-Wave-TTS backend unavailable: {exc}") from exc

    content_type = backend_response.headers.get("content-type", "application/octet-stream")
    headers = {
        "Content-Length": str(len(backend_response.content)),
        "x-openai-model": payload.model,
        "x-openai-voice": voice,
    }
    return Response(
        content=backend_response.content,
        status_code=backend_response.status_code,
        media_type=content_type.split(";", 1)[0],
        headers=headers,
    )
