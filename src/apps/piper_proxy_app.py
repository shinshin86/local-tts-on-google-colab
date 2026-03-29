from __future__ import annotations

import logging
import os

import requests
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logger = logging.getLogger("uvicorn.error")

BACKEND_URL = os.environ.get("BACKEND_URL", "http://127.0.0.1:5000")
OPENAI_MODEL_ID = os.environ.get("OPENAI_MODEL_ID", "piper")
DEFAULT_VOICE = os.environ.get("PROXY_DEFAULT_VOICE", "")
DEFAULT_SPEAKER_ID = int(os.environ.get("PROXY_SPEAKER_ID", "-1"))
ENGINE_NAME = os.environ.get("PROXY_ENGINE", "Piper")

app = FastAPI(title=f"{ENGINE_NAME} OpenAI Compatible TTS")

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
    return {"ok": True, "engine": ENGINE_NAME, "model": OPENAI_MODEL_ID, "backend": BACKEND_URL}


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [{"id": OPENAI_MODEL_ID, "object": "model", "owned_by": "local"}],
    }


@app.get("/v1/voices")
def list_voices():
    try:
        response = requests.get(f"{BACKEND_URL}/voices", timeout=30)
        response.raise_for_status()
        backend_json = response.json()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to fetch voices: {exc}")

    data = []
    if isinstance(backend_json, list):
        for item in backend_json:
            data.append({"id": str(item), "object": "voice"})
    elif isinstance(backend_json, dict):
        for key in backend_json.keys():
            data.append({"id": str(key), "object": "voice"})
    else:
        data.append({"id": str(backend_json), "object": "voice"})
    return {"object": "list", "data": data}


@app.post("/v1/audio/speech")
async def audio_speech(payload: AudioSpeechRequest):
    if payload.response_format.lower() != "wav":
        raise HTTPException(status_code=400, detail="This wrapper currently supports only wav.")

    backend_payload = {
        "text": payload.input,
        "length_scale": 1.0 / max(float(payload.speed), 0.25),
    }
    voice = payload.voice or DEFAULT_VOICE
    if voice:
        backend_payload["voice"] = voice
    if DEFAULT_SPEAKER_ID >= 0:
        backend_payload["speaker_id"] = DEFAULT_SPEAKER_ID

    try:
        response = requests.post(BACKEND_URL, json=backend_payload, timeout=180)
        response.raise_for_status()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to synthesize audio: {exc}")

    return Response(
        content=response.content,
        media_type="audio/wav",
        headers={
            "Content-Length": str(len(response.content)),
            "x-openai-model": payload.model,
            "x-openai-voice": voice,
        },
    )
