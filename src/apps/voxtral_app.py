from __future__ import annotations

import logging
import os

import httpx
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logger = logging.getLogger("uvicorn.error")

VLLM_BACKEND_URL = os.environ.get("VLLM_BACKEND_URL", "http://127.0.0.1:5001")
OPENAI_MODEL_ID = os.environ.get("OPENAI_MODEL_ID", "voxtral-tts")
VOXTRAL_HF_MODEL = os.environ.get("VOXTRAL_HF_MODEL", "mistralai/Voxtral-4B-TTS-2603")
VOXTRAL_DEFAULT_VOICE = os.environ.get("VOXTRAL_DEFAULT_VOICE", "neutral_female")

VOICES = [
    "ar_male", "casual_female", "casual_male", "cheerful_female",
    "de_female", "de_male", "es_female", "es_male",
    "fr_female", "fr_male", "hi_female", "hi_male",
    "it_female", "it_male", "neutral_female", "neutral_male",
    "nl_female", "nl_male", "pt_female", "pt_male",
]

SUPPORTED_FORMATS = ["wav", "pcm", "flac", "mp3", "aac", "opus"]

MEDIA_TYPES = {
    "wav": "audio/wav",
    "pcm": "audio/pcm",
    "flac": "audio/flac",
    "mp3": "audio/mpeg",
    "aac": "audio/aac",
    "opus": "audio/opus",
}

app = FastAPI(title="Voxtral-TTS OpenAI Compatible TTS")

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
    return {"ok": True, "engine": "Voxtral-TTS", "model": OPENAI_MODEL_ID, "backend": VLLM_BACKEND_URL}


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
        "data": [{"id": v, "object": "voice"} for v in VOICES],
    }


@app.post("/v1/audio/speech")
async def audio_speech(payload: AudioSpeechRequest):
    fmt = payload.response_format.lower()
    if fmt not in SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format: {fmt}. Supported: {SUPPORTED_FORMATS}",
        )

    voice = payload.voice or VOXTRAL_DEFAULT_VOICE
    if voice not in VOICES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown voice: {voice}. Available: {VOICES}",
        )

    backend_payload = {
        "model": VOXTRAL_HF_MODEL,
        "input": payload.input,
        "voice": voice,
        "response_format": fmt,
        "speed": payload.speed,
    }

    try:
        async with httpx.AsyncClient(timeout=300) as client:
            resp = await client.post(
                f"{VLLM_BACKEND_URL}/v1/audio/speech",
                json=backend_payload,
            )
            resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text if exc.response else str(exc)
        raise HTTPException(status_code=502, detail=f"vLLM backend error: {detail}")
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to reach vLLM backend: {exc}")

    return Response(
        content=resp.content,
        media_type=MEDIA_TYPES.get(fmt, "audio/wav"),
        headers={
            "Content-Length": str(len(resp.content)),
            "x-openai-model": payload.model,
            "x-openai-voice": voice,
        },
    )
