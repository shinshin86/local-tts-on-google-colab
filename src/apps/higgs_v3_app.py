from __future__ import annotations

import logging
import os

import httpx
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logger = logging.getLogger("uvicorn.error")

BACKEND_URL = os.environ.get("HIGGS_V3_BACKEND_URL", "http://127.0.0.1:5002")
OPENAI_MODEL_ID = os.environ.get("OPENAI_MODEL_ID", "higgs-audio-v3")
DEFAULT_VOICE = os.environ.get("HIGGS_V3_DEFAULT_VOICE", "default")
PROMPT_WAV = os.environ.get("HIGGS_V3_PROMPT_WAV", "")
PROMPT_TEXT = os.environ.get("HIGGS_V3_PROMPT_TEXT", "")
TEMPERATURE = float(os.environ.get("HIGGS_V3_TEMPERATURE", "0.7"))
TOP_K = int(os.environ.get("HIGGS_V3_TOP_K", "50"))
MAX_NEW_TOKENS = int(os.environ.get("HIGGS_V3_MAX_NEW_TOKENS", "2048"))

# SGLang-Omni's /v1/audio/speech accepts these container formats.
SUPPORTED_FORMATS = ["wav", "pcm", "flac", "mp3", "aac", "opus"]

MEDIA_TYPES = {
    "wav": "audio/wav",
    "pcm": "audio/pcm",
    "flac": "audio/flac",
    "mp3": "audio/mpeg",
    "aac": "audio/aac",
    "opus": "audio/opus",
}

app = FastAPI(title="Higgs-Audio-v3 OpenAI Compatible TTS")

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
    return {"ok": True, "engine": "Higgs-Audio-v3", "model": OPENAI_MODEL_ID, "backend": BACKEND_URL}


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [{"id": OPENAI_MODEL_ID, "object": "model", "owned_by": "local"}],
    }


@app.get("/v1/voices")
def list_voices():
    # default = built-in speaker (no reference); clone = zero-shot from the
    # configured reference clip.
    voices = ["default"]
    if PROMPT_WAV:
        voices.append("clone")
    return {
        "object": "list",
        "data": [{"id": v, "object": "voice"} for v in voices],
    }


@app.post("/v1/audio/speech")
async def audio_speech(payload: AudioSpeechRequest):
    fmt = payload.response_format.lower()
    if fmt not in SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format: {fmt}. Supported: {SUPPORTED_FORMATS}",
        )

    voice = payload.voice or DEFAULT_VOICE

    backend_payload = {
        "input": payload.input,
        "response_format": fmt,
        "temperature": TEMPERATURE,
        "top_k": TOP_K,
        "max_new_tokens": MAX_NEW_TOKENS,
    }

    if voice == "clone":
        if not PROMPT_WAV:
            raise HTTPException(
                status_code=400,
                detail="voice='clone' requires HIGGS_V3_PROMPT_WAV (--higgs-v3-prompt-wav). "
                "Use voice='default' for the built-in speaker.",
            )
        reference = {"audio_path": PROMPT_WAV}
        if PROMPT_TEXT:
            reference["text"] = PROMPT_TEXT
        backend_payload["references"] = [reference]
    elif voice != "default":
        raise HTTPException(
            status_code=400,
            detail=f"Unknown voice: {voice}. Available: ['default', 'clone'].",
        )

    try:
        async with httpx.AsyncClient(timeout=300) as client:
            resp = await client.post(
                f"{BACKEND_URL}/v1/audio/speech",
                json=backend_payload,
            )
            resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text if exc.response else str(exc)
        raise HTTPException(status_code=502, detail=f"SGLang-Omni backend error: {detail}")
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to reach SGLang-Omni backend: {exc}")

    return Response(
        content=resp.content,
        media_type=MEDIA_TYPES.get(fmt, "audio/wav"),
        headers={
            "Content-Length": str(len(resp.content)),
            "x-openai-model": payload.model,
            "x-openai-voice": voice,
        },
    )
