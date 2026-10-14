from __future__ import annotations

import logging
import os

import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict

logger = logging.getLogger("uvicorn.error")

BACKEND_URL = os.environ.get("BACKEND_URL", "http://127.0.0.1:5007")
SPEECH_CPP_MODEL = os.environ.get("SPEECH_CPP_MODEL", "qwen3-tts-0.6b")
OPENAI_MODEL_ID = os.environ.get("OPENAI_MODEL_ID", SPEECH_CPP_MODEL)
DEFAULT_VOICE = os.environ.get("SPEECH_CPP_DEFAULT_VOICE", "default") or "default"
QWEN3_SPEAKER = os.environ.get("SPEECH_CPP_QWEN3_SPEAKER", "ono_anna")
CLONE_ENABLED = os.environ.get("SPEECH_CPP_CLONE_ENABLED", "0") == "1"
DEFAULT_INSTRUCTIONS = os.environ.get("SPEECH_CPP_INSTRUCTIONS", "")

QWEN3_SPEAKERS = ["aiden", "dylan", "eric", "ono_anna", "ryan", "serena", "sohee", "uncle_fu", "vivian"]
IS_QWEN3 = SPEECH_CPP_MODEL.startswith("qwen3-tts")
# Qwen3-TTS 0.6B refuses `instructions`; the 1.7B model and Irodori-TTS take them.
TAKES_INSTRUCTIONS = SPEECH_CPP_MODEL != "qwen3-tts-0.6b"
MEDIA_TYPES = {"wav": "audio/wav", "pcm": "audio/pcm"}

app = FastAPI(title="speech.cpp OpenAI Compatible TTS")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[],
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["x-openai-model", "x-openai-voice", "x-sample-rate", "x-speech-seed"],
)


class AudioSpeechRequest(BaseModel):
    # Other speech.cpp request options (seed, steps, language, temperature, ...) pass through.
    model_config = ConfigDict(extra="allow")

    model: str = OPENAI_MODEL_ID
    input: str
    voice: str | None = None
    response_format: str = "wav"
    speed: float = 1.0
    instructions: str | None = None


def available_voices() -> list[str]:
    if IS_QWEN3:
        return ["default", *QWEN3_SPEAKERS]
    return ["default", "clone"] if CLONE_ENABLED else ["default"]


def resolve_backend_voice(voice: str) -> str:
    if IS_QWEN3:
        if voice == "default":
            return QWEN3_SPEAKER
        if voice in QWEN3_SPEAKERS:
            return voice
        if voice == "clone":
            raise HTTPException(
                status_code=400,
                detail="Qwen3-TTS CustomVoice has no voice cloning. Use 'default' or one of: "
                + ", ".join(QWEN3_SPEAKERS)
                + ". Select an Irodori-TTS model for 'clone'.",
            )
    else:
        if voice == "default":
            # Irodori-TTS without a reference: the model chooses the voice (optionally steered by instructions).
            return "none"
        if voice == "clone":
            if not CLONE_ENABLED:
                raise HTTPException(
                    status_code=400,
                    detail="voice='clone' requires --speech-cpp-prompt-wav (SPEECH_CPP_PROMPT_WAV) to be set.",
                )
            return "clone"
    raise HTTPException(
        status_code=400,
        detail=f"Unsupported voice '{voice}'. Available voices: {', '.join(available_voices())}.",
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception while serving request")
    return JSONResponse(
        status_code=500,
        content={"error": type(exc).__name__, "detail": str(exc)},
    )


@app.get("/")
def root():
    return {"ok": True, "engine": "speech.cpp", "model": OPENAI_MODEL_ID, "speech_cpp_model": SPEECH_CPP_MODEL}


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [{"id": OPENAI_MODEL_ID, "object": "model", "owned_by": "local"}],
    }


@app.get("/v1/voices")
def list_voices():
    return {"object": "list", "data": [{"id": v, "object": "voice"} for v in available_voices()]}


@app.post("/v1/audio/speech")
def audio_speech(payload: AudioSpeechRequest):
    response_format = payload.response_format.lower()
    if response_format not in MEDIA_TYPES:
        raise HTTPException(status_code=400, detail="speech.cpp supports only response_format 'wav' or 'pcm'.")

    voice = payload.voice or DEFAULT_VOICE
    backend_payload = {
        **(payload.model_extra or {}),
        "input": payload.input,
        "voice": resolve_backend_voice(voice),
        "response_format": response_format,
    }
    if abs(float(payload.speed) - 1.0) > 1e-6:
        if IS_QWEN3:
            raise HTTPException(status_code=400, detail="Qwen3-TTS in speech.cpp does not support speed; use 1.0.")
        backend_payload["speed"] = float(payload.speed)
    instructions = payload.instructions if payload.instructions is not None else DEFAULT_INSTRUCTIONS
    if instructions and (payload.instructions is not None or TAKES_INSTRUCTIONS):
        backend_payload["instructions"] = instructions

    try:
        upstream = requests.post(
            f"{BACKEND_URL}/v1/audio/speech", json=backend_payload, stream=True, timeout=(10, 600)
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"speech.cpp server is unreachable: {exc}")
    if upstream.status_code != 200:
        try:
            body = upstream.json()
        except ValueError:
            body = {"error": upstream.text}
        upstream.close()
        return JSONResponse(status_code=upstream.status_code, content=body)

    headers = {"x-openai-model": payload.model, "x-openai-voice": voice}
    for name in ("X-Sample-Rate", "X-Speech-Seed", "X-Speech-Stop", "Content-Length"):
        if name in upstream.headers:
            headers[name] = upstream.headers[name]

    def iter_audio():
        try:
            yield from upstream.iter_content(chunk_size=None)
        finally:
            upstream.close()

    return StreamingResponse(iter_audio(), media_type=MEDIA_TYPES[response_format], headers=headers)
