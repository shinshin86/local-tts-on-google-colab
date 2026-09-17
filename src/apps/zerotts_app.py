from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from zerotts import ZeroTTS

logger = logging.getLogger("uvicorn.error")

OPENAI_MODEL_ID = os.environ.get("OPENAI_MODEL_ID", "zerotts")
ZEROTTS_HF_MODEL = os.environ.get("ZEROTTS_HF_MODEL", "zeroweight-ai/ZeroTTS")
ZEROTTS_DEFAULT_VOICE = os.environ.get("ZEROTTS_DEFAULT_VOICE", "maichi")
ZEROTTS_CFG_SCALE = float(os.environ.get("ZEROTTS_CFG_SCALE", "1.0"))
ZEROTTS_AUDIO_TEMPERATURE = float(os.environ.get("ZEROTTS_AUDIO_TEMPERATURE", "0.8"))
ZEROTTS_AUDIO_TOPK = int(os.environ.get("ZEROTTS_AUDIO_TOPK", "25"))
ZEROTTS_AUDIO_TOPP = float(os.environ.get("ZEROTTS_AUDIO_TOPP", "0.95"))
ZEROTTS_AUDIO_REPETITION_PENALTY = float(
    os.environ.get("ZEROTTS_AUDIO_REPETITION_PENALTY", "1.2")
)

app = FastAPI(title="ZeroTTS OpenAI Compatible TTS")
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


_tts: ZeroTTS | None = None


def get_tts() -> ZeroTTS:
    global _tts
    if _tts is None:
        logger.info("Loading ZeroTTS model: %s", ZEROTTS_HF_MODEL)
        _tts = ZeroTTS.from_pretrained(ZEROTTS_HF_MODEL)
        logger.info("ZeroTTS model loaded successfully")
    return _tts


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception while serving request")
    return JSONResponse(status_code=500, content={"error": type(exc).__name__, "detail": str(exc)})


@app.get("/")
def root():
    return {"ok": True, "engine": "ZeroTTS", "model": OPENAI_MODEL_ID}


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [{"id": OPENAI_MODEL_ID, "object": "model", "owned_by": "zeroweight-ai"}],
    }


@app.get("/v1/voices")
def list_voices():
    return {
        "object": "list",
        "data": [{"id": voice, "object": "voice"} for voice in get_tts().list_voices()],
    }


@app.post("/v1/audio/speech")
async def audio_speech(payload: AudioSpeechRequest):
    if payload.response_format.lower() != "wav":
        raise HTTPException(status_code=400, detail="This wrapper currently supports only wav.")

    tts = get_tts()
    voice = payload.voice or ZEROTTS_DEFAULT_VOICE
    available_voices = tts.list_voices()
    if voice not in available_voices:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown voice: {voice}. Available: {available_voices}",
        )
    audio = tts.synthesize(
        payload.input,
        voice=voice,
        cfg_scale=ZEROTTS_CFG_SCALE,
        audio_temperature=ZEROTTS_AUDIO_TEMPERATURE,
        audio_topk=ZEROTTS_AUDIO_TOPK,
        audio_topp=ZEROTTS_AUDIO_TOPP,
        audio_repetition_penalty=ZEROTTS_AUDIO_REPETITION_PENALTY,
    )

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        tts.save_audio(audio, tmp_path)
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
