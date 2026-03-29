from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

import torch
import torchaudio
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from TTS.api import TTS

logger = logging.getLogger("uvicorn.error")

OPENAI_MODEL_ID = os.environ.get("OPENAI_MODEL_ID", "xtts_v2")
XTTS_LANGUAGE = os.environ.get("XTTS_LANGUAGE", "ja")
SPEAKER_WAV = os.environ.get("XTTS_SPEAKER_WAV", "")

app = FastAPI(title="Coqui XTTS v2 OpenAI Compatible TTS")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[],
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["x-openai-model", "x-openai-voice"],
)

SUPPORTED_LANGUAGES = [
    "en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru",
    "nl", "cs", "ar", "zh-cn", "ja", "ko", "hu", "hi",
]

_tts: TTS | None = None


def get_tts() -> TTS:
    global _tts
    if _tts is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    return _tts


def _find_speaker_wav() -> str | None:
    if SPEAKER_WAV and Path(SPEAKER_WAV).exists():
        return SPEAKER_WAV
    return None


class AudioSpeechRequest(BaseModel):
    model: str = OPENAI_MODEL_ID
    input: str
    voice: str | None = None
    response_format: str = "wav"
    speed: float = 1.0
    language: str | None = None


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception while serving request")
    return JSONResponse(
        status_code=500,
        content={"error": type(exc).__name__, "detail": str(exc)},
    )


@app.get("/")
def root():
    return {"ok": True, "engine": "XTTS-v2", "model": OPENAI_MODEL_ID}


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [{"id": OPENAI_MODEL_ID, "object": "model", "owned_by": "local"}],
    }


@app.get("/v1/voices")
def list_voices():
    tts = get_tts()
    speakers = []
    if hasattr(tts, "speakers") and tts.speakers:
        speakers = tts.speakers
    return {
        "object": "list",
        "data": [{"id": s, "object": "voice"} for s in speakers],
    }


@app.post("/v1/audio/speech")
async def audio_speech(payload: AudioSpeechRequest):
    if payload.response_format.lower() != "wav":
        raise HTTPException(status_code=400, detail="This wrapper currently supports only wav.")

    tts = get_tts()
    language = payload.language or XTTS_LANGUAGE

    if language not in SUPPORTED_LANGUAGES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported language: {language}. Supported: {SUPPORTED_LANGUAGES}",
        )

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        speaker_wav = _find_speaker_wav()
        speaker = payload.voice if payload.voice and not speaker_wav else None

        if speaker_wav:
            tts.tts_to_file(
                text=payload.input,
                file_path=tmp_path,
                speaker_wav=speaker_wav,
                language=language,
            )
        elif speaker:
            tts.tts_to_file(
                text=payload.input,
                file_path=tmp_path,
                speaker=speaker,
                language=language,
            )
        else:
            tts.tts_to_file(
                text=payload.input,
                file_path=tmp_path,
                language=language,
            )

        audio_bytes = Path(tmp_path).read_bytes()
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    if len(audio_bytes) == 0:
        raise HTTPException(status_code=500, detail="No audio was generated.")

    return Response(
        content=audio_bytes,
        media_type="audio/wav",
        headers={
            "Content-Length": str(len(audio_bytes)),
            "x-openai-model": payload.model,
            "x-openai-voice": payload.voice or "default",
        },
    )
