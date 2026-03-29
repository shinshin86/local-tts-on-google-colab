from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from qwen_tts import Qwen3TTSModel

logger = logging.getLogger("uvicorn.error")

OPENAI_MODEL_ID = os.environ.get("OPENAI_MODEL_ID", "qwen3-tts")
QWEN3_HF_MODEL = os.environ.get("QWEN3_HF_MODEL", "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")
QWEN3_LANGUAGE = os.environ.get("QWEN3_LANGUAGE", "Japanese")
QWEN3_DEFAULT_SPEAKER = os.environ.get("QWEN3_DEFAULT_SPEAKER", "ono_anna")

SUPPORTED_LANGUAGES = [
    "Chinese", "English", "Japanese", "Korean",
    "German", "French", "Russian", "Portuguese", "Spanish", "Italian",
]

SPEAKERS = [
    "aiden", "dylan", "eric", "ono_anna", "ryan",
    "serena", "sohee", "uncle_fu", "vivian",
]

app = FastAPI(title="Qwen3-TTS OpenAI Compatible TTS")

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


_model: Qwen3TTSModel | None = None


def get_model() -> Qwen3TTSModel:
    global _model
    if _model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        _model = Qwen3TTSModel.from_pretrained(
            QWEN3_HF_MODEL,
            device_map=device,
            dtype=dtype,
        )
    return _model


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception while serving request")
    return JSONResponse(
        status_code=500,
        content={"error": type(exc).__name__, "detail": str(exc)},
    )


@app.get("/")
def root():
    return {"ok": True, "engine": "Qwen3-TTS", "model": OPENAI_MODEL_ID}


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
        "data": [{"id": s, "object": "voice"} for s in SPEAKERS],
    }


@app.post("/v1/audio/speech")
async def audio_speech(payload: AudioSpeechRequest):
    if payload.response_format.lower() != "wav":
        raise HTTPException(status_code=400, detail="This wrapper currently supports only wav.")

    language = payload.language or QWEN3_LANGUAGE
    if language not in SUPPORTED_LANGUAGES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported language: {language}. Supported: {SUPPORTED_LANGUAGES}",
        )

    speaker = payload.voice or QWEN3_DEFAULT_SPEAKER
    if speaker not in SPEAKERS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown speaker: {speaker}. Available: {SPEAKERS}",
        )

    model = get_model()
    wavs, sample_rate = model.generate_custom_voice(
        text=payload.input,
        language=language,
        speaker=speaker,
    )

    if not wavs or len(wavs) == 0:
        raise HTTPException(status_code=500, detail="No audio was generated.")

    audio = wavs[0] if isinstance(wavs[0], np.ndarray) else wavs[0].cpu().numpy()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        sf.write(tmp_path, audio, sample_rate)
        audio_bytes = Path(tmp_path).read_bytes()
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return Response(
        content=audio_bytes,
        media_type="audio/wav",
        headers={
            "Content-Length": str(len(audio_bytes)),
            "x-openai-model": payload.model,
            "x-openai-voice": speaker,
        },
    )
