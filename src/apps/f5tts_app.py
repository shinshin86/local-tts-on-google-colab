from __future__ import annotations

import logging
import os
import tempfile
from importlib.resources import files
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logger = logging.getLogger("uvicorn.error")

OPENAI_MODEL_ID = os.environ.get("OPENAI_MODEL_ID", "f5-tts")
F5TTS_MODEL = os.environ.get("F5TTS_MODEL", "F5TTS_v1_Base")
F5TTS_CKPT_FILE = os.environ.get("F5TTS_CKPT_FILE", "")
F5TTS_VOCAB_FILE = os.environ.get("F5TTS_VOCAB_FILE", "")

DEFAULT_REF_AUDIO = str(files("f5_tts").joinpath("infer/examples/basic/basic_ref_en.wav"))
DEFAULT_REF_TEXT = "Some call me nature, others call me mother nature."

app = FastAPI(title="F5-TTS OpenAI Compatible TTS")

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


_tts = None


def get_tts():
    global _tts
    if _tts is None:
        from f5_tts.api import F5TTS

        logger.info("Loading F5-TTS model: %s", F5TTS_MODEL)
        _tts = F5TTS(
            model=F5TTS_MODEL,
            ckpt_file=F5TTS_CKPT_FILE,
            vocab_file=F5TTS_VOCAB_FILE,
        )
        logger.info("F5-TTS model loaded successfully")
    return _tts


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception while serving request")
    return JSONResponse(
        status_code=500,
        content={"error": type(exc).__name__, "detail": str(exc)},
    )


@app.get("/")
def root():
    return {"ok": True, "engine": "F5-TTS", "model": OPENAI_MODEL_ID}


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [{"id": OPENAI_MODEL_ID, "object": "model", "owned_by": "SWivid"}],
    }


@app.get("/v1/voices")
def list_voices():
    return {
        "object": "list",
        "data": [{"id": "default", "object": "voice"}],
    }


@app.post("/v1/audio/speech")
async def audio_speech(payload: AudioSpeechRequest):
    if payload.response_format.lower() != "wav":
        raise HTTPException(status_code=400, detail="This wrapper currently supports only wav.")

    tts = get_tts()
    wav, sr, _ = tts.infer(
        ref_file=DEFAULT_REF_AUDIO,
        ref_text=DEFAULT_REF_TEXT,
        gen_text=payload.input,
        speed=payload.speed,
    )

    if wav is None or len(wav) == 0:
        raise HTTPException(status_code=500, detail="No audio was generated.")

    audio = wav if isinstance(wav, np.ndarray) else np.array(wav)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        sf.write(tmp_path, audio, sr)
        audio_bytes = Path(tmp_path).read_bytes()
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    voice = payload.voice or "default"
    return Response(
        content=audio_bytes,
        media_type="audio/wav",
        headers={
            "Content-Length": str(len(audio_bytes)),
            "x-openai-model": payload.model,
            "x-openai-voice": voice,
        },
    )
