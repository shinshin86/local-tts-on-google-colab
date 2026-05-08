from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from styletts2 import tts as styletts2_tts

logger = logging.getLogger("uvicorn.error")

OPENAI_MODEL_ID = os.environ.get("OPENAI_MODEL_ID", "styletts2")
DEFAULT_VOICE = os.environ.get("STYLETTS2_DEFAULT_VOICE", "default")
PROMPT_WAV = os.environ.get("STYLETTS2_PROMPT_WAV", "")
ALPHA = float(os.environ.get("STYLETTS2_ALPHA", "0.3"))
BETA = float(os.environ.get("STYLETTS2_BETA", "0.7"))
DIFFUSION_STEPS = int(os.environ.get("STYLETTS2_DIFFUSION_STEPS", "5"))
EMBEDDING_SCALE = float(os.environ.get("STYLETTS2_EMBEDDING_SCALE", "1"))

app = FastAPI(title="StyleTTS 2 OpenAI Compatible TTS")

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


_tts_engine: styletts2_tts.StyleTTS2 | None = None


def get_engine() -> styletts2_tts.StyleTTS2:
    global _tts_engine
    if _tts_engine is None:
        logger.info("Loading StyleTTS 2 (default LibriTTS checkpoint)")
        _tts_engine = styletts2_tts.StyleTTS2()
        logger.info("StyleTTS 2 loaded")
    return _tts_engine


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
        "engine": "StyleTTS2",
        "model": OPENAI_MODEL_ID,
        "default_voice": DEFAULT_VOICE,
    }


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [{"id": OPENAI_MODEL_ID, "object": "model", "owned_by": "yl4579"}],
    }


@app.get("/v1/voices")
def list_voices():
    voices = [{"id": "default", "object": "voice"}]
    if PROMPT_WAV:
        voices.append({"id": "clone", "object": "voice", "ref": PROMPT_WAV})
    return {"object": "list", "data": voices}


@app.post("/v1/audio/speech")
async def audio_speech(payload: AudioSpeechRequest):
    if payload.response_format.lower() != "wav":
        raise HTTPException(status_code=400, detail="This wrapper currently supports only wav.")

    voice = payload.voice or DEFAULT_VOICE

    if voice == "clone":
        if not PROMPT_WAV:
            raise HTTPException(
                status_code=400,
                detail="voice='clone' requires --styletts2-prompt-wav at startup.",
            )
        target_voice_path: str | None = PROMPT_WAV
    elif voice == "default":
        target_voice_path = None
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown voice: {voice}. Available: default, clone (when prompt configured)",
        )

    engine = get_engine()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        kwargs = {
            "output_wav_file": tmp_path,
            "alpha": ALPHA,
            "beta": BETA,
            "diffusion_steps": DIFFUSION_STEPS,
            "embedding_scale": EMBEDDING_SCALE,
        }
        if target_voice_path:
            kwargs["target_voice_path"] = target_voice_path
        engine.inference(payload.input, **kwargs)
        audio_bytes = Path(tmp_path).read_bytes()
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    if not audio_bytes:
        raise HTTPException(status_code=500, detail="No audio was generated.")

    return Response(
        content=audio_bytes,
        media_type="audio/wav",
        headers={
            "Content-Length": str(len(audio_bytes)),
            "x-openai-model": payload.model,
            "x-openai-voice": voice,
        },
    )
