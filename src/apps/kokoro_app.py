from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from kokoro import KPipeline
from pydantic import BaseModel

logger = logging.getLogger("uvicorn.error")

OPENAI_MODEL_ID = os.environ.get("OPENAI_MODEL_ID", "kokoro-82m")
DEFAULT_VOICE = os.environ.get("KOKORO_DEFAULT_VOICE", "jf_alpha")
DEFAULT_LANG_CODE = os.environ.get("KOKORO_DEFAULT_LANG_CODE", "j")

VOICE_PRESETS = [
    "af_heart",
    "af_bella",
    "am_adam",
    "bf_emma",
    "bm_george",
    "jf_alpha",
    "jf_gongitsune",
    "jm_kumo",
    "zf_xiaobei",
]

LANG_CODE_BY_PREFIX = {
    "a": "a",
    "b": "b",
    "e": "e",
    "f": "f",
    "h": "h",
    "i": "i",
    "j": "j",
    "p": "p",
    "z": "z",
}

app = FastAPI(title="Kokoro OpenAI Compatible TTS")

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


_pipelines = {}


def infer_lang_code(voice: str) -> str:
    prefix = (voice or DEFAULT_VOICE).split("_", 1)[0]
    return LANG_CODE_BY_PREFIX.get(prefix[:1], DEFAULT_LANG_CODE)


def get_pipeline(lang_code: str) -> KPipeline:
    pipeline = _pipelines.get(lang_code)
    if pipeline is None:
        pipeline = KPipeline(lang_code=lang_code)
        _pipelines[lang_code] = pipeline
    return pipeline


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception while serving request")
    return JSONResponse(
        status_code=500,
        content={"error": type(exc).__name__, "detail": str(exc)},
    )


@app.get("/")
def root():
    return {"ok": True, "engine": "Kokoro", "model": OPENAI_MODEL_ID}


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
        "data": [{"id": voice, "object": "voice"} for voice in VOICE_PRESETS],
    }


@app.post("/v1/audio/speech")
async def audio_speech(payload: AudioSpeechRequest):
    if payload.response_format.lower() != "wav":
        raise HTTPException(status_code=400, detail="This wrapper currently supports only wav.")

    voice = payload.voice or DEFAULT_VOICE
    lang_code = infer_lang_code(voice)
    pipeline = get_pipeline(lang_code)
    generator = pipeline(payload.input, voice=voice, speed=float(payload.speed))
    chunks = [audio for _, _, audio in generator]
    if not chunks:
        raise HTTPException(status_code=500, detail="No audio was generated.")

    audio = np.concatenate(chunks)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        sf.write(tmp_path, audio, 24000)
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
