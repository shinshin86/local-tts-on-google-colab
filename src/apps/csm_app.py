from __future__ import annotations

import io
import logging
import os
import sys
from pathlib import Path

import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

CSM_REPO_DIR = os.environ.get("CSM_REPO_DIR", "")
if CSM_REPO_DIR:
    sys.path.insert(0, CSM_REPO_DIR)

from generator import Segment, load_csm_1b  # noqa: E402

logger = logging.getLogger("uvicorn.error")

OPENAI_MODEL_ID = os.environ.get("OPENAI_MODEL_ID", "csm-1b")
DEFAULT_VOICE = os.environ.get("CSM_DEFAULT_VOICE", "default")
DEFAULT_SPEAKER = int(os.environ.get("CSM_DEFAULT_SPEAKER", "0"))
MAX_AUDIO_LENGTH_MS = float(os.environ.get("CSM_MAX_AUDIO_LENGTH_MS", "10000"))
TEMPERATURE = float(os.environ.get("CSM_TEMPERATURE", "0.9"))

app = FastAPI(title="Sesame CSM-1B OpenAI Compatible TTS")

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


_generator = None


def get_generator():
    global _generator
    if _generator is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Loading CSM-1B on %s", device)
        _generator = load_csm_1b(device=device)
        logger.info("CSM-1B loaded (sample_rate=%d)", _generator.sample_rate)
    return _generator


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
        "engine": "Sesame-CSM-1B",
        "model": OPENAI_MODEL_ID,
        "default_voice": DEFAULT_VOICE,
        "default_speaker": DEFAULT_SPEAKER,
    }


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [{"id": OPENAI_MODEL_ID, "object": "model", "owned_by": "sesame"}],
    }


@app.get("/v1/voices")
def list_voices():
    voices = [
        {"id": "default", "object": "voice", "speaker_id": DEFAULT_SPEAKER},
        {"id": "speaker_0", "object": "voice", "speaker_id": 0},
        {"id": "speaker_1", "object": "voice", "speaker_id": 1},
    ]
    return {"object": "list", "data": voices}


def _resolve_speaker(voice: str) -> int:
    if voice == "default":
        return DEFAULT_SPEAKER
    if voice.startswith("speaker_"):
        try:
            return int(voice.split("_", 1)[1])
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid voice: {voice}. Expected speaker_<int>.",
            ) from exc
    raise HTTPException(
        status_code=400,
        detail=f"Unknown voice: {voice}. Available: default, speaker_<int>",
    )


@app.post("/v1/audio/speech")
async def audio_speech(payload: AudioSpeechRequest):
    if payload.response_format.lower() != "wav":
        raise HTTPException(status_code=400, detail="This wrapper currently supports only wav.")

    voice = payload.voice or DEFAULT_VOICE
    speaker = _resolve_speaker(voice)

    generator = get_generator()
    context: list[Segment] = []
    audio_tensor = generator.generate(
        text=payload.input,
        speaker=speaker,
        context=context,
        max_audio_length_ms=MAX_AUDIO_LENGTH_MS,
        temperature=TEMPERATURE,
    )

    if audio_tensor is None or audio_tensor.numel() == 0:
        raise HTTPException(status_code=500, detail="No audio was generated.")

    audio_np = audio_tensor.detach().cpu().numpy().astype("float32").squeeze()

    buf = io.BytesIO()
    sf.write(buf, audio_np, generator.sample_rate, format="WAV", subtype="PCM_16")
    audio_bytes = buf.getvalue()

    return Response(
        content=audio_bytes,
        media_type="audio/wav",
        headers={
            "Content-Length": str(len(audio_bytes)),
            "x-openai-model": payload.model,
            "x-openai-voice": voice,
        },
    )
