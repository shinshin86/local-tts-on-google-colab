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

SPARK_REPO_DIR = os.environ.get("SPARK_REPO_DIR", "")
if SPARK_REPO_DIR:
    sys.path.insert(0, SPARK_REPO_DIR)

from cli.SparkTTS import SparkTTS  # noqa: E402

logger = logging.getLogger("uvicorn.error")

OPENAI_MODEL_ID = os.environ.get("OPENAI_MODEL_ID", "spark-tts")
MODEL_DIR = os.environ.get("SPARK_MODEL_DIR", "")
DEFAULT_VOICE = os.environ.get("SPARK_DEFAULT_VOICE", "default")
DEFAULT_GENDER = os.environ.get("SPARK_DEFAULT_GENDER", "female")
DEFAULT_PITCH = os.environ.get("SPARK_DEFAULT_PITCH", "moderate")
DEFAULT_SPEED = os.environ.get("SPARK_DEFAULT_SPEED", "moderate")
PROMPT_WAV = os.environ.get("SPARK_PROMPT_WAV", "")
PROMPT_TEXT = os.environ.get("SPARK_PROMPT_TEXT", "")

LEVELS = {"very_low", "low", "moderate", "high", "very_high"}
GENDERS = {"male", "female"}

app = FastAPI(title="Spark-TTS OpenAI Compatible TTS")

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


_model: SparkTTS | None = None


def get_model() -> SparkTTS:
    global _model
    if _model is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logger.info("Loading Spark-TTS from %s on %s", MODEL_DIR, device)
        _model = SparkTTS(Path(MODEL_DIR), device=device)
        logger.info("Spark-TTS loaded (sample_rate=%d)", _model.sample_rate)
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
    return {
        "ok": True,
        "engine": "Spark-TTS",
        "model": OPENAI_MODEL_ID,
        "model_dir": MODEL_DIR,
        "default_voice": DEFAULT_VOICE,
    }


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [{"id": OPENAI_MODEL_ID, "object": "model", "owned_by": "sparkaudio"}],
    }


@app.get("/v1/voices")
def list_voices():
    voices = [
        {
            "id": "default",
            "object": "voice",
            "mode": "control",
            "gender": DEFAULT_GENDER,
            "pitch": DEFAULT_PITCH,
            "speed": DEFAULT_SPEED,
        }
    ]
    if PROMPT_WAV:
        voices.append({"id": "clone", "object": "voice", "ref": PROMPT_WAV})
    return {"object": "list", "data": voices}


@app.post("/v1/audio/speech")
async def audio_speech(payload: AudioSpeechRequest):
    if payload.response_format.lower() != "wav":
        raise HTTPException(status_code=400, detail="This wrapper currently supports only wav.")

    voice = payload.voice or DEFAULT_VOICE
    model = get_model()

    if voice == "clone":
        if not PROMPT_WAV:
            raise HTTPException(
                status_code=400,
                detail="voice='clone' requires --spark-prompt-wav at startup.",
            )
        waveform = model.inference(
            payload.input,
            prompt_speech_path=PROMPT_WAV,
            prompt_text=PROMPT_TEXT or None,
        )
    elif voice == "default":
        if DEFAULT_GENDER not in GENDERS:
            raise HTTPException(
                status_code=500,
                detail=f"Invalid SPARK_DEFAULT_GENDER: {DEFAULT_GENDER}. Use male/female.",
            )
        if DEFAULT_PITCH not in LEVELS or DEFAULT_SPEED not in LEVELS:
            raise HTTPException(
                status_code=500,
                detail=f"Invalid pitch/speed level. Use one of {sorted(LEVELS)}.",
            )
        waveform = model.inference(
            payload.input,
            prompt_speech_path=None,
            gender=DEFAULT_GENDER,
            pitch=DEFAULT_PITCH,
            speed=DEFAULT_SPEED,
        )
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown voice: {voice}. Available: default, clone (when prompt configured)",
        )

    if isinstance(waveform, torch.Tensor):
        audio = waveform.detach().cpu().numpy().squeeze()
    else:
        audio = waveform

    buf = io.BytesIO()
    sf.write(buf, audio, model.sample_rate, format="WAV", subtype="PCM_16")
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
