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
from voxcpm import VoxCPM

logger = logging.getLogger("uvicorn.error")

OPENAI_MODEL_ID = os.environ.get("OPENAI_MODEL_ID", "voxcpm2")
VOXCPM_HF_MODEL = os.environ.get("VOXCPM_HF_MODEL", "openbmb/VoxCPM2")
VOXCPM_CFG_VALUE = float(os.environ.get("VOXCPM_CFG_VALUE", "2.0"))
VOXCPM_INFERENCE_TIMESTEPS = int(os.environ.get("VOXCPM_INFERENCE_TIMESTEPS", "10"))

app = FastAPI(title="VoxCPM2 OpenAI Compatible TTS")

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


_model: VoxCPM | None = None


def get_model() -> VoxCPM:
    global _model
    if _model is None:
        logger.info("Loading VoxCPM2 model: %s", VOXCPM_HF_MODEL)
        _model = VoxCPM.from_pretrained(VOXCPM_HF_MODEL, load_denoiser=False)
        logger.info("VoxCPM2 model loaded successfully")
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
    return {"ok": True, "engine": "VoxCPM2", "model": OPENAI_MODEL_ID}


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [{"id": OPENAI_MODEL_ID, "object": "model", "owned_by": "openbmb"}],
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

    model = get_model()
    wav = model.generate(
        text=payload.input,
        cfg_value=VOXCPM_CFG_VALUE,
        inference_timesteps=VOXCPM_INFERENCE_TIMESTEPS,
    )

    if wav is None or (hasattr(wav, "__len__") and len(wav) == 0):
        raise HTTPException(status_code=500, detail="No audio was generated.")

    audio = wav if isinstance(wav, np.ndarray) else wav.cpu().numpy()
    sample_rate = model.tts_model.sample_rate

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        sf.write(tmp_path, audio, sample_rate)
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
