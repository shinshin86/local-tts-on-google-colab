from __future__ import annotations

import io
import math
import os
from functools import lru_cache
from threading import Lock

import soundfile as sf
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from ptts import TTS
from pydantic import BaseModel


MODEL = "hf://gradium/phonon-31m-en@8f7a37c94a4c58b04c8287300c53f11d2e208eae"
OPENAI_MODEL_ID = os.environ.get("OPENAI_MODEL_ID", "phonon")
TEMPERATURE = float(os.environ.get("PHONON_TEMPERATURE", "0.3"))

if not math.isfinite(TEMPERATURE) or TEMPERATURE < 0:
    raise ValueError("--phonon-temperature must be finite and nonnegative.")

inference_lock = Lock()
app = FastAPI(title="Phonon OpenAI Compatible TTS")
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


@lru_cache(maxsize=1)
def get_model() -> TTS:
    model = TTS(MODEL, lang="en", device="cpu", quant="q8", temperature=TEMPERATURE)
    if "default" not in model.voices:
        raise RuntimeError("The bundled Phonon voice could not be loaded.")
    return model


@app.get("/")
def root():
    return {
        "ok": True,
        "engine": "Phonon",
        "model": OPENAI_MODEL_ID,
        "language": "en",
        "default_voice": "default",
    }


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [{"id": OPENAI_MODEL_ID, "object": "model", "owned_by": "gradium"}],
    }


@app.get("/v1/voices")
def list_voices():
    return {"object": "list", "data": [{"id": "default", "object": "voice", "language": "en"}]}


@app.post("/v1/audio/speech")
def audio_speech(payload: AudioSpeechRequest):
    if payload.response_format.lower() != "wav":
        raise HTTPException(status_code=400, detail="Phonon supports only WAV output.")
    if payload.speed != 1.0:
        raise HTTPException(status_code=400, detail="Phonon requires speed=1.0.")
    if not payload.input.strip():
        raise HTTPException(status_code=400, detail="Input text is required.")
    if (payload.voice or "default").strip().lower() != "default":
        raise HTTPException(status_code=400, detail="Use voice='default' for the bundled Phonon speaker.")

    with inference_lock:
        model = get_model()
        waveform = model.synth(payload.input.strip())
    if not waveform.size:
        raise HTTPException(status_code=500, detail="No audio was generated.")

    buffer = io.BytesIO()
    sf.write(buffer, waveform, model.sample_rate, format="WAV", subtype="PCM_16")
    return Response(
        content=buffer.getvalue(),
        media_type="audio/wav",
        headers={"x-openai-model": OPENAI_MODEL_ID, "x-openai-voice": "default"},
    )
