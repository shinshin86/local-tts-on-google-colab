from __future__ import annotations

import io
import os
from threading import Lock

import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from tiny_voca.infer import load_model, speak_text


OPENAI_MODEL_ID = os.environ.get("OPENAI_MODEL_ID", "tiny-voca")
DEFAULT_VOICE = os.environ.get("TINY_VOCA_DEFAULT_VOICE", "F2")

# Keep PyTorch's CPU thread pool small for this model and share the loaded model.
torch.set_num_threads(min(torch.get_num_threads(), 4))
tts = load_model(os.environ["TINY_VOCA_MODEL_PATH"], device="cpu")
inference_lock = Lock()

app = FastAPI(title="tiny-voca OpenAI Compatible TTS")
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


@app.get("/")
def root():
    return {
        "ok": True,
        "engine": "tiny-voca",
        "model": OPENAI_MODEL_ID,
        "language": "en",
        "default_voice": DEFAULT_VOICE,
    }


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [{"id": OPENAI_MODEL_ID, "object": "model", "owned_by": "n4ze3m"}],
    }


@app.get("/v1/voices")
def list_voices():
    return {
        "object": "list",
        "data": [{"id": voice, "object": "voice", "language": "en"} for voice in tts.config.speakers],
    }


@app.post("/v1/audio/speech")
def audio_speech(payload: AudioSpeechRequest):
    if payload.response_format.lower() != "wav":
        raise HTTPException(status_code=400, detail="tiny-voca supports only WAV output.")
    if not 0.25 <= payload.speed <= 4.0:
        raise HTTPException(status_code=400, detail="speed must be between 0.25 and 4.0.")
    if not payload.input.strip():
        raise HTTPException(status_code=400, detail="Input text is required.")

    voice = (payload.voice or "default").strip().upper()
    if voice == "DEFAULT":
        voice = DEFAULT_VOICE
    if voice not in tts.config.speakers:
        raise HTTPException(status_code=400, detail="Use one of the built-in voices: F2 or M2.")

    try:
        with inference_lock, torch.inference_mode():
            waveform, _ = speak_text(tts, payload.input, voice=voice, speed=payload.speed)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not waveform.numel():
        raise HTTPException(status_code=500, detail="No audio was generated.")

    buffer = io.BytesIO()
    sf.write(buffer, waveform.numpy(), tts.config.sample_rate, format="WAV", subtype="PCM_16")
    return Response(
        content=buffer.getvalue(),
        media_type="audio/wav",
        headers={"x-openai-model": OPENAI_MODEL_ID, "x-openai-voice": voice},
    )
