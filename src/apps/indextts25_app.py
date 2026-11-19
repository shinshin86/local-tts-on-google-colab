from __future__ import annotations

import io
import os
from functools import lru_cache
from pathlib import Path
from threading import Lock

import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from indextts.infer_v2_5 import IndexTTS2
from pydantic import BaseModel


OPENAI_MODEL_ID = os.environ.get("OPENAI_MODEL_ID", "indextts-2.5")
MODEL_DIR = Path(os.environ["INDEXTTS25_MODEL_DIR"])
LANGUAGE = os.environ.get("INDEXTTS25_LANGUAGE", "ja")
DEFAULT_VOICE = os.environ.get("INDEXTTS25_DEFAULT_VOICE", "default")
DEFAULT_PROMPT_WAV = os.environ["INDEXTTS25_DEFAULT_PROMPT_WAV"]
PROMPT_WAV = os.environ.get("INDEXTTS25_PROMPT_WAV", "")

inference_lock = Lock()
app = FastAPI(title="IndexTTS-2.5 OpenAI Compatible TTS")
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
def get_model() -> IndexTTS2:
    use_cuda = torch.cuda.is_available()
    return IndexTTS2(
        cfg_path=str(MODEL_DIR / "config.yaml"),
        model_dir=str(MODEL_DIR),
        device="cuda:0" if use_cuda else "cpu",
        use_bf16=use_cuda and torch.cuda.is_bf16_supported(including_emulation=False),
        use_cuda_kernel=False,
    )


@app.get("/")
def root():
    return {
        "ok": True,
        "engine": "IndexTTS-2.5",
        "model": OPENAI_MODEL_ID,
        "language": LANGUAGE,
        "default_voice": DEFAULT_VOICE,
    }


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [{"id": OPENAI_MODEL_ID, "object": "model", "owned_by": "indexteam"}],
    }


@app.get("/v1/voices")
def list_voices():
    voices = [{"id": "default", "object": "voice"}]
    if PROMPT_WAV:
        voices.append({"id": "clone", "object": "voice"})
    return {"object": "list", "data": voices}


@app.post("/v1/audio/speech")
def audio_speech(payload: AudioSpeechRequest):
    if payload.response_format.lower() != "wav":
        raise HTTPException(status_code=400, detail="IndexTTS-2.5 supports only WAV output.")
    if not 0.5 <= payload.speed <= 2.0:
        raise HTTPException(status_code=400, detail="speed must be between 0.5 and 2.0.")
    if not payload.input.strip():
        raise HTTPException(status_code=400, detail="Input text is required.")

    voice = payload.voice or DEFAULT_VOICE
    if voice not in {"default", "clone"}:
        raise HTTPException(status_code=400, detail="Use voice='default' or voice='clone'.")
    if voice == "clone" and (not PROMPT_WAV or not Path(PROMPT_WAV).is_file()):
        raise HTTPException(
            status_code=400,
            detail="voice='clone' requires a readable --indextts25-prompt-wav file.",
        )
    prompt = PROMPT_WAV if voice == "clone" else DEFAULT_PROMPT_WAV

    # Upstream caches the last prompt on the model. Keep requests serialized.
    with inference_lock:
        result = get_model().infer(
            spk_audio_prompt=prompt,
            text=payload.input,
            output_path=None,
            lang=LANGUAGE,
            duration_factor=1.0 / payload.speed,
        )
    if result is None or not result[1].size:
        raise HTTPException(status_code=500, detail="No audio was generated.")

    sample_rate, audio = result
    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format="WAV", subtype="PCM_16")
    return Response(
        content=buffer.getvalue(),
        media_type="audio/wav",
        headers={"x-openai-model": OPENAI_MODEL_ID, "x-openai-voice": voice},
    )
