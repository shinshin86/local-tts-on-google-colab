from __future__ import annotations

import io
import os
from functools import lru_cache
from threading import Lock

import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from nemo.collections.tts.modules.magpietts_inference.utils import ModelLoadConfig, load_magpie_model
from pydantic import BaseModel


OPENAI_MODEL_ID = os.environ.get("OPENAI_MODEL_ID", "magpie-tts")
MODEL_PATH = os.environ["MAGPIE_TTS_MODEL_PATH"]
CODEC_PATH = os.environ["MAGPIE_TTS_CODEC_PATH"]
LANGUAGE = os.environ.get("MAGPIE_TTS_LANGUAGE", "ja")
DEFAULT_VOICE = os.environ.get("MAGPIE_TTS_DEFAULT_VOICE", "sofia")
TEXT_NORMALIZATION = os.environ.get("MAGPIE_TTS_TEXT_NORMALIZATION", "1") == "1"

# The fixed speaker indices are specific to the pinned v2607 checkpoint.
SPEAKERS = {"aria": 0, "jason": 1, "john": 2, "leo": 3, "sofia": 4}
LANGUAGE_CODES = {"ar": "ar-MSA", "pt": "pt-BR"}
inference_lock = Lock()

app = FastAPI(title="Magpie-TTS OpenAI Compatible TTS")
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
def get_model():
    if not torch.cuda.is_available():
        raise RuntimeError("Magpie-TTS requires a CUDA GPU in this wrapper.")
    model, _ = load_magpie_model(
        ModelLoadConfig(nemo_file=MODEL_PATH, codecmodel_path=CODEC_PATH),
        device="cuda",
    )
    return model


@app.get("/")
def root():
    return {
        "ok": True,
        "engine": "Magpie-TTS",
        "model": OPENAI_MODEL_ID,
        "language": LANGUAGE,
        "default_voice": DEFAULT_VOICE,
    }


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [{"id": OPENAI_MODEL_ID, "object": "model", "owned_by": "nvidia"}],
    }


@app.get("/v1/voices")
def list_voices():
    return {"object": "list", "data": [{"id": voice, "object": "voice"} for voice in SPEAKERS]}


@app.post("/v1/audio/speech")
def audio_speech(payload: AudioSpeechRequest):
    if payload.response_format.lower() != "wav":
        raise HTTPException(status_code=400, detail="Magpie-TTS supports only WAV output.")
    if payload.speed != 1.0:
        raise HTTPException(status_code=400, detail="Magpie-TTS requires speed=1.0.")
    if not payload.input.strip():
        raise HTTPException(status_code=400, detail="Input text is required.")

    voice = (payload.voice or "default").strip().lower()
    if voice == "default":
        voice = DEFAULT_VOICE
    if voice not in SPEAKERS:
        raise HTTPException(
            status_code=400,
            detail=f"Use one of the built-in voices: {', '.join(SPEAKERS)}.",
        )

    # Serialize inference because NeMo caches text normalizers and decoder state.
    with inference_lock:
        model = get_model()
        audio, lengths = model.do_tts(
            transcript=payload.input.strip(),
            language=LANGUAGE_CODES.get(LANGUAGE, LANGUAGE),
            apply_TN=TEXT_NORMALIZATION,
            speaker_index=SPEAKERS[voice],
        )
        length = int(lengths[0])
        if length <= 0:
            raise HTTPException(status_code=500, detail="No audio was generated.")
        waveform = audio[0, :length].detach().float().cpu().numpy()

    buffer = io.BytesIO()
    sf.write(buffer, waveform, model.output_sample_rate, format="WAV", subtype="PCM_16")
    return Response(
        content=buffer.getvalue(),
        media_type="audio/wav",
        headers={"x-openai-model": OPENAI_MODEL_ID, "x-openai-voice": voice},
    )
