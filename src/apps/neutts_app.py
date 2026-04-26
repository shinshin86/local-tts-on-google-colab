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
from neutts import NeuTTS
from pydantic import BaseModel

logger = logging.getLogger("uvicorn.error")

OPENAI_MODEL_ID = os.environ.get("OPENAI_MODEL_ID", "neutts")
NEUTTS_BACKBONE_REPO = os.environ.get("NEUTTS_BACKBONE_REPO", "neuphonic/neutts-air")
NEUTTS_CODEC_REPO = os.environ.get("NEUTTS_CODEC_REPO", "neuphonic/neucodec")
NEUTTS_DEFAULT_VOICE = os.environ.get("NEUTTS_DEFAULT_VOICE", "jo")
NEUTTS_SAMPLES_DIR = Path(os.environ.get("NEUTTS_SAMPLES_DIR", "samples"))

# Bundled reference voices shipped in the NeuTTS upstream repo.
# Each voice is a (display language, wav filename, transcript filename) tuple.
# The model's language must match the reference audio's language for best results;
# the default backbone (neutts-air) is English-only, so dave/jo are the safe picks.
VOICE_PRESETS: dict[str, dict] = {
    "dave":     {"language": "en", "wav": "dave.wav",     "txt": "dave.txt"},
    "jo":       {"language": "en", "wav": "jo.wav",       "txt": "jo.txt"},
    "greta":    {"language": "de", "wav": "greta.wav",    "txt": "greta.txt"},
    "juliette": {"language": "fr", "wav": "juliette.wav", "txt": "juliette.txt"},
    "mateo":    {"language": "es", "wav": "mateo.wav",    "txt": "mateo.txt"},
}

app = FastAPI(title="NeuTTS OpenAI Compatible TTS")

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


_model: NeuTTS | None = None
_ref_cache: dict[str, tuple[object, str]] = {}


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_model() -> NeuTTS:
    global _model
    if _model is None:
        device = _device()
        logger.info(
            "Loading NeuTTS: backbone=%s codec=%s device=%s",
            NEUTTS_BACKBONE_REPO,
            NEUTTS_CODEC_REPO,
            device,
        )
        _model = NeuTTS(
            backbone_repo=NEUTTS_BACKBONE_REPO,
            backbone_device=device,
            codec_repo=NEUTTS_CODEC_REPO,
            codec_device=device,
        )
        logger.info("NeuTTS model loaded")
    return _model


def get_reference(voice: str, model: NeuTTS) -> tuple[object, str]:
    if voice not in _ref_cache:
        preset = VOICE_PRESETS[voice]
        wav_path = NEUTTS_SAMPLES_DIR / preset["wav"]
        txt_path = NEUTTS_SAMPLES_DIR / preset["txt"]
        if not wav_path.exists() or not txt_path.exists():
            raise HTTPException(
                status_code=500,
                detail=f"Reference files for voice '{voice}' not found under {NEUTTS_SAMPLES_DIR}",
            )
        ref_text = txt_path.read_text(encoding="utf-8").strip()
        logger.info("Encoding reference voice '%s' from %s", voice, wav_path)
        ref_codes = model.encode_reference(str(wav_path))
        _ref_cache[voice] = (ref_codes, ref_text)
    return _ref_cache[voice]


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
        "engine": "NeuTTS",
        "model": OPENAI_MODEL_ID,
        "backbone": NEUTTS_BACKBONE_REPO,
        "default_voice": NEUTTS_DEFAULT_VOICE,
    }


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [{"id": OPENAI_MODEL_ID, "object": "model", "owned_by": "neuphonic"}],
    }


@app.get("/v1/voices")
def list_voices():
    return {
        "object": "list",
        "data": [
            {"id": name, "object": "voice", "language": preset["language"]}
            for name, preset in VOICE_PRESETS.items()
        ],
    }


@app.post("/v1/audio/speech")
async def audio_speech(payload: AudioSpeechRequest):
    if payload.response_format.lower() != "wav":
        raise HTTPException(status_code=400, detail="This wrapper currently supports only wav.")

    voice = payload.voice or NEUTTS_DEFAULT_VOICE
    if voice not in VOICE_PRESETS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown voice '{voice}'. Available: {sorted(VOICE_PRESETS.keys())}",
        )

    model = get_model()
    ref_codes, ref_text = get_reference(voice, model)

    wav = model.infer(payload.input, ref_codes, ref_text)
    if wav is None or (hasattr(wav, "__len__") and len(wav) == 0):
        raise HTTPException(status_code=500, detail="No audio was generated.")

    audio = wav if isinstance(wav, np.ndarray) else np.asarray(wav)
    sample_rate = 24000

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        sf.write(tmp_path, audio, sample_rate)
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
