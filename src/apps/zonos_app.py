from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

import soundfile as sf
import torch
import torchaudio
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from zonos.conditioning import make_cond_dict
from zonos.model import Zonos

logger = logging.getLogger("uvicorn.error")

OPENAI_MODEL_ID = os.environ.get("OPENAI_MODEL_ID", "zonos")
ZONOS_HF_MODEL = os.environ.get("ZONOS_HF_MODEL", "Zyphra/Zonos-v0.1-transformer")
ZONOS_LANGUAGE = os.environ.get("ZONOS_LANGUAGE", "ja")
ZONOS_PROMPT_WAV = os.environ.get("ZONOS_PROMPT_WAV", "")
ZONOS_DEFAULT_PROMPT_WAV = os.environ.get("ZONOS_DEFAULT_PROMPT_WAV", "")
ZONOS_DEFAULT_VOICE = os.environ.get("ZONOS_DEFAULT_VOICE", "default")

# Languages natively supported by the Zonos v0.1 release.
SUPPORTED_LANGUAGES = ["en", "ja", "zh", "fr", "de"]

app = FastAPI(title="Zonos OpenAI Compatible TTS")

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


_model: Zonos | None = None
_speaker_cache: dict[str, torch.Tensor] = {}


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_model() -> Zonos:
    global _model
    if _model is None:
        device = _device()
        logger.info("Loading Zonos model %s on %s", ZONOS_HF_MODEL, device)
        _model = Zonos.from_pretrained(ZONOS_HF_MODEL, device=device)
        logger.info("Zonos model loaded")
    return _model


def get_speaker(model: Zonos, voice: str) -> torch.Tensor:
    if voice in _speaker_cache:
        return _speaker_cache[voice]
    if voice == "clone":
        wav_path = ZONOS_PROMPT_WAV
    else:
        wav_path = ZONOS_DEFAULT_PROMPT_WAV
    if not wav_path or not Path(wav_path).exists():
        raise HTTPException(
            status_code=500,
            detail=f"Reference audio not found for voice '{voice}': {wav_path}",
        )
    logger.info("Encoding speaker embedding for voice '%s' from %s", voice, wav_path)
    wav, sampling_rate = torchaudio.load(wav_path)
    speaker = model.make_speaker_embedding(wav, sampling_rate)
    _speaker_cache[voice] = speaker
    return speaker


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
        "engine": "Zonos",
        "model": OPENAI_MODEL_ID,
        "hf_model": ZONOS_HF_MODEL,
        "language": ZONOS_LANGUAGE,
        "default_voice": ZONOS_DEFAULT_VOICE,
    }


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [{"id": OPENAI_MODEL_ID, "object": "model", "owned_by": "zyphra"}],
    }


@app.get("/v1/voices")
def list_voices():
    voices = [{"id": "default", "object": "voice", "language": ZONOS_LANGUAGE}]
    if ZONOS_PROMPT_WAV:
        voices.append({"id": "clone", "object": "voice", "language": ZONOS_LANGUAGE})
    return {"object": "list", "data": voices}


@app.post("/v1/audio/speech")
async def audio_speech(payload: AudioSpeechRequest):
    if payload.response_format.lower() != "wav":
        raise HTTPException(status_code=400, detail="This wrapper currently supports only wav.")

    if ZONOS_LANGUAGE not in SUPPORTED_LANGUAGES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported language '{ZONOS_LANGUAGE}'. Supported: {SUPPORTED_LANGUAGES}",
        )

    voice = payload.voice or ZONOS_DEFAULT_VOICE
    if voice not in {"default", "clone"}:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown voice '{voice}'. Use 'default' or 'clone'.",
        )
    if voice == "clone" and not ZONOS_PROMPT_WAV:
        raise HTTPException(
            status_code=400,
            detail="voice='clone' requires --zonos-prompt-wav at startup.",
        )

    model = get_model()
    speaker = get_speaker(model, voice)
    cond_dict = make_cond_dict(text=payload.input, speaker=speaker, language=ZONOS_LANGUAGE)
    conditioning = model.prepare_conditioning(cond_dict)
    codes = model.generate(conditioning)
    wavs = model.autoencoder.decode(codes).cpu()

    audio = wavs[0]
    if audio.ndim == 2:
        audio = audio[0]
    sample_rate = int(model.autoencoder.sampling_rate)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        sf.write(tmp_path, audio.numpy(), sample_rate)
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
