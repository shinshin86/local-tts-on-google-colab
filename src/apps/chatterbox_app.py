from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logger = logging.getLogger("uvicorn.error")

OPENAI_MODEL_ID = os.environ.get("OPENAI_MODEL_ID", "chatterbox")
CHATTERBOX_LANGUAGE = os.environ.get("CHATTERBOX_LANGUAGE", "ja")
CHATTERBOX_PROMPT_WAV = os.environ.get("CHATTERBOX_PROMPT_WAV", "")
CHATTERBOX_DEFAULT_VOICE = os.environ.get("CHATTERBOX_DEFAULT_VOICE", "default")

# Languages supported by Chatterbox Multilingual.
SUPPORTED_LANGUAGES = [
    "ar", "da", "de", "el", "en", "es", "fi", "fr", "he", "hi", "it",
    "ja", "ko", "ms", "nl", "no", "pl", "pt", "ru", "sv", "sw", "tr", "zh",
]

app = FastAPI(title="Chatterbox OpenAI Compatible TTS")

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


_model: ChatterboxMultilingualTTS | None = None


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_model() -> ChatterboxMultilingualTTS:
    global _model
    if _model is None:
        device = _device()
        logger.info("Loading Chatterbox multilingual model on %s", device)
        _model = ChatterboxMultilingualTTS.from_pretrained(device=device)
        logger.info("Chatterbox model loaded (sr=%s)", getattr(_model, "sr", "?"))
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
        "engine": "Chatterbox",
        "model": OPENAI_MODEL_ID,
        "language": CHATTERBOX_LANGUAGE,
        "default_voice": CHATTERBOX_DEFAULT_VOICE,
    }


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [{"id": OPENAI_MODEL_ID, "object": "model", "owned_by": "resemble-ai"}],
    }


@app.get("/v1/voices")
def list_voices():
    voices = [{"id": "default", "object": "voice", "language": CHATTERBOX_LANGUAGE}]
    if CHATTERBOX_PROMPT_WAV:
        voices.append({"id": "clone", "object": "voice", "language": CHATTERBOX_LANGUAGE})
    return {"object": "list", "data": voices}


@app.post("/v1/audio/speech")
async def audio_speech(payload: AudioSpeechRequest):
    if payload.response_format.lower() != "wav":
        raise HTTPException(status_code=400, detail="This wrapper currently supports only wav.")

    if CHATTERBOX_LANGUAGE not in SUPPORTED_LANGUAGES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported language '{CHATTERBOX_LANGUAGE}'. Supported: {SUPPORTED_LANGUAGES}",
        )

    voice = payload.voice or CHATTERBOX_DEFAULT_VOICE
    if voice not in {"default", "clone"}:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown voice '{voice}'. Use 'default' or 'clone'.",
        )
    if voice == "clone" and not CHATTERBOX_PROMPT_WAV:
        raise HTTPException(
            status_code=400,
            detail="voice='clone' requires --chatterbox-prompt-wav at startup.",
        )

    model = get_model()
    kwargs = {"language_id": CHATTERBOX_LANGUAGE}
    if voice == "clone":
        kwargs["audio_prompt_path"] = CHATTERBOX_PROMPT_WAV

    wav = model.generate(text=payload.input, **kwargs)

    if hasattr(wav, "detach"):
        wav_np = wav.detach().to("cpu").numpy()
    else:
        wav_np = np.asarray(wav)
    if wav_np.ndim == 2:
        wav_np = wav_np[0]

    sample_rate = int(getattr(model, "sr", 24000))

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        sf.write(tmp_path, wav_np, sample_rate)
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
