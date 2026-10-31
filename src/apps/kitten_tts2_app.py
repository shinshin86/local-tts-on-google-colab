from __future__ import annotations

import io
import json
import logging
import os
import threading
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logger = logging.getLogger("uvicorn.error")

OPENAI_MODEL_ID = os.environ.get("OPENAI_MODEL_ID", "kitten-tts-2")
MODEL_DIR = Path(os.environ["KITTEN_TTS2_MODEL_DIR"])
WEIGHTS = os.environ.get("KITTEN_TTS2_WEIGHTS", "packed")
DECODER = os.environ.get("KITTEN_TTS2_DECODER", "default")
DEFAULT_VOICE = os.environ.get("KITTEN_TTS2_DEFAULT_VOICE", "JapaneseAnimeHeroine")
PROMPT_WAV = os.environ.get("KITTEN_TTS2_PROMPT_WAV", "")
PROMPT_TEXT = os.environ.get("KITTEN_TTS2_PROMPT_TEXT", "")
DEFAULT_PRESET = os.environ.get("KITTEN_TTS2_PRESET", "stable")
DEFAULT_SEED = int(os.environ.get("KITTEN_TTS2_SEED", "-1"))


def _read_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# Read the voice list and presets straight from the downloaded repo so that
# /v1/voices answers without loading the 1.7B model.
_CONFIG = _read_json(MODEL_DIR / "config.json")
VOICES = sorted(_read_json(MODEL_DIR / _CONFIG.get("voices", "voices/voices.json")))
PRESETS = sorted(_CONFIG.get("decode_presets", {}))

app = FastAPI(title="KittenTTS 2 OpenAI Compatible TTS")

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
    # KittenTTS 2 extensions (all optional).
    preset: str | None = None
    seed: int | None = None
    # None = auto: the upstream normalizer is English-tuned, so it is only
    # applied when the input is plain ASCII.
    normalize: bool | None = None


_tts = None
_load_lock = threading.Lock()
_generate_lock = threading.Lock()


def get_tts():
    global _tts
    with _load_lock:
        if _tts is None:
            from kittenml import KittenTTS

            _tts = KittenTTS(str(MODEL_DIR), weights=WEIGHTS, decoder=DECODER)
            logger.info("KittenTTS 2 loaded on %s (sample rate %s)", _tts.device, _tts.sample_rate)
    return _tts


def clone_available() -> bool:
    return bool(PROMPT_WAV) and Path(PROMPT_WAV).is_file()


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
        "engine": "KittenTTS-2",
        "model": OPENAI_MODEL_ID,
        "weights": WEIGHTS,
        "decoder": DECODER,
        "default_voice": DEFAULT_VOICE,
        "clone_available": clone_available(),
        "presets": PRESETS,
    }


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [{"id": OPENAI_MODEL_ID, "object": "model", "owned_by": "local"}],
    }


@app.get("/v1/voices")
def list_voices():
    extra = [{"id": "clone", "object": "voice"}] if clone_available() else []
    return {
        "object": "list",
        "data": [
            {"id": "default", "object": "voice"},
            *extra,
            *[{"id": voice, "object": "voice"} for voice in VOICES],
        ],
    }


@app.post("/v1/audio/speech")
def audio_speech(payload: AudioSpeechRequest):
    if payload.response_format.lower() != "wav":
        raise HTTPException(status_code=400, detail="This wrapper currently supports only wav.")
    text = payload.input.strip()
    if not text:
        raise HTTPException(status_code=400, detail="input must not be empty.")
    if payload.speed != 1.0:
        raise HTTPException(status_code=400, detail="KittenTTS 2 supports speed 1.0 only.")
    preset = payload.preset or DEFAULT_PRESET
    if preset not in PRESETS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown preset '{preset}'. Available presets: {', '.join(PRESETS)}",
        )

    voice = payload.voice or "default"
    if voice == "default":
        voice = DEFAULT_VOICE
    speaker: dict
    if voice == "clone":
        if not clone_available():
            raise HTTPException(
                status_code=400,
                detail=(
                    "voice 'clone' needs a reference clip. Start the server with "
                    "--kitten-tts2-prompt-wav (and ideally --kitten-tts2-prompt-text)."
                ),
            )
        speaker = {"reference": PROMPT_WAV, "reference_text": PROMPT_TEXT or None}
    elif voice in VOICES:
        speaker = {"voice": voice}
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown voice '{voice}'. See /v1/voices for the available voices.",
        )

    seed = payload.seed if payload.seed is not None else DEFAULT_SEED
    advanced = {"seed": seed} if seed is not None and seed >= 0 else None
    normalize = payload.normalize if payload.normalize is not None else text.isascii()

    tts = get_tts()
    with _generate_lock:
        try:
            audio = tts.generate(
                text, preset=preset, normalize=normalize, advanced=advanced, **speaker
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    audio = np.asarray(audio, dtype=np.float32).squeeze()
    if audio.size == 0:
        raise HTTPException(status_code=500, detail="No audio was generated.")

    buffer = io.BytesIO()
    sf.write(buffer, audio, tts.sample_rate, format="WAV", subtype="PCM_16")
    audio_bytes = buffer.getvalue()
    return Response(
        content=audio_bytes,
        media_type="audio/wav",
        headers={
            "Content-Length": str(len(audio_bytes)),
            "x-openai-model": payload.model,
            "x-openai-voice": voice,
        },
    )
