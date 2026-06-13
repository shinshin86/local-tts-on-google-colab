from __future__ import annotations

import base64
import io
import logging
import os
from pathlib import Path

import numpy as np
import requests
import soundfile as sf
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logger = logging.getLogger("uvicorn.error")

OPENAI_MODEL_ID = os.environ.get("OPENAI_MODEL_ID", "zonos2")
BACKEND_URL = os.environ.get("ZONOS2_BACKEND_URL", "http://127.0.0.1:5003")
ZONOS2_LANGUAGE = os.environ.get("ZONOS2_LANGUAGE", "ja")
ZONOS2_DEFAULT_REF = os.environ.get("ZONOS2_DEFAULT_REF", "")
ZONOS2_PROMPT_WAV = os.environ.get("ZONOS2_PROMPT_WAV", "")
ZONOS2_DEFAULT_VOICE = os.environ.get("ZONOS2_DEFAULT_VOICE", "default")
ZONOS2_ACCURATE_MODE = os.environ.get("ZONOS2_ACCURATE_MODE", "1") not in {"0", "false", "False", ""}
_seed_raw = os.environ.get("ZONOS2_SEED", "")
ZONOS2_SEED: int | None = int(_seed_raw) if _seed_raw.strip() else None

# Mini-SGLang `/tts/generate` always streams raw little-endian float32 PCM,
# mono, at 44.1 kHz (see the X-Audio-* response headers upstream sends).
BACKEND_SAMPLE_RATE = 44100

# Convenience aliases → the text-normalization language codes the backend
# `/tts/generate` endpoint understands (en_us, en_gb, fr_fr, de, es, it,
# pt_br, ja, cmn, ko). Pass any of those codes through verbatim.
LANGUAGE_ALIASES = {
    "en": "en_us",
    "en-us": "en_us",
    "en-gb": "en_gb",
    "zh": "cmn",
    "zh-cn": "cmn",
    "fr": "fr_fr",
    "pt": "pt_br",
    "jp": "ja",
}


def _resolve_language(language: str) -> str:
    key = language.strip().lower()
    return LANGUAGE_ALIASES.get(key, key or "ja")


app = FastAPI(title="ZONOS2 OpenAI Compatible TTS")

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


def _reference_path(voice: str) -> str:
    if voice == "clone":
        return ZONOS2_PROMPT_WAV
    return ZONOS2_DEFAULT_REF


def _encode_reference(wav_path: str) -> tuple[str, str]:
    path = Path(wav_path)
    if not wav_path or not path.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Reference audio not found: {wav_path}",
        )
    audio_b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return audio_b64, path.name


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
        "engine": "ZONOS2",
        "model": OPENAI_MODEL_ID,
        "backend": BACKEND_URL,
        "language": ZONOS2_LANGUAGE,
        "default_voice": ZONOS2_DEFAULT_VOICE,
    }


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [{"id": OPENAI_MODEL_ID, "object": "model", "owned_by": "zyphra"}],
    }


@app.get("/v1/voices")
def list_voices():
    voices = [{"id": "default", "object": "voice", "language": ZONOS2_LANGUAGE}]
    if ZONOS2_PROMPT_WAV:
        voices.append({"id": "clone", "object": "voice", "language": ZONOS2_LANGUAGE})
    return {"object": "list", "data": voices}


@app.post("/v1/audio/speech")
async def audio_speech(payload: AudioSpeechRequest):
    if payload.response_format.lower() != "wav":
        raise HTTPException(status_code=400, detail="This wrapper currently supports only wav.")

    voice = payload.voice or ZONOS2_DEFAULT_VOICE
    if voice not in {"default", "clone"}:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown voice '{voice}'. Use 'default' or 'clone'.",
        )
    if voice == "clone" and not ZONOS2_PROMPT_WAV:
        raise HTTPException(
            status_code=400,
            detail="voice='clone' requires --zonos2-prompt-wav at startup.",
        )

    speaker_b64, speaker_name = _encode_reference(_reference_path(voice))

    backend_payload = {
        "text": payload.input,
        "language": _resolve_language(ZONOS2_LANGUAGE),
        "accurate_mode": ZONOS2_ACCURATE_MODE,
        "stream": False,
        "speaker_audio_base64": speaker_b64,
        "speaker_audio_name": speaker_name,
    }
    if ZONOS2_SEED is not None:
        backend_payload["seed"] = ZONOS2_SEED

    try:
        # First call blocks while the Mini-SGLang workers finish loading the
        # model, so keep the timeout generous.
        response = requests.post(
            f"{BACKEND_URL}/tts/generate",
            json=backend_payload,
            timeout=900,
        )
        response.raise_for_status()
    except requests.HTTPError as exc:
        detail = exc.response.text if exc.response is not None else str(exc)
        raise HTTPException(status_code=502, detail=f"Backend error: {detail}")
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to synthesize audio: {exc}")

    pcm = np.frombuffer(response.content, dtype="<f4")
    if pcm.size == 0:
        raise HTTPException(status_code=500, detail="No audio was generated.")

    buf = io.BytesIO()
    # Write PCM_16 (not FLOAT) so the WAV is readable by the Python stdlib
    # `wave` module and the broadest set of OpenAI-style clients.
    sf.write(buf, pcm, BACKEND_SAMPLE_RATE, format="WAV", subtype="PCM_16")
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
