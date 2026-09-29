from __future__ import annotations

import io
import logging
import math
import os
import wave
from pathlib import Path

import requests
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logger = logging.getLogger("uvicorn.error")

OPENAI_MODEL_ID = os.environ.get("OPENAI_MODEL_ID", "breeze-tts-2")
BACKEND_URL = os.environ.get("BREEZE_TTS2_BACKEND_URL", "http://127.0.0.1:5007").rstrip("/")
PROMPT_WAV = os.environ.get("BREEZE_TTS2_PROMPT_WAV", "")
PROMPT_TEXT = os.environ.get("BREEZE_TTS2_PROMPT_TEXT", "")
DEFAULT_VOICE = os.environ.get("BREEZE_TTS2_DEFAULT_VOICE", "default")
DEFAULT_INSTRUCTIONS = os.environ.get("BREEZE_TTS2_DEFAULT_INSTRUCTIONS", "")
DEFAULT_SEED = int(os.environ.get("BREEZE_TTS2_SEED", "42"))
INSTRUCTION_CFG_SCALE = float(os.environ.get("BREEZE_TTS2_INSTRUCTION_CFG_SCALE", "4.0"))

app = FastAPI(title="Breeze TTS 2 OpenAI Compatible TTS")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[],
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["x-openai-model", "x-openai-voice", "x-breeze-mode"],
)


class AudioSpeechRequest(BaseModel):
    model: str = OPENAI_MODEL_ID
    input: str
    voice: str | None = None
    instructions: str | None = None
    response_format: str = "wav"
    speed: float = 1.0
    # Breeze-specific optional controls. OpenAI-compatible clients can omit them.
    seed: int | None = None
    cfg_scale: float | None = None


def _clone_is_configured() -> bool:
    return bool(PROMPT_WAV and PROMPT_TEXT.strip())


def _available_voices() -> list[str]:
    voices = ["default"]
    if _clone_is_configured():
        voices.append("clone")
    return voices


def _resolve_voice(voice: str | None) -> str:
    requested = voice or DEFAULT_VOICE
    if requested not in {"default", "clone"}:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown voice '{requested}'. Available: {', '.join(_available_voices())}",
        )
    if requested == "clone":
        if not _clone_is_configured():
            raise HTTPException(
                status_code=400,
                detail=(
                    "voice='clone' requires both --breeze-tts2-prompt-wav and "
                    "--breeze-tts2-prompt-text at startup."
                ),
            )
        if not Path(PROMPT_WAV).is_file():
            raise HTTPException(
                status_code=400,
                detail=f"The configured Breeze TTS 2 reference audio does not exist: {PROMPT_WAV}",
            )
    return requested


def _resolve_instructions(value: str | None) -> str:
    # An explicit empty string suppresses a configured startup default.
    return DEFAULT_INSTRUCTIONS.strip() if value is None else value.strip()


def _resolve_cfg_scale(value: float | None, instructions: str) -> float:
    cfg_scale = value if value is not None else (INSTRUCTION_CFG_SCALE if instructions else 1.0)
    if not math.isfinite(cfg_scale) or cfg_scale <= 0:
        raise HTTPException(status_code=400, detail="cfg_scale must be greater than 0.")
    if not instructions and cfg_scale != 1.0:
        raise HTTPException(
            status_code=400,
            detail="cfg_scale must be 1.0 when instructions are empty.",
        )
    return cfg_scale


def _mode(voice: str, instructions: str) -> str:
    if voice == "clone":
        return "direction" if instructions else "clone"
    return "design" if instructions else "plain"


def _pcm_to_wav(pcm: bytes, sample_rate: int) -> bytes:
    if not pcm:
        raise HTTPException(status_code=502, detail="Breeze TTS 2 backend returned no audio.")
    if len(pcm) % 2:
        raise HTTPException(status_code=502, detail="Breeze TTS 2 backend returned invalid PCM audio.")
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return output.getvalue()


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception while serving request")
    return JSONResponse(status_code=500, content={"error": type(exc).__name__, "detail": str(exc)})


@app.get("/")
def root():
    return {
        "ok": True,
        "engine": "Breeze-TTS-2",
        "model": OPENAI_MODEL_ID,
        "backend": BACKEND_URL,
        "default_voice": DEFAULT_VOICE,
        "default_instructions_configured": bool(DEFAULT_INSTRUCTIONS),
        "voices": _available_voices(),
    }


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [{"id": OPENAI_MODEL_ID, "object": "model", "owned_by": "breezeblue"}],
    }


@app.get("/v1/voices")
def list_voices():
    descriptions = {
        "default": "Plain TTS, or Voice Design when instructions are provided.",
        "clone": "Voice Clone, or Voice Direction when instructions are provided.",
    }
    return {
        "object": "list",
        "data": [
            {"id": voice, "object": "voice", "description": descriptions[voice]}
            for voice in _available_voices()
        ],
    }


@app.post("/v1/audio/speech")
async def audio_speech(payload: AudioSpeechRequest):
    if payload.response_format.lower() != "wav":
        raise HTTPException(status_code=400, detail="This wrapper currently supports only wav.")
    if payload.speed != 1.0:
        raise HTTPException(
            status_code=400,
            detail="Breeze TTS 2 has no numeric speed control; describe the pace in instructions.",
        )
    if not payload.input.strip():
        raise HTTPException(status_code=400, detail="input must not be empty.")

    voice = _resolve_voice(payload.voice)
    instructions = _resolve_instructions(payload.instructions)
    cfg_scale = _resolve_cfg_scale(payload.cfg_scale, instructions)
    mode = _mode(voice, instructions)

    form = {
        "text": payload.input,
        "instruction": instructions,
        "cfg_scale": str(cfg_scale),
        "ref_text": PROMPT_TEXT if voice == "clone" else "",
        "seed": str(DEFAULT_SEED if payload.seed is None else payload.seed),
    }
    files = None
    reference_file = None
    try:
        if voice == "clone":
            reference_file = open(PROMPT_WAV, "rb")
            files = {"ref_audio": (Path(PROMPT_WAV).name, reference_file, "audio/wav")}
        response = requests.post(
            f"{BACKEND_URL}/v1/audio/speech",
            data=form,
            files=files,
            timeout=900,
        )
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Failed to call Breeze TTS 2 backend: {exc}")
    finally:
        if reference_file is not None:
            reference_file.close()

    if not response.ok:
        try:
            detail = response.json()
        except ValueError:
            detail = response.text
        raise HTTPException(status_code=response.status_code, detail=detail)

    sample_rate = int(response.headers.get("X-Sample-Rate", "24000"))
    audio_bytes = _pcm_to_wav(response.content, sample_rate)
    return Response(
        content=audio_bytes,
        media_type="audio/wav",
        headers={
            "Content-Length": str(len(audio_bytes)),
            "x-openai-model": payload.model,
            "x-openai-voice": voice,
            "x-breeze-mode": mode,
        },
    )
