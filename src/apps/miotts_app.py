from __future__ import annotations

import base64
import json
import logging
import os
from pathlib import Path

import requests
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logger = logging.getLogger("uvicorn.error")

OPENAI_MODEL_ID = os.environ.get("OPENAI_MODEL_ID", "miotts")
# MioTTS synthesis backend (run_server.py) — exposes /v1/tts and /v1/tts/file.
BACKEND_URL = os.environ.get("MIOTTS_BACKEND_URL", "http://127.0.0.1:5005")
MIOTTS_DEFAULT_VOICE = os.environ.get("MIOTTS_DEFAULT_VOICE", "default")
MIOTTS_DEFAULT_PRESET = os.environ.get("MIOTTS_DEFAULT_PRESET", "jp_female")
MIOTTS_PROMPT_WAV = os.environ.get("MIOTTS_PROMPT_WAV", "")
# Shipped presets, used both for validation and the /v1/voices listing.
MIOTTS_PRESETS = [
    p.strip()
    for p in os.environ.get(
        "MIOTTS_PRESETS", "jp_female,jp_male,en_female,en_male"
    ).split(",")
    if p.strip()
]


def _opt_float(name: str) -> float | None:
    raw = os.environ.get(name, "")
    return float(raw) if raw.strip() else None


def _opt_int(name: str) -> int | None:
    raw = os.environ.get(name, "")
    return int(raw) if raw.strip() else None


MIOTTS_TEMPERATURE = _opt_float("MIOTTS_TEMPERATURE")
MIOTTS_TOP_P = _opt_float("MIOTTS_TOP_P")
MIOTTS_REPETITION_PENALTY = _opt_float("MIOTTS_REPETITION_PENALTY")
MIOTTS_MAX_TOKENS = _opt_int("MIOTTS_MAX_TOKENS")

app = FastAPI(title="MioTTS OpenAI Compatible TTS")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[],
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=[
        "x-openai-model",
        "x-openai-voice",
        "x-miotts-timings",
        "x-miotts-token-count",
    ],
)


class AudioSpeechRequest(BaseModel):
    model: str = OPENAI_MODEL_ID
    input: str
    voice: str | None = None
    response_format: str = "wav"
    speed: float = 1.0


def _llm_params() -> dict:
    params: dict[str, float | int] = {}
    if MIOTTS_TEMPERATURE is not None:
        params["temperature"] = MIOTTS_TEMPERATURE
    if MIOTTS_TOP_P is not None:
        params["top_p"] = MIOTTS_TOP_P
    if MIOTTS_REPETITION_PENALTY is not None:
        params["repetition_penalty"] = MIOTTS_REPETITION_PENALTY
    if MIOTTS_MAX_TOKENS is not None:
        params["max_tokens"] = MIOTTS_MAX_TOKENS
    return params


def _build_reference(voice: str) -> dict:
    if voice == "clone":
        path = Path(MIOTTS_PROMPT_WAV)
        if not MIOTTS_PROMPT_WAV or not path.exists():
            raise HTTPException(
                status_code=400,
                detail=(
                    "voice='clone' requires --miotts-prompt-wav pointing at an "
                    f"existing reference audio file (got: {MIOTTS_PROMPT_WAV!r})."
                ),
            )
        data = base64.b64encode(path.read_bytes()).decode("ascii")
        return {"type": "base64", "data": data}
    # default -> the configured shipped preset; an explicit preset name -> itself.
    preset_id = MIOTTS_DEFAULT_PRESET if voice == "default" else voice
    return {"type": "preset", "preset_id": preset_id}


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
        "engine": "MioTTS",
        "model": OPENAI_MODEL_ID,
        "backend": BACKEND_URL,
        "default_voice": MIOTTS_DEFAULT_VOICE,
        "default_preset": MIOTTS_DEFAULT_PRESET,
        "presets": MIOTTS_PRESETS,
    }


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [{"id": OPENAI_MODEL_ID, "object": "model", "owned_by": "aratako"}],
    }


@app.get("/v1/voices")
def list_voices():
    voices = [{"id": "default", "object": "voice", "preset": MIOTTS_DEFAULT_PRESET}]
    for preset in MIOTTS_PRESETS:
        voices.append({"id": preset, "object": "voice", "preset": preset})
    if MIOTTS_PROMPT_WAV:
        voices.append({"id": "clone", "object": "voice", "preset": None})
    return {"object": "list", "data": voices}


def _valid_voices() -> set[str]:
    valid = {"default", *MIOTTS_PRESETS}
    if MIOTTS_PROMPT_WAV:
        valid.add("clone")
    return valid


@app.post("/v1/audio/speech")
async def audio_speech(payload: AudioSpeechRequest):
    if payload.response_format.lower() != "wav":
        raise HTTPException(status_code=400, detail="This wrapper currently supports only wav.")

    voice = payload.voice or MIOTTS_DEFAULT_VOICE
    valid = _valid_voices()
    if voice not in valid:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown voice '{voice}'. Available: {sorted(valid)}.",
        )

    backend_payload = {
        "text": payload.input,
        "reference": _build_reference(voice),
        # base64 output gives us the synthesis timings + token count, which we
        # surface as response headers for speed observability.
        "output": {"format": "base64"},
    }
    llm = _llm_params()
    if llm:
        backend_payload["llm"] = llm

    try:
        response = requests.post(
            f"{BACKEND_URL}/v1/tts",
            json=backend_payload,
            timeout=600,
        )
        response.raise_for_status()
    except requests.HTTPError as exc:
        detail = exc.response.text if exc.response is not None else str(exc)
        raise HTTPException(status_code=502, detail=f"Backend error: {detail}")
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to synthesize audio: {exc}")

    try:
        result = response.json()
        audio_bytes = base64.b64decode(result["audio"])
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Malformed backend response: {exc}")

    if not audio_bytes:
        raise HTTPException(status_code=500, detail="No audio was generated.")

    headers = {
        "Content-Length": str(len(audio_bytes)),
        "x-openai-model": payload.model,
        "x-openai-voice": voice,
    }
    if "timings" in result and result["timings"] is not None:
        headers["x-miotts-timings"] = json.dumps(result["timings"], ensure_ascii=False)
    if "token_count" in result and result["token_count"] is not None:
        headers["x-miotts-token-count"] = str(result["token_count"])

    return Response(content=audio_bytes, media_type="audio/wav", headers=headers)
