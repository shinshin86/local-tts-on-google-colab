from __future__ import annotations

import logging
import os

import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict

logger = logging.getLogger("uvicorn.error")

BACKEND_URL = os.environ.get("BACKEND_URL", "http://127.0.0.1:5010")
BACKEND_MODEL_ID = os.environ.get("BACKEND_MODEL_ID", "drama3-small")
OPENAI_MODEL_ID = os.environ.get("OPENAI_MODEL_ID", BACKEND_MODEL_ID)
DEFAULT_VOICE = os.environ.get("DRAMA3_DEFAULT_VOICE", "default") or "default"
CLONE_ENABLED = os.environ.get("DRAMA3_CLONE_ENABLED", "0") == "1"
DEFAULT_INSTRUCTIONS = os.environ.get("DRAMA3_INSTRUCTIONS", "")

app = FastAPI(title="drama3-small (audio.cpp) OpenAI Compatible TTS")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[],
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["x-openai-model", "x-openai-voice"],
)


# Request fields forwarded to audio.cpp. Anything that can name a file on the Colab runtime
# (path-type voice_ref, multi_reference_cond, previous_takes, previous_takes_file) is not forwarded,
# because the public trycloudflare URL would let any caller point the server at local files.
PASSTHROUGH_FIELDS = {"seed", "temperature", "top_p", "top_k", "max_tokens", "reference_text"}
PASSTHROUGH_OPTIONS = {"max_pause_frames", "pause_end_ratio", "pause_cap_ratio", "text_chunk_size"}


class AudioSpeechRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str = OPENAI_MODEL_ID
    input: str
    voice: str | None = None
    response_format: str = "wav"
    speed: float = 1.0
    instructions: str | None = None


def available_voices() -> list[str]:
    return ["default", "clone"] if CLONE_ENABLED else ["default"]


def apply_instructions(text: str, instructions: str) -> str:
    """Drama 3 reads an English style direction in square brackets at the start of a line."""
    instructions = instructions.strip()
    if not instructions:
        return text
    if not (instructions.startswith("[") and instructions.endswith("]")):
        instructions = f"[{instructions}]"
    return f"{instructions} {text}"


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
        "engine": "drama3-small",
        "model": OPENAI_MODEL_ID,
        "default_voice": DEFAULT_VOICE,
        "clone_enabled": CLONE_ENABLED,
        "attribution": "Built with Fish Audio",
    }


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [{"id": OPENAI_MODEL_ID, "object": "model", "owned_by": "local"}],
    }


@app.get("/v1/voices")
def list_voices():
    return {"object": "list", "data": [{"id": v, "object": "voice"} for v in available_voices()]}


@app.post("/v1/audio/speech")
def audio_speech(payload: AudioSpeechRequest):
    if payload.response_format.lower() != "wav":
        raise HTTPException(status_code=400, detail="drama3-small supports only response_format 'wav'.")
    if abs(float(payload.speed) - 1.0) > 1e-6:
        raise HTTPException(status_code=400, detail="drama3-small does not support speed; use 1.0.")

    voice = payload.voice or DEFAULT_VOICE
    if voice not in available_voices():
        if voice == "clone":
            raise HTTPException(
                status_code=400,
                detail="voice='clone' requires --drama3-prompt-wav (DRAMA3_PROMPT_WAV) to be set.",
            )
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported voice '{voice}'. Available voices: {', '.join(available_voices())}.",
        )

    extra = payload.model_extra or {}
    instructions = payload.instructions if payload.instructions is not None else DEFAULT_INSTRUCTIONS
    backend_payload = {
        **{k: v for k, v in extra.items() if k in PASSTHROUGH_FIELDS},
        "model": BACKEND_MODEL_ID,
        "input": apply_instructions(payload.input, instructions),
        "response_format": "wav",
    }
    if voice == "clone":
        backend_payload["voice"] = "clone"
    options = extra.get("options")
    if isinstance(options, dict):
        allowed = {k: v for k, v in options.items() if k in PASSTHROUGH_OPTIONS}
        if allowed:
            backend_payload["options"] = allowed
    voice_ref = extra.get("voice_ref")
    if voice_ref is not None:
        # An inline reference clones that voice for this request only (audio.cpp caps it at 5 MiB).
        if not (isinstance(voice_ref, dict) and voice_ref.get("type") == "base64" and voice_ref.get("data")):
            raise HTTPException(
                status_code=400,
                detail='voice_ref must be {"type": "base64", "data": "<WAV base64>"}; server-side paths are not accepted.',
            )
        if not str(extra.get("reference_text") or "").strip():
            raise HTTPException(status_code=400, detail="voice_ref needs reference_text (the transcript of the reference audio).")
        backend_payload["voice_ref"] = {"type": "base64", "data": voice_ref["data"]}
        backend_payload.pop("voice", None)

    try:
        upstream = requests.post(f"{BACKEND_URL}/v1/audio/speech", json=backend_payload, timeout=(10, 900))
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"audio.cpp server is unreachable: {exc}")
    if upstream.status_code != 200:
        try:
            body = upstream.json()
        except ValueError:
            body = {"error": upstream.text}
        return JSONResponse(status_code=upstream.status_code, content=body)

    return Response(
        content=upstream.content,
        media_type="audio/wav",
        headers={"x-openai-model": payload.model, "x-openai-voice": voice},
    )
