from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

import outetts
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logger = logging.getLogger("uvicorn.error")

OPENAI_MODEL_ID = os.environ.get("OPENAI_MODEL_ID", "outetts")
OUTETTS_MODEL_SIZE = os.environ.get("OUTETTS_MODEL_SIZE", "0.6B")
OUTETTS_BACKEND = os.environ.get("OUTETTS_BACKEND", "HF")
OUTETTS_DEFAULT_SPEAKER = os.environ.get("OUTETTS_DEFAULT_SPEAKER", "EN-FEMALE-1-NEUTRAL")
OUTETTS_PROMPT_WAV = os.environ.get("OUTETTS_PROMPT_WAV", "")
OUTETTS_PROMPT_TEXT = os.environ.get("OUTETTS_PROMPT_TEXT", "")
OUTETTS_DEFAULT_VOICE = os.environ.get("OUTETTS_DEFAULT_VOICE", "default")

app = FastAPI(title="OuteTTS OpenAI Compatible TTS")

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


_interface: outetts.Interface | None = None
_speaker_cache: dict[str, object] = {}


def _resolve_model_enum() -> object:
    size = OUTETTS_MODEL_SIZE.upper().replace(".", "_")
    name = f"VERSION_1_0_SIZE_{size}"
    if not hasattr(outetts.Models, name):
        raise RuntimeError(
            f"Unknown OuteTTS model size '{OUTETTS_MODEL_SIZE}'. Expected e.g. '0.6B' or '1B'."
        )
    return getattr(outetts.Models, name)


def _resolve_backend_enum() -> object:
    name = OUTETTS_BACKEND.upper()
    if not hasattr(outetts.Backend, name):
        raise RuntimeError(
            f"Unknown OuteTTS backend '{OUTETTS_BACKEND}'. Expected e.g. 'HF' or 'LLAMACPP'."
        )
    return getattr(outetts.Backend, name)


def get_interface() -> outetts.Interface:
    global _interface
    if _interface is None:
        model_enum = _resolve_model_enum()
        backend_enum = _resolve_backend_enum()
        logger.info("Loading OuteTTS: model=%s backend=%s", model_enum, backend_enum)
        config = outetts.ModelConfig.auto_config(model=model_enum, backend=backend_enum)
        _interface = outetts.Interface(config=config)
        logger.info("OuteTTS interface ready")
    return _interface


def get_speaker(voice: str):
    if voice in _speaker_cache:
        return _speaker_cache[voice]
    interface = get_interface()
    if voice == "clone":
        if not OUTETTS_PROMPT_WAV:
            raise HTTPException(
                status_code=400,
                detail="voice='clone' requires --outetts-prompt-wav at startup.",
            )
        kwargs = {"audio_path": OUTETTS_PROMPT_WAV}
        if OUTETTS_PROMPT_TEXT:
            kwargs["transcript"] = OUTETTS_PROMPT_TEXT
        speaker = interface.create_speaker(**kwargs)
    else:
        speaker = interface.load_default_speaker(OUTETTS_DEFAULT_SPEAKER)
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
        "engine": "OuteTTS",
        "model": OPENAI_MODEL_ID,
        "size": OUTETTS_MODEL_SIZE,
        "backend": OUTETTS_BACKEND,
        "default_speaker": OUTETTS_DEFAULT_SPEAKER,
        "default_voice": OUTETTS_DEFAULT_VOICE,
    }


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [{"id": OPENAI_MODEL_ID, "object": "model", "owned_by": "outeai"}],
    }


@app.get("/v1/voices")
def list_voices():
    voices = [{"id": "default", "object": "voice", "speaker": OUTETTS_DEFAULT_SPEAKER}]
    if OUTETTS_PROMPT_WAV:
        voices.append({"id": "clone", "object": "voice"})
    return {"object": "list", "data": voices}


@app.post("/v1/audio/speech")
async def audio_speech(payload: AudioSpeechRequest):
    if payload.response_format.lower() != "wav":
        raise HTTPException(status_code=400, detail="This wrapper currently supports only wav.")

    voice = payload.voice or OUTETTS_DEFAULT_VOICE
    if voice not in {"default", "clone"}:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown voice '{voice}'. Use 'default' or 'clone'.",
        )

    interface = get_interface()
    speaker = get_speaker(voice)

    output = interface.generate(
        config=outetts.GenerationConfig(
            text=payload.input,
            generation_type=outetts.GenerationType.CHUNKED,
            speaker=speaker,
        )
    )

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        output.save(tmp_path)
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
