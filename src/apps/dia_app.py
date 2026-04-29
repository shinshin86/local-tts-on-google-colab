from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

import torch
from dia.model import Dia
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logger = logging.getLogger("uvicorn.error")

OPENAI_MODEL_ID = os.environ.get("OPENAI_MODEL_ID", "dia")
DIA_HF_MODEL = os.environ.get("DIA_HF_MODEL", "nari-labs/Dia-1.6B-0626")
DIA_COMPUTE_DTYPE = os.environ.get("DIA_COMPUTE_DTYPE", "float16")
DIA_PROMPT_WAV = os.environ.get("DIA_PROMPT_WAV", "")
DIA_PROMPT_TEXT = os.environ.get("DIA_PROMPT_TEXT", "")
DIA_DEFAULT_VOICE = os.environ.get("DIA_DEFAULT_VOICE", "default")

app = FastAPI(title="Dia OpenAI Compatible TTS")

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


_model: Dia | None = None


def get_model() -> Dia:
    global _model
    if _model is None:
        compute_dtype = DIA_COMPUTE_DTYPE
        if compute_dtype == "float16" and not torch.cuda.is_available():
            # float16 on CPU is unsupported by most ops; fall back to float32.
            compute_dtype = "float32"
        logger.info("Loading Dia model %s (compute_dtype=%s)", DIA_HF_MODEL, compute_dtype)
        _model = Dia.from_pretrained(DIA_HF_MODEL, compute_dtype=compute_dtype)
        logger.info("Dia model loaded")
    return _model


def _ensure_speaker_tags(text: str) -> str:
    # Dia expects speaker tags like [S1] / [S2] in the input. If the user passes
    # plain text, prepend [S1] so single-speaker TTS still works.
    stripped = text.lstrip()
    if stripped.startswith("[S1]") or stripped.startswith("[S2]"):
        return text
    return f"[S1] {text}"


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
        "engine": "Dia",
        "model": OPENAI_MODEL_ID,
        "hf_model": DIA_HF_MODEL,
        "default_voice": DIA_DEFAULT_VOICE,
    }


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [{"id": OPENAI_MODEL_ID, "object": "model", "owned_by": "nari-labs"}],
    }


@app.get("/v1/voices")
def list_voices():
    voices = [{"id": "default", "object": "voice", "language": "en"}]
    if DIA_PROMPT_WAV and DIA_PROMPT_TEXT:
        voices.append({"id": "clone", "object": "voice", "language": "en"})
    return {"object": "list", "data": voices}


@app.post("/v1/audio/speech")
async def audio_speech(payload: AudioSpeechRequest):
    if payload.response_format.lower() != "wav":
        raise HTTPException(status_code=400, detail="This wrapper currently supports only wav.")

    voice = payload.voice or DIA_DEFAULT_VOICE
    if voice not in {"default", "clone"}:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown voice '{voice}'. Use 'default' or 'clone'.",
        )
    if voice == "clone" and not (DIA_PROMPT_WAV and DIA_PROMPT_TEXT):
        raise HTTPException(
            status_code=400,
            detail="voice='clone' requires both --dia-prompt-wav and --dia-prompt-text at startup.",
        )

    model = get_model()

    if voice == "clone":
        # Per Dia docs, prepend the prompt transcript so the model conditions on it.
        clone_text = _ensure_speaker_tags(DIA_PROMPT_TEXT)
        new_text = _ensure_speaker_tags(payload.input)
        text = f"{clone_text} {new_text}"
        output = model.generate(text, audio_prompt=DIA_PROMPT_WAV, use_torch_compile=False)
    else:
        text = _ensure_speaker_tags(payload.input)
        output = model.generate(text, use_torch_compile=False)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        model.save_audio(tmp_path, output)
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
