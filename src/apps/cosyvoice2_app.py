from __future__ import annotations

import io
import logging
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

COSYVOICE_REPO_DIR = os.environ.get("COSYVOICE_REPO_DIR", "")
if COSYVOICE_REPO_DIR:
    sys.path.insert(0, COSYVOICE_REPO_DIR)
    sys.path.insert(0, str(Path(COSYVOICE_REPO_DIR) / "third_party" / "Matcha-TTS"))

from cosyvoice.cli.cosyvoice import CosyVoice2  # noqa: E402
from cosyvoice.utils.file_utils import load_wav  # noqa: E402

logger = logging.getLogger("uvicorn.error")

OPENAI_MODEL_ID = os.environ.get("OPENAI_MODEL_ID", "cosyvoice2")
MODEL_DIR = os.environ.get("COSYVOICE_MODEL_DIR", "")
PROMPT_WAV = os.environ.get("COSYVOICE_PROMPT_WAV", "")
PROMPT_TEXT = os.environ.get("COSYVOICE_PROMPT_TEXT", "")
DEFAULT_VOICE = os.environ.get("COSYVOICE_DEFAULT_VOICE", "default")

PROMPT_SR = 16000

app = FastAPI(title="CosyVoice2 OpenAI Compatible TTS")

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


_model: CosyVoice2 | None = None
_default_prompt = None


def _bundled_default_prompt_path() -> str:
    return str(Path(COSYVOICE_REPO_DIR) / "asset" / "zero_shot_prompt.wav")


def get_model() -> CosyVoice2:
    global _model
    if _model is None:
        logger.info("Loading CosyVoice2 from %s", MODEL_DIR)
        _model = CosyVoice2(MODEL_DIR, load_jit=False, load_trt=False, fp16=False)
        logger.info("CosyVoice2 loaded (sample_rate=%d)", _model.sample_rate)
    return _model


def get_default_prompt():
    global _default_prompt
    if _default_prompt is None:
        _default_prompt = load_wav(_bundled_default_prompt_path(), PROMPT_SR)
    return _default_prompt


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
        "engine": "CosyVoice2",
        "model": OPENAI_MODEL_ID,
        "model_dir": MODEL_DIR,
        "default_voice": DEFAULT_VOICE,
    }


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [{"id": OPENAI_MODEL_ID, "object": "model", "owned_by": "funaudiollm"}],
    }


@app.get("/v1/voices")
def list_voices():
    voices = [{"id": "default", "object": "voice", "ref": "asset/zero_shot_prompt.wav"}]
    if PROMPT_WAV:
        voices.append({"id": "clone", "object": "voice", "ref": PROMPT_WAV})
    return {"object": "list", "data": voices}


@app.post("/v1/audio/speech")
async def audio_speech(payload: AudioSpeechRequest):
    if payload.response_format.lower() != "wav":
        raise HTTPException(status_code=400, detail="This wrapper currently supports only wav.")

    voice = payload.voice or DEFAULT_VOICE
    model = get_model()

    if voice == "clone":
        if not PROMPT_WAV:
            raise HTTPException(
                status_code=400,
                detail="voice='clone' requires --cosyvoice-prompt-wav at startup.",
            )
        prompt_wav = load_wav(PROMPT_WAV, PROMPT_SR)
        if PROMPT_TEXT:
            it = model.inference_zero_shot(
                payload.input, PROMPT_TEXT, prompt_wav, stream=False, speed=payload.speed
            )
        else:
            it = model.inference_cross_lingual(
                payload.input, prompt_wav, stream=False, speed=payload.speed
            )
    elif voice == "default":
        # Bundled reference is Chinese; cross_lingual handles arbitrary input language.
        it = model.inference_cross_lingual(
            payload.input, get_default_prompt(), stream=False, speed=payload.speed
        )
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown voice: {voice}. Available: default, clone (when prompt configured)",
        )

    chunks = []
    for out in it:
        chunks.append(out["tts_speech"].cpu().numpy().squeeze())
    if not chunks:
        raise HTTPException(status_code=500, detail="No audio was generated.")
    audio = np.concatenate(chunks, axis=-1)

    buf = io.BytesIO()
    sf.write(buf, audio, model.sample_rate, format="WAV", subtype="PCM_16")
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
