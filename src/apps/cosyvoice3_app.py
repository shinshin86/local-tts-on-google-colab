from __future__ import annotations

import io
import logging
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

COSYVOICE_REPO_DIR = os.environ.get("COSYVOICE3_REPO_DIR", "")
if COSYVOICE_REPO_DIR:
    sys.path.insert(0, COSYVOICE_REPO_DIR)
    sys.path.insert(0, str(Path(COSYVOICE_REPO_DIR) / "third_party" / "Matcha-TTS"))

from cosyvoice.cli.cosyvoice import AutoModel  # noqa: E402

logger = logging.getLogger("uvicorn.error")

OPENAI_MODEL_ID = os.environ.get("OPENAI_MODEL_ID", "cosyvoice3")
MODEL_DIR = os.environ.get("COSYVOICE3_MODEL_DIR", "")
PROMPT_WAV = os.environ.get("COSYVOICE3_PROMPT_WAV", "")
PROMPT_TEXT = os.environ.get("COSYVOICE3_PROMPT_TEXT", "")
INSTRUCT = os.environ.get("COSYVOICE3_INSTRUCT", "")
DEFAULT_VOICE = os.environ.get("COSYVOICE3_DEFAULT_VOICE", "default")

SYSTEM_PROMPT = "You are a helpful assistant."

app = FastAPI(title="CosyVoice3 OpenAI Compatible TTS")

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


_model: Any | None = None


def _bundled_default_prompt_path() -> str:
    return str(Path(COSYVOICE_REPO_DIR) / "asset" / "zero_shot_prompt.wav")


def _with_system_prompt(text: str) -> str:
    return f"{SYSTEM_PROMPT}<|endofprompt|>{text}"


def _instruction_prompt(instruct: str) -> str:
    return f"{SYSTEM_PROMPT} {instruct.strip()}<|endofprompt|>"


def get_model() -> Any:
    global _model
    if _model is None:
        logger.info("Loading CosyVoice3 from %s", MODEL_DIR)
        _model = AutoModel(
            model_dir=MODEL_DIR,
            load_trt=False,
            load_vllm=False,
            fp16=False,
        )
        logger.info("CosyVoice3 loaded (sample_rate=%d)", _model.sample_rate)
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
        "engine": "CosyVoice3",
        "model": OPENAI_MODEL_ID,
        "model_dir": MODEL_DIR,
        "default_voice": DEFAULT_VOICE,
        "instruction_enabled": bool(INSTRUCT),
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
    if not 0.25 <= payload.speed <= 4.0:
        raise HTTPException(status_code=400, detail="speed must be between 0.25 and 4.0.")

    voice = payload.voice or DEFAULT_VOICE
    if voice == "clone":
        if not PROMPT_WAV:
            raise HTTPException(
                status_code=400,
                detail="voice='clone' requires --cosyvoice3-prompt-wav at startup.",
            )
        prompt_path = PROMPT_WAV
    elif voice == "default":
        prompt_path = _bundled_default_prompt_path()
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown voice: {voice}. Available: default, clone (when prompt configured)",
        )

    model = get_model()
    if INSTRUCT:
        it = model.inference_instruct2(
            payload.input,
            _instruction_prompt(INSTRUCT),
            prompt_path,
            stream=False,
            speed=payload.speed,
        )
    elif voice == "clone" and PROMPT_TEXT:
        it = model.inference_zero_shot(
            _with_system_prompt(payload.input),
            _with_system_prompt(PROMPT_TEXT),
            prompt_path,
            stream=False,
            speed=payload.speed,
        )
    else:
        it = model.inference_cross_lingual(
            _with_system_prompt(payload.input),
            prompt_path,
            stream=False,
            speed=payload.speed,
        )

    chunks = [out["tts_speech"].cpu().numpy().squeeze() for out in it]
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
