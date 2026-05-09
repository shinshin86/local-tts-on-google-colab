from __future__ import annotations

import io
import logging
import os

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# PyTorch 2.6 flipped `torch.load`'s default to `weights_only=True`. ChatTTS still
# ships pickled .pt checkpoints, so loading fails unless we restore the legacy
# behavior. Patch before importing ChatTTS, which calls `torch.load` during
# `Chat.load()`.
_orig_torch_load = torch.load


def _chattts_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)


torch.load = _chattts_torch_load

import ChatTTS  # noqa: E402, N814

logger = logging.getLogger("uvicorn.error")

OPENAI_MODEL_ID = os.environ.get("OPENAI_MODEL_ID", "chattts")
DEFAULT_VOICE = os.environ.get("CHATTTS_DEFAULT_VOICE", "default")
DEFAULT_SEED = int(os.environ.get("CHATTTS_SEED", "2"))
DEFAULT_TEMPERATURE = float(os.environ.get("CHATTTS_TEMPERATURE", "0.3"))

# ChatTTS samples a speaker embedding per request. We expose two modes:
# - `default`: deterministic sampling from CHATTTS_SEED for reproducibility
# - `random`:  fresh random speaker on every request (chat.sample_random_speaker())
VOICE_PRESETS = ["default", "random"]

app = FastAPI(title="ChatTTS OpenAI Compatible TTS")

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


_chat: ChatTTS.Chat | None = None
_default_speaker: str | None = None


def get_chat() -> ChatTTS.Chat:
    global _chat
    if _chat is None:
        logger.info("Loading ChatTTS")
        _chat = ChatTTS.Chat()
        _chat.load(compile=False)
        logger.info("ChatTTS loaded")
    return _chat


def get_default_speaker() -> str:
    global _default_speaker
    if _default_speaker is None:
        chat = get_chat()
        torch.manual_seed(DEFAULT_SEED)
        _default_speaker = chat.sample_random_speaker()
    return _default_speaker


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
        "engine": "ChatTTS",
        "model": OPENAI_MODEL_ID,
        "default_voice": DEFAULT_VOICE,
        "seed": DEFAULT_SEED,
    }


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [{"id": OPENAI_MODEL_ID, "object": "model", "owned_by": "2noise"}],
    }


@app.get("/v1/voices")
def list_voices():
    return {
        "object": "list",
        "data": [{"id": voice, "object": "voice"} for voice in VOICE_PRESETS],
    }


@app.post("/v1/audio/speech")
async def audio_speech(payload: AudioSpeechRequest):
    if payload.response_format.lower() != "wav":
        raise HTTPException(status_code=400, detail="This wrapper currently supports only wav.")

    voice = payload.voice or DEFAULT_VOICE
    chat = get_chat()

    if voice == "default":
        speaker = get_default_speaker()
    elif voice == "random":
        speaker = chat.sample_random_speaker()
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown voice: {voice}. Available: {VOICE_PRESETS}",
        )

    params_infer_code = ChatTTS.Chat.InferCodeParams(
        spk_emb=speaker,
        temperature=DEFAULT_TEMPERATURE,
    )
    wavs = chat.infer([payload.input], params_infer_code=params_infer_code)
    if not wavs:
        raise HTTPException(status_code=500, detail="No audio was generated.")

    audio = np.asarray(wavs[0], dtype=np.float32).squeeze()
    if audio.size == 0:
        raise HTTPException(status_code=500, detail="No audio samples were produced.")

    sample_rate = 24000

    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, format="WAV", subtype="PCM_16")
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
