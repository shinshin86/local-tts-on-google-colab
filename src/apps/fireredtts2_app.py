from __future__ import annotations

import io
import logging
import os
import re
from typing import Any

import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fireredtts2.fireredtts2 import FireRedTTS2
from pydantic import BaseModel

logger = logging.getLogger("uvicorn.error")

OPENAI_MODEL_ID = os.environ.get("OPENAI_MODEL_ID", "fireredtts2")
FIREREDTTS2_HF_MODEL = os.environ.get("FIREREDTTS2_HF_MODEL", "FireRedTeam/FireRedTTS2")
FIREREDTTS2_MODEL_DIR = os.environ.get("FIREREDTTS2_MODEL_DIR", "")
FIREREDTTS2_GENERATION_MODE = os.environ.get("FIREREDTTS2_GENERATION_MODE", "monologue")
FIREREDTTS2_DEFAULT_VOICE = os.environ.get("FIREREDTTS2_DEFAULT_VOICE", "random")
FIREREDTTS2_PROMPT_WAV = os.environ.get("FIREREDTTS2_PROMPT_WAV", "")
FIREREDTTS2_PROMPT_TEXT = os.environ.get("FIREREDTTS2_PROMPT_TEXT", "")
FIREREDTTS2_TEMPERATURE = float(os.environ.get("FIREREDTTS2_TEMPERATURE", "0.75"))
FIREREDTTS2_TOPK = int(os.environ.get("FIREREDTTS2_TOPK", "20"))
FIREREDTTS2_USE_BF16 = os.environ.get("FIREREDTTS2_USE_BF16", "1") == "1"
SAMPLE_RATE = 24000

app = FastAPI(title="FireRedTTS2 OpenAI Compatible TTS")
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


def get_model() -> Any:
    global _model
    if _model is None:
        if not torch.cuda.is_available():
            raise RuntimeError("FireRedTTS2 requires a CUDA GPU in this wrapper.")
        logger.info(
            "Loading FireRedTTS2 from %s (mode=%s, bf16=%s)",
            FIREREDTTS2_MODEL_DIR,
            FIREREDTTS2_GENERATION_MODE,
            FIREREDTTS2_USE_BF16,
        )
        _model = FireRedTTS2(
            pretrained_dir=FIREREDTTS2_MODEL_DIR,
            gen_type=FIREREDTTS2_GENERATION_MODE,
            device="cuda",
            use_bf16=FIREREDTTS2_USE_BF16,
        )
        logger.info("FireRedTTS2 ready")
    return _model


def _available_voices() -> list[str]:
    voices = ["random"]
    if (
        FIREREDTTS2_GENERATION_MODE == "monologue"
        and FIREREDTTS2_PROMPT_WAV
        and FIREREDTTS2_PROMPT_TEXT
    ):
        voices.append("clone")
    return voices


def _resolve_voice(voice: str | None) -> str:
    requested = FIREREDTTS2_DEFAULT_VOICE if not voice or voice == "default" else voice
    if requested not in _available_voices():
        raise HTTPException(
            status_code=400,
            detail=f"Unknown or unavailable voice '{requested}'. Available: {', '.join(_available_voices())}",
        )
    return requested


def _dialogue_segments(text: str) -> list[str]:
    segments = [part.strip() for part in re.findall(r"(\[S[1-4]\][^\[\]]+)", text)]
    if not segments:
        segments = [f"[S1]{text.strip()}"]
    return segments


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception while serving request")
    return JSONResponse(status_code=500, content={"error": type(exc).__name__, "detail": str(exc)})


@app.get("/")
def root():
    return {
        "ok": True,
        "engine": "FireRedTTS2",
        "model": OPENAI_MODEL_ID,
        "hf_model": FIREREDTTS2_HF_MODEL,
        "generation_mode": FIREREDTTS2_GENERATION_MODE,
        "default_voice": FIREREDTTS2_DEFAULT_VOICE,
        "voices": _available_voices(),
    }


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [{"id": OPENAI_MODEL_ID, "object": "model", "owned_by": "fireredteam"}],
    }


@app.get("/v1/voices")
def list_voices():
    descriptions = {
        "random": "Random speaker; dialogue mode assigns timbres by [S1]...[S4].",
        "clone": "Zero-shot clone from the configured reference audio and transcript.",
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
        raise HTTPException(status_code=400, detail="FireRedTTS2 does not expose speed control.")

    voice = _resolve_voice(payload.voice)
    model = get_model()
    if FIREREDTTS2_GENERATION_MODE == "dialogue":
        audio = model.generate_dialogue(
            text_list=_dialogue_segments(payload.input),
            temperature=FIREREDTTS2_TEMPERATURE,
            topk=FIREREDTTS2_TOPK,
        )
    else:
        audio = model.generate_monologue(
            text=payload.input,
            prompt_wav=FIREREDTTS2_PROMPT_WAV if voice == "clone" else None,
            prompt_text=FIREREDTTS2_PROMPT_TEXT if voice == "clone" else None,
            temperature=FIREREDTTS2_TEMPERATURE,
            topk=FIREREDTTS2_TOPK,
        )

    waveform = audio.detach().float().cpu().numpy().squeeze()
    if waveform.size == 0:
        raise HTTPException(status_code=500, detail="No audio was generated.")
    buf = io.BytesIO()
    sf.write(buf, waveform, SAMPLE_RATE, format="WAV", subtype="PCM_16")
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
