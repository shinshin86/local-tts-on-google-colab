from __future__ import annotations

import io
import logging
import os
import wave
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sarashina_tts.generate.generate import SarashinaTTSGenerator

logger = logging.getLogger("uvicorn.error")

OPENAI_MODEL_ID = os.environ.get("OPENAI_MODEL_ID", "sarashina-tts")
SARASHINA_HF_MODEL = os.environ.get("SARASHINA_HF_MODEL", "sbintuitions/sarashina2.2-tts")
SARASHINA_USE_VLLM = os.environ.get("SARASHINA_USE_VLLM", "0") == "1"
SARASHINA_PROMPT_WAV = os.environ.get("SARASHINA_PROMPT_WAV", "").strip()
SARASHINA_PROMPT_TEXT = os.environ.get("SARASHINA_PROMPT_TEXT", "").strip()
SARASHINA_DEFAULT_VOICE = os.environ.get("SARASHINA_DEFAULT_VOICE", "default").strip() or "default"
SARASHINA_MODEL_DIR = os.environ.get("SARASHINA_MODEL_DIR", "").strip()

SAMPLE_RATE = 24000

# Voice presets:
#   - "default": plain TTS without any voice prompt (no zero-shot cloning).
#   - "clone"  : zero-shot voice cloning, only available when both
#                SARASHINA_PROMPT_WAV and SARASHINA_PROMPT_TEXT are configured.
VOICE_DEFAULT = "default"
VOICE_CLONE = "clone"

app = FastAPI(title="Sarashina-TTS OpenAI Compatible TTS")

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


_generator: SarashinaTTSGenerator | None = None
_clone_cache: tuple[Any, Any, Any] | None = None


def _has_clone_voice() -> bool:
    return bool(SARASHINA_PROMPT_WAV) and bool(SARASHINA_PROMPT_TEXT)


def get_generator() -> SarashinaTTSGenerator:
    global _generator
    if _generator is None:
        kwargs: dict[str, Any] = {
            "model_id": SARASHINA_HF_MODEL,
            "use_vllm": SARASHINA_USE_VLLM,
        }
        if SARASHINA_MODEL_DIR:
            kwargs["model_dir"] = SARASHINA_MODEL_DIR
        logger.info(
            "Loading Sarashina-TTS: model_id=%s use_vllm=%s",
            SARASHINA_HF_MODEL,
            SARASHINA_USE_VLLM,
        )
        _generator = SarashinaTTSGenerator(**kwargs)
        logger.info("Sarashina-TTS generator loaded")
    return _generator


def get_clone_features(generator: SarashinaTTSGenerator) -> tuple[Any, Any, Any]:
    global _clone_cache
    if _clone_cache is None:
        wav_path = Path(SARASHINA_PROMPT_WAV)
        if not wav_path.is_file():
            raise HTTPException(
                status_code=500,
                detail=f"Configured SARASHINA_PROMPT_WAV not found: {wav_path}",
            )
        logger.info("Encoding voice clone reference from %s", wav_path)
        flow_embedding = generator._extract_zero_shot_embedding(str(wav_path))
        prompt_tokens = generator._extract_audio_prompt_tokens(str(wav_path))
        prompt_feat = generator._extract_audio_prompt_feat(str(wav_path))
        _clone_cache = (flow_embedding, prompt_tokens, prompt_feat)
    return _clone_cache


def _to_wav_bytes(audio: torch.Tensor) -> bytes:
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)
    elif audio.ndim != 2:
        raise HTTPException(status_code=500, detail=f"Unexpected audio tensor shape: {tuple(audio.shape)}")

    audio = torch.clamp(audio.detach().to("cpu", torch.float32), -1.0, 1.0)
    audio_i16 = (audio * 32767.0).round().to(torch.int16)

    num_channels, _ = audio_i16.shape
    interleaved = audio_i16.transpose(0, 1).contiguous().numpy().tobytes()

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(num_channels)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(interleaved)
    return buf.getvalue()


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
        "engine": "Sarashina-TTS",
        "model": OPENAI_MODEL_ID,
        "hf_model": SARASHINA_HF_MODEL,
        "default_voice": SARASHINA_DEFAULT_VOICE,
        "clone_voice_available": _has_clone_voice(),
    }


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [{"id": OPENAI_MODEL_ID, "object": "model", "owned_by": "sbintuitions"}],
    }


@app.get("/v1/voices")
def list_voices():
    voices = [{"id": VOICE_DEFAULT, "object": "voice", "language": "ja"}]
    if _has_clone_voice():
        voices.append({"id": VOICE_CLONE, "object": "voice", "language": "ja"})
    return {"object": "list", "data": voices}


@app.post("/v1/audio/speech")
async def audio_speech(payload: AudioSpeechRequest):
    if payload.response_format.lower() != "wav":
        raise HTTPException(status_code=400, detail="This wrapper currently supports only wav.")

    voice = (payload.voice or SARASHINA_DEFAULT_VOICE).strip() or VOICE_DEFAULT
    if voice not in {VOICE_DEFAULT, VOICE_CLONE}:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown voice '{voice}'. Available: ['default'] + (['clone'] if configured)",
        )
    if voice == VOICE_CLONE and not _has_clone_voice():
        raise HTTPException(
            status_code=400,
            detail="Voice 'clone' requested but SARASHINA_PROMPT_WAV / SARASHINA_PROMPT_TEXT are not configured.",
        )

    generator = get_generator()

    decode_kwargs = {"speed": payload.speed} if payload.speed and payload.speed > 0 else {}

    if voice == VOICE_CLONE:
        flow_embedding, prompt_tokens, prompt_feat = get_clone_features(generator)
        wavs = generator.generate(
            texts=[payload.input],
            flow_embedding=flow_embedding,
            audio_prompt_text=SARASHINA_PROMPT_TEXT,
            audio_prompt_tokens=prompt_tokens,
            audio_prompt_feat=prompt_feat,
            audio_prompt_path=SARASHINA_PROMPT_WAV,
            decode_kwargs=decode_kwargs or None,
        )
    else:
        wavs = generator.generate(
            texts=[payload.input],
            flow_embedding=None,
            decode_kwargs=decode_kwargs or None,
        )

    if not wavs:
        raise HTTPException(status_code=500, detail="No audio was generated.")

    audio = wavs[0]
    if not torch.is_tensor(audio) or audio.numel() == 0:
        raise HTTPException(status_code=500, detail="Generated audio is empty.")

    audio_bytes = _to_wav_bytes(audio)

    return Response(
        content=audio_bytes,
        media_type="audio/wav",
        headers={
            "Content-Length": str(len(audio_bytes)),
            "x-openai-model": payload.model,
            "x-openai-voice": voice,
        },
    )
