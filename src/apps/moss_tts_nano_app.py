from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from transformers import AutoConfig, AutoModelForCausalLM

logger = logging.getLogger("uvicorn.error")

OPENAI_MODEL_ID = os.environ.get("OPENAI_MODEL_ID", "moss-tts-nano")
MOSS_HF_MODEL = os.environ.get("MOSS_TTS_NANO_HF_MODEL", "OpenMOSS-Team/MOSS-TTS-Nano-100M")
MOSS_MODE = os.environ.get("MOSS_TTS_NANO_MODE", "continuation")
MOSS_ATTN_IMPL = os.environ.get("MOSS_TTS_NANO_ATTN_IMPL", "sdpa")

app = FastAPI(title="MOSS-TTS-Nano OpenAI Compatible TTS")

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


_model = None
_device: str = "cpu"
_dtype = torch.float32


def get_model():
    global _model, _device, _dtype
    if _model is None:
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        _dtype = torch.bfloat16 if _device == "cuda" else torch.float32
        logger.info("Loading MOSS-TTS-Nano model: %s (device=%s, dtype=%s, attn=%s)", MOSS_HF_MODEL, _device, _dtype, MOSS_ATTN_IMPL)
        cfg = AutoConfig.from_pretrained(MOSS_HF_MODEL, trust_remote_code=True)
        cfg.attn_implementation = MOSS_ATTN_IMPL
        if hasattr(cfg, "local_transformer_attn_implementation"):
            cfg.local_transformer_attn_implementation = MOSS_ATTN_IMPL
        gpt2_cfg = getattr(cfg, "gpt2_config", None)
        if isinstance(gpt2_cfg, dict):
            gpt2_cfg["attn_implementation"] = MOSS_ATTN_IMPL
            gpt2_cfg["_attn_implementation"] = MOSS_ATTN_IMPL
        elif gpt2_cfg is not None:
            try:
                gpt2_cfg._attn_implementation = MOSS_ATTN_IMPL
                gpt2_cfg.attn_implementation = MOSS_ATTN_IMPL
            except Exception:
                pass
        model = AutoModelForCausalLM.from_pretrained(MOSS_HF_MODEL, config=cfg, trust_remote_code=True)
        if hasattr(model, "set_attention_implementation"):
            try:
                model.set_attention_implementation(MOSS_ATTN_IMPL, local_attn_implementation=MOSS_ATTN_IMPL)
            except Exception:
                pass
        model.to(device=_device, dtype=_dtype)
        model.eval()
        _model = model
        logger.info("MOSS-TTS-Nano model loaded")
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
    return {"ok": True, "engine": "MOSS-TTS-Nano", "model": OPENAI_MODEL_ID}


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [{"id": OPENAI_MODEL_ID, "object": "model", "owned_by": "openmoss"}],
    }


@app.get("/v1/voices")
def list_voices():
    return {
        "object": "list",
        "data": [{"id": "default", "object": "voice"}],
    }


@app.post("/v1/audio/speech")
async def audio_speech(payload: AudioSpeechRequest):
    if payload.response_format.lower() != "wav":
        raise HTTPException(status_code=400, detail="This wrapper currently supports only wav.")

    model = get_model()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        model.inference(
            text=payload.input,
            output_audio_path=tmp_path,
            mode=MOSS_MODE,
            prompt_audio_path=None,
        )
        audio_bytes = Path(tmp_path).read_bytes()
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    if not audio_bytes:
        raise HTTPException(status_code=500, detail="No audio was generated.")

    voice = payload.voice or "default"
    return Response(
        content=audio_bytes,
        media_type="audio/wav",
        headers={
            "Content-Length": str(len(audio_bytes)),
            "x-openai-model": payload.model,
            "x-openai-voice": voice,
        },
    )
