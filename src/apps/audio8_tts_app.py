from __future__ import annotations

import io
import logging
import os
from pathlib import Path

import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from transformers import AutoModel, AutoProcessor

logger = logging.getLogger("uvicorn.error")

OPENAI_MODEL_ID = os.environ.get("OPENAI_MODEL_ID", "audio8-tts")
AUDIO8_HF_MODEL = os.environ.get("AUDIO8_HF_MODEL", "Audio8/Audio8-TTS-Preview-0.6b")
AUDIO8_PROMPT_WAV = os.environ.get("AUDIO8_PROMPT_WAV", "")
AUDIO8_PROMPT_TEXT = os.environ.get("AUDIO8_PROMPT_TEXT", "")
AUDIO8_DEFAULT_VOICE = os.environ.get("AUDIO8_DEFAULT_VOICE", "default")
AUDIO8_DEVICE = os.environ.get("AUDIO8_DEVICE", "auto")
AUDIO8_DTYPE = os.environ.get("AUDIO8_DTYPE", "auto")
AUDIO8_MAX_NEW_TOKENS = int(os.environ.get("AUDIO8_MAX_NEW_TOKENS", "1024"))
AUDIO8_TEMPERATURE = float(os.environ.get("AUDIO8_TEMPERATURE", "0.8"))
AUDIO8_TOP_P = float(os.environ.get("AUDIO8_TOP_P", "0.95"))
AUDIO8_TOP_K = int(os.environ.get("AUDIO8_TOP_K", "50"))

app = FastAPI(title="Audio8-TTS OpenAI Compatible TTS")
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


_processor = None
_model = None
_device: torch.device | None = None


def _resolve_device() -> torch.device:
    value = AUDIO8_DEVICE
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


def _resolve_dtype(device: torch.device) -> torch.dtype:
    if AUDIO8_DTYPE == "auto":
        return torch.bfloat16 if device.type == "cuda" else torch.float32
    dtype = getattr(torch, AUDIO8_DTYPE)
    if device.type == "cpu" and dtype != torch.float32:
        raise RuntimeError("CPU inference requires AUDIO8_DTYPE=float32.")
    return dtype


def get_runtime():
    global _processor, _model, _device
    if _model is None:
        _device = _resolve_device()
        dtype = _resolve_dtype(_device)
        logger.info("Loading Audio8 TTS model: %s", AUDIO8_HF_MODEL)
        _processor = AutoProcessor.from_pretrained(AUDIO8_HF_MODEL, trust_remote_code=True)
        _model = AutoModel.from_pretrained(
            AUDIO8_HF_MODEL,
            trust_remote_code=True,
            dtype=dtype,
        ).eval().to(_device)
        logger.info("Audio8 TTS model loaded on %s with %s", _device, dtype)
    return _processor, _model, _device


def clone_is_configured() -> bool:
    return bool(AUDIO8_PROMPT_WAV and AUDIO8_PROMPT_TEXT)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception while serving request")
    return JSONResponse(status_code=500, content={"error": type(exc).__name__, "detail": str(exc)})


@app.get("/")
def root():
    return {"ok": True, "engine": "Audio8-TTS", "model": OPENAI_MODEL_ID}


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [{"id": OPENAI_MODEL_ID, "object": "model", "owned_by": "Audio8"}],
    }


@app.get("/v1/voices")
def list_voices():
    voices = ["default"]
    if clone_is_configured():
        voices.append("clone")
    return {"object": "list", "data": [{"id": voice, "object": "voice"} for voice in voices]}


@app.post("/v1/audio/speech")
async def audio_speech(payload: AudioSpeechRequest):
    if payload.response_format.lower() != "wav":
        raise HTTPException(status_code=400, detail="This wrapper currently supports only wav.")
    voice = payload.voice or AUDIO8_DEFAULT_VOICE
    if voice not in {"default", "clone"}:
        raise HTTPException(status_code=400, detail="voice must be 'default' or 'clone'.")
    if voice == "clone" and not clone_is_configured():
        raise HTTPException(
            status_code=400,
            detail="voice='clone' requires both --audio8-prompt-wav and --audio8-prompt-text.",
        )
    if voice == "clone" and not Path(AUDIO8_PROMPT_WAV).is_file():
        raise HTTPException(status_code=400, detail="The configured Audio8 reference audio does not exist.")

    processor, model, device = get_runtime()
    processor_kwargs = {"text": [payload.input], "return_tensors": "pt"}
    if voice == "clone":
        processor_kwargs.update(
            reference_audio=[AUDIO8_PROMPT_WAV],
            reference_text=[AUDIO8_PROMPT_TEXT],
        )
    inputs = processor(**processor_kwargs)
    inputs = {name: value.to(device) for name, value in inputs.items()}
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=AUDIO8_MAX_NEW_TOKENS,
            temperature=AUDIO8_TEMPERATURE,
            top_p=AUDIO8_TOP_P,
            top_k=AUDIO8_TOP_K,
            do_sample=True,
            return_dict_in_generate=True,
        )
        waveforms, waveform_lengths = model.decode_audio(output.codes)

    length = int(waveform_lengths[0])
    audio = waveforms[0, :length].float().cpu().numpy()
    buffer = io.BytesIO()
    sf.write(buffer, audio, int(model.config.codec_sample_rate), format="WAV")
    audio_bytes = buffer.getvalue()
    return Response(
        content=audio_bytes,
        media_type="audio/wav",
        headers={
            "Content-Length": str(len(audio_bytes)),
            "x-openai-model": payload.model,
            "x-openai-voice": voice,
        },
    )
