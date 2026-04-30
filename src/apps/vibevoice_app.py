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
from vibevoice.modular.modeling_vibevoice import VibeVoiceForConditionalGeneration
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor

logger = logging.getLogger("uvicorn.error")

OPENAI_MODEL_ID = os.environ.get("OPENAI_MODEL_ID", "vibevoice")
VIBEVOICE_HF_MODEL = os.environ.get("VIBEVOICE_HF_MODEL", "microsoft/VibeVoice-1.5B")
VIBEVOICE_VOICES_DIR = Path(os.environ.get("VIBEVOICE_VOICES_DIR", "demo/voices"))
VIBEVOICE_DEFAULT_SPEAKER = os.environ.get("VIBEVOICE_DEFAULT_SPEAKER", "en-Alice_woman")
VIBEVOICE_PROMPT_WAV = os.environ.get("VIBEVOICE_PROMPT_WAV", "")
VIBEVOICE_DEFAULT_VOICE = os.environ.get("VIBEVOICE_DEFAULT_VOICE", "default")
VIBEVOICE_DDPM_STEPS = int(os.environ.get("VIBEVOICE_DDPM_STEPS", "10"))
VIBEVOICE_CFG_SCALE = float(os.environ.get("VIBEVOICE_CFG_SCALE", "1.3"))

app = FastAPI(title="VibeVoice OpenAI Compatible TTS")

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


_processor: VibeVoiceProcessor | None = None
_model: VibeVoiceForConditionalGeneration | None = None


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_model() -> tuple[VibeVoiceProcessor, VibeVoiceForConditionalGeneration]:
    global _processor, _model
    if _processor is None or _model is None:
        device = _device()
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        logger.info("Loading VibeVoice processor + model: %s on %s (%s)", VIBEVOICE_HF_MODEL, device, dtype)
        _processor = VibeVoiceProcessor.from_pretrained(VIBEVOICE_HF_MODEL)
        _model = VibeVoiceForConditionalGeneration.from_pretrained(
            VIBEVOICE_HF_MODEL,
            torch_dtype=dtype,
            device_map=device,
        )
        _model.set_ddpm_inference_steps(num_steps=VIBEVOICE_DDPM_STEPS)
        logger.info("VibeVoice ready (ddpm_steps=%s, cfg_scale=%s)", VIBEVOICE_DDPM_STEPS, VIBEVOICE_CFG_SCALE)
    return _processor, _model


def _resolve_speaker_path(voice: str) -> Path:
    if voice == "clone":
        if not VIBEVOICE_PROMPT_WAV:
            raise HTTPException(
                status_code=400,
                detail="voice='clone' requires --vibevoice-prompt-wav at startup.",
            )
        return Path(VIBEVOICE_PROMPT_WAV)
    candidate = VIBEVOICE_VOICES_DIR / f"{VIBEVOICE_DEFAULT_SPEAKER}.wav"
    if not candidate.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Bundled speaker file not found: {candidate}",
        )
    return candidate


def _format_text(text: str) -> str:
    # VibeVoice expects "Speaker N: ..." per line. If the user did not provide a
    # speaker prefix, prepend "Speaker 1:" so single-speaker TTS still works.
    stripped = text.lstrip()
    if stripped.lower().startswith("speaker "):
        return text
    return f"Speaker 1: {text}"


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
        "engine": "VibeVoice",
        "model": OPENAI_MODEL_ID,
        "hf_model": VIBEVOICE_HF_MODEL,
        "default_speaker": VIBEVOICE_DEFAULT_SPEAKER,
        "default_voice": VIBEVOICE_DEFAULT_VOICE,
    }


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [{"id": OPENAI_MODEL_ID, "object": "model", "owned_by": "microsoft"}],
    }


@app.get("/v1/voices")
def list_voices():
    voices = [{"id": "default", "object": "voice", "speaker": VIBEVOICE_DEFAULT_SPEAKER}]
    if VIBEVOICE_PROMPT_WAV:
        voices.append({"id": "clone", "object": "voice"})
    return {"object": "list", "data": voices}


@app.post("/v1/audio/speech")
async def audio_speech(payload: AudioSpeechRequest):
    if payload.response_format.lower() != "wav":
        raise HTTPException(status_code=400, detail="This wrapper currently supports only wav.")

    voice = payload.voice or VIBEVOICE_DEFAULT_VOICE
    if voice not in {"default", "clone"}:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown voice '{voice}'. Use 'default' or 'clone'.",
        )

    processor, model = get_model()
    speaker_path = _resolve_speaker_path(voice)

    text = _format_text(payload.input)
    inputs = processor(
        text=[text],
        voice_samples=[[str(speaker_path)]],
        padding=True,
        return_tensors="pt",
    )

    outputs = model.generate(
        **inputs,
        max_new_tokens=None,
        cfg_scale=VIBEVOICE_CFG_SCALE,
        tokenizer=processor.tokenizer,
        generation_config={"do_sample": False},
    )

    if not getattr(outputs, "speech_outputs", None):
        raise HTTPException(status_code=500, detail="No audio was generated.")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        processor.save_audio(outputs.speech_outputs[0], tmp_path)
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
