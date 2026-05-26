from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

import torch
import torchaudio
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from transformers import AutoModel, AutoProcessor

logger = logging.getLogger("uvicorn.error")

OPENAI_MODEL_ID = os.environ.get("OPENAI_MODEL_ID", "moss-tts-v1.5")
MOSS_HF_MODEL = os.environ.get("MOSS_TTS_V1_5_HF_MODEL", "OpenMOSS-Team/MOSS-TTS-v1.5")
MOSS_LANGUAGE = os.environ.get("MOSS_TTS_V1_5_LANGUAGE", "Japanese")
MOSS_PROMPT_WAV = os.environ.get("MOSS_TTS_V1_5_PROMPT_WAV", "")
MOSS_ATTN_IMPL = os.environ.get("MOSS_TTS_V1_5_ATTN_IMPL", "sdpa")
MOSS_MAX_NEW_TOKENS = int(os.environ.get("MOSS_TTS_V1_5_MAX_NEW_TOKENS", "4096"))

app = FastAPI(title="MOSS-TTS-v1.5 OpenAI Compatible TTS")

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
_device: str = "cpu"
_dtype = torch.float32


def get_pipeline():
    global _processor, _model, _device, _dtype
    if _model is None:
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        _dtype = torch.bfloat16 if _device == "cuda" else torch.float32
        logger.info(
            "Loading MOSS-TTS-v1.5: %s (device=%s, dtype=%s, attn=%s)",
            MOSS_HF_MODEL,
            _device,
            _dtype,
            MOSS_ATTN_IMPL,
        )
        processor = AutoProcessor.from_pretrained(MOSS_HF_MODEL, trust_remote_code=True)
        processor.audio_tokenizer = processor.audio_tokenizer.to(_device)
        model = AutoModel.from_pretrained(
            MOSS_HF_MODEL,
            trust_remote_code=True,
            attn_implementation=MOSS_ATTN_IMPL,
            torch_dtype=_dtype,
        ).to(_device)
        model.eval()
        _processor = processor
        _model = model
        logger.info("MOSS-TTS-v1.5 loaded")
    return _processor, _model


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception while serving request")
    return JSONResponse(
        status_code=500,
        content={"error": type(exc).__name__, "detail": str(exc)},
    )


@app.get("/")
def root():
    return {"ok": True, "engine": "MOSS-TTS-v1.5", "model": OPENAI_MODEL_ID}


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [{"id": OPENAI_MODEL_ID, "object": "model", "owned_by": "openmoss"}],
    }


@app.get("/v1/voices")
def list_voices():
    voices = [{"id": "default", "object": "voice"}]
    if MOSS_PROMPT_WAV:
        voices.append({"id": "clone", "object": "voice"})
    return {"object": "list", "data": voices}


@app.post("/v1/audio/speech")
async def audio_speech(payload: AudioSpeechRequest):
    if payload.response_format.lower() != "wav":
        raise HTTPException(status_code=400, detail="This wrapper currently supports only wav.")

    voice = (payload.voice or "default").strip()
    if voice == "clone" and not MOSS_PROMPT_WAV:
        raise HTTPException(
            status_code=400,
            detail="voice='clone' requires MOSS_TTS_V1_5_PROMPT_WAV. Pass --moss-tts-v1-5-prompt-wav at launch.",
        )

    processor, model = get_pipeline()

    message_kwargs = {"text": payload.input, "language": MOSS_LANGUAGE}
    if voice == "clone":
        message_kwargs["reference"] = [MOSS_PROMPT_WAV]

    conversations = [[processor.build_user_message(**message_kwargs)]]
    batch = processor(conversations, mode="generation")
    input_ids = batch["input_ids"].to(_device)
    attention_mask = batch["attention_mask"].to(_device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=MOSS_MAX_NEW_TOKENS,
        )

    decoded = processor.decode(outputs)
    if not decoded:
        raise HTTPException(status_code=500, detail="MOSS-TTS-v1.5 produced no output.")

    audio = decoded[0].audio_codes_list[0]
    sample_rate = processor.model_config.sampling_rate

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        audio_cpu = audio.detach().to("cpu", dtype=torch.float32)
        if audio_cpu.ndim == 1:
            audio_cpu = audio_cpu.unsqueeze(0)
        torchaudio.save(tmp_path, audio_cpu, sample_rate)
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
