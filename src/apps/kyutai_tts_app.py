from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from moshi.models.loaders import CheckpointInfo
from moshi.models.tts import TTSModel
from pydantic import BaseModel

logger = logging.getLogger("uvicorn.error")

OPENAI_MODEL_ID = os.environ.get("OPENAI_MODEL_ID", "kyutai-tts")
HF_REPO = os.environ.get("KYUTAI_HF_REPO", "kyutai/tts-1.6b-en_fr")
VOICE_REPO = os.environ.get("KYUTAI_VOICE_REPO", "kyutai/tts-voices")
DEFAULT_VOICE_PATH = os.environ.get(
    "KYUTAI_VOICE", "expresso/ex03-ex01_happy_001_channel1_334s.wav"
)
PROMPT_WAV = os.environ.get("KYUTAI_PROMPT_WAV", "")
DEFAULT_VOICE = os.environ.get("KYUTAI_DEFAULT_VOICE", "default")

app = FastAPI(title="Kyutai TTS OpenAI Compatible TTS")

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


_tts_model: TTSModel | None = None


def get_model() -> TTSModel:
    global _tts_model
    if _tts_model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Loading Kyutai TTS model %s on %s", HF_REPO, device)
        checkpoint_info = CheckpointInfo.from_hf_repo(HF_REPO)
        _tts_model = TTSModel.from_checkpoint_info(
            checkpoint_info, n_q=32, temp=0.6, device=device, voice_repo=VOICE_REPO
        )
        logger.info("Kyutai TTS model loaded (sample_rate=%s)", _tts_model.mimi.sample_rate)
    return _tts_model


def resolve_voice_path(model: TTSModel, voice: str) -> str:
    # When voice is "default", use the configured voice name from the voice repo.
    # When voice is "clone", use the local prompt wav (path or .safetensors file).
    # Any other value is treated as a path inside the voice repo.
    if voice == "clone":
        if not PROMPT_WAV:
            raise HTTPException(
                status_code=400,
                detail="voice='clone' requires --kyutai-prompt-wav at startup.",
            )
        return PROMPT_WAV
    if voice == "default":
        target = DEFAULT_VOICE_PATH
    else:
        target = voice
    if target.endswith(".safetensors"):
        return target
    return model.get_voice_path(target)


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
        "engine": "Kyutai-TTS",
        "model": OPENAI_MODEL_ID,
        "hf_repo": HF_REPO,
        "voice_repo": VOICE_REPO,
        "default_voice": DEFAULT_VOICE,
    }


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [{"id": OPENAI_MODEL_ID, "object": "model", "owned_by": "kyutai"}],
    }


@app.get("/v1/voices")
def list_voices():
    voices = [
        {"id": "default", "object": "voice", "voice_path": DEFAULT_VOICE_PATH}
    ]
    if PROMPT_WAV:
        voices.append({"id": "clone", "object": "voice", "voice_path": PROMPT_WAV})
    return {"object": "list", "data": voices}


@app.post("/v1/audio/speech")
async def audio_speech(payload: AudioSpeechRequest):
    if payload.response_format.lower() != "wav":
        raise HTTPException(status_code=400, detail="This wrapper currently supports only wav.")

    model = get_model()
    voice = payload.voice or DEFAULT_VOICE
    voice_path = resolve_voice_path(model, voice)

    entries = model.prepare_script([payload.input], padding_between=1)
    condition_attributes = model.make_condition_attributes([voice_path], cfg_coef=2.0)

    result = model.generate([entries], [condition_attributes])

    with model.mimi.streaming(1), torch.no_grad():
        pcms = []
        for frame in result.frames[model.delay_steps:]:
            pcm = model.mimi.decode(frame[:, 1:, :]).cpu().numpy()
            pcms.append(np.clip(pcm[0, 0], -1, 1))
        if not pcms:
            raise HTTPException(status_code=500, detail="No audio was generated.")
        audio = np.concatenate(pcms, axis=-1)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        sf.write(tmp_path, audio, model.mimi.sample_rate)
        audio_bytes = Path(tmp_path).read_bytes()
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return Response(
        content=audio_bytes,
        media_type="audio/wav",
        headers={
            "Content-Length": str(len(audio_bytes)),
            "x-openai-model": payload.model,
            "x-openai-voice": voice,
        },
    )
