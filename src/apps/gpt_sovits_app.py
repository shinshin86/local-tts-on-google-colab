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

GPT_SOVITS_REPO_DIR = os.environ.get("GPT_SOVITS_REPO_DIR", "")
if GPT_SOVITS_REPO_DIR:
    sys.path.insert(0, GPT_SOVITS_REPO_DIR)
    sys.path.insert(0, str(Path(GPT_SOVITS_REPO_DIR) / "GPT_SoVITS"))

from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config  # noqa: E402

logger = logging.getLogger("uvicorn.error")

OPENAI_MODEL_ID = os.environ.get("OPENAI_MODEL_ID", "gpt-sovits-v2")
VERSION = os.environ.get("GPT_SOVITS_VERSION", "v2")
DEFAULT_VOICE = os.environ.get("GPT_SOVITS_DEFAULT_VOICE", "default")
PROMPT_WAV = os.environ.get("GPT_SOVITS_PROMPT_WAV", "")
PROMPT_TEXT = os.environ.get("GPT_SOVITS_PROMPT_TEXT", "")
PROMPT_LANG = os.environ.get("GPT_SOVITS_PROMPT_LANG", "en")
TARGET_LANG = os.environ.get("GPT_SOVITS_TARGET_LANG", "en")

# GPT-SoVITS supports zh / en / ja / ko / yue (Cantonese). The TTS_Config picks
# the right BERT / HuBERT for the selected version.
SUPPORTED_LANGS = ["zh", "en", "ja", "ko", "yue", "auto"]

CONFIG_PATH = str(
    Path(GPT_SOVITS_REPO_DIR) / "GPT_SoVITS" / "configs" / "tts_infer.yaml"
)

app = FastAPI(title="GPT-SoVITS OpenAI Compatible TTS")

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


_pipeline: TTS | None = None


def get_pipeline() -> TTS:
    global _pipeline
    if _pipeline is None:
        logger.info("Loading GPT-SoVITS TTS_Config from %s (version=%s)", CONFIG_PATH, VERSION)
        cfg = TTS_Config(CONFIG_PATH)
        # The shipped tts_infer.yaml has multiple version blocks (v1, v2, v3, v4, etc.).
        # Pin to whichever the user requested.
        if hasattr(cfg, "version"):
            cfg.version = VERSION
        _pipeline = TTS(cfg)
        logger.info("GPT-SoVITS pipeline ready")
    return _pipeline


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
        "engine": "GPT-SoVITS",
        "model": OPENAI_MODEL_ID,
        "version": VERSION,
        "default_voice": DEFAULT_VOICE,
        "prompt_lang": PROMPT_LANG,
        "target_lang": TARGET_LANG,
    }


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [{"id": OPENAI_MODEL_ID, "object": "model", "owned_by": "rvc-boss"}],
    }


@app.get("/v1/voices")
def list_voices():
    voices = []
    if PROMPT_WAV and PROMPT_TEXT:
        voices.append({"id": "default", "object": "voice", "ref": PROMPT_WAV})
    voices.append({"id": "clone", "object": "voice"})
    return {"object": "list", "data": voices}


@app.post("/v1/audio/speech")
async def audio_speech(payload: AudioSpeechRequest):
    if payload.response_format.lower() != "wav":
        raise HTTPException(status_code=400, detail="This wrapper currently supports only wav.")

    voice = payload.voice or DEFAULT_VOICE

    if not PROMPT_WAV or not PROMPT_TEXT:
        raise HTTPException(
            status_code=400,
            detail=(
                "GPT-SoVITS requires a reference audio. "
                "Pass --gpt-sovits-prompt-wav and --gpt-sovits-prompt-text at startup."
            ),
        )

    # GPT-SoVITS is fundamentally a few-shot voice cloning model: every request
    # needs a reference audio. We accept both `default` and `clone` here for
    # consistency with the rest of the engines, but they take the same code path.
    if voice not in ("default", "clone"):
        raise HTTPException(
            status_code=400,
            detail=f"Unknown voice: {voice}. Available: default, clone",
        )

    pipeline = get_pipeline()
    request_dict = {
        "text": payload.input,
        "text_lang": TARGET_LANG,
        "ref_audio_path": PROMPT_WAV,
        "prompt_text": PROMPT_TEXT,
        "prompt_lang": PROMPT_LANG,
        "top_k": 15,
        "top_p": 1.0,
        "temperature": 1.0,
        "text_split_method": "cut5",
        "batch_size": 1,
        "speed_factor": float(payload.speed),
        "streaming_mode": False,
        "media_type": "wav",
        "parallel_infer": True,
        "repetition_penalty": 1.35,
    }

    sample_rate: int | None = None
    chunks: list[np.ndarray] = []
    for sr, audio_chunk in pipeline.run(request_dict):
        sample_rate = sr
        if audio_chunk is not None and len(audio_chunk) > 0:
            chunks.append(np.asarray(audio_chunk))

    if not chunks or sample_rate is None:
        raise HTTPException(status_code=500, detail="No audio was generated.")

    audio = np.concatenate(chunks, axis=-1)

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
