from __future__ import annotations

import io
import logging
import os
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from boson_multimodal.data_types import (
    AudioContent,
    ChatMLSample,
    Message,
)
from boson_multimodal.serve.serve_engine import HiggsAudioServeEngine

logger = logging.getLogger("uvicorn.error")

OPENAI_MODEL_ID = os.environ.get("OPENAI_MODEL_ID", "higgs-audio-v2")
HIGGS_REPO_DIR = os.environ.get("HIGGS_REPO_DIR", "")
HF_MODEL = os.environ.get("HIGGS_HF_MODEL", "bosonai/higgs-audio-v2-generation-3B-base")
HF_TOKENIZER = os.environ.get("HIGGS_HF_TOKENIZER", "bosonai/higgs-audio-v2-tokenizer")
DEFAULT_VOICE = os.environ.get("HIGGS_DEFAULT_VOICE", "default")
DEFAULT_REF_VOICE = os.environ.get("HIGGS_DEFAULT_REF_VOICE", "belinda")
PROMPT_WAV = os.environ.get("HIGGS_PROMPT_WAV", "")
PROMPT_TEXT = os.environ.get("HIGGS_PROMPT_TEXT", "")
MAX_NEW_TOKENS = int(os.environ.get("HIGGS_MAX_NEW_TOKENS", "1024"))
TEMPERATURE = float(os.environ.get("HIGGS_TEMPERATURE", "0.7"))

# Bundled reference voices ship under examples/voice_prompts/<name>.wav and
# <name>.txt in the upstream repo.
VOICE_PROMPTS_DIR = Path(HIGGS_REPO_DIR) / "examples" / "voice_prompts" if HIGGS_REPO_DIR else None

DEFAULT_SYSTEM_MESSAGE = (
    "You are an AI assistant designed to convert text into speech. "
    "If the user's message includes a [SPEAKER*] tag, do not read out the tag and "
    "generate speech for the following text, using the specified voice. "
    "If no speaker tag is present, select a suitable voice on your own."
)

app = FastAPI(title="Higgs Audio v2 OpenAI Compatible TTS")

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


_engine: HiggsAudioServeEngine | None = None


def get_engine() -> HiggsAudioServeEngine:
    global _engine
    if _engine is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Loading HiggsAudioServeEngine on %s (%s)", device, HF_MODEL)
        _engine = HiggsAudioServeEngine(
            model_name_or_path=HF_MODEL,
            audio_tokenizer_name_or_path=HF_TOKENIZER,
            device=device,
        )
        logger.info("Higgs Audio v2 ready")
    return _engine


def _bundled_voice_paths(name: str) -> tuple[Path, Path] | None:
    if VOICE_PROMPTS_DIR is None:
        return None
    wav = VOICE_PROMPTS_DIR / f"{name}.wav"
    txt = VOICE_PROMPTS_DIR / f"{name}.txt"
    if not wav.exists() or not txt.exists():
        return None
    return wav, txt


def _build_chat_ml_sample(text: str, ref_wav: Path | None, ref_text: str | None) -> ChatMLSample:
    messages: list[Message] = [Message(role="system", content=DEFAULT_SYSTEM_MESSAGE)]
    if ref_wav is not None and ref_text is not None:
        messages.append(Message(role="user", content=ref_text))
        messages.append(Message(role="assistant", content=AudioContent(audio_url=str(ref_wav))))
    messages.append(Message(role="user", content=text))
    return ChatMLSample(messages=messages)


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
        "engine": "Higgs-Audio-v2",
        "model": OPENAI_MODEL_ID,
        "default_voice": DEFAULT_VOICE,
        "default_ref_voice": DEFAULT_REF_VOICE,
    }


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [{"id": OPENAI_MODEL_ID, "object": "model", "owned_by": "boson-ai"}],
    }


@app.get("/v1/voices")
def list_voices():
    voices = [{"id": "default", "object": "voice", "ref": DEFAULT_REF_VOICE}]
    if PROMPT_WAV:
        voices.append({"id": "clone", "object": "voice", "ref": PROMPT_WAV})
    if VOICE_PROMPTS_DIR is not None and VOICE_PROMPTS_DIR.exists():
        for wav_path in sorted(VOICE_PROMPTS_DIR.glob("*.wav")):
            txt_path = wav_path.with_suffix(".txt")
            if txt_path.exists():
                voices.append({"id": wav_path.stem, "object": "voice"})
    return {"object": "list", "data": voices}


@app.post("/v1/audio/speech")
async def audio_speech(payload: AudioSpeechRequest):
    if payload.response_format.lower() != "wav":
        raise HTTPException(status_code=400, detail="This wrapper currently supports only wav.")

    voice = payload.voice or DEFAULT_VOICE

    if voice == "clone":
        if not PROMPT_WAV or not PROMPT_TEXT:
            raise HTTPException(
                status_code=400,
                detail="voice='clone' requires --higgs-prompt-wav and --higgs-prompt-text at startup.",
            )
        ref_wav: Path | None = Path(PROMPT_WAV)
        ref_text: str | None = PROMPT_TEXT
    elif voice == "default":
        bundled = _bundled_voice_paths(DEFAULT_REF_VOICE)
        if bundled is None:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Default reference voice '{DEFAULT_REF_VOICE}' not found in "
                    f"{VOICE_PROMPTS_DIR}. Pass --higgs-default-ref-voice to a valid name."
                ),
            )
        ref_wav, ref_text_path = bundled
        ref_text = ref_text_path.read_text(encoding="utf-8").strip()
    else:
        bundled = _bundled_voice_paths(voice)
        if bundled is None:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown voice: {voice}. Available: default, clone, or any preset under examples/voice_prompts/.",
            )
        ref_wav, ref_text_path = bundled
        ref_text = ref_text_path.read_text(encoding="utf-8").strip()

    sample = _build_chat_ml_sample(payload.input, ref_wav, ref_text)
    engine = get_engine()
    response = engine.generate(
        chat_ml_sample=sample,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
    )

    if response.audio is None or response.sampling_rate is None:
        raise HTTPException(status_code=500, detail="No audio was generated.")

    audio = np.asarray(response.audio, dtype=np.float32)

    buf = io.BytesIO()
    sf.write(buf, audio, response.sampling_rate, format="WAV", subtype="PCM_16")
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
