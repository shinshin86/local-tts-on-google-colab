from __future__ import annotations

import io
import logging
import os

import soundfile as sf
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from dots_tts.runtime import DotsTtsRuntime

logger = logging.getLogger("uvicorn.error")

OPENAI_MODEL_ID = os.environ.get("OPENAI_MODEL_ID", "dots-tts")
HF_MODEL = os.environ.get("DOTS_TTS_HF_MODEL", "rednote-hilab/dots.tts-base")
DEFAULT_VOICE = os.environ.get("DOTS_TTS_DEFAULT_VOICE", "default")
PROMPT_WAV = os.environ.get("DOTS_TTS_PROMPT_WAV", "")
PROMPT_TEXT = os.environ.get("DOTS_TTS_PROMPT_TEXT", "")
# Language tag mode passed to DotsTtsRuntime: "none" / "" -> no tag,
# "auto_detect" -> infer from the input text, or a code/name such as EN / JA.
LANGUAGE = os.environ.get("DOTS_TTS_LANGUAGE", "auto_detect")
PRECISION = os.environ.get("DOTS_TTS_PRECISION", "bfloat16")
NUM_STEPS = int(os.environ.get("DOTS_TTS_NUM_STEPS", "10"))
GUIDANCE_SCALE = float(os.environ.get("DOTS_TTS_GUIDANCE_SCALE", "1.2"))
SPEAKER_SCALE = float(os.environ.get("DOTS_TTS_SPEAKER_SCALE", "1.5"))
MAX_GENERATE_LENGTH = int(os.environ.get("DOTS_TTS_MAX_GENERATE_LENGTH", "500"))

app = FastAPI(title="dots.tts OpenAI Compatible TTS")

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
    # Optional per-request override of the language tag mode.
    language: str | None = None


_runtime: DotsTtsRuntime | None = None


def get_runtime() -> DotsTtsRuntime:
    global _runtime
    if _runtime is None:
        logger.info("Loading dots.tts (%s) precision=%s", HF_MODEL, PRECISION)
        # from_pretrained accepts a HF repo id directly (snapshot_download under
        # the hood) and moves the model to CUDA when available. ~9.5GB download
        # on first load.
        _runtime = DotsTtsRuntime.from_pretrained(
            HF_MODEL,
            precision=PRECISION,
            max_generate_length=MAX_GENERATE_LENGTH,
        )
        logger.info("dots.tts loaded (sample_rate=%d)", _runtime.sample_rate)
    return _runtime


def _normalize_language(language: str | None) -> str | None:
    if language is None:
        return None
    stripped = language.strip()
    if not stripped or stripped.lower() == "none":
        return None
    return stripped


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
        "engine": "dots.tts",
        "model": OPENAI_MODEL_ID,
        "hf_model": HF_MODEL,
        "default_voice": DEFAULT_VOICE,
        "clone_ready": bool(PROMPT_WAV),
    }


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [{"id": OPENAI_MODEL_ID, "object": "model", "owned_by": "rednote-hilab"}],
    }


@app.get("/v1/voices")
def list_voices():
    # default = no reference (random-voice sampling; only meaningful on a
    # fine-tuned single-speaker checkpoint, otherwise a random timbre).
    # clone = zero-shot cloning from the configured reference clip.
    voices = [{"id": "default", "object": "voice"}]
    if PROMPT_WAV:
        voices.append({"id": "clone", "object": "voice", "prompt_wav": PROMPT_WAV})
    return {"object": "list", "data": voices}


@app.post("/v1/audio/speech")
async def audio_speech(payload: AudioSpeechRequest):
    if payload.response_format.lower() != "wav":
        raise HTTPException(status_code=400, detail="This wrapper currently supports only wav.")

    voice = payload.voice or DEFAULT_VOICE
    language = _normalize_language(payload.language) if payload.language is not None else _normalize_language(LANGUAGE)

    prompt_audio_path: str | None = None
    prompt_text: str | None = None
    if voice == "clone":
        if not PROMPT_WAV:
            raise HTTPException(
                status_code=400,
                detail="voice='clone' requires DOTS_TTS_PROMPT_WAV (--dots-tts-prompt-wav). "
                "Optionally set --dots-tts-prompt-text for continuation cloning. "
                "Use voice='default' for random-voice sampling.",
            )
        prompt_audio_path = PROMPT_WAV
        # With a transcript -> continuation cloning (recommended). Without it ->
        # x-vector-only cloning (timbre from the speaker embedding).
        prompt_text = PROMPT_TEXT or None
    elif voice != "default":
        raise HTTPException(
            status_code=400,
            detail=f"Unknown voice: {voice}. Available: ['default', 'clone'].",
        )

    runtime = get_runtime()
    result = runtime.generate(
        text=payload.input,
        prompt_audio_path=prompt_audio_path,
        prompt_text=prompt_text,
        language=language,
        num_steps=NUM_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        speaker_scale=SPEAKER_SCALE,
    )

    audio = result["audio"]
    if audio is None or audio.numel() == 0:
        raise HTTPException(status_code=500, detail="No audio was generated.")

    audio_np = audio.float().cpu().squeeze().numpy()

    buf = io.BytesIO()
    sf.write(buf, audio_np, result["sample_rate"], format="WAV", subtype="PCM_16")
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
