from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

import soundfile as sf
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pocket_tts import TTSModel
from pydantic import BaseModel

logger = logging.getLogger("uvicorn.error")

OPENAI_MODEL_ID = os.environ.get("OPENAI_MODEL_ID", "pocket-tts")
LANGUAGE = os.environ.get("POCKET_LANGUAGE", "english")
DEFAULT_SPEAKER = os.environ.get("POCKET_DEFAULT_SPEAKER", "alba")
PROMPT_WAV = os.environ.get("POCKET_PROMPT_WAV", "")
DEFAULT_VOICE = os.environ.get("POCKET_DEFAULT_VOICE", "default")

PREDEFINED_VOICES = [
    "alba",
    "anna",
    "azelma",
    "bill_boerst",
    "caro_davy",
    "charles",
    "cosette",
    "eponine",
    "eve",
    "fantine",
    "george",
    "jane",
    "jean",
    "javert",
    "marius",
    "mary",
    "michael",
    "paul",
    "peter_yearsley",
    "stuart_bell",
    "vera",
]

app = FastAPI(title="Pocket TTS OpenAI Compatible TTS")

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
_voice_cache: dict[str, dict] = {}


def get_model() -> TTSModel:
    global _tts_model
    if _tts_model is None:
        logger.info("Loading Pocket TTS model (language=%s)", LANGUAGE)
        _tts_model = TTSModel.load_model(language=LANGUAGE)
        logger.info("Pocket TTS model loaded (sample_rate=%s)", _tts_model.sample_rate)
    return _tts_model


def resolve_voice(model: TTSModel, voice: str) -> dict:
    # default → use the configured DEFAULT_SPEAKER preset
    # clone → load the local PROMPT_WAV (audio file or .safetensors)
    # any other value is treated as either a predefined preset name or a path.
    if voice == "clone":
        if not PROMPT_WAV:
            raise HTTPException(
                status_code=400,
                detail="voice='clone' requires --pocket-prompt-wav at startup.",
            )
        target = PROMPT_WAV
    elif voice == "default":
        target = DEFAULT_SPEAKER
    else:
        target = voice
    state = _voice_cache.get(target)
    if state is None:
        logger.info("Loading voice state for '%s'", target)
        state = model.get_state_for_audio_prompt(target)
        _voice_cache[target] = state
    return state


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
        "engine": "Pocket-TTS",
        "model": OPENAI_MODEL_ID,
        "language": LANGUAGE,
        "default_speaker": DEFAULT_SPEAKER,
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
    voices = [{"id": "default", "object": "voice", "speaker": DEFAULT_SPEAKER}]
    if PROMPT_WAV:
        voices.append({"id": "clone", "object": "voice", "speaker": PROMPT_WAV})
    voices.extend({"id": name, "object": "voice"} for name in PREDEFINED_VOICES)
    return {"object": "list", "data": voices}


@app.post("/v1/audio/speech")
async def audio_speech(payload: AudioSpeechRequest):
    if payload.response_format.lower() != "wav":
        raise HTTPException(status_code=400, detail="This wrapper currently supports only wav.")

    model = get_model()
    voice = payload.voice or DEFAULT_VOICE
    voice_state = resolve_voice(model, voice)

    audio = model.generate_audio(voice_state, payload.input)
    audio_np = audio.detach().cpu().numpy()
    if audio_np.ndim == 2:
        # [channels, samples] → [samples, channels] for soundfile
        audio_np = audio_np.T

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        sf.write(tmp_path, audio_np, model.sample_rate)
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
