from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

import soundfile as sf
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from supertonic import TTS

logger = logging.getLogger("uvicorn.error")

OPENAI_MODEL_ID = os.environ.get("OPENAI_MODEL_ID", "supertonic-3")
SUPERTONIC_MODEL = os.environ.get("SUPERTONIC_MODEL", "supertonic-3")
DEFAULT_VOICE = os.environ.get("SUPERTONIC_DEFAULT_VOICE", "M1")
DEFAULT_LANG = os.environ.get("SUPERTONIC_DEFAULT_LANG", "en")
TOTAL_STEPS = int(os.environ.get("SUPERTONIC_TOTAL_STEPS", "5"))


def detect_lang(text: str) -> str | None:
    """Light-weight script detection for callers that don't pass `language`.

    OpenAI's /v1/audio/speech schema has no language field, so this lets the
    default smoke test (Japanese text, no language) work without forcing the
    caller to send a non-standard parameter.
    """
    for ch in text:
        code = ord(ch)
        # Hangul syllables / Jamo
        if 0xAC00 <= code <= 0xD7A3 or 0x1100 <= code <= 0x11FF or 0x3130 <= code <= 0x318F:
            return "ko"
        # Hiragana / Katakana
        if 0x3040 <= code <= 0x309F or 0x30A0 <= code <= 0x30FF:
            return "ja"
        # CJK Unified Ideographs — ambiguous (ja/zh/ko), but the repo's default
        # test text is Japanese so prefer ja.
        if 0x4E00 <= code <= 0x9FFF:
            return "ja"
    return None

# supertonic-3 voice presets shipped with the model. supertonic-py exposes the
# authoritative list at runtime via tts.voice_style_names; we keep these here
# only to drive the /v1/voices response when the model has not finished loading.
VOICE_PRESETS = ["M1", "M2", "M3", "M4", "M5", "F1", "F2", "F3", "F4", "F5"]

# supertonic-3 accepts 31 ISO codes plus a "na" fallback for unknown text.
SUPPORTED_LANGS = {
    "en", "ko", "ja", "ar", "bg", "cs", "da", "de", "el", "es", "et", "fi",
    "fr", "hi", "hr", "hu", "id", "it", "lt", "lv", "nl", "pl", "pt", "ro",
    "ru", "sk", "sl", "sv", "tr", "uk", "vi", "na",
}

app = FastAPI(title="Supertonic OpenAI Compatible TTS")

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
    # Non-standard: Supertonic needs a language hint per request.
    language: str | None = None


_tts: TTS | None = None
_styles: dict[str, object] = {}


def get_tts() -> TTS:
    global _tts
    if _tts is None:
        _tts = TTS(model=SUPERTONIC_MODEL, auto_download=True)
    return _tts


def get_voice_style(voice_name: str):
    style = _styles.get(voice_name)
    if style is None:
        style = get_tts().get_voice_style(voice_name=voice_name)
        _styles[voice_name] = style
    return style


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception while serving request")
    return JSONResponse(
        status_code=500,
        content={"error": type(exc).__name__, "detail": str(exc)},
    )


@app.get("/")
def root():
    return {"ok": True, "engine": "Supertonic", "model": OPENAI_MODEL_ID}


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [{"id": OPENAI_MODEL_ID, "object": "model", "owned_by": "local"}],
    }


@app.get("/v1/voices")
def list_voices():
    # Prefer the authoritative list from the loaded model when available.
    voices = VOICE_PRESETS
    if _tts is not None and getattr(_tts, "voice_style_names", None):
        voices = list(_tts.voice_style_names)
    return {
        "object": "list",
        "data": [{"id": voice, "object": "voice"} for voice in voices],
    }


@app.post("/v1/audio/speech")
async def audio_speech(payload: AudioSpeechRequest):
    if payload.response_format.lower() != "wav":
        raise HTTPException(status_code=400, detail="This wrapper currently supports only wav.")

    voice = payload.voice or DEFAULT_VOICE
    # Map "default" to the configured default voice; Supertonic itself uses
    # preset names like M1/F1.
    if voice == "default":
        voice = DEFAULT_VOICE
    if voice == "clone":
        raise HTTPException(
            status_code=400,
            detail=(
                "Supertonic does not support runtime voice cloning. "
                "Use a preset voice (M1-M5 / F1-F5)."
            ),
        )

    lang = (payload.language or detect_lang(payload.input) or DEFAULT_LANG).lower()
    if lang not in SUPPORTED_LANGS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported language '{lang}'. Pass one of {sorted(SUPPORTED_LANGS)} "
                "via the 'language' field, or use 'na' for unknown languages."
            ),
        )

    tts = get_tts()
    try:
        style = get_voice_style(voice)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown voice '{voice}'. Available presets: "
                f"{', '.join(tts.voice_style_names)}."
            ),
        ) from exc

    wav, _duration = tts.synthesize(
        payload.input,
        voice_style=style,
        total_steps=TOTAL_STEPS,
        speed=float(payload.speed),
        lang=lang,
    )

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        # wav has shape (1, num_samples); soundfile expects (num_samples,).
        sf.write(tmp_path, wav.squeeze(), tts.sample_rate)
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
