from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from melo.api import TTS
from openvoice import se_extractor
from openvoice.api import ToneColorConverter
from pydantic import BaseModel

logger = logging.getLogger("uvicorn.error")

OPENAI_MODEL_ID = os.environ.get("OPENAI_MODEL_ID", "openvoice-v2")
OPENVOICE_LANGUAGE = os.environ.get("OPENVOICE_LANGUAGE", "JP")
OPENVOICE_CKPT_DIR = Path(os.environ.get("OPENVOICE_CKPT_DIR", "checkpoints_v2"))
OPENVOICE_REPO_DIR = Path(os.environ.get("OPENVOICE_REPO_DIR", "OpenVoice"))
OPENVOICE_PROMPT_WAV = os.environ.get("OPENVOICE_PROMPT_WAV", "")
OPENVOICE_DEFAULT_VOICE = os.environ.get("OPENVOICE_DEFAULT_VOICE", "default")

# Languages exposed by OpenVoice V2 + MeloTTS base speakers.
SUPPORTED_LANGUAGES = ["EN", "ES", "FR", "ZH", "JP", "KR"]
# Bundled reference audio shipped in the OpenVoice repo, used as the fallback
# when the user does not provide --openvoice-prompt-wav.
DEFAULT_REFERENCE = OPENVOICE_REPO_DIR / "resources" / "example_reference.mp3"

app = FastAPI(title="OpenVoice V2 OpenAI Compatible TTS")

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


_tone_converter: ToneColorConverter | None = None
_base_tts: TTS | None = None
_base_speaker_id: int | None = None
_source_se: torch.Tensor | None = None
_target_se_cache: dict[str, torch.Tensor] = {}


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_tone_converter() -> ToneColorConverter:
    global _tone_converter
    if _tone_converter is None:
        device = _device()
        cfg = OPENVOICE_CKPT_DIR / "converter" / "config.json"
        ckpt = OPENVOICE_CKPT_DIR / "converter" / "checkpoint.pth"
        logger.info("Loading ToneColorConverter from %s", cfg)
        _tone_converter = ToneColorConverter(str(cfg), device=device)
        _tone_converter.load_ckpt(str(ckpt))
    return _tone_converter


def get_base_tts() -> tuple[TTS, int, torch.Tensor]:
    global _base_tts, _base_speaker_id, _source_se
    if _base_tts is None:
        device = _device()
        if OPENVOICE_LANGUAGE not in SUPPORTED_LANGUAGES:
            raise RuntimeError(
                f"Unsupported language '{OPENVOICE_LANGUAGE}'. Supported: {SUPPORTED_LANGUAGES}"
            )
        logger.info("Loading MeloTTS base for language=%s", OPENVOICE_LANGUAGE)
        _base_tts = TTS(language=OPENVOICE_LANGUAGE, device=device)
        speaker_ids = _base_tts.hps.data.spk2id
        speaker_key = next(iter(speaker_ids.keys()))
        _base_speaker_id = speaker_ids[speaker_key]
        se_filename = speaker_key.lower().replace("_", "-")
        se_path = OPENVOICE_CKPT_DIR / "base_speakers" / "ses" / f"{se_filename}.pth"
        logger.info("Loading source speaker embedding from %s", se_path)
        _source_se = torch.load(str(se_path), map_location=device)
    assert _base_tts is not None and _base_speaker_id is not None and _source_se is not None
    return _base_tts, _base_speaker_id, _source_se


def get_target_se(voice: str) -> torch.Tensor:
    if voice in _target_se_cache:
        return _target_se_cache[voice]
    converter = get_tone_converter()
    if voice == "clone":
        ref_path = OPENVOICE_PROMPT_WAV
    else:
        ref_path = str(DEFAULT_REFERENCE)
    if not ref_path or not Path(ref_path).exists():
        raise HTTPException(
            status_code=500,
            detail=f"Reference audio not found for voice '{voice}': {ref_path}",
        )
    logger.info("Extracting target SE for voice '%s' from %s", voice, ref_path)
    target_se, _ = se_extractor.get_se(ref_path, converter, vad=True)
    _target_se_cache[voice] = target_se
    return target_se


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
        "engine": "OpenVoice V2",
        "model": OPENAI_MODEL_ID,
        "language": OPENVOICE_LANGUAGE,
        "default_voice": OPENVOICE_DEFAULT_VOICE,
    }


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [{"id": OPENAI_MODEL_ID, "object": "model", "owned_by": "myshell-ai"}],
    }


@app.get("/v1/voices")
def list_voices():
    voices = [{"id": "default", "object": "voice", "language": OPENVOICE_LANGUAGE}]
    if OPENVOICE_PROMPT_WAV:
        voices.append({"id": "clone", "object": "voice", "language": OPENVOICE_LANGUAGE})
    return {"object": "list", "data": voices}


@app.post("/v1/audio/speech")
async def audio_speech(payload: AudioSpeechRequest):
    if payload.response_format.lower() != "wav":
        raise HTTPException(status_code=400, detail="This wrapper currently supports only wav.")

    voice = payload.voice or OPENVOICE_DEFAULT_VOICE
    if voice not in {"default", "clone"}:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown voice '{voice}'. Use 'default' or 'clone'.",
        )
    if voice == "clone" and not OPENVOICE_PROMPT_WAV:
        raise HTTPException(
            status_code=400,
            detail="voice='clone' requires --openvoice-prompt-wav at startup.",
        )

    tts, speaker_id, source_se = get_base_tts()
    converter = get_tone_converter()
    target_se = get_target_se(voice)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_src:
        src_path = tmp_src.name
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_out:
        out_path = tmp_out.name

    try:
        tts.tts_to_file(payload.input, speaker_id, src_path, speed=payload.speed)
        converter.convert(
            audio_src_path=src_path,
            src_se=source_se,
            tgt_se=target_se,
            output_path=out_path,
            message="@MyShell",
        )
        audio_bytes = Path(out_path).read_bytes()
    finally:
        Path(src_path).unlink(missing_ok=True)
        Path(out_path).unlink(missing_ok=True)

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
