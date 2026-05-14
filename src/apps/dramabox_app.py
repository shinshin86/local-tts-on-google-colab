from __future__ import annotations

import logging
import os
import sys
import tempfile
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logger = logging.getLogger("uvicorn.error")

OPENAI_MODEL_ID = os.environ.get("OPENAI_MODEL_ID", "dramabox")
DRAMABOX_REPO_DIR = os.environ.get("DRAMABOX_REPO_DIR", "")
DRAMABOX_HF_MODEL = os.environ.get("DRAMABOX_HF_MODEL", "ResembleAI/Dramabox")
DRAMABOX_GEMMA_REPO = os.environ.get("DRAMABOX_GEMMA_REPO", "unsloth/gemma-3-12b-it-bnb-4bit")
DEFAULT_VOICE = os.environ.get("DRAMABOX_DEFAULT_VOICE", "default")
DEFAULT_REF_VOICE = os.environ.get("DRAMABOX_DEFAULT_REF_VOICE", "female_american")
PROMPT_WAV = os.environ.get("DRAMABOX_PROMPT_WAV", "")
DTYPE = os.environ.get("DRAMABOX_DTYPE", "bf16")
CFG_SCALE = float(os.environ.get("DRAMABOX_CFG_SCALE", "2.5"))
STG_SCALE = float(os.environ.get("DRAMABOX_STG_SCALE", "1.5"))
DURATION_MULTIPLIER = float(os.environ.get("DRAMABOX_DURATION_MULTIPLIER", "1.1"))
SEED = int(os.environ.get("DRAMABOX_SEED", "42"))
COMPILE_MODEL = os.environ.get("DRAMABOX_COMPILE", "0") == "1"
BNB_4BIT = os.environ.get("DRAMABOX_BNB_4BIT", "1") == "1"

# Upstream layout: <repo>/src/inference_server.py imports from <repo>/ltx2 and
# <repo>/src via sys.path. Mirror that here so we can load TTSServer.
if DRAMABOX_REPO_DIR:
    for sub in ("ltx2", "src"):
        path = str(Path(DRAMABOX_REPO_DIR) / sub)
        if path not in sys.path:
            sys.path.insert(0, path)

VOICES_DIR = Path(DRAMABOX_REPO_DIR) / "assets" / "voices" if DRAMABOX_REPO_DIR else None

app = FastAPI(title="DramaBox OpenAI Compatible TTS")

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


_server = None  # type: ignore[var-annotated]


def _resolve_bundled_voice(name: str) -> Path | None:
    if VOICES_DIR is None:
        return None
    for ext in (".wav", ".mp3", ".flac", ".ogg"):
        candidate = VOICES_DIR / f"{name}{ext}"
        if candidate.exists():
            return candidate
    return None


def get_server():
    global _server
    if _server is not None:
        return _server

    # Lazy imports: TTSServer transitively pulls in torch / bnb / gemma weights.
    from huggingface_hub import hf_hub_download, snapshot_download

    logger.info("DramaBox: fetching checkpoints from %s ...", DRAMABOX_HF_MODEL)
    transformer_path = hf_hub_download(DRAMABOX_HF_MODEL, "dramabox-dit-v1.safetensors")
    audio_components_path = hf_hub_download(
        DRAMABOX_HF_MODEL, "dramabox-audio-components.safetensors"
    )
    logger.info("DramaBox: fetching Gemma snapshot from %s ...", DRAMABOX_GEMMA_REPO)
    gemma_root = snapshot_download(DRAMABOX_GEMMA_REPO)

    from inference_server import TTSServer  # type: ignore[import-not-found]

    logger.info(
        "DramaBox: loading TTSServer (dtype=%s, compile=%s, bnb_4bit=%s)",
        DTYPE,
        COMPILE_MODEL,
        BNB_4BIT,
    )
    _server = TTSServer(
        checkpoint=transformer_path,
        full_checkpoint=audio_components_path,
        gemma_root=gemma_root,
        device="cuda",
        dtype=DTYPE,
        compile_model=COMPILE_MODEL,
        bnb_4bit=BNB_4BIT,
    )
    logger.info("DramaBox: TTSServer ready")
    return _server


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
        "engine": "DramaBox",
        "model": OPENAI_MODEL_ID,
        "default_voice": DEFAULT_VOICE,
        "default_ref_voice": DEFAULT_REF_VOICE,
    }


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [{"id": OPENAI_MODEL_ID, "object": "model", "owned_by": "resemble-ai"}],
    }


@app.get("/v1/voices")
def list_voices():
    voices = [{"id": "default", "object": "voice", "ref": DEFAULT_REF_VOICE}]
    if PROMPT_WAV:
        voices.append({"id": "clone", "object": "voice", "ref": PROMPT_WAV})
    if VOICES_DIR is not None and VOICES_DIR.exists():
        seen = set()
        for path in sorted(VOICES_DIR.iterdir()):
            if path.suffix.lower() in {".wav", ".mp3", ".flac", ".ogg"} and path.stem not in seen:
                voices.append({"id": path.stem, "object": "voice"})
                seen.add(path.stem)
    return {"object": "list", "data": voices}


@app.post("/v1/audio/speech")
async def audio_speech(payload: AudioSpeechRequest):
    if payload.response_format.lower() != "wav":
        raise HTTPException(status_code=400, detail="This wrapper currently supports only wav.")

    voice = payload.voice or DEFAULT_VOICE

    if voice == "clone":
        if not PROMPT_WAV:
            raise HTTPException(
                status_code=400,
                detail="voice='clone' requires --dramabox-prompt-wav at startup.",
            )
        voice_ref: Path | None = Path(PROMPT_WAV)
    elif voice == "default":
        voice_ref = _resolve_bundled_voice(DEFAULT_REF_VOICE)
        if voice_ref is None:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Default reference voice '{DEFAULT_REF_VOICE}' not found in "
                    f"{VOICES_DIR}. Pass --dramabox-default-ref-voice to a valid preset name."
                ),
            )
    else:
        voice_ref = _resolve_bundled_voice(voice)
        if voice_ref is None:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Unknown voice: {voice}. Available: default, clone, "
                    "or any preset under DramaBox/assets/voices/."
                ),
            )

    server = get_server()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        # Perth implicit watermarking stays on (Resemble AI requirement;
        # see CLAUDE.md "watermarks must not be removed").
        server.generate_to_file(
            prompt=payload.input,
            output=tmp_path,
            voice_ref=str(voice_ref) if voice_ref else None,
            cfg_scale=CFG_SCALE,
            stg_scale=STG_SCALE,
            duration_multiplier=DURATION_MULTIPLIER,
            seed=SEED,
            watermark=True,
        )
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
