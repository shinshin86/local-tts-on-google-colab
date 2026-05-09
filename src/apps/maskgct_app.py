from __future__ import annotations

import io
import logging
import os
import sys
from pathlib import Path

import safetensors
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from huggingface_hub import hf_hub_download
from pydantic import BaseModel

MASKGCT_REPO_DIR = os.environ.get("MASKGCT_REPO_DIR", "")
if MASKGCT_REPO_DIR:
    sys.path.insert(0, MASKGCT_REPO_DIR)
    # Upstream's mandarin g2p module uses relative paths like
    # `./models/tts/maskgct/g2p/sources/...` and calls `exit()` if the resource
    # files are missing. Chdir into the Amphion repo so those relative paths
    # resolve before triggering the import below.
    os.chdir(MASKGCT_REPO_DIR)

from models.tts.maskgct.maskgct_utils import (  # noqa: E402
    MaskGCT_Inference_Pipeline,
    build_acoustic_codec,
    build_s2a_model,
    build_semantic_codec,
    build_semantic_model,
    build_t2s_model,
    load_config,
)

logger = logging.getLogger("uvicorn.error")

OPENAI_MODEL_ID = os.environ.get("OPENAI_MODEL_ID", "maskgct")
DEFAULT_VOICE = os.environ.get("MASKGCT_DEFAULT_VOICE", "default")
PROMPT_WAV = os.environ.get("MASKGCT_PROMPT_WAV", "")
PROMPT_TEXT = os.environ.get("MASKGCT_PROMPT_TEXT", "")
PROMPT_LANG = os.environ.get("MASKGCT_PROMPT_LANG", "en")
TARGET_LANG = os.environ.get("MASKGCT_TARGET_LANG", "en")

# Bundled reference shipped in the Amphion repo, used as the "default" voice.
BUNDLED_PROMPT_WAV = str(
    Path(MASKGCT_REPO_DIR) / "models" / "tts" / "maskgct" / "wav" / "prompt.wav"
)
BUNDLED_PROMPT_TEXT = " We do not break. We never give in. We never back down."

app = FastAPI(title="MaskGCT OpenAI Compatible TTS")

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


_pipeline: MaskGCT_Inference_Pipeline | None = None


def get_pipeline() -> MaskGCT_Inference_Pipeline:
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg_path = str(
        Path(MASKGCT_REPO_DIR) / "models" / "tts" / "maskgct" / "config" / "maskgct.json"
    )
    cfg = load_config(cfg_path)
    logger.info("Building MaskGCT components on %s", device)

    semantic_model, semantic_mean, semantic_std = build_semantic_model(device)
    semantic_codec = build_semantic_codec(cfg.model.semantic_codec, device)
    codec_encoder, codec_decoder = build_acoustic_codec(cfg.model.acoustic_codec, device)
    t2s_model = build_t2s_model(cfg.model.t2s_model, device)
    s2a_model_1layer = build_s2a_model(cfg.model.s2a_model.s2a_1layer, device)
    s2a_model_full = build_s2a_model(cfg.model.s2a_model.s2a_full, device)

    logger.info("Downloading MaskGCT checkpoints from amphion/MaskGCT")
    semantic_code_ckpt = hf_hub_download("amphion/MaskGCT", filename="semantic_codec/model.safetensors")
    codec_encoder_ckpt = hf_hub_download("amphion/MaskGCT", filename="acoustic_codec/model.safetensors")
    codec_decoder_ckpt = hf_hub_download("amphion/MaskGCT", filename="acoustic_codec/model_1.safetensors")
    t2s_model_ckpt = hf_hub_download("amphion/MaskGCT", filename="t2s_model/model.safetensors")
    s2a_1layer_ckpt = hf_hub_download(
        "amphion/MaskGCT", filename="s2a_model/s2a_model_1layer/model.safetensors"
    )
    s2a_full_ckpt = hf_hub_download(
        "amphion/MaskGCT", filename="s2a_model/s2a_model_full/model.safetensors"
    )

    safetensors.torch.load_model(semantic_codec, semantic_code_ckpt)
    safetensors.torch.load_model(codec_encoder, codec_encoder_ckpt)
    safetensors.torch.load_model(codec_decoder, codec_decoder_ckpt)
    safetensors.torch.load_model(t2s_model, t2s_model_ckpt)
    safetensors.torch.load_model(s2a_model_1layer, s2a_1layer_ckpt)
    safetensors.torch.load_model(s2a_model_full, s2a_full_ckpt)

    _pipeline = MaskGCT_Inference_Pipeline(
        semantic_model,
        semantic_codec,
        codec_encoder,
        codec_decoder,
        t2s_model,
        s2a_model_1layer,
        s2a_model_full,
        semantic_mean,
        semantic_std,
        device,
    )
    logger.info("MaskGCT pipeline ready")
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
        "engine": "MaskGCT",
        "model": OPENAI_MODEL_ID,
        "default_voice": DEFAULT_VOICE,
        "prompt_lang": PROMPT_LANG,
        "target_lang": TARGET_LANG,
    }


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [{"id": OPENAI_MODEL_ID, "object": "model", "owned_by": "amphion"}],
    }


@app.get("/v1/voices")
def list_voices():
    voices = [{"id": "default", "object": "voice", "ref": BUNDLED_PROMPT_WAV}]
    if PROMPT_WAV:
        voices.append({"id": "clone", "object": "voice", "ref": PROMPT_WAV})
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
                detail="voice='clone' requires --maskgct-prompt-wav and --maskgct-prompt-text at startup.",
            )
        prompt_wav = PROMPT_WAV
        prompt_text = PROMPT_TEXT
    elif voice == "default":
        prompt_wav = BUNDLED_PROMPT_WAV
        prompt_text = BUNDLED_PROMPT_TEXT
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown voice: {voice}. Available: default, clone (when prompt configured)",
        )

    pipeline = get_pipeline()
    audio = pipeline.maskgct_inference(
        prompt_wav,
        prompt_text,
        payload.input,
        PROMPT_LANG,
        TARGET_LANG,
        target_len=None,
    )

    if audio is None or len(audio) == 0:
        raise HTTPException(status_code=500, detail="No audio was generated.")

    buf = io.BytesIO()
    sf.write(buf, audio, 24000, format="WAV", subtype="PCM_16")
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
