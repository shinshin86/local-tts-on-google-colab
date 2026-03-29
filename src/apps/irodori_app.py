from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from huggingface_hub import hf_hub_download
from irodori_tts.inference_runtime import (
    InferenceRuntime,
    RuntimeKey,
    SamplingRequest,
    default_runtime_device,
    resolve_cfg_scales,
    save_wav,
)

logger = logging.getLogger("uvicorn.error")

HF_CHECKPOINT = os.environ.get("IRODORI_HF_CHECKPOINT", "Aratako/Irodori-TTS-500M")
MODEL_DEVICE = os.environ.get("IRODORI_MODEL_DEVICE", default_runtime_device())
CODEC_DEVICE = os.environ.get("IRODORI_CODEC_DEVICE", default_runtime_device())
MODEL_PRECISION = os.environ.get("IRODORI_MODEL_PRECISION", "fp32")
CODEC_PRECISION = os.environ.get("IRODORI_CODEC_PRECISION", "fp32")
CODEC_REPO = os.environ.get("IRODORI_CODEC_REPO", "facebook/dacvae-watermarked")
OPENAI_MODEL_ID = os.environ.get("OPENAI_MODEL_ID", HF_CHECKPOINT)

app = FastAPI(title="Irodori OpenAI Compatible TTS")

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


_runtime = None


def get_runtime():
    global _runtime
    if _runtime is None:
        checkpoint_path = hf_hub_download(
            repo_id=HF_CHECKPOINT,
            filename="model.safetensors",
        )
        _runtime = InferenceRuntime.from_key(
            RuntimeKey(
                checkpoint=checkpoint_path,
                model_device=MODEL_DEVICE,
                codec_repo=CODEC_REPO,
                model_precision=MODEL_PRECISION,
                codec_device=CODEC_DEVICE,
                codec_precision=CODEC_PRECISION,
                enable_watermark=False,
                compile_model=False,
                compile_dynamic=False,
            )
        )
    return _runtime


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception while serving request")
    return JSONResponse(
        status_code=500,
        content={"error": type(exc).__name__, "detail": str(exc)},
    )


@app.get("/")
def root():
    return {"ok": True, "engine": "Irodori-TTS", "model": OPENAI_MODEL_ID}


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": OPENAI_MODEL_ID,
                "object": "model",
                "owned_by": "local",
            }
        ],
    }


@app.get("/v1/voices")
def list_voices():
    return {"object": "list", "data": []}


@app.post("/v1/audio/speech")
async def audio_speech(payload: AudioSpeechRequest):
    if payload.response_format.lower() != "wav":
        raise HTTPException(status_code=400, detail="This wrapper currently supports only wav.")

    runtime = get_runtime()
    cfg_scale_text, cfg_scale_speaker, _ = resolve_cfg_scales(
        cfg_guidance_mode="independent",
        cfg_scale_text=3.0,
        cfg_scale_speaker=5.0,
        cfg_scale=None,
    )

    result = runtime.synthesize(
        SamplingRequest(
            text=payload.input,
            ref_wav=None,
            ref_latent=None,
            no_ref=True,
            ref_normalize_db=None,
            ref_ensure_max=False,
            num_candidates=1,
            decode_mode="sequential",
            seconds=30.0,
            max_ref_seconds=30.0,
            max_text_len=None,
            num_steps=40,
            cfg_scale_text=cfg_scale_text,
            cfg_scale_speaker=cfg_scale_speaker,
            cfg_guidance_mode="independent",
            cfg_scale=None,
            cfg_min_t=0.5,
            cfg_max_t=1.0,
            truncation_factor=None,
            rescale_k=None,
            rescale_sigma=None,
            context_kv_cache=True,
            speaker_kv_scale=None,
            speaker_kv_min_t=None,
            speaker_kv_max_layers=None,
            seed=None,
            trim_tail=True,
            tail_window_size=20,
            tail_std_threshold=0.05,
            tail_mean_threshold=0.1,
        ),
        log_fn=None,
    )

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        save_wav(tmp_path, result.audio, result.sample_rate)
        audio_bytes = Path(tmp_path).read_bytes()
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return Response(
        content=audio_bytes,
        media_type="audio/wav",
        headers={
            "Content-Length": str(len(audio_bytes)),
            "x-openai-model": payload.model,
            "x-openai-voice": payload.voice or "",
        },
    )
