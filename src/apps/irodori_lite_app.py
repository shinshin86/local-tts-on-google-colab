from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import irodori_tts_lite
from irodori_tts.inference_runtime import (
    InferenceRuntime,
    RuntimeKey,
    SamplingRequest,
    default_runtime_device,
    resolve_cfg_scales,
    save_wav,
)

logger = logging.getLogger("uvicorn.error")

# Default: kizuna-intelligence/Irodori-TTS-Lite-int4 (voice-design, no Duration Predictor)
# Alternate: kizuna-intelligence/Irodori-TTS-500M-v3-int4 (set IRODORI_LITE_CHECKPOINT_FILE=model.safetensors)
HF_CHECKPOINT = os.environ.get(
    "IRODORI_LITE_HF_CHECKPOINT", "kizuna-intelligence/Irodori-TTS-Lite-int4"
)
CHECKPOINT_FILE = os.environ.get("IRODORI_LITE_CHECKPOINT_FILE", "dit_int4.safetensors")
MODEL_DEVICE = os.environ.get("IRODORI_LITE_MODEL_DEVICE", default_runtime_device())
CODEC_DEVICE = os.environ.get("IRODORI_LITE_CODEC_DEVICE", default_runtime_device())
CODEC_REPO = os.environ.get(
    "IRODORI_LITE_CODEC_REPO", "Aratako/Semantic-DACVAE-Japanese-32dim"
)
CODEC_INT4 = os.environ.get("IRODORI_LITE_CODEC_INT4", "0") == "1"
OPENAI_MODEL_ID = os.environ.get("OPENAI_MODEL_ID", HF_CHECKPOINT)

# v3-int4 ships the Duration Predictor; the default voice-design int4 does not, so we
# derive seconds from phoneme count via pyopenjtalk (mirrors the upstream example).
IS_V3 = "v3" in HF_CHECKPOINT.lower()

app = FastAPI(title="Irodori-TTS-Lite OpenAI Compatible TTS")

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
_patched = False


def estimate_seconds(text: str) -> float:
    try:
        import pyopenjtalk

        phs = pyopenjtalk.g2p(text, kana=False).split()
        return max(2.0, len(phs) / 11.0 + 0.6)
    except Exception:
        return max(2.0, len(text) / 6.0 + 0.6)


def get_runtime():
    global _runtime, _patched
    if not _patched:
        configure_kwargs = {"use_fused": True, "force_fp16": True}
        if CODEC_INT4:
            configure_kwargs["codec_int4"] = True
        irodori_tts_lite.configure(**configure_kwargs)
        irodori_tts_lite.patch()
        _patched = True

    if _runtime is None:
        if "/" in HF_CHECKPOINT and not HF_CHECKPOINT.startswith("/"):
            checkpoint_uri = f"hf://{HF_CHECKPOINT}/{CHECKPOINT_FILE}"
        else:
            checkpoint_uri = HF_CHECKPOINT
        checkpoint_path = irodori_tts_lite.resolve_checkpoint(checkpoint_uri)
        _runtime = InferenceRuntime.from_key(
            RuntimeKey(
                checkpoint=checkpoint_path,
                model_device=MODEL_DEVICE,
                codec_repo=CODEC_REPO,
                model_precision="fp16",
                codec_device=CODEC_DEVICE,
                codec_precision="fp16",
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
    return {"ok": True, "engine": "Irodori-TTS-Lite", "model": OPENAI_MODEL_ID}


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
    cfg_scale_text, _cfg_scale_caption, cfg_scale_speaker, _ = resolve_cfg_scales(
        cfg_guidance_mode="independent",
        cfg_scale_text=3.0,
        cfg_scale_caption=3.0,
        cfg_scale_speaker=5.0,
        cfg_scale=None,
    )

    seconds = None if IS_V3 else estimate_seconds(payload.input)
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
            seconds=seconds,
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
