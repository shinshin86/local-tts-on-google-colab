from __future__ import annotations

import copy
import io
import logging
import os
import pickle
import tempfile
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from vibevoice.modular.modeling_vibevoice_streaming_inference import (
    VibeVoiceStreamingForConditionalGenerationInference,
)
from vibevoice.processor.vibevoice_streaming_processor import VibeVoiceStreamingProcessor

logger = logging.getLogger("uvicorn.error")

OPENAI_MODEL_ID = os.environ.get("OPENAI_MODEL_ID", "vibevoice-realtime")
VIBEVOICE_HF_MODEL = os.environ.get(
    "VIBEVOICE_HF_MODEL", "microsoft/VibeVoice-Realtime-0.5B"
)
VIBEVOICE_MODEL_DIR = os.environ.get("VIBEVOICE_MODEL_DIR", VIBEVOICE_HF_MODEL)
VIBEVOICE_PROCESSOR_DIR = os.environ.get("VIBEVOICE_PROCESSOR_DIR", VIBEVOICE_MODEL_DIR)
VIBEVOICE_VOICES_DIR = Path(os.environ.get("VIBEVOICE_VOICES_DIR", "demo/voices/streaming_model"))
VIBEVOICE_DEFAULT_SPEAKER = os.environ.get("VIBEVOICE_DEFAULT_SPEAKER", "jp-Spk1_woman")
VIBEVOICE_DDPM_STEPS = int(os.environ.get("VIBEVOICE_DDPM_STEPS", "5"))
VIBEVOICE_CFG_SCALE = float(os.environ.get("VIBEVOICE_CFG_SCALE", "1.5"))

app = FastAPI(title="VibeVoice-Realtime OpenAI Compatible TTS")

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


_processor: VibeVoiceStreamingProcessor | None = None
_model: VibeVoiceStreamingForConditionalGenerationInference | None = None


# The official prompt caches contain BaseModelOutputWithPast and DynamicCache
# dict subclasses. PyTorch's weights-only loader cannot rebuild them (upstream
# issue #392), so restrict full pickle semantics to the exact classes and tensor
# primitives used by those caches. Any other global remains blocked.
_VOICE_PRESET_SAFE_GLOBALS = {
    ("collections", "OrderedDict"),
    ("transformers.modeling_outputs", "BaseModelOutputWithPast"),
    ("transformers.cache_utils", "DynamicCache"),
}


class _VoicePresetUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) in _VOICE_PRESET_SAFE_GLOBALS:
            return super().find_class(module, name)
        if module == "torch._utils" and name.startswith("_rebuild_"):
            return super().find_class(module, name)
        if module == "torch" and name.endswith("Storage"):
            return super().find_class(module, name)
        raise pickle.UnpicklingError(
            f"Refusing to load disallowed global '{module}.{name}' from voice preset"
        )


class _RestrictedPickleModule:
    Unpickler = _VoicePresetUnpickler

    @staticmethod
    def load(file, **kwargs):
        return _VoicePresetUnpickler(file, **kwargs).load()

    @staticmethod
    def loads(data, **kwargs):
        return _VoicePresetUnpickler(io.BytesIO(data), **kwargs).load()


def _voice_presets() -> dict[str, Path]:
    return {path.stem.lower(): path for path in sorted(VIBEVOICE_VOICES_DIR.rglob("*.pt"))}


def _resolve_voice(voice: str | None) -> tuple[str, Path]:
    requested = VIBEVOICE_DEFAULT_SPEAKER if not voice or voice == "default" else voice
    presets = _voice_presets()
    path = presets.get(requested.lower())
    if path is None:
        available = ", ".join(sorted(presets))
        raise HTTPException(
            status_code=400,
            detail=f"Unknown voice '{requested}'. Available presets: {available}",
        )
    return path.stem, path


def get_model() -> tuple[VibeVoiceStreamingProcessor, VibeVoiceStreamingForConditionalGenerationInference]:
    global _processor, _model
    if _processor is None or _model is None:
        if not torch.cuda.is_available():
            raise RuntimeError("VibeVoice-Realtime currently requires a CUDA GPU in this wrapper.")
        logger.info("Loading VibeVoice-Realtime: %s", VIBEVOICE_HF_MODEL)
        _processor = VibeVoiceStreamingProcessor.from_pretrained(VIBEVOICE_PROCESSOR_DIR)
        _model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
            VIBEVOICE_MODEL_DIR,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            attn_implementation="sdpa",
        )
        _model.eval()
        _model.set_ddpm_inference_steps(num_steps=VIBEVOICE_DDPM_STEPS)
        logger.info(
            "VibeVoice-Realtime ready (ddpm_steps=%s, cfg_scale=%s)",
            VIBEVOICE_DDPM_STEPS,
            VIBEVOICE_CFG_SCALE,
        )
    return _processor, _model


def _load_cached_prompt(path: Path) -> dict[str, Any]:
    return torch.load(
        path,
        map_location="cuda",
        pickle_module=_RestrictedPickleModule,
        weights_only=False,
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception while serving request")
    return JSONResponse(status_code=500, content={"error": type(exc).__name__, "detail": str(exc)})


@app.get("/")
def root():
    return {
        "ok": True,
        "engine": "VibeVoice-Realtime",
        "model": OPENAI_MODEL_ID,
        "hf_model": VIBEVOICE_HF_MODEL,
        "default_speaker": VIBEVOICE_DEFAULT_SPEAKER,
        "voice_count": len(_voice_presets()),
    }


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [{"id": OPENAI_MODEL_ID, "object": "model", "owned_by": "microsoft"}],
    }


@app.get("/v1/voices")
def list_voices():
    voices = [
        {"id": path.stem, "object": "voice", "language": path.stem.split("-", 1)[0]}
        for path in _voice_presets().values()
    ]
    return {"object": "list", "data": voices}


@app.post("/v1/audio/speech")
async def audio_speech(payload: AudioSpeechRequest):
    if payload.response_format.lower() != "wav":
        raise HTTPException(status_code=400, detail="This wrapper currently supports only wav.")
    if payload.speed != 1.0:
        raise HTTPException(status_code=400, detail="VibeVoice-Realtime does not support speed control.")

    voice, voice_path = _resolve_voice(payload.voice)
    processor, model = get_model()
    cached_prompt = _load_cached_prompt(voice_path)
    inputs = processor.process_input_with_cached_prompt(
        text=payload.input,
        cached_prompt=cached_prompt,
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
    )
    for key, value in inputs.items():
        if torch.is_tensor(value):
            inputs[key] = value.to("cuda")

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=None,
            cfg_scale=VIBEVOICE_CFG_SCALE,
            tokenizer=processor.tokenizer,
            generation_config={"do_sample": False},
            all_prefilled_outputs=copy.deepcopy(cached_prompt),
        )

    if not getattr(outputs, "speech_outputs", None) or outputs.speech_outputs[0] is None:
        raise HTTPException(status_code=500, detail="No audio was generated.")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        processor.save_audio(outputs.speech_outputs[0], output_path=str(tmp_path))
        audio_bytes = tmp_path.read_bytes()
    finally:
        tmp_path.unlink(missing_ok=True)

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
