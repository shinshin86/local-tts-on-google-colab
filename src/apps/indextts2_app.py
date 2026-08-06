from __future__ import annotations

import logging
import os
import tempfile
import threading
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from indextts.infer_v2 import IndexTTS2
from pydantic import BaseModel

logger = logging.getLogger("uvicorn.error")

OPENAI_MODEL_ID = os.environ.get("OPENAI_MODEL_ID", "indextts2")
INDEXTTS2_HF_MODEL = os.environ.get("INDEXTTS2_HF_MODEL", "IndexTeam/IndexTTS-2")
INDEXTTS2_MODEL_DIR = os.environ.get("INDEXTTS2_MODEL_DIR", "")
INDEXTTS2_DEFAULT_PROMPT_WAV = os.environ.get("INDEXTTS2_DEFAULT_PROMPT_WAV", "")
INDEXTTS2_PROMPT_WAV = os.environ.get("INDEXTTS2_PROMPT_WAV", "")
INDEXTTS2_DEFAULT_VOICE = os.environ.get("INDEXTTS2_DEFAULT_VOICE", "default")
INDEXTTS2_EMOTION_WAV = os.environ.get("INDEXTTS2_EMOTION_WAV", "")
INDEXTTS2_EMOTION_TEXT = os.environ.get("INDEXTTS2_EMOTION_TEXT", "")
INDEXTTS2_EMOTION_VECTOR = os.environ.get("INDEXTTS2_EMOTION_VECTOR", "")
INDEXTTS2_EMOTION_ALPHA = float(os.environ.get("INDEXTTS2_EMOTION_ALPHA", "0.6"))
INDEXTTS2_USE_RANDOM = os.environ.get("INDEXTTS2_USE_RANDOM", "0") == "1"
INDEXTTS2_USE_FP16 = os.environ.get("INDEXTTS2_USE_FP16", "1") == "1"

app = FastAPI(title="IndexTTS2 OpenAI Compatible TTS")
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
    emotion_text: str | None = None
    emotion_vector: list[float] | None = None
    emotion_weight: float | None = None
    emotion_random: bool | None = None


_model: Any | None = None
_model_lock = threading.Lock()
_inference_lock = threading.Lock()


def _parse_startup_vector() -> list[float] | None:
    if not INDEXTTS2_EMOTION_VECTOR.strip():
        return None
    try:
        vector = [float(item.strip()) for item in INDEXTTS2_EMOTION_VECTOR.split(",")]
    except ValueError as exc:
        raise RuntimeError("INDEXTTS2_EMOTION_VECTOR must contain eight comma-separated numbers.") from exc
    return _validate_vector(vector)


def _validate_vector(vector: list[float]) -> list[float]:
    if len(vector) != 8:
        raise HTTPException(status_code=400, detail="emotion_vector must contain exactly eight values.")
    if any(value < 0.0 or value > 1.0 for value in vector):
        raise HTTPException(status_code=400, detail="emotion_vector values must be between 0.0 and 1.0.")
    return vector


def get_model() -> Any:
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                device = "cuda:0" if torch.cuda.is_available() else "cpu"
                use_fp16 = INDEXTTS2_USE_FP16 and device.startswith("cuda")
                model_dir = Path(INDEXTTS2_MODEL_DIR)
                aux_paths = {
                    "w2v_bert": str(model_dir / "hf_cache" / "w2v-bert-2.0"),
                    "semantic_codec": str(model_dir / "hf_cache" / "semantic_codec_model.safetensors"),
                    "campplus": str(model_dir / "hf_cache" / "campplus_cn_common.bin"),
                    "bigvgan": str(model_dir / "hf_cache" / "bigvgan"),
                }
                logger.info("Loading IndexTTS2 from %s (device=%s, fp16=%s)", model_dir, device, use_fp16)
                _model = IndexTTS2(
                    cfg_path=str(model_dir / "config.yaml"),
                    model_dir=str(model_dir),
                    use_fp16=use_fp16,
                    device=device,
                    use_cuda_kernel=False,
                    use_deepspeed=False,
                    use_accel=False,
                    use_torch_compile=False,
                    aux_paths=aux_paths,
                )
                logger.info("IndexTTS2 ready")
    return _model


def _available_voices() -> list[str]:
    voices = ["default"]
    if INDEXTTS2_PROMPT_WAV:
        voices.append("clone")
    return voices


def _resolve_voice(voice: str | None) -> tuple[str, str]:
    requested = INDEXTTS2_DEFAULT_VOICE if not voice else voice
    if requested == "default":
        prompt = INDEXTTS2_DEFAULT_PROMPT_WAV
    elif requested == "clone" and INDEXTTS2_PROMPT_WAV:
        prompt = INDEXTTS2_PROMPT_WAV
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown or unavailable voice '{requested}'. Available: {', '.join(_available_voices())}",
        )
    if not prompt or not Path(prompt).is_file():
        raise HTTPException(status_code=500, detail=f"Reference audio is missing for voice '{requested}'.")
    return requested, prompt


def _emotion_kwargs(payload: AudioSpeechRequest) -> dict[str, Any]:
    if payload.emotion_text is not None and payload.emotion_vector is not None:
        raise HTTPException(status_code=400, detail="Use either emotion_text or emotion_vector, not both.")
    weight = INDEXTTS2_EMOTION_ALPHA if payload.emotion_weight is None else payload.emotion_weight
    if weight < 0.0 or weight > 1.0:
        raise HTTPException(status_code=400, detail="emotion_weight must be between 0.0 and 1.0.")
    use_random = INDEXTTS2_USE_RANDOM if payload.emotion_random is None else payload.emotion_random

    if payload.emotion_text is not None:
        return {
            "use_emo_text": True,
            "emo_text": payload.emotion_text,
            "emo_alpha": weight,
            "use_random": use_random,
        }
    if payload.emotion_vector is not None:
        return {
            "emo_vector": _validate_vector(payload.emotion_vector),
            "emo_alpha": weight,
            "use_random": use_random,
        }
    if INDEXTTS2_EMOTION_TEXT:
        return {
            "use_emo_text": True,
            "emo_text": INDEXTTS2_EMOTION_TEXT,
            "emo_alpha": weight,
            "use_random": use_random,
        }
    startup_vector = _parse_startup_vector()
    if startup_vector is not None:
        return {
            "emo_vector": startup_vector,
            "emo_alpha": weight,
            "use_random": use_random,
        }
    if INDEXTTS2_EMOTION_WAV:
        if not Path(INDEXTTS2_EMOTION_WAV).is_file():
            raise HTTPException(status_code=500, detail="Configured emotion reference audio is missing.")
        return {
            "emo_audio_prompt": INDEXTTS2_EMOTION_WAV,
            "emo_alpha": weight,
            "use_random": use_random,
        }
    return {"use_random": use_random}


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception while serving request")
    return JSONResponse(status_code=500, content={"error": type(exc).__name__, "detail": str(exc)})


@app.get("/")
def root():
    return {
        "ok": True,
        "engine": "IndexTTS2",
        "model": OPENAI_MODEL_ID,
        "hf_model": INDEXTTS2_HF_MODEL,
        "default_voice": INDEXTTS2_DEFAULT_VOICE,
        "voices": _available_voices(),
        "duration_control_available": False,
    }


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [{"id": OPENAI_MODEL_ID, "object": "model", "owned_by": "indexteam"}],
    }


@app.get("/v1/voices")
def list_voices():
    descriptions = {
        "default": "Official IndexTTS2 demo reference voice.",
        "clone": "Zero-shot clone from the configured reference audio.",
    }
    return {
        "object": "list",
        "data": [
            {"id": voice, "object": "voice", "description": descriptions[voice]}
            for voice in _available_voices()
        ],
    }


@app.post("/v1/audio/speech")
async def audio_speech(payload: AudioSpeechRequest):
    if payload.response_format.lower() != "wav":
        raise HTTPException(status_code=400, detail="This wrapper currently supports only wav.")
    if payload.speed != 1.0:
        raise HTTPException(
            status_code=400,
            detail="The public IndexTTS2 release does not expose duration or speed control.",
        )
    if not payload.input.strip():
        raise HTTPException(status_code=400, detail="input must not be empty.")

    voice, prompt_wav = _resolve_voice(payload.voice)
    emotion_kwargs = _emotion_kwargs(payload)
    model = get_model()

    output_path = ""
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            output_path = tmp.name
        with _inference_lock:
            model.infer(
                spk_audio_prompt=prompt_wav,
                text=payload.input,
                output_path=output_path,
                verbose=False,
                **emotion_kwargs,
            )
        audio = Path(output_path).read_bytes()
    finally:
        if output_path:
            Path(output_path).unlink(missing_ok=True)

    return Response(
        content=audio,
        media_type="audio/wav",
        headers={"x-openai-model": OPENAI_MODEL_ID, "x-openai-voice": voice},
    )
