from __future__ import annotations

import logging
import os
from io import BytesIO
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from scipy.io import wavfile
from style_bert_vits2.constants import (
    DEFAULT_ASSIST_TEXT_WEIGHT,
    DEFAULT_NOISE,
    DEFAULT_NOISEW,
    DEFAULT_SDP_RATIO,
    DEFAULT_SPLIT_INTERVAL,
    DEFAULT_STYLE_WEIGHT,
    Languages,
)
from style_bert_vits2.nlp import bert_models
from style_bert_vits2.nlp.japanese import pyopenjtalk_worker as pyopenjtalk
from style_bert_vits2.nlp.japanese.user_dict import update_dict
from style_bert_vits2.tts_model import TTSModel, TTSModelHolder

logger = logging.getLogger("uvicorn.error")

OPENAI_MODEL_ID = os.environ.get("OPENAI_MODEL_ID", "style-bert-vits2")
MODEL_ROOT = Path(os.environ.get("STYLE_BERT_MODEL_ROOT", "./model_assets"))
MODEL_NAME = os.environ.get("STYLE_BERT_MODEL_NAME", "jvnv-F2-jp")
SPEAKER_ID = int(os.environ.get("STYLE_BERT_SPEAKER_ID", "0"))
STYLE_NAME = os.environ.get("STYLE_BERT_STYLE", "Neutral")
DEVICE = os.environ.get("STYLE_BERT_DEVICE", "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu")

app = FastAPI(title="Style-Bert-VITS2 OpenAI Compatible TTS")

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


_holder = None
_models = []


def get_model() -> TTSModel:
    global _holder, _models
    if _holder is None:
        pyopenjtalk.initialize_worker()
        update_dict()
        bert_models.load_model(Languages.JP)
        bert_models.load_tokenizer(Languages.JP)
        _holder = TTSModelHolder(MODEL_ROOT, DEVICE)
        if len(_holder.model_names) == 0:
            raise RuntimeError(f"No Style-Bert-VITS2 models found in {MODEL_ROOT}")
        _models = []
        for model_name, model_paths in _holder.model_files_dict.items():
            model = TTSModel(
                model_path=model_paths[0],
                config_path=_holder.root_dir / model_name / "config.json",
                style_vec_path=_holder.root_dir / model_name / "style_vectors.npy",
                device=_holder.device,
            )
            _models.append(model)

    for index, info in enumerate(_holder.models_info):
        if info.name == MODEL_NAME:
            return _models[index]
    return _models[0]


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception while serving request")
    return JSONResponse(
        status_code=500,
        content={"error": type(exc).__name__, "detail": str(exc)},
    )


@app.get("/")
def root():
    return {"ok": True, "engine": "Style-Bert-VITS2", "model": OPENAI_MODEL_ID}


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [{"id": OPENAI_MODEL_ID, "object": "model", "owned_by": "local"}],
    }


@app.get("/v1/voices")
def list_voices():
    model = get_model()
    return {
        "object": "list",
        "data": [
            {"id": str(speaker_id), "name": speaker_name, "object": "voice"}
            for speaker_id, speaker_name in model.id2spk.items()
        ],
    }


@app.post("/v1/audio/speech")
async def audio_speech(payload: AudioSpeechRequest):
    if payload.response_format.lower() != "wav":
        raise HTTPException(status_code=400, detail="This wrapper currently supports only wav.")

    model = get_model()
    speaker_id = SPEAKER_ID
    if payload.voice:
        try:
            speaker_id = int(payload.voice)
        except ValueError:
            if payload.voice in model.spk2id:
                speaker_id = model.spk2id[payload.voice]
            else:
                raise HTTPException(status_code=400, detail=f"Unknown voice '{payload.voice}'.")

    if speaker_id not in model.id2spk:
        raise HTTPException(status_code=400, detail=f"Unknown speaker_id '{speaker_id}'.")

    length = 1.0 / max(float(payload.speed), 0.25)
    sample_rate, audio = model.infer(
        text=payload.input,
        language=Languages.JP,
        speaker_id=speaker_id,
        reference_audio_path=None,
        sdp_ratio=DEFAULT_SDP_RATIO,
        noise=DEFAULT_NOISE,
        noise_w=DEFAULT_NOISEW,
        length=length,
        line_split=False,
        split_interval=DEFAULT_SPLIT_INTERVAL,
        assist_text=None,
        assist_text_weight=DEFAULT_ASSIST_TEXT_WEIGHT,
        use_assist_text=False,
        style=STYLE_NAME,
        style_weight=DEFAULT_STYLE_WEIGHT,
    )

    with BytesIO() as wav_content:
        wavfile.write(wav_content, sample_rate, audio)
        audio_bytes = wav_content.getvalue()

    return Response(
        content=audio_bytes,
        media_type="audio/wav",
        headers={
            "Content-Length": str(len(audio_bytes)),
            "x-openai-model": payload.model,
            "x-openai-voice": str(speaker_id),
        },
    )
