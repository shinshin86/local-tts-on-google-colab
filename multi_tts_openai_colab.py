"""
Google Colab向け:
選択したローカルTTSを OpenAI互換 `/v1/audio/speech` として一時公開する単一セル用スクリプト。

想定用途:
- 動作確認
- 1ランタイムで1エンジンずつ起動
- 出力は WAV のみ

使い方:
1. このファイル全体を Colab のコードセルに貼り付けます。
2. フォームの値を選びます。
3. セルを実行します。
"""

#@title Multi Engine Local TTS -> OpenAI Compatible `/v1/audio/speech`
ENGINE = "Irodori-TTS"  #@param ["Irodori-TTS", "Kokoro", "MeloTTS", "Style-Bert-VITS2", "Piper"]
EXPOSE_PUBLIC_URL = True  #@param {type:"boolean"}
TEST_TEXT = "こんにちは。これは OpenAI 互換 TTS の動作確認です。"  #@param {type:"string"}
TEST_SPEED = 1.0  #@param {type:"number"}
TEST_VOICE = ""  #@param {type:"string"}
OPENAI_MODEL_ID = ""  #@param {type:"string"}

#@markdown ---
#@markdown Irodori-TTS
IRODORI_HF_CHECKPOINT = "Aratako/Irodori-TTS-500M"  #@param {type:"string"}
IRODORI_MODEL_PRECISION = "fp32"  #@param ["fp32", "bf16", "fp16"]
IRODORI_CODEC_PRECISION = "fp32"  #@param ["fp32", "bf16", "fp16"]

#@markdown ---
#@markdown Kokoro
KOKORO_DEFAULT_VOICE = "jf_alpha"  #@param ["jf_alpha", "jf_gongitsune", "jm_kumo", "af_heart", "af_bella", "am_adam", "bf_emma", "bm_george", "zf_xiaobei"]
KOKORO_DEFAULT_LANG_CODE = "j"  #@param ["j", "a", "b", "e", "f", "h", "i", "p", "z"]

#@markdown ---
#@markdown MeloTTS
MELO_LANGUAGE = "JP"  #@param ["JP", "EN", "ZH", "ES", "FR", "KR"]
MELO_DEFAULT_VOICE = "JP"  #@param ["JP", "EN-Default", "EN-US", "EN-BR", "EN_INDIA", "EN-AU", "ZH", "ES", "FR", "KR"]

#@markdown ---
#@markdown Style-Bert-VITS2
STYLE_BERT_MODEL_REPO = "litagin/style_bert_vits2_jvnv"  #@param {type:"string"}
STYLE_BERT_MODEL_SUBDIR = "jvnv-F2-jp"  #@param {type:"string"}
STYLE_BERT_MODEL_NAME = "jvnv-F2-jp"  #@param {type:"string"}
STYLE_BERT_SPEAKER_ID = 0  #@param {type:"integer"}
STYLE_BERT_STYLE = "Neutral"  #@param {type:"string"}

#@markdown ---
#@markdown Piper
PIPER_VOICE = "en_US-lessac-medium"  #@param {type:"string"}
PIPER_SPEAKER_ID = -1  #@param {type:"integer"}

import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path


KOKORO_VOICE_PRESETS = [
    "jf_alpha",
    "jf_gongitsune",
    "jm_kumo",
    "af_heart",
    "af_bella",
    "am_adam",
    "bf_emma",
    "bm_george",
    "zf_xiaobei",
]

MELO_VOICE_PRESETS = [
    "JP",
    "EN-Default",
    "EN-US",
    "EN-BR",
    "EN_INDIA",
    "EN-AU",
    "ZH",
    "ES",
    "FR",
    "KR",
]


ROOT_DIR = Path("/content/openai-compatible-local-tts")
ENGINES_DIR = ROOT_DIR / "engines"
LOG_DIR = ROOT_DIR / "logs"
OUTPUT_DIR = ROOT_DIR / "outputs"
CLOUDFLARED_PATH = ROOT_DIR / "cloudflared"
APP_PORT = 8000
PIPER_BACKEND_PORT = 5000

for path in (ROOT_DIR, ENGINES_DIR, LOG_DIR, OUTPUT_DIR):
    path.mkdir(parents=True, exist_ok=True)


IRODORI_APP_CODE = r'''
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
'''


KOKORO_APP_CODE = r'''
from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from kokoro import KPipeline
from pydantic import BaseModel

logger = logging.getLogger("uvicorn.error")

OPENAI_MODEL_ID = os.environ.get("OPENAI_MODEL_ID", "kokoro-82m")
DEFAULT_VOICE = os.environ.get("KOKORO_DEFAULT_VOICE", "jf_alpha")
DEFAULT_LANG_CODE = os.environ.get("KOKORO_DEFAULT_LANG_CODE", "j")

VOICE_PRESETS = [
    "af_heart",
    "af_bella",
    "am_adam",
    "bf_emma",
    "bm_george",
    "jf_alpha",
    "jf_gongitsune",
    "jm_kumo",
    "zf_xiaobei",
]

LANG_CODE_BY_PREFIX = {
    "a": "a",
    "b": "b",
    "e": "e",
    "f": "f",
    "h": "h",
    "i": "i",
    "j": "j",
    "p": "p",
    "z": "z",
}

app = FastAPI(title="Kokoro OpenAI Compatible TTS")

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


_pipelines = {}


def infer_lang_code(voice: str) -> str:
    prefix = (voice or DEFAULT_VOICE).split("_", 1)[0]
    return LANG_CODE_BY_PREFIX.get(prefix[:1], DEFAULT_LANG_CODE)


def get_pipeline(lang_code: str) -> KPipeline:
    pipeline = _pipelines.get(lang_code)
    if pipeline is None:
        pipeline = KPipeline(lang_code=lang_code)
        _pipelines[lang_code] = pipeline
    return pipeline


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception while serving request")
    return JSONResponse(
        status_code=500,
        content={"error": type(exc).__name__, "detail": str(exc)},
    )


@app.get("/")
def root():
    return {"ok": True, "engine": "Kokoro", "model": OPENAI_MODEL_ID}


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [{"id": OPENAI_MODEL_ID, "object": "model", "owned_by": "local"}],
    }


@app.get("/v1/voices")
def list_voices():
    return {
        "object": "list",
        "data": [{"id": voice, "object": "voice"} for voice in VOICE_PRESETS],
    }


@app.post("/v1/audio/speech")
async def audio_speech(payload: AudioSpeechRequest):
    if payload.response_format.lower() != "wav":
        raise HTTPException(status_code=400, detail="This wrapper currently supports only wav.")

    voice = payload.voice or DEFAULT_VOICE
    lang_code = infer_lang_code(voice)
    pipeline = get_pipeline(lang_code)
    generator = pipeline(payload.input, voice=voice, speed=float(payload.speed))
    chunks = [audio for _, _, audio in generator]
    if not chunks:
        raise HTTPException(status_code=500, detail="No audio was generated.")

    audio = np.concatenate(chunks)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        sf.write(tmp_path, audio, 24000)
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
'''


MELO_APP_CODE = r'''
from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from melo.api import TTS
from pydantic import BaseModel

logger = logging.getLogger("uvicorn.error")

OPENAI_MODEL_ID = os.environ.get("OPENAI_MODEL_ID", "melotts")
DEFAULT_LANGUAGE = os.environ.get("MELO_LANGUAGE", "JP")
DEFAULT_VOICE = os.environ.get("MELO_DEFAULT_VOICE", "JP")
DEVICE = os.environ.get("MELO_DEVICE", "auto")

app = FastAPI(title="MeloTTS OpenAI Compatible TTS")

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
    language: str | None = None


_models = {}


def get_model(language: str) -> TTS:
    model = _models.get(language)
    if model is None:
        model = TTS(language=language, device=DEVICE)
        _models[language] = model
    return model


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception while serving request")
    return JSONResponse(
        status_code=500,
        content={"error": type(exc).__name__, "detail": str(exc)},
    )


@app.get("/")
def root():
    return {"ok": True, "engine": "MeloTTS", "model": OPENAI_MODEL_ID}


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [{"id": OPENAI_MODEL_ID, "object": "model", "owned_by": "local"}],
    }


@app.get("/v1/voices")
def list_voices(language: str | None = None):
    target_language = language or DEFAULT_LANGUAGE
    model = get_model(target_language)
    voices = list(model.hps.data.spk2id.keys())
    return {
        "object": "list",
        "data": [{"id": voice, "object": "voice"} for voice in voices],
    }


@app.post("/v1/audio/speech")
async def audio_speech(payload: AudioSpeechRequest):
    if payload.response_format.lower() != "wav":
        raise HTTPException(status_code=400, detail="This wrapper currently supports only wav.")

    target_language = payload.language or DEFAULT_LANGUAGE
    model = get_model(target_language)
    speaker_ids = model.hps.data.spk2id
    voice = payload.voice or DEFAULT_VOICE or next(iter(speaker_ids))
    if voice not in speaker_ids:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown voice '{voice}'. Available: {', '.join(speaker_ids.keys())}",
        )

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        model.tts_to_file(payload.input, speaker_ids[voice], tmp_path, speed=float(payload.speed))
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
'''


STYLE_BERT_APP_CODE = r'''
from __future__ import annotations

import logging
import os
import tempfile
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
'''


PIPER_PROXY_APP_CODE = r'''
from __future__ import annotations

import logging
import os

import requests
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logger = logging.getLogger("uvicorn.error")

BACKEND_URL = os.environ.get("BACKEND_URL", "http://127.0.0.1:5000")
OPENAI_MODEL_ID = os.environ.get("OPENAI_MODEL_ID", "piper")
DEFAULT_VOICE = os.environ.get("PROXY_DEFAULT_VOICE", "")
DEFAULT_SPEAKER_ID = int(os.environ.get("PROXY_SPEAKER_ID", "-1"))
ENGINE_NAME = os.environ.get("PROXY_ENGINE", "Piper")

app = FastAPI(title=f"{ENGINE_NAME} OpenAI Compatible TTS")

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


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception while serving request")
    return JSONResponse(
        status_code=500,
        content={"error": type(exc).__name__, "detail": str(exc)},
    )


@app.get("/")
def root():
    return {"ok": True, "engine": ENGINE_NAME, "model": OPENAI_MODEL_ID, "backend": BACKEND_URL}


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [{"id": OPENAI_MODEL_ID, "object": "model", "owned_by": "local"}],
    }


@app.get("/v1/voices")
def list_voices():
    try:
        response = requests.get(f"{BACKEND_URL}/voices", timeout=30)
        response.raise_for_status()
        backend_json = response.json()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to fetch voices: {exc}")

    data = []
    if isinstance(backend_json, list):
        for item in backend_json:
            data.append({"id": str(item), "object": "voice"})
    elif isinstance(backend_json, dict):
        for key in backend_json.keys():
            data.append({"id": str(key), "object": "voice"})
    else:
        data.append({"id": str(backend_json), "object": "voice"})
    return {"object": "list", "data": data}


@app.post("/v1/audio/speech")
async def audio_speech(payload: AudioSpeechRequest):
    if payload.response_format.lower() != "wav":
        raise HTTPException(status_code=400, detail="This wrapper currently supports only wav.")

    backend_payload = {
        "text": payload.input,
        "length_scale": 1.0 / max(float(payload.speed), 0.25),
    }
    voice = payload.voice or DEFAULT_VOICE
    if voice:
        backend_payload["voice"] = voice
    if DEFAULT_SPEAKER_ID >= 0:
        backend_payload["speaker_id"] = DEFAULT_SPEAKER_ID

    try:
        response = requests.post(BACKEND_URL, json=backend_payload, timeout=180)
        response.raise_for_status()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to synthesize audio: {exc}")

    return Response(
        content=response.content,
        media_type="audio/wav",
        headers={
            "Content-Length": str(len(response.content)),
            "x-openai-model": payload.model,
            "x-openai-voice": voice,
        },
    )
'''


def run(
    cmd,
    *,
    cwd=None,
    env=None,
    check=True,
    capture_output=False,
):
    if isinstance(cmd, (list, tuple)):
        printable = shlex.join(str(part) for part in cmd)
    else:
        printable = cmd
    print(f"$ {printable}")
    return subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        check=check,
        text=True,
        capture_output=capture_output,
    )


def popen(cmd, *, cwd=None, env=None, log_path: Path):
    if isinstance(cmd, (list, tuple)):
        printable = shlex.join(str(part) for part in cmd)
    else:
        printable = cmd
    print(f"$ {printable}  > {log_path}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, "a", buffering=1, encoding="utf-8")
    return subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )


def wait_http(url: str, timeout: int = 180):
    start = time.time()
    while time.time() - start < timeout:
        completed = run(
            ["curl", "-fsS", url],
            check=False,
            capture_output=True,
        )
        if completed.returncode == 0:
            return True
        time.sleep(2)
    return False


def write_text(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def ensure_uv():
    run([sys.executable, "-m", "pip", "install", "-q", "-U", "uv"])


def ensure_cloudflared():
    if CLOUDFLARED_PATH.exists():
        return
    run(
        [
            "wget",
            "-q",
            "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64",
            "-O",
            str(CLOUDFLARED_PATH),
        ]
    )
    run(["chmod", "+x", str(CLOUDFLARED_PATH)])


def kill_old_processes():
    patterns = [
        "uvicorn app:app",
        "uvicorn openai_wrapper_app:app",
        "python -m piper.http_server",
        "cloudflared tunnel",
        "server_fastapi.py",
    ]
    for pattern in patterns:
        run(["pkill", "-f", pattern], check=False)
    run(["bash", "-lc", f"fuser -k {APP_PORT}/tcp || true"], check=False)
    run(["bash", "-lc", f"fuser -k {PIPER_BACKEND_PORT}/tcp || true"], check=False)


def ensure_git_clone(repo_url: str, target_dir: Path):
    if target_dir.exists():
        print(f"reuse: {target_dir}")
        return
    run(["git", "clone", repo_url, str(target_dir)])


def ensure_venv(engine_dir: Path) -> Path:
    venv_dir = engine_dir / ".venv"
    if not venv_dir.exists():
        run(["uv", "venv", str(venv_dir)])
    return venv_dir / "bin" / "python"


def uv_pip_install(python_bin: Path, packages, *, cwd=None):
    run(["uv", "pip", "install", "--python", str(python_bin), *packages], cwd=cwd)


def tail_log(log_path: Path, lines: int = 80):
    if not log_path.exists():
        print(f"log not found: {log_path}")
        return
    print(f"\n=== tail: {log_path.name} ===")
    completed = run(["tail", "-n", str(lines), str(log_path)], capture_output=True)
    print(completed.stdout)


def pretty_print_json_url(url: str, title: str):
    print(f"\n=== {title} ===")
    completed = run(["curl", "-fsS", url], capture_output=True, check=False)
    if completed.returncode != 0:
        print(f"failed to fetch: {url}")
        return
    try:
        print(json.dumps(json.loads(completed.stdout), ensure_ascii=False, indent=2))
    except json.JSONDecodeError:
        print(completed.stdout)


def resolve_selected_voice() -> str:
    if TEST_VOICE:
        return TEST_VOICE
    if ENGINE == "Kokoro":
        return KOKORO_DEFAULT_VOICE
    if ENGINE == "MeloTTS":
        return MELO_DEFAULT_VOICE
    return ""


def print_engine_voice_hints():
    print("\n=== Voice Selection Hint ===")
    if ENGINE == "Kokoro":
        print("Kokoro はフォームで音声を選択できます。")
        print("候補:", ", ".join(KOKORO_VOICE_PRESETS))
    elif ENGINE == "MeloTTS":
        print("MeloTTS はフォームで代表的な voice を選択できます。")
        print("候補:", ", ".join(MELO_VOICE_PRESETS))
        print(f"現在の language: {MELO_LANGUAGE}")
    elif ENGINE == "Style-Bert-VITS2":
        print("Style-Bert-VITS2 は speaker_id ベースです。起動後の /v1/voices を確認して speaker_id を選んでください。")
    elif ENGINE == "Piper":
        print("Piper は voice 名がそのままモデル指定です。必要なら PIPER_VOICE を変更してください。")
    else:
        print("Irodori-TTS は現状 voice 切り替えを持たない想定です。")


def launch_cloudflared() -> str | None:
    ensure_cloudflared()
    proc = subprocess.Popen(
        [str(CLOUDFLARED_PATH), "tunnel", "--url", f"http://127.0.0.1:{APP_PORT}", "--no-autoupdate"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=str(ROOT_DIR),
    )
    public_url = None
    start = time.time()
    while time.time() - start < 60:
        line = proc.stdout.readline()
        if line:
            print(line, end="")
            match = re.search(r"https://[-a-zA-Z0-9]+\.trycloudflare\.com", line)
            if match:
                public_url = match.group(0)
                break
    return public_url


def install_irodori() -> dict:
    repo_dir = ENGINES_DIR / "Irodori-TTS"
    ensure_git_clone("https://github.com/Aratako/Irodori-TTS", repo_dir)
    write_text(repo_dir / "app.py", IRODORI_APP_CODE)
    run(["uv", "sync"], cwd=str(repo_dir))
    python_bin = repo_dir / ".venv" / "bin" / "python"
    uv_pip_install(
        python_bin,
        ["fastapi", "uvicorn", "huggingface_hub"],
        cwd=str(repo_dir),
    )
    uv_pip_install(
        python_bin,
        ["git+https://github.com/facebookresearch/dacvae.git"],
        cwd=str(repo_dir),
    )
    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "IRODORI_HF_CHECKPOINT": IRODORI_HF_CHECKPOINT,
        "IRODORI_MODEL_PRECISION": IRODORI_MODEL_PRECISION,
        "IRODORI_CODEC_PRECISION": IRODORI_CODEC_PRECISION,
        "OPENAI_MODEL_ID": OPENAI_MODEL_ID or IRODORI_HF_CHECKPOINT,
    }
    proc = popen(
        [str(repo_dir / ".venv" / "bin" / "uvicorn"), "app:app", "--host", "0.0.0.0", "--port", str(APP_PORT), "--log-level", "debug", "--access-log"],
        cwd=str(repo_dir),
        env=env,
        log_path=LOG_DIR / "irodori-uvicorn.log",
    )
    return {"proc": proc, "app_dir": repo_dir, "log_path": LOG_DIR / "irodori-uvicorn.log"}


def install_kokoro() -> dict:
    engine_dir = ENGINES_DIR / "kokoro-openai"
    engine_dir.mkdir(parents=True, exist_ok=True)
    python_bin = ensure_venv(engine_dir)
    run(["apt-get", "-qq", "-y", "install", "espeak-ng"])
    uv_pip_install(
        python_bin,
        ["fastapi", "uvicorn", "kokoro>=0.9.2", "soundfile", "numpy", "misaki[ja]", "misaki[zh]", "misaki[en]", "unidic"],
    )
    run([str(python_bin), "-m", "unidic", "download"], cwd=str(engine_dir))
    write_text(engine_dir / "app.py", KOKORO_APP_CODE)
    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "OPENAI_MODEL_ID": OPENAI_MODEL_ID or "kokoro-82m",
        "KOKORO_DEFAULT_VOICE": KOKORO_DEFAULT_VOICE,
        "KOKORO_DEFAULT_LANG_CODE": KOKORO_DEFAULT_LANG_CODE,
    }
    proc = popen(
        [str(engine_dir / ".venv" / "bin" / "uvicorn"), "app:app", "--host", "0.0.0.0", "--port", str(APP_PORT), "--log-level", "info"],
        cwd=str(engine_dir),
        env=env,
        log_path=LOG_DIR / "kokoro-uvicorn.log",
    )
    return {"proc": proc, "app_dir": engine_dir, "log_path": LOG_DIR / "kokoro-uvicorn.log"}


def install_melo() -> dict:
    # NOTE: MeloTTS は現在 Colab (uv + venv) 環境でのインストールに問題があります。
    # MeloTTS が古い transformers==4.27.4 / tokenizers==0.13.3 に依存しており、
    # tokenizers のビルドに Rust コンパイラが必要、かつ fugashi のビルドに libmecab-dev が必要です。
    # --no-deps で回避を試みても pykakasi 等の未宣言依存が多く、安定動作しません。
    # MeloTTS 側の依存管理が改善されるまで、このエンジンは動作しない可能性があります。
    repo_dir = ENGINES_DIR / "MeloTTS"
    ensure_git_clone("https://github.com/myshell-ai/MeloTTS.git", repo_dir)
    python_bin = ensure_venv(repo_dir)
    uv_pip_install(
        python_bin,
        ["-e", ".", "fastapi", "uvicorn", "huggingface_hub"],
        cwd=str(repo_dir),
    )
    run([str(python_bin), "-m", "unidic", "download"], cwd=str(repo_dir))
    write_text(repo_dir / "openai_wrapper_app.py", MELO_APP_CODE)
    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "OPENAI_MODEL_ID": OPENAI_MODEL_ID or f"melotts-{MELO_LANGUAGE.lower()}",
        "MELO_LANGUAGE": MELO_LANGUAGE,
        "MELO_DEFAULT_VOICE": MELO_DEFAULT_VOICE,
        "MELO_DEVICE": "auto",
    }
    proc = popen(
        [str(repo_dir / ".venv" / "bin" / "uvicorn"), "openai_wrapper_app:app", "--host", "0.0.0.0", "--port", str(APP_PORT), "--log-level", "info"],
        cwd=str(repo_dir),
        env=env,
        log_path=LOG_DIR / "melo-uvicorn.log",
    )
    return {"proc": proc, "app_dir": repo_dir, "log_path": LOG_DIR / "melo-uvicorn.log"}


def install_style_bert() -> dict:
    # NOTE: Style-Bert-VITS2 は現在 Colab (uv + venv) 環境でのセットアップに問題があります。
    # 1. pyopenjtalk が pkg_resources に依存 → setuptools v82 で削除済み → setuptools<81 が必要
    # 2. `-e .` インストールで torch が正しく入らない → transformers が torch を認識できない
    # 3. scipy も別途インストールが必要
    # venv 内の依存整合性を取るのが困難なため、現状このエンジンは動作しない可能性があります。
    repo_dir = ENGINES_DIR / "Style-Bert-VITS2"
    ensure_git_clone("https://github.com/litagin02/Style-Bert-VITS2.git", repo_dir)
    python_bin = ensure_venv(repo_dir)
    uv_pip_install(
        python_bin,
        ["-e", ".", "fastapi", "uvicorn", "huggingface_hub", "scipy"],
        cwd=str(repo_dir),
    )
    download_code = (
        "from huggingface_hub import snapshot_download; "
        f"snapshot_download(repo_id={STYLE_BERT_MODEL_REPO!r}, local_dir={str(repo_dir / 'model_assets')!r}, allow_patterns={[f'{STYLE_BERT_MODEL_SUBDIR}/*']!r})"
    )
    run([str(python_bin), "-c", download_code], cwd=str(repo_dir))
    write_text(repo_dir / "openai_wrapper_app.py", STYLE_BERT_APP_CODE)
    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "OPENAI_MODEL_ID": OPENAI_MODEL_ID or STYLE_BERT_MODEL_NAME,
        "STYLE_BERT_MODEL_ROOT": str(repo_dir / "model_assets"),
        "STYLE_BERT_MODEL_NAME": STYLE_BERT_MODEL_NAME,
        "STYLE_BERT_SPEAKER_ID": str(STYLE_BERT_SPEAKER_ID),
        "STYLE_BERT_STYLE": STYLE_BERT_STYLE,
        "STYLE_BERT_DEVICE": "cuda" if os.path.exists("/usr/bin/nvidia-smi") else "cpu",
    }
    proc = popen(
        [str(repo_dir / ".venv" / "bin" / "uvicorn"), "openai_wrapper_app:app", "--host", "0.0.0.0", "--port", str(APP_PORT), "--log-level", "info"],
        cwd=str(repo_dir),
        env=env,
        log_path=LOG_DIR / "style-bert-uvicorn.log",
    )
    return {"proc": proc, "app_dir": repo_dir, "log_path": LOG_DIR / "style-bert-uvicorn.log"}


def install_piper() -> dict:
    engine_dir = ENGINES_DIR / "piper-openai"
    engine_dir.mkdir(parents=True, exist_ok=True)
    python_bin = ensure_venv(engine_dir)
    uv_pip_install(
        python_bin,
        ["fastapi", "uvicorn", "requests", "piper-tts[http]"],
    )
    run([str(python_bin), "-m", "piper.download_voices", PIPER_VOICE], cwd=str(engine_dir))
    backend_proc = popen(
        [str(python_bin), "-m", "piper.http_server", "-m", PIPER_VOICE, "--host", "127.0.0.1", "--port", str(PIPER_BACKEND_PORT)],
        cwd=str(engine_dir),
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
        log_path=LOG_DIR / "piper-backend.log",
    )
    if not wait_http(f"http://127.0.0.1:{PIPER_BACKEND_PORT}/voices", timeout=120):
        tail_log(LOG_DIR / "piper-backend.log")
        raise RuntimeError("Piper backend did not become ready.")
    write_text(engine_dir / "app.py", PIPER_PROXY_APP_CODE)
    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "BACKEND_URL": f"http://127.0.0.1:{PIPER_BACKEND_PORT}",
        "OPENAI_MODEL_ID": OPENAI_MODEL_ID or PIPER_VOICE,
        "PROXY_DEFAULT_VOICE": PIPER_VOICE,
        "PROXY_SPEAKER_ID": str(PIPER_SPEAKER_ID),
        "PROXY_ENGINE": "Piper",
    }
    proc = popen(
        [str(engine_dir / ".venv" / "bin" / "uvicorn"), "app:app", "--host", "0.0.0.0", "--port", str(APP_PORT), "--log-level", "info"],
        cwd=str(engine_dir),
        env=env,
        log_path=LOG_DIR / "piper-proxy-uvicorn.log",
    )
    return {
        "proc": proc,
        "backend_proc": backend_proc,
        "app_dir": engine_dir,
        "log_path": LOG_DIR / "piper-proxy-uvicorn.log",
        "backend_log_path": LOG_DIR / "piper-backend.log",
    }


def synth_test_wav(base_url: str):
    output_path = OUTPUT_DIR / f"{ENGINE.lower().replace(' ', '-').replace('/', '-')}.wav"
    selected_voice = resolve_selected_voice()
    payload = {
        "model": OPENAI_MODEL_ID or ENGINE,
        "input": TEST_TEXT,
        "speed": TEST_SPEED,
        "response_format": "wav",
    }
    if selected_voice:
        payload["voice"] = selected_voice
    run(
        [
            "curl",
            "-sS",
            "-X",
            "POST",
            f"{base_url}/v1/audio/speech",
            "-H",
            "Content-Type: application/json",
            "-d",
            json.dumps(payload, ensure_ascii=False),
            "--output",
            str(output_path),
        ]
    )
    print(f"test wav: {output_path}")


def main():
    print(f"engine: {ENGINE}")
    ensure_uv()
    kill_old_processes()
    print_engine_voice_hints()

    installers = {
        "Irodori-TTS": install_irodori,
        "Kokoro": install_kokoro,
        "MeloTTS": install_melo,
        "Style-Bert-VITS2": install_style_bert,
        "Piper": install_piper,
    }
    state = installers[ENGINE]()

    if not wait_http(f"http://127.0.0.1:{APP_PORT}/", timeout=180):
        tail_log(state["log_path"])
        if "backend_log_path" in state:
            tail_log(state["backend_log_path"])
        raise RuntimeError(f"{ENGINE} OpenAI wrapper did not become ready.")

    local_base_url = f"http://127.0.0.1:{APP_PORT}"
    print("\n=== Local Ready ===")
    print("Base URL :", local_base_url + "/v1")
    print("Speech   :", local_base_url + "/v1/audio/speech")
    print("Models   :", local_base_url + "/v1/models")
    print("Voices   :", local_base_url + "/v1/voices")

    pretty_print_json_url(local_base_url + "/", "Root")
    pretty_print_json_url(local_base_url + "/v1/models", "Models")
    pretty_print_json_url(local_base_url + "/v1/voices", "Voices")
    synth_test_wav(local_base_url)

    public_url = None
    if EXPOSE_PUBLIC_URL:
        public_url = launch_cloudflared()

    print("\n=== Log Tail ===")
    tail_log(state["log_path"], lines=60)
    if "backend_log_path" in state:
        tail_log(state["backend_log_path"], lines=40)

    if public_url:
        print("\n=== Public Ready ===")
        print("Base URL :", public_url + "/v1")
        print("Speech   :", public_url + "/v1/audio/speech")
        print("\nTest curl:")
        print(
            "curl -X POST "
            + shlex.quote(public_url + "/v1/audio/speech")
            + " -H 'Content-Type: application/json' "
            + "-d "
            + shlex.quote(
                json.dumps(
                    {
                        "model": OPENAI_MODEL_ID or ENGINE,
                        "input": TEST_TEXT,
                        "speed": TEST_SPEED,
                        "response_format": "wav",
                        **({"voice": resolve_selected_voice()} if resolve_selected_voice() else {}),
                    },
                    ensure_ascii=False,
                )
            )
            + " --output out.wav"
        )
    elif EXPOSE_PUBLIC_URL:
        print("cloudflared のURL取得に失敗しました。ログを確認してください。")


main()
