from __future__ import annotations

import io
import math
import os
from threading import Lock

import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoModel


OPENAI_MODEL_ID = os.environ.get("OPENAI_MODEL_ID", "zonos3-mini")
MODEL_DIR = os.environ["ZONOS3_MINI_MODEL_DIR"]
LANGUAGE = os.environ.get("ZONOS3_MINI_LANGUAGE", "ja").strip().lower().replace("-", "_")
PROMPT_WAV = os.environ.get("ZONOS3_MINI_PROMPT_WAV", "")
DEFAULT_VOICE = os.environ.get("ZONOS3_MINI_DEFAULT_VOICE", "default")
SPEAKING_RATE = float(os.environ.get("ZONOS3_MINI_SPEAKING_RATE", "0"))
SPEAKER_CFG_SCALE = float(os.environ.get("ZONOS3_MINI_SPEAKER_CFG_SCALE", "1"))
RELEASE_VOICE = "Savannah"

if not math.isfinite(SPEAKING_RATE) or SPEAKING_RATE < 0:
    raise ValueError("--zonos3-mini-speaking-rate must be finite and nonnegative.")
if not math.isfinite(SPEAKER_CFG_SCALE) or SPEAKER_CFG_SCALE < 0:
    raise ValueError("--zonos3-mini-speaker-cfg-scale must be finite and nonnegative.")
if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(including_emulation=False):
    raise RuntimeError("ZONOS3-Mini requires a CUDA GPU with native bfloat16 support, such as an L4.")

model = AutoModel.from_pretrained(
    MODEL_DIR, trust_remote_code=True, local_files_only=True, dtype=torch.bfloat16,
).to("cuda").eval()
if LANGUAGE not in model.config.generation["languages"]:
    raise ValueError(f"Unsupported --zonos3-mini-language: {LANGUAGE!r}")

# Load auxiliary assets before accepting requests and serialize GPU generation.
_ = model.tokenizer
model.voice(RELEASE_VOICE)
generation_lock = Lock()

app = FastAPI(title="ZONOS3-Mini OpenAI Compatible TTS")
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


@app.get("/")
def root():
    return {
        "ok": True,
        "engine": "ZONOS3-Mini",
        "model": OPENAI_MODEL_ID,
        "language": LANGUAGE,
        "default_voice": DEFAULT_VOICE,
    }


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [{"id": OPENAI_MODEL_ID, "object": "model", "owned_by": "zyphra"}],
    }


@app.get("/v1/voices")
def list_voices():
    voices = [{"id": "default", "object": "voice", "language": LANGUAGE}]
    if PROMPT_WAV:
        voices.append({"id": "clone", "object": "voice", "language": LANGUAGE})
    return {"object": "list", "data": voices}


@app.post("/v1/audio/speech")
def audio_speech(payload: AudioSpeechRequest):
    if payload.response_format.lower() != "wav":
        raise HTTPException(status_code=400, detail="ZONOS3-Mini supports only WAV output.")
    if payload.speed != 1.0:
        raise HTTPException(
            status_code=400,
            detail="Use speed=1.0 and configure pace with --zonos3-mini-speaking-rate.",
        )
    if not payload.input.strip():
        raise HTTPException(status_code=400, detail="Input text is required.")

    voice = payload.voice or DEFAULT_VOICE
    if voice not in {"default", "clone"}:
        raise HTTPException(status_code=400, detail="Use voice='default' or voice='clone'.")
    if voice == "clone":
        if not PROMPT_WAV:
            raise HTTPException(
                status_code=400,
                detail="voice='clone' requires --zonos3-mini-prompt-wav at startup.",
            )
        try:
            with sf.SoundFile(PROMPT_WAV) as ref:
                sample_rate = ref.samplerate
                wav = ref.read(frames=60 * sample_rate, dtype="float32", always_2d=True)
        except (OSError, sf.SoundFileError) as exc:
            raise HTTPException(status_code=400, detail="Could not read the configured reference audio.") from exc
        if not wav.size:
            raise HTTPException(status_code=400, detail="The configured reference audio is empty.")
        speaker = {"speaker_audio": wav.T, "speaker_sample_rate": sample_rate}
    else:
        speaker = {"voice": RELEASE_VOICE}

    condition_values = {"speaking_rate": SPEAKING_RATE} if SPEAKING_RATE else None
    try:
        with generation_lock:
            result = model.generate(
                payload.input,
                language=LANGUAGE,
                speaker_cfg_scale=SPEAKER_CFG_SCALE,
                condition_values=condition_values,
                **speaker,
            )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not result.waveform.numel():
        raise HTTPException(status_code=500, detail="No audio was generated.")

    buffer = io.BytesIO()
    sf.write(buffer, result.waveform.cpu().numpy(), result.sampling_rate, format="WAV", subtype="PCM_16")
    return Response(
        content=buffer.getvalue(),
        media_type="audio/wav",
        headers={"x-openai-model": payload.model, "x-openai-voice": voice},
    )
