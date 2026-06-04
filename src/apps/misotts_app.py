from __future__ import annotations

import io
import logging
import os
import sys

import soundfile as sf
import torch
import torchaudio
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

MISOTTS_REPO_DIR = os.environ.get("MISOTTS_REPO_DIR", "")
if MISOTTS_REPO_DIR:
    sys.path.insert(0, MISOTTS_REPO_DIR)

from generator import DEFAULT_MISO_TTS_REPO_ID, Segment, load_miso_8b  # noqa: E402

logger = logging.getLogger("uvicorn.error")

OPENAI_MODEL_ID = os.environ.get("OPENAI_MODEL_ID", "miso-tts-8b")
HF_MODEL = os.environ.get("MISOTTS_HF_MODEL", "") or DEFAULT_MISO_TTS_REPO_ID
DEFAULT_VOICE = os.environ.get("MISOTTS_DEFAULT_VOICE", "default")
DEFAULT_SPEAKER = int(os.environ.get("MISOTTS_DEFAULT_SPEAKER", "0"))
PROMPT_WAV = os.environ.get("MISOTTS_PROMPT_WAV", "")
PROMPT_TEXT = os.environ.get("MISOTTS_PROMPT_TEXT", "")
MAX_AUDIO_LENGTH_MS = float(os.environ.get("MISOTTS_MAX_AUDIO_LENGTH_MS", "30000"))
TEMPERATURE = float(os.environ.get("MISOTTS_TEMPERATURE", "0.9"))
TOPK = int(os.environ.get("MISOTTS_TOPK", "50"))
# Upstream generator.py hardcodes AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B"),
# which is HF-gated. Redirect it to an ungated mirror that ships byte-identical tokenizer
# files (same 128k Llama 3 vocab, same special-token ids) so the engine runs without the HF
# access-request gate / HF_TOKEN. The Llama 3.2 Community License still governs the tokenizer.
# Set MISOTTS_TOKENIZER_REPO="meta-llama/Llama-3.2-1B" (with HF_TOKEN) to use the official source.
TOKENIZER_REPO = os.environ.get("MISOTTS_TOKENIZER_REPO", "unsloth/Llama-3.2-1B")


def _redirect_llama_tokenizer():
    if not TOKENIZER_REPO or TOKENIZER_REPO == "meta-llama/Llama-3.2-1B":
        return
    import transformers

    _orig = transformers.AutoTokenizer.from_pretrained

    def _patched(name, *args, **kwargs):
        if name == "meta-llama/Llama-3.2-1B":
            logger.info("Redirecting Llama tokenizer %s -> %s", name, TOKENIZER_REPO)
            name = TOKENIZER_REPO
        return _orig(name, *args, **kwargs)

    transformers.AutoTokenizer.from_pretrained = _patched


app = FastAPI(title="MisoTTS 8B OpenAI Compatible TTS")

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


_generator = None


def get_generator():
    global _generator
    if _generator is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _redirect_llama_tokenizer()
        logger.info("Loading MisoTTS 8B (%s) on %s", HF_MODEL, device)
        _generator = load_miso_8b(device=device, model_path_or_repo_id=HF_MODEL)
        if device == "cuda":
            # The Generator ctor calls setup_caches() AFTER the model was already
            # moved to CUDA, and torchtune's KVCache allocates k_cache/v_cache/
            # cache_pos with dtype only (no device) -> they land on CPU and trigger
            # "Expected all tensors to be on the same device" in kv_cache.update.
            # Re-issue .to(device) to relocate those cache buffers (params and the
            # causal masks are already on device, so this is a no-op for them).
            _generator._model.to(device)
        logger.info("MisoTTS 8B loaded (sample_rate=%d)", _generator.sample_rate)
    return _generator


def _load_prompt_segment(generator, wav_path: str, text: str, speaker: int) -> Segment:
    audio, sr = torchaudio.load(wav_path)
    # Collapse to mono and resample to the generator's native rate (24 kHz).
    audio = audio.mean(dim=0)
    if sr != generator.sample_rate:
        audio = torchaudio.functional.resample(
            audio, orig_freq=sr, new_freq=generator.sample_rate
        )
    return Segment(speaker=speaker, text=text, audio=audio)


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
        "engine": "MisoTTS-8B",
        "model": OPENAI_MODEL_ID,
        "hf_model": HF_MODEL,
        "default_voice": DEFAULT_VOICE,
        "default_speaker": DEFAULT_SPEAKER,
        "clone_ready": bool(PROMPT_WAV),
    }


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [{"id": OPENAI_MODEL_ID, "object": "model", "owned_by": "misolabs"}],
    }


@app.get("/v1/voices")
def list_voices():
    voices = [
        {"id": "default", "object": "voice", "speaker_id": DEFAULT_SPEAKER},
        {"id": "speaker_0", "object": "voice", "speaker_id": 0},
        {"id": "speaker_1", "object": "voice", "speaker_id": 1},
    ]
    if PROMPT_WAV:
        voices.append({"id": "clone", "object": "voice", "prompt_wav": PROMPT_WAV})
    return {"object": "list", "data": voices}


def _resolve_speaker(voice: str) -> int:
    if voice == "default":
        return DEFAULT_SPEAKER
    if voice.startswith("speaker_"):
        try:
            return int(voice.split("_", 1)[1])
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid voice: {voice}. Expected speaker_<int>.",
            ) from exc
    raise HTTPException(
        status_code=400,
        detail=f"Unknown voice: {voice}. Available: default, clone, speaker_<int>",
    )


@app.post("/v1/audio/speech")
async def audio_speech(payload: AudioSpeechRequest):
    if payload.response_format.lower() != "wav":
        raise HTTPException(status_code=400, detail="This wrapper currently supports only wav.")

    voice = payload.voice or DEFAULT_VOICE
    generator = get_generator()

    context: list[Segment] = []
    if voice == "clone":
        if not PROMPT_WAV:
            raise HTTPException(
                status_code=400,
                detail="voice='clone' requires MISOTTS_PROMPT_WAV "
                "(set --misotts-prompt-wav). Use voice='default' otherwise.",
            )
        speaker = DEFAULT_SPEAKER
        context = [_load_prompt_segment(generator, PROMPT_WAV, PROMPT_TEXT, speaker)]
    else:
        speaker = _resolve_speaker(voice)

    audio_tensor = generator.generate(
        text=payload.input,
        speaker=speaker,
        context=context,
        max_audio_length_ms=MAX_AUDIO_LENGTH_MS,
        temperature=TEMPERATURE,
        topk=TOPK,
    )

    if audio_tensor is None or audio_tensor.numel() == 0:
        raise HTTPException(status_code=500, detail="No audio was generated.")

    audio_np = audio_tensor.detach().cpu().numpy().astype("float32").squeeze()

    buf = io.BytesIO()
    sf.write(buf, audio_np, generator.sample_rate, format="WAV", subtype="PCM_16")
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
