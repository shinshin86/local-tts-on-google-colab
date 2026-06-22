from __future__ import annotations

import io
import logging
import os

import soundfile as sf
import torch
import torchaudio
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from transformers import (
    AutoFeatureExtractor,
    AutoModelForCausalLM,
    AutoTokenizer,
    MimiModel,
)

logger = logging.getLogger("uvicorn.error")

OPENAI_MODEL_ID = os.environ.get("OPENAI_MODEL_ID", "vyvo-multilingual")
HF_MODEL = os.environ.get("VYVO_HF_MODEL", "Vyvo/Vyvo-Multilingual-v0.1")
MIMI_REPO = os.environ.get("VYVO_MIMI_REPO", "kyutai/mimi")
DEFAULT_VOICE = os.environ.get("VYVO_DEFAULT_VOICE", "clone")
PROMPT_WAV = os.environ.get("VYVO_PROMPT_WAV", "")
PROMPT_TEXT = os.environ.get("VYVO_PROMPT_TEXT", "")
TEMPERATURE = float(os.environ.get("VYVO_TEMPERATURE", "0.7"))
TOP_P = float(os.environ.get("VYVO_TOP_P", "0.9"))
REPETITION_PENALTY = float(os.environ.get("VYVO_REPETITION_PENALTY", "1.1"))
MAX_NEW_TOKENS = int(os.environ.get("VYVO_MAX_NEW_TOKENS", "9600"))
MIN_NEW_TOKENS = int(os.environ.get("VYVO_MIN_NEW_TOKENS", "960"))

# Token layout (must match Vyvo training; lifted verbatim from the model card).
# Qwen3 base vocab ends at BASE; audio ids and the special tokens live above it.
BASE = 151669
NUM_CODEBOOKS = 32
CODEBOOK_SIZE = 2048
AUDIO_OFFSET = 10
SOS, EOS, SOH, EOH, SOA = 1, 2, 3, 4, 5  # offsets above BASE

app = FastAPI(title="Vyvo-Multilingual OpenAI Compatible TTS")

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


class _Runtime:
    def __init__(self, tokenizer, model, mimi, feature_extractor, device):
        self.tokenizer = tokenizer
        self.model = model
        self.mimi = mimi
        self.feature_extractor = feature_extractor
        self.device = device

    @property
    def sample_rate(self) -> int:
        return int(self.mimi.config.sampling_rate)


_runtime: _Runtime | None = None


def get_runtime() -> _Runtime:
    global _runtime
    if _runtime is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        logger.info("Loading Vyvo (%s) + Mimi (%s) on %s", HF_MODEL, MIMI_REPO, device)
        tokenizer = AutoTokenizer.from_pretrained(HF_MODEL)
        model = AutoModelForCausalLM.from_pretrained(HF_MODEL, dtype=dtype).to(device).eval()
        mimi = MimiModel.from_pretrained(MIMI_REPO).to(device).eval()
        feature_extractor = AutoFeatureExtractor.from_pretrained(MIMI_REPO)
        _runtime = _Runtime(tokenizer, model, mimi, feature_extractor, device)
        logger.info("Vyvo loaded (sample_rate=%d)", _runtime.sample_rate)
    return _runtime


def _encode_reference(rt: _Runtime, path: str) -> list[int]:
    # Load a wav and encode it to Mimi audio tokens (as LM token ids).
    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    target_sr = rt.feature_extractor.sampling_rate
    if sr != target_sr:
        wav = (
            torchaudio.functional.resample(torch.from_numpy(wav).unsqueeze(0), sr, target_sr)
            .squeeze(0)
            .numpy()
        )

    inputs = rt.feature_extractor(raw_audio=wav, sampling_rate=target_sr, return_tensors="pt")
    codes = (
        rt.mimi.encode(inputs["input_values"].to(rt.device), num_quantizers=NUM_CODEBOOKS)
        .audio_codes[0]
        .cpu()
    )

    frame_interleaved = codes.transpose(0, 1).reshape(-1).tolist()
    return [
        code + AUDIO_OFFSET + (i % NUM_CODEBOOKS) * CODEBOOK_SIZE + BASE
        for i, code in enumerate(frame_interleaved)
    ]


def _build_prompt(rt: _Runtime, reference_tokens: list[int], reference_text: str, target_text: str) -> list[int]:
    # [SOH] ref_text + target_text [eot] [EOH] [SOA] [SOS] <reference audio>
    text_ids = rt.tokenizer(reference_text + " " + target_text, add_special_tokens=False).input_ids
    head = [BASE + SOH] + text_ids + [rt.tokenizer.eos_token_id, BASE + EOH, BASE + SOA, BASE + SOS]
    return head + reference_tokens


def _decode_audio(rt: _Runtime, generated_ids: list[int]):
    # Turn generated LM token ids back into a waveform; stop at first invalid id.
    codes: list[int] = []
    for i, token in enumerate(generated_ids):
        value = token - BASE - AUDIO_OFFSET - (i % NUM_CODEBOOKS) * CODEBOOK_SIZE
        if 0 <= value < CODEBOOK_SIZE:
            codes.append(value)
        else:
            break

    frames = len(codes) // NUM_CODEBOOKS
    if frames == 0:
        return None
    codes_t = torch.tensor(codes[: frames * NUM_CODEBOOKS]).view(frames, NUM_CODEBOOKS)
    codes_t = codes_t.t().unsqueeze(0).to(rt.device)
    return rt.mimi.decode(codes_t).audio_values.detach().squeeze().cpu().float().numpy()


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
        "engine": "Vyvo-Multilingual",
        "model": OPENAI_MODEL_ID,
        "hf_model": HF_MODEL,
        "mimi_repo": MIMI_REPO,
        "default_voice": DEFAULT_VOICE,
        "clone_ready": bool(PROMPT_WAV and PROMPT_TEXT),
    }


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [{"id": OPENAI_MODEL_ID, "object": "model", "owned_by": "vyvo"}],
    }


@app.get("/v1/voices")
def list_voices():
    # Vyvo has no built-in speaker — it always clones from a reference clip.
    # `default` and `clone` take the same code path and both need a reference.
    voices = [{"id": "clone", "object": "voice"}]
    if PROMPT_WAV and PROMPT_TEXT:
        voices[0]["ref"] = PROMPT_WAV
    return {"object": "list", "data": voices}


@app.post("/v1/audio/speech")
async def audio_speech(payload: AudioSpeechRequest):
    if payload.response_format.lower() != "wav":
        raise HTTPException(status_code=400, detail="This wrapper currently supports only wav.")

    voice = payload.voice or DEFAULT_VOICE
    if voice not in ("default", "clone"):
        raise HTTPException(
            status_code=400,
            detail=f"Unknown voice: {voice}. Available: default, clone",
        )

    # Vyvo is fundamentally a zero-shot cloning model with no built-in speaker:
    # every request needs a reference audio + its transcript.
    if not PROMPT_WAV or not PROMPT_TEXT:
        raise HTTPException(
            status_code=400,
            detail=(
                "Vyvo-Multilingual requires a reference audio. "
                "Pass --vyvo-prompt-wav and --vyvo-prompt-text at startup, "
                "then call with voice='clone'."
            ),
        )

    rt = get_runtime()
    with torch.inference_mode():
        reference_tokens = _encode_reference(rt, PROMPT_WAV)
        prompt = _build_prompt(rt, reference_tokens, PROMPT_TEXT, payload.input)
        input_ids = torch.tensor([prompt], device=rt.device)
        output = rt.model.generate(
            input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=MAX_NEW_TOKENS,
            min_new_tokens=MIN_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            repetition_penalty=REPETITION_PENALTY,
            eos_token_id=BASE + EOS,
            pad_token_id=rt.tokenizer.eos_token_id,
        )
        audio_np = _decode_audio(rt, output[0, input_ids.shape[1] :].tolist())
    if audio_np is None or audio_np.size == 0:
        raise HTTPException(status_code=500, detail="No audio was generated.")

    buf = io.BytesIO()
    sf.write(buf, audio_np, rt.sample_rate, format="WAV", subtype="PCM_16")
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
