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
from huggingface_hub import snapshot_download
from pydantic import BaseModel
from transformers import AutoTokenizer

# uvicorn runs with cwd = the cloned Ming-omni-tts repo root, so these top-level
# modules (and, for the MoE checkpoint, the tokenizer files) resolve directly.
from modeling_bailingmm import BailingMMNativeForConditionalGeneration
from spkemb_extractor import SpkembExtractor

logger = logging.getLogger("uvicorn.error")

OPENAI_MODEL_ID = os.environ.get("OPENAI_MODEL_ID", "ming-omni-tts")
HF_MODEL = os.environ.get("MING_OMNI_TTS_HF_MODEL", "inclusionAI/Ming-omni-tts-16.8B-A3B")
DEFAULT_VOICE = os.environ.get("MING_OMNI_TTS_DEFAULT_VOICE", "default")
PROMPT_WAV = os.environ.get("MING_OMNI_TTS_PROMPT_WAV", "")
PROMPT_TEXT = os.environ.get("MING_OMNI_TTS_PROMPT_TEXT", "")
GEN_PROMPT = os.environ.get(
    "MING_OMNI_TTS_GEN_PROMPT", "Please generate speech based on the following description.\n"
)
MAX_DECODE_STEPS = int(os.environ.get("MING_OMNI_TTS_MAX_DECODE_STEPS", "200"))
CFG = float(os.environ.get("MING_OMNI_TTS_CFG", "2.0"))
SIGMA = float(os.environ.get("MING_OMNI_TTS_SIGMA", "0.25"))
TEMPERATURE = float(os.environ.get("MING_OMNI_TTS_TEMPERATURE", "0"))


class MingRuntime:
    """Trimmed port of cookbooks/test.py's MingAudio — the zero-shot TTS path
    only (no TTA / music / text-only generation). Text normalization is skipped
    on purpose: upstream notes it is unsupported for the MoE checkpoint."""

    def __init__(self, model_path: str):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = BailingMMNativeForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        self.model = self.model.eval().to(torch.bfloat16).to(self.device)

        if self.model.model_type == "dense":
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        else:
            # MoE checkpoint: the tokenizer (with custom tokenization_bailing.py)
            # ships in the repo root, which is the cwd.
            self.tokenizer = AutoTokenizer.from_pretrained(".", trust_remote_code=True)
        self.model.tokenizer = self.tokenizer
        self.sample_rate = self.model.config.audio_tokenizer_config.sample_rate
        self.patch_size = self.model.config.ditar_config["patch_size"]

        local_model_path = model_path if os.path.isdir(model_path) else snapshot_download(repo_id=model_path)
        self.spkemb_extractor = SpkembExtractor(f"{local_model_path}/campplus.onnx")

    def pad_waveform(self, waveform):
        # Pad to a multiple of the patch size (12.5 Hz tokenizer framerate).
        pad_align = int(1 / 12.5 * self.patch_size * self.sample_rate)
        new_len = (waveform.size(-1) + pad_align - 1) // pad_align * pad_align
        if new_len != waveform.size(1):
            new_wav = torch.zeros(1, new_len, dtype=waveform.dtype, device=waveform.device)
            new_wav[:, : waveform.size(1)] = waveform.clone()
            waveform = new_wav
        return waveform

    def preprocess_one_prompt_wav(self, waveform_path, use_spk_emb):
        if waveform_path is None:
            return None, None
        waveform, sr = torchaudio.load(waveform_path)
        waveform1 = waveform.clone()
        if sr != self.sample_rate:
            waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)(waveform)
        if use_spk_emb:
            waveform1 = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(waveform1)
            spk_emb = self.spkemb_extractor(waveform1)
        else:
            spk_emb = None
        return waveform, spk_emb

    def speech_generation(
        self,
        text: str,
        use_spk_emb: bool = False,
        use_zero_spk_emb: bool = False,
        prompt_wav_path: str | None = None,
        prompt_text: str | None = None,
    ):
        if prompt_wav_path is None:
            prompt_waveform, spk_emb = None, None
            if use_zero_spk_emb:
                spk_emb = [torch.zeros(1, 192, device=self.device, dtype=torch.bfloat16)]
        else:
            waveform, spk_emb_one = self.preprocess_one_prompt_wav(prompt_wav_path, use_spk_emb)
            prompt_waveform = self.pad_waveform(waveform)
            spk_emb = [spk_emb_one] if spk_emb_one is not None else None

        waveform = self.model.generate(
            prompt=GEN_PROMPT,
            text=text,
            spk_emb=spk_emb,
            instruction=None,
            prompt_waveform=prompt_waveform,
            prompt_text=prompt_text,
            max_decode_steps=MAX_DECODE_STEPS,
            cfg=CFG,
            sigma=SIGMA,
            temperature=TEMPERATURE,
            use_zero_spk_emb=use_zero_spk_emb,
        )
        return waveform


app = FastAPI(title="Ming-omni-TTS OpenAI Compatible TTS")

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


_runtime: MingRuntime | None = None


def get_runtime() -> MingRuntime:
    global _runtime
    if _runtime is None:
        logger.info("Loading Ming-omni-TTS (%s)", HF_MODEL)
        # ~34GB of weights download on first load (ungated, no HF_TOKEN).
        _runtime = MingRuntime(HF_MODEL)
        logger.info("Ming-omni-TTS loaded (sample_rate=%d)", _runtime.sample_rate)
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
    return {
        "ok": True,
        "engine": "Ming-omni-TTS",
        "model": OPENAI_MODEL_ID,
        "hf_model": HF_MODEL,
        "default_voice": DEFAULT_VOICE,
        "clone_ready": bool(PROMPT_WAV),
    }


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [{"id": OPENAI_MODEL_ID, "object": "model", "owned_by": "inclusionAI"}],
    }


@app.get("/v1/voices")
def list_voices():
    # default = built-in voice (zero speaker-embedding, no reference);
    # clone = zero-shot cloning from the configured reference clip.
    voices = [{"id": "default", "object": "voice"}]
    if PROMPT_WAV:
        voices.append({"id": "clone", "object": "voice", "prompt_wav": PROMPT_WAV})
    return {"object": "list", "data": voices}


@app.post("/v1/audio/speech")
async def audio_speech(payload: AudioSpeechRequest):
    if payload.response_format.lower() != "wav":
        raise HTTPException(status_code=400, detail="This wrapper currently supports only wav.")

    voice = payload.voice or DEFAULT_VOICE

    if voice == "default":
        use_spk_emb = False
        use_zero_spk_emb = True
        prompt_wav_path = None
        prompt_text = None
    elif voice == "clone":
        if not PROMPT_WAV:
            raise HTTPException(
                status_code=400,
                detail="voice='clone' requires MING_OMNI_TTS_PROMPT_WAV (--ming-omni-tts-prompt-wav). "
                "Optionally set --ming-omni-tts-prompt-text. Use voice='default' for the built-in voice.",
            )
        use_spk_emb = True
        use_zero_spk_emb = False
        prompt_wav_path = PROMPT_WAV
        prompt_text = PROMPT_TEXT or None
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown voice: {voice}. Available: ['default', 'clone'].",
        )

    runtime = get_runtime()
    waveform = runtime.speech_generation(
        text=payload.input,
        use_spk_emb=use_spk_emb,
        use_zero_spk_emb=use_zero_spk_emb,
        prompt_wav_path=prompt_wav_path,
        prompt_text=prompt_text,
    )

    if waveform is None or waveform.numel() == 0:
        raise HTTPException(status_code=500, detail="No audio was generated.")

    audio_np = waveform.float().cpu().squeeze().numpy()

    buf = io.BytesIO()
    sf.write(buf, audio_np, runtime.sample_rate, format="WAV", subtype="PCM_16")
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
