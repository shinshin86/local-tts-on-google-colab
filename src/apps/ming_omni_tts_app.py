from __future__ import annotations

import copy
import io
import json
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
# Per-task prompt that selects what the model generates. `speech` is the
# default; `music` and `tta` (text-to-audio / sound events) reuse the same
# /v1/audio/speech endpoint via the request `task` field (or the startup
# default below). The speech prompt stays configurable for backward compat.
SPEECH_PROMPT = os.environ.get(
    "MING_OMNI_TTS_GEN_PROMPT", "Please generate speech based on the following description.\n"
)
TASK_PROMPTS = {
    "speech": SPEECH_PROMPT,
    "music": "Please generate music based on the following description.\n",
    "tta": "Please generate audio events based on given text.\n",
}
DEFAULT_TASK = os.environ.get("MING_OMNI_TTS_TASK", "speech").strip().lower() or "speech"
# Natural-language voice design (zero-shot): these map to the model's
# instruction JSON keys 风格 / 情感 / 方言. Empty -> not sent. They layer on top
# of voice=default (zero speaker-embedding) or voice=clone (reference + design).
DEFAULT_STYLE = os.environ.get("MING_OMNI_TTS_STYLE", "")
DEFAULT_EMOTION = os.environ.get("MING_OMNI_TTS_EMOTION", "")
DEFAULT_DIALECT = os.environ.get("MING_OMNI_TTS_DIALECT", "")

MAX_DECODE_STEPS = int(os.environ.get("MING_OMNI_TTS_MAX_DECODE_STEPS", "200"))
CFG = float(os.environ.get("MING_OMNI_TTS_CFG", "2.0"))
SIGMA = float(os.environ.get("MING_OMNI_TTS_SIGMA", "0.25"))
TEMPERATURE = float(os.environ.get("MING_OMNI_TTS_TEMPERATURE", "0"))

# music / tta are non-speech generation and need different decode dynamics than
# zero-temperature TTS (matching upstream cookbooks/test.py). Speech keeps the
# configurable globals above.
TASK_DECODE = {
    "music": {"max_decode_steps": max(MAX_DECODE_STEPS, 400), "cfg": 4.5, "sigma": 0.3, "temperature": 2.5},
    "tta": {"max_decode_steps": MAX_DECODE_STEPS, "cfg": 4.5, "sigma": 0.3, "temperature": 2.5},
}

# instruction template (Chinese keys), ported from cookbooks/test.py.
BASE_CAPTION_TEMPLATE = {
    "audio_sequence": [
        {
            "序号": 1,
            "说话人": "speaker_1",
            "方言": None,
            "风格": None,
            "语速": None,
            "基频": None,
            "音量": None,
            "情感": None,
            "BGM": {
                "Genre": None,
                "Mood": None,
                "Instrument": None,
                "Theme": None,
                "ENV": None,
                "SNR": None,
            },
            "IP": None,
        }
    ]
}


class MingRuntime:
    """Trimmed port of cookbooks/test.py's MingAudio. Supports the speech / music
    / tta generation paths and natural-language voice design via the instruction
    JSON. Text normalization is skipped (unsupported for the MoE checkpoint)."""

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

    def create_instruction(self, user_input: dict) -> str:
        new_caption = copy.deepcopy(BASE_CAPTION_TEMPLATE)
        target = new_caption["audio_sequence"][0]
        for key, value in user_input.items():
            if key in target:
                target[key] = value
        return json.dumps(new_caption, ensure_ascii=False)

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

    def generate(
        self,
        text: str,
        task_prompt: str,
        instruction: dict | None = None,
        use_spk_emb: bool = False,
        use_zero_spk_emb: bool = False,
        prompt_wav_path: str | None = None,
        prompt_text: str | None = None,
        max_decode_steps: int = MAX_DECODE_STEPS,
        cfg: float = CFG,
        sigma: float = SIGMA,
        temperature: float = TEMPERATURE,
    ):
        if prompt_wav_path is None:
            prompt_waveform, spk_emb = None, None
            if use_zero_spk_emb:
                spk_emb = [torch.zeros(1, 192, device=self.device, dtype=torch.bfloat16)]
        else:
            waveform, spk_emb_one = self.preprocess_one_prompt_wav(prompt_wav_path, use_spk_emb)
            prompt_waveform = self.pad_waveform(waveform)
            spk_emb = [spk_emb_one] if spk_emb_one is not None else None

        instruction_json = self.create_instruction(instruction) if instruction else None

        return self.model.generate(
            prompt=task_prompt,
            text=text,
            spk_emb=spk_emb,
            instruction=instruction_json,
            prompt_waveform=prompt_waveform,
            prompt_text=prompt_text,
            max_decode_steps=max_decode_steps,
            cfg=cfg,
            sigma=sigma,
            temperature=temperature,
            use_zero_spk_emb=use_zero_spk_emb,
        )


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
    # Ming-omni extensions (all optional; None -> fall back to startup defaults,
    # so a plain OpenAI request behaves exactly as before).
    task: str | None = None  # speech | music | tta
    style: str | None = None  # 风格: natural-language voice design (e.g. gentle young female)
    emotion: str | None = None  # 情感
    dialect: str | None = None  # 方言 (e.g. Cantonese)


_runtime: MingRuntime | None = None


def get_runtime() -> MingRuntime:
    global _runtime
    if _runtime is None:
        logger.info("Loading Ming-omni-TTS (%s)", HF_MODEL)
        # ~34GB of weights download on first load (ungated, no HF_TOKEN).
        _runtime = MingRuntime(HF_MODEL)
        logger.info("Ming-omni-TTS loaded (sample_rate=%d)", _runtime.sample_rate)
    return _runtime


def _pick(value: str | None, default: str) -> str:
    return value if value is not None else default


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
        "default_task": DEFAULT_TASK,
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

    task = (payload.task or DEFAULT_TASK).strip().lower()
    if task not in TASK_PROMPTS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task: {task}. Available: {sorted(TASK_PROMPTS)}.",
        )
    task_prompt = TASK_PROMPTS[task]

    runtime = get_runtime()

    if task in ("music", "tta"):
        # Non-speech generation: `input` is the description; no speaker, no
        # instruction. Decode dynamics differ from TTS (upstream-recommended).
        decode = TASK_DECODE[task]
        waveform = runtime.generate(
            text=payload.input,
            task_prompt=task_prompt,
            use_zero_spk_emb=True,
            **decode,
        )
        voice = task
    else:
        # speech: build the (optional) voice-design instruction.
        instruction: dict = {}
        style = _pick(payload.style, DEFAULT_STYLE)
        emotion = _pick(payload.emotion, DEFAULT_EMOTION)
        dialect = _pick(payload.dialect, DEFAULT_DIALECT)
        if style:
            instruction["风格"] = style
        if emotion:
            instruction["情感"] = emotion
        if dialect:
            instruction["方言"] = dialect

        voice = payload.voice or DEFAULT_VOICE
        if voice == "default":
            use_spk_emb, use_zero_spk_emb = False, True
            prompt_wav_path, prompt_text = None, None
        elif voice == "clone":
            if not PROMPT_WAV:
                raise HTTPException(
                    status_code=400,
                    detail="voice='clone' requires MING_OMNI_TTS_PROMPT_WAV (--ming-omni-tts-prompt-wav). "
                    "Optionally set --ming-omni-tts-prompt-text. Use voice='default' for the built-in voice.",
                )
            use_spk_emb, use_zero_spk_emb = True, False
            prompt_wav_path, prompt_text = PROMPT_WAV, (PROMPT_TEXT or None)
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown voice: {voice}. Available: ['default', 'clone'].",
            )

        waveform = runtime.generate(
            text=payload.input,
            task_prompt=task_prompt,
            instruction=instruction or None,
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
