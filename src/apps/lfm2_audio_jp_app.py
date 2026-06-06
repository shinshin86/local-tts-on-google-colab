from __future__ import annotations

import io
import logging
import os

import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from liquid_audio import ChatState, LFM2AudioModel, LFM2AudioProcessor

logger = logging.getLogger("uvicorn.error")

OPENAI_MODEL_ID = os.environ.get("OPENAI_MODEL_ID", "lfm2-audio-jp")
HF_MODEL = os.environ.get("LFM2_AUDIO_JP_HF_MODEL", "LiquidAI/LFM2.5-Audio-1.5B-JP")
# The JP checkpoint ships a single Japanese voice selected via this system prompt
# (there are no named-voice presets like the English base model).
SYSTEM_PROMPT = os.environ.get("LFM2_AUDIO_JP_SYSTEM_PROMPT", "Perform TTS in japanese.")
MAX_NEW_TOKENS = int(os.environ.get("LFM2_AUDIO_JP_MAX_NEW_TOKENS", "1024"))
AUDIO_TEMPERATURE = float(os.environ.get("LFM2_AUDIO_JP_AUDIO_TEMPERATURE", "0.8"))
AUDIO_TOP_K = int(os.environ.get("LFM2_AUDIO_JP_AUDIO_TOP_K", "64"))

# liquid-audio's LFM2 detokenizer emits 24 kHz mono.
SAMPLE_RATE = 24_000
# Terminal audio frame emitted by generate_sequential when it leaves AUDIO_OUT.
AUDIO_END_CODE = 2048

app = FastAPI(title="LFM2.5-Audio-JP OpenAI Compatible TTS")

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


_processor: LFM2AudioProcessor | None = None
_model: LFM2AudioModel | None = None


def get_model():
    global _processor, _model
    if _model is None:
        logger.info("Loading LFM2.5-Audio-JP (%s)", HF_MODEL)
        # from_pretrained defaults to device="cuda", dtype=bfloat16. ChatState and
        # decode build their tensors on the processor/model device automatically.
        _processor = LFM2AudioProcessor.from_pretrained(HF_MODEL).eval()
        _model = LFM2AudioModel.from_pretrained(HF_MODEL).eval()
        logger.info("LFM2.5-Audio-JP loaded (sample_rate=%d)", SAMPLE_RATE)
    return _processor, _model


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
        "engine": "LFM2.5-Audio-JP",
        "model": OPENAI_MODEL_ID,
        "hf_model": HF_MODEL,
        "system_prompt": SYSTEM_PROMPT,
    }


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [{"id": OPENAI_MODEL_ID, "object": "model", "owned_by": "liquid-ai"}],
    }


@app.get("/v1/voices")
def list_voices():
    # Single built-in Japanese voice; no reference / cloning support.
    return {"object": "list", "data": [{"id": "default", "object": "voice"}]}


@app.post("/v1/audio/speech")
async def audio_speech(payload: AudioSpeechRequest):
    if payload.response_format.lower() != "wav":
        raise HTTPException(status_code=400, detail="This wrapper currently supports only wav.")

    voice = payload.voice or "default"
    if voice != "default":
        raise HTTPException(
            status_code=400,
            detail=f"Unknown voice: {voice}. LFM2.5-Audio-JP only exposes 'default' "
            "(single built-in Japanese voice; no reference/cloning).",
        )

    processor, model = get_model()

    # Sequential generation: system prompt selects the TTS task/voice, the user
    # turn carries the text, then the assistant turn is decoded into audio.
    chat = ChatState(processor)
    chat.new_turn("system")
    chat.add_text(SYSTEM_PROMPT)
    chat.end_turn()
    chat.new_turn("user")
    chat.add_text(payload.input)
    chat.end_turn()
    chat.new_turn("assistant")

    # generate_sequential yields text tokens (numel==1) and audio frames
    # (numel==8, one entry per Mimi codebook). Keep the audio frames, dropping
    # the terminal end-of-audio marker frame (all entries == 2048), which is out
    # of the detokenizer's valid [0, 2047] range.
    audio_frames: list[torch.Tensor] = []
    with torch.no_grad():
        for t in model.generate_sequential(
            **chat,
            max_new_tokens=MAX_NEW_TOKENS,
            audio_temperature=AUDIO_TEMPERATURE,
            audio_top_k=AUDIO_TOP_K,
        ):
            if t.numel() > 1 and int(t.reshape(-1)[0]) != AUDIO_END_CODE:
                audio_frames.append(t)

    if not audio_frames:
        raise HTTPException(
            status_code=500,
            detail="No audio was generated (the model emitted no audio frames).",
        )

    # (8, T) -> (1, 8, T) integer codes -> (1, T') waveform.
    audio_codes = torch.stack(audio_frames, 1).unsqueeze(0)
    waveform = processor.decode(audio_codes)
    audio_np = waveform.float().cpu()[0].numpy()

    buf = io.BytesIO()
    sf.write(buf, audio_np, SAMPLE_RATE, format="WAV", subtype="PCM_16")
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
