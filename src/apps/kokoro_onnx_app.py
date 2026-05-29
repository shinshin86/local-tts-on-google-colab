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
from pydantic import BaseModel

logger = logging.getLogger("uvicorn.error")

OPENAI_MODEL_ID = os.environ.get("OPENAI_MODEL_ID", "kokoro-82m-onnx")
HF_MODEL = os.environ.get("KOKORO_ONNX_HF_MODEL", "nvidia/kokoro-82M-onnx-opt")
DEFAULT_VOICE = os.environ.get("KOKORO_ONNX_DEFAULT_VOICE", "jf_alpha")
DEFAULT_LANG_CODE = os.environ.get("KOKORO_ONNX_DEFAULT_LANG_CODE", "j")
# auto = CUDA preferred, CPU fallback; cuda = same; cpu = CPU only.
PROVIDER = os.environ.get("KOKORO_ONNX_PROVIDER", "auto").lower()

SAMPLE_RATE = 24000
# Kokoro's ONNX context length is 512; leave room for the pad token 0 at both
# ends, so a single forward pass takes at most 510 phoneme tokens.
MAX_CONTENT_TOKENS = 510

# Voice name prefix -> Kokoro lang_code. Matches the upstream Kokoro convention
# (a=American English, b=British English, e=Spanish, f=French, h=Hindi,
# i=Italian, j=Japanese, p=Brazilian Portuguese, z=Mandarin Chinese).
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

# lang_codes routed through misaki's EspeakG2P, with the espeak-ng voice name.
ESPEAK_LANG = {"e": "es", "f": "fr-fr", "h": "hi", "i": "it", "p": "pt-br"}

# Representative subset used for /v1/voices fallback before the model finishes
# loading; the authoritative list is read from the model's voices.txt at runtime.
VOICE_PRESETS = [
    "af_heart",
    "af_bella",
    "am_adam",
    "am_michael",
    "bf_emma",
    "bm_george",
    "jf_alpha",
    "jf_gongitsune",
    "jm_kumo",
    "zf_xiaobei",
    "zm_yunjian",
    "ef_dora",
    "ff_siwis",
    "hf_alpha",
    "if_sara",
    "pf_dora",
]

app = FastAPI(title="Kokoro-ONNX OpenAI Compatible TTS")

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


_state: dict = {}


def _resolve_providers() -> list[str]:
    if PROVIDER == "cpu":
        return ["CPUExecutionProvider"]
    # "auto" / "cuda": prefer CUDA, fall back to CPU if the GPU provider cannot
    # be loaded (e.g. no GPU runtime or CUDA/cuDNN mismatch).
    return ["CUDAExecutionProvider", "CPUExecutionProvider"]


def _ensure_loaded() -> dict:
    if _state:
        return _state

    import onnxruntime as ort
    from huggingface_hub import snapshot_download

    model_dir = snapshot_download(HF_MODEL)

    onnx_path = os.path.join(model_dir, "kokoro-82m-v1.0.onnx")

    # tokens.txt: "<symbol> <id>" per line. The symbol itself may be a space,
    # so split on the last whitespace only.
    vocab: dict[str, int] = {}
    for line in Path(os.path.join(model_dir, "tokens.txt")).read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        sym, idx = line.rsplit(" ", 1)
        vocab[sym] = int(idx)

    # voices.txt: "<index>=<name>" per line.
    name2idx: dict[str, int] = {}
    for line in Path(os.path.join(model_dir, "voices.txt")).read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        i, name = line.split("=", 1)
        name2idx[name.strip()] = int(i)

    # voices.bin: float32, shape (n_speakers, 510, 256).
    voices = np.fromfile(os.path.join(model_dir, "voices.bin"), dtype=np.float32).reshape(
        len(name2idx), -1, 256
    )

    sess = ort.InferenceSession(onnx_path, providers=_resolve_providers())

    _state.update(
        model_dir=model_dir,
        vocab=vocab,
        name2idx=name2idx,
        voices=voices,
        sess=sess,
        space_id=vocab.get(" "),
        g2p={},
    )
    logger.info(
        "Kokoro-ONNX loaded: provider=%s voices=%d", sess.get_providers()[0], len(name2idx)
    )
    return _state


def infer_lang_code(voice: str) -> str:
    prefix = (voice or DEFAULT_VOICE).split("_", 1)[0]
    return LANG_CODE_BY_PREFIX.get(prefix[:1], DEFAULT_LANG_CODE)


def _get_g2p(lang_code: str):
    state = _ensure_loaded()
    g2p = state["g2p"].get(lang_code)
    if g2p is not None:
        return g2p

    if lang_code in ("a", "b"):
        from misaki import en
        from misaki.espeak import EspeakFallback

        british = lang_code == "b"
        try:
            fallback = EspeakFallback(british=british)
        except Exception:  # noqa: BLE001 - fallback is best-effort for OOV words
            fallback = None
        g2p = en.G2P(trf=False, british=british, fallback=fallback)
    elif lang_code == "j":
        from misaki import ja

        g2p = ja.JAG2P()
    elif lang_code == "z":
        from misaki import zh

        g2p = zh.ZHG2P()
    elif lang_code in ESPEAK_LANG:
        from misaki.espeak import EspeakG2P

        g2p = EspeakG2P(language=ESPEAK_LANG[lang_code])
    else:
        from misaki import en

        g2p = en.G2P(trf=False, british=False)

    state["g2p"][lang_code] = g2p
    return g2p


def _phonemize(text: str, lang_code: str) -> str:
    g2p = _get_g2p(lang_code)
    phonemes, _ = g2p(text)
    return phonemes or ""


def _phonemes_to_ids(phonemes: str, vocab: dict[str, int]) -> list[int]:
    return [vocab[ch] for ch in phonemes if ch in vocab]


def _chunk_ids(ids: list[int], space_id: int | None, limit: int = MAX_CONTENT_TOKENS) -> list[list[int]]:
    """Split a token-id stream into <=limit chunks, preferring word boundaries
    (the space token) so we never cut a word in half."""
    if len(ids) <= limit:
        return [ids]
    chunks: list[list[int]] = []
    i, n = 0, len(ids)
    while i < n:
        end = min(i + limit, n)
        if end < n and space_id is not None:
            cut = end
            while cut > i and ids[cut - 1] != space_id:
                cut -= 1
            if cut > i:
                end = cut
        chunks.append(ids[i:end])
        i = end
    return chunks


def _generate(ids: list[int], voice_idx: int, speed: float) -> np.ndarray:
    state = _ensure_loaded()
    voices = state["voices"]
    sess = state["sess"]
    audio_chunks: list[np.ndarray] = []
    for chunk in _chunk_ids(ids, state["space_id"]):
        if not chunk:
            continue
        tokens = np.array([[0, *chunk, 0]], dtype=np.int64)
        # Style vector is selected by the (unpadded) phoneme-token count.
        style = voices[voice_idx, min(len(chunk), voices.shape[1] - 1)].reshape(1, 256).astype(
            np.float32
        )
        out = sess.run(
            ["audio"],
            {"tokens": tokens, "style": style, "speed": np.array([speed], dtype=np.float32)},
        )[0]
        audio_chunks.append(np.asarray(out, dtype=np.float32).reshape(-1))
    if not audio_chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(audio_chunks)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception while serving request")
    return JSONResponse(
        status_code=500,
        content={"error": type(exc).__name__, "detail": str(exc)},
    )


@app.get("/")
def root():
    return {"ok": True, "engine": "Kokoro-ONNX", "model": OPENAI_MODEL_ID}


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [{"id": OPENAI_MODEL_ID, "object": "model", "owned_by": "local"}],
    }


@app.get("/v1/voices")
def list_voices():
    voices = VOICE_PRESETS
    if _state:
        voices = sorted(_state["name2idx"], key=lambda name: _state["name2idx"][name])
    return {
        "object": "list",
        "data": [{"id": voice, "object": "voice"} for voice in voices],
    }


@app.post("/v1/audio/speech")
async def audio_speech(payload: AudioSpeechRequest):
    if payload.response_format.lower() != "wav":
        raise HTTPException(status_code=400, detail="This wrapper currently supports only wav.")

    state = _ensure_loaded()
    name2idx = state["name2idx"]

    voice = payload.voice or DEFAULT_VOICE
    if voice == "default":
        voice = DEFAULT_VOICE
    if voice == "clone":
        raise HTTPException(
            status_code=400,
            detail=(
                "Kokoro-ONNX does not support runtime voice cloning. "
                "Use one of the built-in preset voices (see /v1/voices)."
            ),
        )
    if voice not in name2idx:
        sample = ", ".join(list(name2idx)[:12])
        raise HTTPException(
            status_code=400,
            detail=f"Unknown voice '{voice}'. Available voices include: {sample}, ... (see /v1/voices).",
        )

    lang_code = infer_lang_code(voice)
    phonemes = _phonemize(payload.input, lang_code)
    ids = _phonemes_to_ids(phonemes, state["vocab"])
    if not ids:
        raise HTTPException(
            status_code=400,
            detail="Input produced no phonemes; please provide non-empty text.",
        )

    audio = _generate(ids, name2idx[voice], float(payload.speed))
    if audio.size == 0:
        raise HTTPException(status_code=500, detail="No audio was generated.")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        sf.write(tmp_path, audio, SAMPLE_RATE)
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
