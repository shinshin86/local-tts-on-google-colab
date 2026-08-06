from __future__ import annotations

import os
from pathlib import Path

from src.config import Settings
from src.runtime import popen, run, uv_pip_install, write_text


INDEXTTS2_REPO_URL = "https://github.com/index-tts/index-tts.git"
INDEXTTS2_REPO_REF = "90ca4d608209584bad3a5bd5becc0b80c146e60f"
INDEXTTS2_MODEL_ID = "IndexTeam/IndexTTS-2"
INDEXTTS2_MODEL_REF = "740dcaff396282ffb241903d150ac011cd4b1ede"
INDEXTTS2_DEMO_REF = "b01840e8e4fd9753743a6d0466cd73ae1d634a68"
W2V_BERT_REF = "da985ba0987f70aaeb84a80f2851cfac8c697a7b"
MASKGCT_REF = "265c6cef07625665d0c28d2faafb1415562379dc"
CAMPPLUS_REF = "e4b6ede7ce16997aff4ae69fbca1f0175e2afede"
BIGVGAN_REF = "633ff708ed5b74903e86ff1298cf4a98e921c513"


def _ensure_repo(repo_dir: Path) -> None:
    if not repo_dir.exists():
        run(["git", "clone", INDEXTTS2_REPO_URL, str(repo_dir)])
    run(["git", "fetch", "origin", INDEXTTS2_REPO_REF], cwd=str(repo_dir))
    run(["git", "checkout", "--detach", INDEXTTS2_REPO_REF], cwd=str(repo_dir))


def _ensure_py311_venv(engine_dir: Path) -> Path:
    venv_dir = engine_dir / ".venv"
    if not venv_dir.exists():
        run(["uv", "venv", "--python", "3.11", str(venv_dir)])
    return venv_dir / "bin" / "python"


def _download_models(python_bin: Path, model_id: str, model_dir: Path, engine_dir: Path) -> None:
    model_revision = INDEXTTS2_MODEL_REF if model_id == INDEXTTS2_MODEL_ID else None
    code = f"""
from pathlib import Path
import shutil
from huggingface_hub import hf_hub_download, snapshot_download

model_dir = Path({str(model_dir)!r})
cache_dir = model_dir / "hf_cache"
model_dir.mkdir(parents=True, exist_ok=True)
cache_dir.mkdir(parents=True, exist_ok=True)

snapshot_download(
    repo_id={model_id!r},
    revision={model_revision!r},
    local_dir=str(model_dir),
)
snapshot_download(
    repo_id="facebook/w2v-bert-2.0",
    revision={W2V_BERT_REF!r},
    local_dir=str(cache_dir / "w2v-bert-2.0"),
)

semantic_source = hf_hub_download(
    repo_id="amphion/MaskGCT",
    filename="semantic_codec/model.safetensors",
    revision={MASKGCT_REF!r},
    local_dir=str(cache_dir / "maskgct"),
)
shutil.copyfile(semantic_source, cache_dir / "semantic_codec_model.safetensors")

hf_hub_download(
    repo_id="funasr/campplus",
    filename="campplus_cn_common.bin",
    revision={CAMPPLUS_REF!r},
    local_dir=str(cache_dir),
)
snapshot_download(
    repo_id="nvidia/bigvgan_v2_22khz_80band_256x",
    revision={BIGVGAN_REF!r},
    allow_patterns=["config.json", "bigvgan_generator.pt"],
    local_dir=str(cache_dir / "bigvgan"),
)

hf_hub_download(
    repo_id="IndexTeam/IndexTTS-2-Demo",
    repo_type="space",
    filename="examples/voice_01.wav",
    revision={INDEXTTS2_DEMO_REF!r},
    local_dir={str(engine_dir / "official-demo")!r},
)
"""
    run([str(python_bin), "-c", code])


def install(settings: Settings) -> dict:
    engine_dir = settings.engines_dir / "indextts2"
    engine_dir.mkdir(parents=True, exist_ok=True)

    repo_dir = engine_dir / "index-tts"
    _ensure_repo(repo_dir)

    python_bin = _ensure_py311_venv(engine_dir)
    run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(python_bin),
            "--index-url",
            "https://download.pytorch.org/whl/cu128",
            "torch==2.8.0",
            "torchaudio==2.8.0",
        ]
    )
    uv_pip_install(python_bin, ["-e", str(repo_dir)])
    uv_pip_install(python_bin, ["fastapi", "uvicorn", "soundfile", "huggingface-hub"])

    model_dir = engine_dir / "models" / "IndexTTS-2"
    _download_models(python_bin, settings.indextts2_hf_model, model_dir, engine_dir)

    write_text(engine_dir / "app.py", settings.read_repo_text("src/apps/indextts2_app.py"))

    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "PYTHONPATH": str(repo_dir),
        "OPENAI_MODEL_ID": settings.openai_model_id or "indextts2",
        "INDEXTTS2_HF_MODEL": settings.indextts2_hf_model,
        "INDEXTTS2_MODEL_DIR": str(model_dir),
        "INDEXTTS2_DEFAULT_PROMPT_WAV": str(
            engine_dir / "official-demo" / "examples" / "voice_01.wav"
        ),
        "INDEXTTS2_PROMPT_WAV": settings.indextts2_prompt_wav,
        "INDEXTTS2_DEFAULT_VOICE": settings.indextts2_default_voice,
        "INDEXTTS2_EMOTION_WAV": settings.indextts2_emotion_wav,
        "INDEXTTS2_EMOTION_TEXT": settings.indextts2_emotion_text,
        "INDEXTTS2_EMOTION_VECTOR": settings.indextts2_emotion_vector,
        "INDEXTTS2_EMOTION_ALPHA": str(settings.indextts2_emotion_alpha),
        "INDEXTTS2_USE_RANDOM": "1" if settings.indextts2_use_random else "0",
        "INDEXTTS2_USE_FP16": "1" if settings.indextts2_use_fp16 else "0",
    }
    proc = popen(
        [
            str(engine_dir / ".venv" / "bin" / "uvicorn"),
            "app:app",
            "--host",
            "0.0.0.0",
            "--port",
            str(settings.app_port),
            "--log-level",
            "info",
        ],
        cwd=str(engine_dir),
        env=env,
        log_path=settings.log_dir / "indextts2-uvicorn.log",
    )
    return {
        "proc": proc,
        "app_dir": engine_dir,
        "log_path": settings.log_dir / "indextts2-uvicorn.log",
    }
