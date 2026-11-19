from __future__ import annotations

import os
from pathlib import Path

from src.config import Settings
from src.runtime import ensure_git_clone, popen, run, uv_pip_install, write_text

from .indextts2 import BIGVGAN_REF, CAMPPLUS_REF, INDEXTTS2_REPO_URL, W2V_BERT_REF


CODE_REF = "d9e41aac89fd00b3d71497fddb287b7f24613712"
MODEL_ID = "IndexTeam/IndexTTS-2.5"
MODEL_REF = "c39ce5ba981572cb187443877ff559dfb246ce63"
DEMO_REF = "a2b37aada1fe3fad15036051a5594005e445146a"


def _download_models(python_bin: Path, model_dir: Path, demo_dir: Path) -> None:
    code = """
from pathlib import Path
import sys
from huggingface_hub import hf_hub_download, snapshot_download

model_dir, demo_dir = map(Path, sys.argv[1:])
cache_dir = model_dir / "hf_cache"
snapshot_download(
    repo_id=%r, revision=%r, local_dir=model_dir,
    ignore_patterns=["qwen0.6bemo4-merge/*"],
)
snapshot_download(
    repo_id="facebook/w2v-bert-2.0", revision=%r, local_dir=cache_dir / "w2v-bert-2.0",
)
hf_hub_download(
    repo_id="funasr/campplus", filename="campplus_cn_common.bin",
    revision=%r, local_dir=cache_dir,
)
snapshot_download(
    repo_id="nvidia/bigvgan_v2_22khz_80band_256x", revision=%r,
    allow_patterns=["config.json", "bigvgan_generator.pt", "README.md", "LICENSE*"],
    local_dir=cache_dir / "bigvgan",
)
snapshot_download(
    repo_id="IndexTeam/IndexTTS-2.5-Demo", repo_type="space", revision=%r,
    allow_patterns=["examples/voice_01.wav", "LICENSE*"], local_dir=demo_dir,
)
""" % (MODEL_ID, MODEL_REF, W2V_BERT_REF, CAMPPLUS_REF, BIGVGAN_REF, DEMO_REF)
    run([str(python_bin), "-c", code, str(model_dir), str(demo_dir)])


def install(settings: Settings) -> dict:
    engine_dir = settings.engines_dir / "indextts25"
    engine_dir.mkdir(parents=True, exist_ok=True)
    repo_dir = engine_dir / "index-tts"
    ensure_git_clone(INDEXTTS2_REPO_URL, repo_dir)
    run(["git", "fetch", "origin", CODE_REF], cwd=str(repo_dir))
    run(["git", "checkout", "--detach", CODE_REF], cwd=str(repo_dir))

    venv_dir = engine_dir / ".venv"
    if not venv_dir.exists():
        run(["uv", "venv", "--python", "3.11", str(venv_dir)])
    python_bin = venv_dir / "bin" / "python"
    uv_pip_install(
        python_bin,
        ["--index-url", "https://download.pytorch.org/whl/cu128", "torch==2.8.0", "torchaudio==2.8.0"],
    )
    uv_pip_install(
        python_bin,
        ["-e", str(repo_dir), "fastapi", "uvicorn", "soundfile", "huggingface-hub"],
    )

    model_dir = engine_dir / "models" / MODEL_REF
    demo_dir = engine_dir / "official-demo"
    _download_models(python_bin, model_dir, demo_dir)
    write_text(engine_dir / "app.py", settings.read_repo_text("src/apps/indextts25_app.py"))

    prompt_wav = settings.indextts25_prompt_wav
    if prompt_wav:
        prompt_wav = str(Path(prompt_wav).expanduser().resolve())

    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "OPENAI_MODEL_ID": settings.openai_model_id or "indextts-2.5",
        "INDEXTTS25_MODEL_DIR": str(model_dir),
        "INDEXTTS25_LANGUAGE": settings.indextts25_language,
        "INDEXTTS25_DEFAULT_VOICE": settings.indextts25_default_voice,
        "INDEXTTS25_DEFAULT_PROMPT_WAV": str(demo_dir / "examples" / "voice_01.wav"),
        "INDEXTTS25_PROMPT_WAV": prompt_wav,
    }
    log_path = settings.log_dir / "indextts25-uvicorn.log"
    proc = popen(
        [
            str(python_bin), "-m", "uvicorn", "app:app",
            "--host", "0.0.0.0", "--port", str(settings.app_port), "--log-level", "info",
        ],
        cwd=str(engine_dir),
        env=env,
        log_path=log_path,
    )
    return {"proc": proc, "app_dir": engine_dir, "log_path": log_path}
