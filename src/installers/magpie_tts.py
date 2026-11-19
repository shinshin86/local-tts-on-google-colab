from __future__ import annotations

import os

from src.config import Settings
from src.runtime import ensure_git_clone, popen, run, uv_pip_install, write_text


# Match the NeMo Speech and text-processing revisions used by NVIDIA's demo.
NEMO_REPO = "https://github.com/NVIDIA-NeMo/Speech.git"
NEMO_REF = "24c4f58a764350a62248094c70e487892e9d1a00"
TEXT_PROCESSING = (
    "nemo-text-processing @ git+https://github.com/NVIDIA/NeMo-text-processing.git"
    "@1f1263579fe57ba7ed783cad3dddee710fcc5064"
)
MAGPIE_MODEL = "nvidia/magpie_tts_multilingual_357m"
MAGPIE_REF = "5023df68bd3f5b5ce6d666a50979bc501af145cc"  # v2607
MAGPIE_FILE = "magpie_tts_multilingual_357m.nemo"
CODEC_MODEL = "nvidia/nemo-nano-codec-22khz-1.89kbps-21.5fps"
CODEC_REF = "fc00890b604aa2de298d2641ffc6c5f6caf8c4d7"
CODEC_FILE = "nemo-nano-codec-22khz-1.89kbps-21.5fps.nemo"


def install(settings: Settings) -> dict:
    engine_dir = settings.engines_dir / "magpie-tts"
    engine_dir.mkdir(parents=True, exist_ok=True)
    repo_dir = engine_dir / "nemo-speech"
    ensure_git_clone(NEMO_REPO, repo_dir)
    run(["git", "fetch", "origin", NEMO_REF], cwd=str(repo_dir))
    run(["git", "checkout", "--detach", NEMO_REF], cwd=str(repo_dir))

    venv_dir = engine_dir / ".venv"
    if not venv_dir.exists():
        run(["uv", "venv", "--python", "3.12", str(venv_dir)])
    python_bin = venv_dir / "bin" / "python"
    uv_pip_install(
        python_bin,
        [
            "--index-url", "https://download.pytorch.org/whl/cu128",
            "torch==2.9.1+cu128", "torchaudio==2.9.1+cu128",
        ],
    )
    uv_pip_install(
        python_bin,
        [
            f"{repo_dir}[tts]", TEXT_PROCESSING, "cuda-bindings<13",
            "fastapi", "uvicorn", "soundfile",
        ],
    )

    # Cache the pinned archives before serving requests. NeMo's loader restores
    # these local files and uses its bundled tokenizer and G2P assets.
    model_dir = engine_dir / "models" / MAGPIE_REF
    codec_dir = engine_dir / "models" / CODEC_REF
    for model, revision, filename, folder in (
        (MAGPIE_MODEL, MAGPIE_REF, MAGPIE_FILE, model_dir),
        (CODEC_MODEL, CODEC_REF, CODEC_FILE, codec_dir),
    ):
        run(
            [
                str(python_bin), "-c",
                "import sys; from huggingface_hub import hf_hub_download; "
                "hf_hub_download(sys.argv[1], sys.argv[2], revision=sys.argv[3], local_dir=sys.argv[4])",
                model, filename, revision, str(folder),
            ]
        )

    write_text(engine_dir / "app.py", settings.read_repo_text("src/apps/magpie_tts_app.py"))
    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "NUMBA_DISABLE_CUDA": "1",
        "OPENAI_MODEL_ID": settings.openai_model_id or "magpie-tts",
        "MAGPIE_TTS_MODEL_PATH": str(model_dir / MAGPIE_FILE),
        "MAGPIE_TTS_CODEC_PATH": str(codec_dir / CODEC_FILE),
        "MAGPIE_TTS_LANGUAGE": settings.magpie_tts_language,
        "MAGPIE_TTS_DEFAULT_VOICE": settings.magpie_tts_default_voice,
        "MAGPIE_TTS_TEXT_NORMALIZATION": "1" if settings.magpie_tts_text_normalization else "0",
    }
    log_path = settings.log_dir / "magpie-tts-uvicorn.log"
    # The loader's fallback phoneme dictionary paths are relative to the repo.
    proc = popen(
        [
            str(python_bin), "-m", "uvicorn", "app:app",
            "--app-dir", str(engine_dir),
            "--host", "0.0.0.0", "--port", str(settings.app_port), "--log-level", "info",
        ],
        cwd=str(repo_dir),
        env=env,
        log_path=log_path,
    )
    return {"proc": proc, "app_dir": engine_dir, "log_path": log_path}
