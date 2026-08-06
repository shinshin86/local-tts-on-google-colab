from __future__ import annotations

import json
import os
from pathlib import Path

from src.config import Settings
from src.runtime import ensure_venv, popen, run, uv_pip_install, write_text


VIBEVOICE_REPO_URL = "https://github.com/microsoft/VibeVoice.git"
VIBEVOICE_REPO_REF = "94da20d98b2fa7688e9cbfaf7692ddb4954f7600"
VIBEVOICE_MODEL_ID = "microsoft/VibeVoice-Realtime-0.5B"
VIBEVOICE_MODEL_REF = "6bce5f06044837fe6d2c5d7a71a84f0416bd57e4"
VIBEVOICE_TOKENIZER_ID = "Qwen/Qwen2.5-0.5B"
VIBEVOICE_TOKENIZER_REF = "060db6499f32faf8b98477b0a26969ef7d8b9987"


def _ensure_repo(repo_dir: Path) -> None:
    if not repo_dir.exists():
        run(["git", "clone", VIBEVOICE_REPO_URL, str(repo_dir)])
    run(["git", "fetch", "origin", VIBEVOICE_REPO_REF], cwd=str(repo_dir))
    run(["git", "checkout", "--detach", VIBEVOICE_REPO_REF], cwd=str(repo_dir))


def install(settings: Settings) -> dict:
    engine_dir = settings.engines_dir / "vibevoice-realtime"
    engine_dir.mkdir(parents=True, exist_ok=True)

    repo_dir = engine_dir / "VibeVoice"
    _ensure_repo(repo_dir)

    python_bin = ensure_venv(engine_dir)
    uv_pip_install(python_bin, ["-e", f"{repo_dir}[streamingtts]"])
    uv_pip_install(python_bin, ["fastapi", "uvicorn", "soundfile", "numpy"])

    model_dir = engine_dir / "models" / "VibeVoice-Realtime-0.5B"
    model_revision = (
        VIBEVOICE_MODEL_REF if settings.vibevoice_hf_model == VIBEVOICE_MODEL_ID else None
    )
    if not (model_dir / "model.safetensors").exists():
        run(
            [
                str(python_bin),
                "-c",
                (
                    "from huggingface_hub import snapshot_download; "
                    f"snapshot_download({settings.vibevoice_hf_model!r}, "
                    f"revision={model_revision!r}, local_dir={str(model_dir)!r})"
                ),
            ]
        )

    processor_dir = model_dir
    if settings.vibevoice_hf_model == VIBEVOICE_MODEL_ID:
        tokenizer_dir = engine_dir / "models" / "Qwen2.5-0.5B-tokenizer"
        if not (tokenizer_dir / "tokenizer.json").exists():
            run(
                [
                    str(python_bin),
                    "-c",
                    (
                        "from huggingface_hub import snapshot_download; "
                        f"snapshot_download({VIBEVOICE_TOKENIZER_ID!r}, "
                        f"revision={VIBEVOICE_TOKENIZER_REF!r}, "
                        f"local_dir={str(tokenizer_dir)!r}, "
                        "allow_patterns=['config.json','merges.txt','tokenizer.json',"
                        "'tokenizer_config.json','vocab.json'])"
                    ),
                ]
            )
        processor_dir = engine_dir / "processor"
        processor_config = json.loads(
            (model_dir / "preprocessor_config.json").read_text(encoding="utf-8")
        )
        processor_config["language_model_pretrained_name"] = str(tokenizer_dir)
        write_text(
            processor_dir / "preprocessor_config.json",
            json.dumps(processor_config, indent=2) + "\n",
        )

    write_text(engine_dir / "app.py", settings.read_repo_text("src/apps/vibevoice_app.py"))

    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "OPENAI_MODEL_ID": settings.openai_model_id or "vibevoice-realtime",
        "VIBEVOICE_HF_MODEL": settings.vibevoice_hf_model,
        "VIBEVOICE_MODEL_DIR": str(model_dir),
        "VIBEVOICE_PROCESSOR_DIR": str(processor_dir),
        "VIBEVOICE_VOICES_DIR": str(repo_dir / "demo" / "voices" / "streaming_model"),
        "VIBEVOICE_DEFAULT_SPEAKER": settings.vibevoice_default_speaker,
        "VIBEVOICE_DDPM_STEPS": str(settings.vibevoice_ddpm_steps),
        "VIBEVOICE_CFG_SCALE": str(settings.vibevoice_cfg_scale),
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
        log_path=settings.log_dir / "vibevoice-realtime-uvicorn.log",
    )
    return {
        "proc": proc,
        "app_dir": engine_dir,
        "log_path": settings.log_dir / "vibevoice-realtime-uvicorn.log",
    }
