from __future__ import annotations

import os

from src.config import Settings
from src.runtime import ensure_git_clone, ensure_venv, popen, run, uv_pip_install, write_text


def install(settings: Settings) -> dict:
    repo_dir = settings.engines_dir / "Style-Bert-VITS2"
    ensure_git_clone("https://github.com/litagin02/Style-Bert-VITS2.git", repo_dir)
    python_bin = ensure_venv(repo_dir)
    uv_pip_install(
        python_bin,
        ["-e", ".", "fastapi", "uvicorn", "huggingface_hub", "scipy"],
        cwd=str(repo_dir),
    )
    download_code = (
        "from huggingface_hub import snapshot_download; "
        f"snapshot_download(repo_id={settings.style_bert_model_repo!r}, "
        f"local_dir={str(repo_dir / 'model_assets')!r}, "
        f"allow_patterns={[f'{settings.style_bert_model_subdir}/*']!r})"
    )
    run([str(python_bin), "-c", download_code], cwd=str(repo_dir))
    write_text(repo_dir / "openai_wrapper_app.py", settings.read_repo_text("src/apps/style_bert_app.py"))
    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "OPENAI_MODEL_ID": settings.openai_model_id or settings.style_bert_model_name,
        "STYLE_BERT_MODEL_ROOT": str(repo_dir / "model_assets"),
        "STYLE_BERT_MODEL_NAME": settings.style_bert_model_name,
        "STYLE_BERT_SPEAKER_ID": str(settings.style_bert_speaker_id),
        "STYLE_BERT_STYLE": settings.style_bert_style,
        "STYLE_BERT_DEVICE": "cuda" if os.path.exists("/usr/bin/nvidia-smi") else "cpu",
    }
    proc = popen(
        [
            str(repo_dir / ".venv" / "bin" / "uvicorn"),
            "openai_wrapper_app:app",
            "--host",
            "0.0.0.0",
            "--port",
            str(settings.app_port),
            "--log-level",
            "info",
        ],
        cwd=str(repo_dir),
        env=env,
        log_path=settings.log_dir / "style-bert-uvicorn.log",
    )
    return {"proc": proc, "app_dir": repo_dir, "log_path": settings.log_dir / "style-bert-uvicorn.log"}
