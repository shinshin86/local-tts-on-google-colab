from __future__ import annotations

from src.config import Settings

from .irodori import install_runtime


def install(settings: Settings) -> dict:
    return install_runtime(
        settings,
        engine_dir_name="Irodori-TTS-MF",
        app_source="src/apps/irodori_mf_app.py",
        checkpoint=settings.irodori_mf_hf_checkpoint,
        codec_repo=settings.irodori_mf_codec_repo,
        model_precision=settings.irodori_mf_model_precision,
        codec_precision=settings.irodori_mf_codec_precision,
        engine_name="Irodori-TTS-MF",
        log_filename="irodori-mf-uvicorn.log",
        num_steps=None,
    )
