from __future__ import annotations

from src.config import Settings

from .irodori import install_runtime


def install(settings: Settings) -> dict:
    return install_runtime(
        settings,
        engine_dir_name="Irodori-TTS-Anime",
        app_source="src/apps/irodori_anime_app.py",
        checkpoint=settings.irodori_anime_hf_checkpoint,
        codec_repo=settings.irodori_anime_codec_repo,
        model_precision=settings.irodori_anime_model_precision,
        codec_precision=settings.irodori_anime_codec_precision,
        engine_name="Irodori-TTS-Anime",
        log_filename="irodori-anime-uvicorn.log",
    )
