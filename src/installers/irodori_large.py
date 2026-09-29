from __future__ import annotations

from src.config import Settings

from .irodori import install_runtime


# First upstream revision with v4-Large, T5Gemma 2, and torchao quantized
# checkpoint support.
IRODORI_V4_LARGE_UPSTREAM_REF = "89f9d8fbd4d51ea019867ee1197725ede1df13c5"


def install(settings: Settings) -> dict:
    return install_runtime(
        settings,
        engine_dir_name="Irodori-TTS-Large",
        app_source="src/apps/irodori_large_app.py",
        checkpoint=settings.irodori_large_hf_checkpoint,
        codec_repo=settings.irodori_large_codec_repo,
        model_precision=settings.irodori_large_model_precision,
        codec_precision=settings.irodori_large_codec_precision,
        engine_name="Irodori-TTS-Large",
        log_filename="irodori-large-uvicorn.log",
        num_steps=40,
        prompt_wav=settings.irodori_large_prompt_wav,
        default_voice=settings.irodori_large_default_voice,
        default_instructions=settings.irodori_large_default_instructions,
        instructions_enabled=True,
        prompt_flag="--irodori-large-prompt-wav",
        upstream_ref=IRODORI_V4_LARGE_UPSTREAM_REF,
    )
