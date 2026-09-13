from __future__ import annotations

import os

os.environ.setdefault("IRODORI_ENGINE_NAME", "Irodori-TTS-Anime")
os.environ.setdefault("IRODORI_DEFAULT_VOICE_ONLY", "1")

from irodori_app_shared import app  # noqa: E402

__all__ = ["app"]
