from __future__ import annotations

import os

os.environ.setdefault("IRODORI_ENGINE_NAME", "Irodori-TTS-MF")
os.environ.setdefault("IRODORI_NUM_STEPS", "")

from irodori_app_shared import app  # noqa: E402,F401
