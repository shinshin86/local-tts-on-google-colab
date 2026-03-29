from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


KOKORO_VOICE_PRESETS = [
    "jf_alpha",
    "jf_gongitsune",
    "jm_kumo",
    "af_heart",
    "af_bella",
    "am_adam",
    "bf_emma",
    "bm_george",
    "zf_xiaobei",
]

MELO_VOICE_PRESETS = [
    "JP",
    "EN-Default",
    "EN-US",
    "EN-BR",
    "EN_INDIA",
    "EN-AU",
    "ZH",
    "ES",
    "FR",
    "KR",
]


def default_repo_dir() -> Path:
    return Path(__file__).resolve().parent.parent


@dataclass
class Settings:
    engine: str
    expose_public_url: bool = True
    dry_run: bool = False
    test_text: str = "こんにちは。これは OpenAI 互換 TTS の動作確認です。"
    test_speed: float = 1.0
    test_voice: str = ""
    openai_model_id: str = ""
    irodori_hf_checkpoint: str = "Aratako/Irodori-TTS-500M"
    irodori_model_precision: str = "fp32"
    irodori_codec_precision: str = "fp32"
    kokoro_default_voice: str = "jf_alpha"
    kokoro_default_lang_code: str = "j"
    melo_language: str = "JP"
    melo_default_voice: str = "JP"
    style_bert_model_repo: str = "litagin/style_bert_vits2_jvnv"
    style_bert_model_subdir: str = "jvnv-F2-jp"
    style_bert_model_name: str = "jvnv-F2-jp"
    style_bert_speaker_id: int = 0
    style_bert_style: str = "Neutral"
    piper_voice: str = "en_US-lessac-medium"
    piper_speaker_id: int = -1
    xtts_language: str = "ja"
    xtts_speaker_wav: str = ""
    qwen3_hf_model: str = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
    qwen3_language: str = "Japanese"
    qwen3_default_speaker: str = "Chelsie"
    root_dir: Path = Path("/content/openai-compatible-local-tts")
    repo_dir: Path = field(default_factory=default_repo_dir)
    app_port: int = 8000
    piper_backend_port: int = 5000

    @property
    def engines_dir(self) -> Path:
        return self.root_dir / "engines"

    @property
    def log_dir(self) -> Path:
        return self.root_dir / "logs"

    @property
    def output_dir(self) -> Path:
        return self.root_dir / "outputs"

    @property
    def cloudflared_path(self) -> Path:
        return self.root_dir / "cloudflared"

    def ensure_directories(self) -> None:
        for path in (self.root_dir, self.engines_dir, self.log_dir, self.output_dir):
            path.mkdir(parents=True, exist_ok=True)

    def read_repo_text(self, relative_path: str) -> str:
        source_path = self.repo_dir / relative_path
        if not source_path.exists():
            raise FileNotFoundError(
                f"Required source file not found: {source_path}. "
                "Run this script from a cloned local-tts-on-google-colab repository."
            )
        return source_path.read_text(encoding="utf-8")
