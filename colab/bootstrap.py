from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from src.config import Settings
from src.installers import INSTALLERS
from src.launcher import launch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True, choices=sorted(INSTALLERS.keys()))
    parser.set_defaults(expose_public_url=True)
    parser.add_argument("--expose-public-url", dest="expose_public_url", action="store_true")
    parser.add_argument("--no-expose-public-url", dest="expose_public_url", action="store_false")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--test-text", default="こんにちは。これは OpenAI 互換 TTS の動作確認です。")
    parser.add_argument("--test-speed", type=float, default=1.0)
    parser.add_argument("--test-voice", default="")
    parser.add_argument("--openai-model-id", default="")
    parser.add_argument("--irodori-hf-checkpoint", default="Aratako/Irodori-TTS-500M")
    parser.add_argument("--irodori-model-precision", default="fp32")
    parser.add_argument("--irodori-codec-precision", default="fp32")
    parser.add_argument("--kokoro-default-voice", default="jf_alpha")
    parser.add_argument("--kokoro-default-lang-code", default="j")
    parser.add_argument("--melo-language", default="JP")
    parser.add_argument("--melo-default-voice", default="JP")
    parser.add_argument("--style-bert-model-repo", default="litagin/style_bert_vits2_jvnv")
    parser.add_argument("--style-bert-model-subdir", default="jvnv-F2-jp")
    parser.add_argument("--style-bert-model-name", default="jvnv-F2-jp")
    parser.add_argument("--style-bert-speaker-id", type=int, default=0)
    parser.add_argument("--style-bert-style", default="Neutral")
    parser.add_argument("--piper-voice", default="en_US-lessac-medium")
    parser.add_argument("--piper-speaker-id", type=int, default=-1)
    parser.add_argument("--root-dir", default="/content/openai-compatible-local-tts")
    return parser.parse_args()


def main():
    args = parse_args()
    settings = Settings(
        engine=args.engine,
        expose_public_url=args.expose_public_url,
        dry_run=args.dry_run,
        test_text=args.test_text,
        test_speed=args.test_speed,
        test_voice=args.test_voice,
        openai_model_id=args.openai_model_id,
        irodori_hf_checkpoint=args.irodori_hf_checkpoint,
        irodori_model_precision=args.irodori_model_precision,
        irodori_codec_precision=args.irodori_codec_precision,
        kokoro_default_voice=args.kokoro_default_voice,
        kokoro_default_lang_code=args.kokoro_default_lang_code,
        melo_language=args.melo_language,
        melo_default_voice=args.melo_default_voice,
        style_bert_model_repo=args.style_bert_model_repo,
        style_bert_model_subdir=args.style_bert_model_subdir,
        style_bert_model_name=args.style_bert_model_name,
        style_bert_speaker_id=args.style_bert_speaker_id,
        style_bert_style=args.style_bert_style,
        piper_voice=args.piper_voice,
        piper_speaker_id=args.piper_speaker_id,
        root_dir=Path(args.root_dir),
        repo_dir=REPO_DIR,
    )
    launch(settings)


if __name__ == "__main__":
    main()
