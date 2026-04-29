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
    parser.add_argument("--f5tts-model", default="F5TTS_v1_Base")
    parser.add_argument("--fish-speech-model", default="fishaudio/s2-pro")
    parser.add_argument("--f5tts-ckpt-file", default="")
    parser.add_argument("--f5tts-vocab-file", default="")
    # V1を利用する場合: "Aratako/Irodori-TTS-500M"
    parser.add_argument("--irodori-hf-checkpoint", default="Aratako/Irodori-TTS-500M-v2")
    # V1を利用する場合: "facebook/dacvae-watermarked"
    parser.add_argument("--irodori-codec-repo", default="Aratako/Semantic-DACVAE-Japanese-32dim")
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
    parser.add_argument("--piper-plus-model", default="tsukuyomi")
    parser.add_argument("--qwen3-hf-model", default="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")
    parser.add_argument("--qwen3-language", default="Japanese")
    parser.add_argument("--qwen3-default-speaker", default="ono_anna")
    parser.add_argument("--moss-tts-nano-hf-model", default="OpenMOSS-Team/MOSS-TTS-Nano-100M")
    parser.add_argument("--moss-tts-nano-mode", default="continuation")
    parser.add_argument("--neutts-backbone-repo", default="neuphonic/neutts-air")
    parser.add_argument("--neutts-codec-repo", default="neuphonic/neucodec")
    parser.add_argument("--neutts-default-voice", default="jo")
    parser.add_argument("--voxcpm-hf-model", default="openbmb/VoxCPM2")
    parser.add_argument("--voxcpm-cfg-value", type=float, default=2.0)
    parser.add_argument("--voxcpm-inference-timesteps", type=int, default=10)
    parser.add_argument("--voxtral-hf-model", default="mistralai/Voxtral-4B-TTS-2603")
    parser.add_argument("--voxtral-default-voice", default="neutral_female")
    parser.add_argument("--voxtral-backend-port", type=int, default=5001)
    parser.add_argument("--sarashina-hf-model", default="sbintuitions/sarashina2.2-tts")
    parser.add_argument("--sarashina-use-vllm", action="store_true")
    parser.add_argument("--sarashina-prompt-wav", default="")
    parser.add_argument("--sarashina-prompt-text", default="")
    parser.add_argument("--sarashina-default-voice", default="default")
    parser.add_argument("--chatterbox-language", default="ja")
    parser.add_argument("--chatterbox-prompt-wav", default="")
    parser.add_argument("--chatterbox-default-voice", default="default")
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
        f5tts_model=args.f5tts_model,
        f5tts_ckpt_file=args.f5tts_ckpt_file,
        f5tts_vocab_file=args.f5tts_vocab_file,
        fish_speech_model=args.fish_speech_model,
        irodori_hf_checkpoint=args.irodori_hf_checkpoint,
        irodori_codec_repo=args.irodori_codec_repo,
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
        piper_plus_model=args.piper_plus_model,
        qwen3_hf_model=args.qwen3_hf_model,
        qwen3_language=args.qwen3_language,
        qwen3_default_speaker=args.qwen3_default_speaker,
        moss_tts_nano_hf_model=args.moss_tts_nano_hf_model,
        moss_tts_nano_mode=args.moss_tts_nano_mode,
        neutts_backbone_repo=args.neutts_backbone_repo,
        neutts_codec_repo=args.neutts_codec_repo,
        neutts_default_voice=args.neutts_default_voice,
        voxcpm_hf_model=args.voxcpm_hf_model,
        voxcpm_cfg_value=args.voxcpm_cfg_value,
        voxcpm_inference_timesteps=args.voxcpm_inference_timesteps,
        voxtral_hf_model=args.voxtral_hf_model,
        voxtral_default_voice=args.voxtral_default_voice,
        voxtral_backend_port=args.voxtral_backend_port,
        sarashina_hf_model=args.sarashina_hf_model,
        sarashina_use_vllm=args.sarashina_use_vllm,
        sarashina_prompt_wav=args.sarashina_prompt_wav,
        sarashina_prompt_text=args.sarashina_prompt_text,
        sarashina_default_voice=args.sarashina_default_voice,
        chatterbox_language=args.chatterbox_language,
        chatterbox_prompt_wav=args.chatterbox_prompt_wav,
        chatterbox_default_voice=args.chatterbox_default_voice,
        root_dir=Path(args.root_dir),
        repo_dir=REPO_DIR,
    )
    launch(settings)


if __name__ == "__main__":
    main()
