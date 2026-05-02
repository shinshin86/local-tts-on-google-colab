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
    parser.add_argument("--kyutai-hf-repo", default="kyutai/tts-1.6b-en_fr")
    parser.add_argument("--kyutai-voice-repo", default="kyutai/tts-voices")
    parser.add_argument(
        "--kyutai-voice", default="expresso/ex03-ex01_happy_001_channel1_334s.wav"
    )
    parser.add_argument("--kyutai-prompt-wav", default="")
    parser.add_argument("--kyutai-default-voice", default="default")
    parser.add_argument("--pocket-language", default="english")
    parser.add_argument("--pocket-default-speaker", default="alba")
    parser.add_argument("--pocket-prompt-wav", default="")
    parser.add_argument("--pocket-default-voice", default="default")
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
    parser.add_argument("--zonos-hf-model", default="Zyphra/Zonos-v0.1-transformer")
    parser.add_argument("--zonos-language", default="ja")
    parser.add_argument("--zonos-prompt-wav", default="")
    parser.add_argument("--zonos-default-voice", default="default")
    parser.add_argument("--outetts-model-size", default="0.6B")
    parser.add_argument("--outetts-backend", default="HF")
    parser.add_argument("--outetts-default-speaker", default="EN-FEMALE-1-NEUTRAL")
    parser.add_argument("--outetts-prompt-wav", default="")
    parser.add_argument("--outetts-prompt-text", default="")
    parser.add_argument("--outetts-default-voice", default="default")
    parser.add_argument("--dia-hf-model", default="nari-labs/Dia-1.6B-0626")
    parser.add_argument("--dia-compute-dtype", default="float16")
    parser.add_argument("--dia-prompt-wav", default="")
    parser.add_argument("--dia-prompt-text", default="")
    parser.add_argument("--dia-default-voice", default="default")
    parser.add_argument("--openvoice-language", default="JP")
    parser.add_argument("--openvoice-prompt-wav", default="")
    parser.add_argument("--openvoice-default-voice", default="default")
    parser.add_argument("--orpheus-hf-model", default="canopylabs/orpheus-tts-0.1-finetune-prod")
    parser.add_argument("--orpheus-default-voice", default="tara")
    parser.add_argument("--orpheus-max-model-len", type=int, default=2048)
    parser.add_argument("--cosyvoice-hf-model", default="FunAudioLLM/CosyVoice2-0.5B")
    parser.add_argument("--cosyvoice-prompt-wav", default="")
    parser.add_argument("--cosyvoice-prompt-text", default="")
    parser.add_argument("--cosyvoice-default-voice", default="default")
    parser.add_argument("--spark-hf-model", default="SparkAudio/Spark-TTS-0.5B")
    parser.add_argument("--spark-default-voice", default="default")
    parser.add_argument("--spark-default-gender", default="female")
    parser.add_argument("--spark-default-pitch", default="moderate")
    parser.add_argument("--spark-default-speed", default="moderate")
    parser.add_argument("--spark-prompt-wav", default="")
    parser.add_argument("--spark-prompt-text", default="")
    parser.add_argument("--vibevoice-hf-model", default="microsoft/VibeVoice-1.5B")
    parser.add_argument("--vibevoice-default-speaker", default="en-Alice_woman")
    parser.add_argument("--vibevoice-prompt-wav", default="")
    parser.add_argument("--vibevoice-default-voice", default="default")
    parser.add_argument("--vibevoice-ddpm-steps", type=int, default=10)
    parser.add_argument("--vibevoice-cfg-scale", type=float, default=1.3)
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
        kyutai_hf_repo=args.kyutai_hf_repo,
        kyutai_voice_repo=args.kyutai_voice_repo,
        kyutai_voice=args.kyutai_voice,
        kyutai_prompt_wav=args.kyutai_prompt_wav,
        kyutai_default_voice=args.kyutai_default_voice,
        pocket_language=args.pocket_language,
        pocket_default_speaker=args.pocket_default_speaker,
        pocket_prompt_wav=args.pocket_prompt_wav,
        pocket_default_voice=args.pocket_default_voice,
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
        zonos_hf_model=args.zonos_hf_model,
        zonos_language=args.zonos_language,
        zonos_prompt_wav=args.zonos_prompt_wav,
        zonos_default_voice=args.zonos_default_voice,
        outetts_model_size=args.outetts_model_size,
        outetts_backend=args.outetts_backend,
        outetts_default_speaker=args.outetts_default_speaker,
        outetts_prompt_wav=args.outetts_prompt_wav,
        outetts_prompt_text=args.outetts_prompt_text,
        outetts_default_voice=args.outetts_default_voice,
        dia_hf_model=args.dia_hf_model,
        dia_compute_dtype=args.dia_compute_dtype,
        dia_prompt_wav=args.dia_prompt_wav,
        dia_prompt_text=args.dia_prompt_text,
        dia_default_voice=args.dia_default_voice,
        openvoice_language=args.openvoice_language,
        openvoice_prompt_wav=args.openvoice_prompt_wav,
        openvoice_default_voice=args.openvoice_default_voice,
        orpheus_hf_model=args.orpheus_hf_model,
        orpheus_default_voice=args.orpheus_default_voice,
        orpheus_max_model_len=args.orpheus_max_model_len,
        cosyvoice_hf_model=args.cosyvoice_hf_model,
        cosyvoice_prompt_wav=args.cosyvoice_prompt_wav,
        cosyvoice_prompt_text=args.cosyvoice_prompt_text,
        cosyvoice_default_voice=args.cosyvoice_default_voice,
        spark_hf_model=args.spark_hf_model,
        spark_default_voice=args.spark_default_voice,
        spark_default_gender=args.spark_default_gender,
        spark_default_pitch=args.spark_default_pitch,
        spark_default_speed=args.spark_default_speed,
        spark_prompt_wav=args.spark_prompt_wav,
        spark_prompt_text=args.spark_prompt_text,
        vibevoice_hf_model=args.vibevoice_hf_model,
        vibevoice_default_speaker=args.vibevoice_default_speaker,
        vibevoice_prompt_wav=args.vibevoice_prompt_wav,
        vibevoice_default_voice=args.vibevoice_default_voice,
        vibevoice_ddpm_steps=args.vibevoice_ddpm_steps,
        vibevoice_cfg_scale=args.vibevoice_cfg_scale,
        root_dir=Path(args.root_dir),
        repo_dir=REPO_DIR,
    )
    launch(settings)


if __name__ == "__main__":
    main()
