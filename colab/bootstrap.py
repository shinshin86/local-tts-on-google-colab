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
    # V1: "Aratako/Irodori-TTS-500M", V2: "Aratako/Irodori-TTS-500M-v2"
    parser.add_argument("--irodori-hf-checkpoint", default="Aratako/Irodori-TTS-500M-v3")
    # V1を利用する場合: "facebook/dacvae-watermarked"
    parser.add_argument("--irodori-codec-repo", default="Aratako/Semantic-DACVAE-Japanese-32dim")
    parser.add_argument("--irodori-model-precision", default="fp32")
    parser.add_argument("--irodori-codec-precision", default="fp32")
    parser.add_argument(
        "--irodori-lite-hf-checkpoint", default="kizuna-intelligence/Irodori-TTS-Lite-int4"
    )
    parser.add_argument("--irodori-lite-checkpoint-file", default="dit_int4.safetensors")
    parser.add_argument(
        "--irodori-lite-codec-repo", default="Aratako/Semantic-DACVAE-Japanese-32dim"
    )
    parser.add_argument("--irodori-lite-codec-int4", action="store_true")
    parser.add_argument("--kokoro-default-voice", default="jf_alpha")
    parser.add_argument("--kokoro-default-lang-code", default="j")
    parser.add_argument("--kokoro-onnx-hf-model", default="nvidia/kokoro-82M-onnx-opt")
    parser.add_argument("--kokoro-onnx-default-voice", default="jf_alpha")
    parser.add_argument("--kokoro-onnx-default-lang-code", default="j")
    parser.add_argument("--kokoro-onnx-provider", default="auto", choices=["auto", "cuda", "cpu"])
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
    parser.add_argument("--moss-tts-v1-5-hf-model", default="OpenMOSS-Team/MOSS-TTS-v1.5")
    parser.add_argument("--moss-tts-v1-5-language", default="Japanese")
    parser.add_argument("--moss-tts-v1-5-prompt-wav", default="")
    parser.add_argument("--moss-tts-v1-5-default-voice", default="default")
    parser.add_argument("--moss-tts-v1-5-attn-impl", default="sdpa")
    parser.add_argument("--moss-tts-v1-5-max-new-tokens", type=int, default=4096)
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
    parser.add_argument("--bark-default-voice", default="v2/en_speaker_6")
    parser.add_argument("--bark-use-small-models", action="store_true")
    parser.add_argument("--chattts-default-voice", default="default")
    parser.add_argument("--chattts-seed", type=int, default=2)
    parser.add_argument("--chattts-temperature", type=float, default=0.3)
    parser.add_argument("--csm-hf-model", default="sesame/csm-1b")
    parser.add_argument("--csm-llama-model", default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--csm-default-voice", default="default")
    parser.add_argument("--csm-default-speaker", type=int, default=0)
    parser.add_argument("--csm-max-audio-length-ms", type=int, default=10000)
    parser.add_argument("--csm-temperature", type=float, default=0.9)
    parser.add_argument("--misotts-hf-model", default="MisoLabs/MisoTTS")
    parser.add_argument("--misotts-default-voice", default="default")
    parser.add_argument("--misotts-default-speaker", type=int, default=0)
    parser.add_argument("--misotts-prompt-wav", default="")
    parser.add_argument("--misotts-prompt-text", default="")
    parser.add_argument("--misotts-max-audio-length-ms", type=int, default=30000)
    parser.add_argument("--misotts-temperature", type=float, default=0.9)
    parser.add_argument("--misotts-topk", type=int, default=50)
    parser.add_argument("--misotts-tokenizer-repo", default="unsloth/Llama-3.2-1B")
    parser.add_argument("--styletts2-default-voice", default="default")
    parser.add_argument("--styletts2-prompt-wav", default="")
    parser.add_argument("--styletts2-alpha", type=float, default=0.3)
    parser.add_argument("--styletts2-beta", type=float, default=0.7)
    parser.add_argument("--styletts2-diffusion-steps", type=int, default=5)
    parser.add_argument("--styletts2-embedding-scale", type=float, default=1.0)
    parser.add_argument("--maskgct-default-voice", default="default")
    parser.add_argument("--maskgct-prompt-wav", default="")
    parser.add_argument("--maskgct-prompt-text", default="")
    parser.add_argument("--maskgct-prompt-lang", default="en")
    parser.add_argument("--maskgct-target-lang", default="en")
    parser.add_argument("--gpt-sovits-version", default="v2")
    parser.add_argument("--gpt-sovits-default-voice", default="default")
    parser.add_argument("--gpt-sovits-prompt-wav", default="")
    parser.add_argument("--gpt-sovits-prompt-text", default="")
    parser.add_argument("--gpt-sovits-prompt-lang", default="en")
    parser.add_argument("--gpt-sovits-target-lang", default="en")
    parser.add_argument("--higgs-hf-model", default="bosonai/higgs-audio-v2-generation-3B-base")
    parser.add_argument("--higgs-hf-tokenizer", default="bosonai/higgs-audio-v2-tokenizer")
    parser.add_argument("--higgs-default-voice", default="default")
    parser.add_argument("--higgs-default-ref-voice", default="belinda")
    parser.add_argument("--higgs-prompt-wav", default="")
    parser.add_argument("--higgs-prompt-text", default="")
    parser.add_argument("--higgs-max-new-tokens", type=int, default=1024)
    parser.add_argument("--higgs-temperature", type=float, default=0.7)
    parser.add_argument("--higgs-v3-hf-model", default="bosonai/higgs-audio-v3-tts-4b")
    parser.add_argument("--higgs-v3-default-voice", default="default")
    parser.add_argument("--higgs-v3-prompt-wav", default="")
    parser.add_argument("--higgs-v3-prompt-text", default="")
    parser.add_argument("--higgs-v3-temperature", type=float, default=0.7)
    parser.add_argument("--higgs-v3-top-k", type=int, default=50)
    parser.add_argument("--higgs-v3-max-new-tokens", type=int, default=2048)
    parser.add_argument("--higgs-v3-backend-port", type=int, default=5002)
    parser.add_argument("--dots-tts-hf-model", default="rednote-hilab/dots.tts-base")
    parser.add_argument("--dots-tts-default-voice", default="default")
    parser.add_argument("--dots-tts-prompt-wav", default="")
    parser.add_argument("--dots-tts-prompt-text", default="")
    parser.add_argument("--dots-tts-language", default="auto_detect")
    parser.add_argument("--dots-tts-precision", default="bfloat16")
    parser.add_argument("--dots-tts-num-steps", type=int, default=10)
    parser.add_argument("--dots-tts-guidance-scale", type=float, default=1.2)
    parser.add_argument("--dots-tts-speaker-scale", type=float, default=1.5)
    parser.add_argument("--dots-tts-max-generate-length", type=int, default=500)
    parser.add_argument("--lfm2-audio-jp-hf-model", default="LiquidAI/LFM2.5-Audio-1.5B-JP")
    parser.add_argument("--lfm2-audio-jp-system-prompt", default="Perform TTS in japanese.")
    parser.add_argument("--lfm2-audio-jp-max-new-tokens", type=int, default=1024)
    parser.add_argument("--lfm2-audio-jp-audio-temperature", type=float, default=0.8)
    parser.add_argument("--lfm2-audio-jp-audio-top-k", type=int, default=64)
    parser.add_argument("--supertonic-model", default="supertonic-3")
    parser.add_argument("--supertonic-default-voice", default="M1")
    parser.add_argument("--supertonic-default-lang", default="en")
    parser.add_argument("--supertonic-total-steps", type=int, default=5)
    parser.add_argument("--dramabox-hf-model", default="ResembleAI/Dramabox")
    parser.add_argument("--dramabox-gemma-repo", default="unsloth/gemma-3-12b-it-bnb-4bit")
    parser.add_argument("--dramabox-default-voice", default="default")
    parser.add_argument("--dramabox-default-ref-voice", default="female_american")
    parser.add_argument("--dramabox-prompt-wav", default="")
    parser.add_argument("--dramabox-dtype", default="bf16")
    parser.add_argument("--dramabox-cfg-scale", type=float, default=2.5)
    parser.add_argument("--dramabox-stg-scale", type=float, default=1.5)
    parser.add_argument("--dramabox-duration-multiplier", type=float, default=1.1)
    parser.add_argument("--dramabox-seed", type=int, default=42)
    parser.add_argument("--dramabox-compile", action="store_true")
    parser.add_argument("--dramabox-no-bnb-4bit", action="store_true")
    parser.add_argument("--scenema-default-voice", default="default")
    parser.add_argument("--scenema-default-gender", default="male")
    parser.add_argument("--scenema-prompt-wav", default="")
    parser.add_argument("--scenema-gemma-quantize", default="nf4")
    parser.add_argument("--scenema-seed", type=int, default=-1)
    parser.add_argument("--scenema-pace", type=float, default=1.5)
    parser.add_argument("--scenema-no-validate", action="store_true")
    parser.add_argument("--scenema-min-match-ratio", type=float, default=0.90)
    parser.add_argument("--scenema-skip-vc", action="store_true")
    parser.add_argument("--scenema-vc-steps", type=int, default=25)
    parser.add_argument("--scenema-vc-cfg-rate", type=float, default=0.5)
    parser.add_argument("--scenema-background-sfx", action="store_true")
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
        irodori_lite_hf_checkpoint=args.irodori_lite_hf_checkpoint,
        irodori_lite_checkpoint_file=args.irodori_lite_checkpoint_file,
        irodori_lite_codec_repo=args.irodori_lite_codec_repo,
        irodori_lite_codec_int4=args.irodori_lite_codec_int4,
        kokoro_default_voice=args.kokoro_default_voice,
        kokoro_default_lang_code=args.kokoro_default_lang_code,
        kokoro_onnx_hf_model=args.kokoro_onnx_hf_model,
        kokoro_onnx_default_voice=args.kokoro_onnx_default_voice,
        kokoro_onnx_default_lang_code=args.kokoro_onnx_default_lang_code,
        kokoro_onnx_provider=args.kokoro_onnx_provider,
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
        moss_tts_v1_5_hf_model=args.moss_tts_v1_5_hf_model,
        moss_tts_v1_5_language=args.moss_tts_v1_5_language,
        moss_tts_v1_5_prompt_wav=args.moss_tts_v1_5_prompt_wav,
        moss_tts_v1_5_default_voice=args.moss_tts_v1_5_default_voice,
        moss_tts_v1_5_attn_impl=args.moss_tts_v1_5_attn_impl,
        moss_tts_v1_5_max_new_tokens=args.moss_tts_v1_5_max_new_tokens,
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
        bark_default_voice=args.bark_default_voice,
        bark_use_small_models=args.bark_use_small_models,
        chattts_default_voice=args.chattts_default_voice,
        chattts_seed=args.chattts_seed,
        chattts_temperature=args.chattts_temperature,
        csm_hf_model=args.csm_hf_model,
        csm_llama_model=args.csm_llama_model,
        csm_default_voice=args.csm_default_voice,
        csm_default_speaker=args.csm_default_speaker,
        csm_max_audio_length_ms=args.csm_max_audio_length_ms,
        csm_temperature=args.csm_temperature,
        misotts_hf_model=args.misotts_hf_model,
        misotts_default_voice=args.misotts_default_voice,
        misotts_default_speaker=args.misotts_default_speaker,
        misotts_prompt_wav=args.misotts_prompt_wav,
        misotts_prompt_text=args.misotts_prompt_text,
        misotts_max_audio_length_ms=args.misotts_max_audio_length_ms,
        misotts_temperature=args.misotts_temperature,
        misotts_topk=args.misotts_topk,
        misotts_tokenizer_repo=args.misotts_tokenizer_repo,
        styletts2_default_voice=args.styletts2_default_voice,
        styletts2_prompt_wav=args.styletts2_prompt_wav,
        styletts2_alpha=args.styletts2_alpha,
        styletts2_beta=args.styletts2_beta,
        styletts2_diffusion_steps=args.styletts2_diffusion_steps,
        styletts2_embedding_scale=args.styletts2_embedding_scale,
        maskgct_default_voice=args.maskgct_default_voice,
        maskgct_prompt_wav=args.maskgct_prompt_wav,
        maskgct_prompt_text=args.maskgct_prompt_text,
        maskgct_prompt_lang=args.maskgct_prompt_lang,
        maskgct_target_lang=args.maskgct_target_lang,
        gpt_sovits_version=args.gpt_sovits_version,
        gpt_sovits_default_voice=args.gpt_sovits_default_voice,
        gpt_sovits_prompt_wav=args.gpt_sovits_prompt_wav,
        gpt_sovits_prompt_text=args.gpt_sovits_prompt_text,
        gpt_sovits_prompt_lang=args.gpt_sovits_prompt_lang,
        gpt_sovits_target_lang=args.gpt_sovits_target_lang,
        higgs_hf_model=args.higgs_hf_model,
        higgs_hf_tokenizer=args.higgs_hf_tokenizer,
        higgs_default_voice=args.higgs_default_voice,
        higgs_default_ref_voice=args.higgs_default_ref_voice,
        higgs_prompt_wav=args.higgs_prompt_wav,
        higgs_prompt_text=args.higgs_prompt_text,
        higgs_max_new_tokens=args.higgs_max_new_tokens,
        higgs_temperature=args.higgs_temperature,
        higgs_v3_hf_model=args.higgs_v3_hf_model,
        higgs_v3_default_voice=args.higgs_v3_default_voice,
        higgs_v3_prompt_wav=args.higgs_v3_prompt_wav,
        higgs_v3_prompt_text=args.higgs_v3_prompt_text,
        higgs_v3_temperature=args.higgs_v3_temperature,
        higgs_v3_top_k=args.higgs_v3_top_k,
        higgs_v3_max_new_tokens=args.higgs_v3_max_new_tokens,
        higgs_v3_backend_port=args.higgs_v3_backend_port,
        dots_tts_hf_model=args.dots_tts_hf_model,
        dots_tts_default_voice=args.dots_tts_default_voice,
        dots_tts_prompt_wav=args.dots_tts_prompt_wav,
        dots_tts_prompt_text=args.dots_tts_prompt_text,
        dots_tts_language=args.dots_tts_language,
        dots_tts_precision=args.dots_tts_precision,
        dots_tts_num_steps=args.dots_tts_num_steps,
        dots_tts_guidance_scale=args.dots_tts_guidance_scale,
        dots_tts_speaker_scale=args.dots_tts_speaker_scale,
        dots_tts_max_generate_length=args.dots_tts_max_generate_length,
        lfm2_audio_jp_hf_model=args.lfm2_audio_jp_hf_model,
        lfm2_audio_jp_system_prompt=args.lfm2_audio_jp_system_prompt,
        lfm2_audio_jp_max_new_tokens=args.lfm2_audio_jp_max_new_tokens,
        lfm2_audio_jp_audio_temperature=args.lfm2_audio_jp_audio_temperature,
        lfm2_audio_jp_audio_top_k=args.lfm2_audio_jp_audio_top_k,
        supertonic_model=args.supertonic_model,
        supertonic_default_voice=args.supertonic_default_voice,
        supertonic_default_lang=args.supertonic_default_lang,
        supertonic_total_steps=args.supertonic_total_steps,
        dramabox_hf_model=args.dramabox_hf_model,
        dramabox_gemma_repo=args.dramabox_gemma_repo,
        dramabox_default_voice=args.dramabox_default_voice,
        dramabox_default_ref_voice=args.dramabox_default_ref_voice,
        dramabox_prompt_wav=args.dramabox_prompt_wav,
        dramabox_dtype=args.dramabox_dtype,
        dramabox_cfg_scale=args.dramabox_cfg_scale,
        dramabox_stg_scale=args.dramabox_stg_scale,
        dramabox_duration_multiplier=args.dramabox_duration_multiplier,
        dramabox_seed=args.dramabox_seed,
        dramabox_compile=args.dramabox_compile,
        dramabox_bnb_4bit=not args.dramabox_no_bnb_4bit,
        scenema_default_voice=args.scenema_default_voice,
        scenema_default_gender=args.scenema_default_gender,
        scenema_prompt_wav=args.scenema_prompt_wav,
        scenema_gemma_quantize=args.scenema_gemma_quantize,
        scenema_seed=args.scenema_seed,
        scenema_pace=args.scenema_pace,
        scenema_validate=not args.scenema_no_validate,
        scenema_min_match_ratio=args.scenema_min_match_ratio,
        scenema_skip_vc=args.scenema_skip_vc,
        scenema_vc_steps=args.scenema_vc_steps,
        scenema_vc_cfg_rate=args.scenema_vc_cfg_rate,
        scenema_background_sfx=args.scenema_background_sfx,
        root_dir=Path(args.root_dir),
        repo_dir=REPO_DIR,
    )
    launch(settings)


if __name__ == "__main__":
    main()
