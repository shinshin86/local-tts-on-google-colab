from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
import time

from src.config import (
    KOKORO_ONNX_VOICE_PRESETS,
    KOKORO_VOICE_PRESETS,
    MELO_VOICE_PRESETS,
    NEUTTS_VOICE_PRESETS,
    Settings,
)
from src.installers import INSTALLERS
from src.runtime import (
    ensure_cloudflared,
    ensure_uv,
    kill_old_processes,
    pretty_print_json_url,
    run,
    tail_log,
    wait_http,
)


def resolve_selected_voice(settings: Settings) -> str:
    if settings.test_voice:
        return settings.test_voice
    if settings.engine == "Kokoro":
        return settings.kokoro_default_voice
    if settings.engine == "Kokoro-ONNX":
        return settings.kokoro_onnx_default_voice
    if settings.engine == "MeloTTS":
        return settings.melo_default_voice
    if settings.engine == "NeuTTS":
        return settings.neutts_default_voice
    if settings.engine == "Sarashina-TTS":
        return settings.sarashina_default_voice
    if settings.engine == "Chatterbox":
        return settings.chatterbox_default_voice
    if settings.engine == "Zonos":
        return settings.zonos_default_voice
    if settings.engine == "OuteTTS":
        return settings.outetts_default_voice
    if settings.engine == "Dia":
        return settings.dia_default_voice
    if settings.engine == "OpenVoice-V2":
        return settings.openvoice_default_voice
    if settings.engine == "VibeVoice":
        return settings.vibevoice_default_voice
    if settings.engine == "Kyutai-TTS":
        return settings.kyutai_default_voice
    if settings.engine == "Pocket-TTS":
        return settings.pocket_default_voice
    if settings.engine == "Orpheus-TTS":
        return settings.orpheus_default_voice
    if settings.engine == "CosyVoice2":
        return settings.cosyvoice_default_voice
    if settings.engine == "Spark-TTS":
        return settings.spark_default_voice
    if settings.engine == "Bark":
        return settings.bark_default_voice
    if settings.engine == "ChatTTS":
        return settings.chattts_default_voice
    if settings.engine == "CSM-1B":
        return settings.csm_default_voice
    if settings.engine == "StyleTTS2":
        return settings.styletts2_default_voice
    if settings.engine == "MaskGCT":
        return settings.maskgct_default_voice
    if settings.engine == "GPT-SoVITS":
        return settings.gpt_sovits_default_voice
    if settings.engine == "Higgs-Audio-v2":
        return settings.higgs_default_voice
    if settings.engine == "Supertonic":
        return settings.supertonic_default_voice
    if settings.engine == "DramaBox":
        return settings.dramabox_default_voice
    if settings.engine == "Scenema":
        return settings.scenema_default_voice
    if settings.engine == "MOSS-TTS-v1.5":
        return settings.moss_tts_v1_5_default_voice
    return ""


def print_engine_voice_hints(settings: Settings):
    print("\n=== Voice Selection Hint ===")
    if settings.engine == "Fish-Speech":
        print("Fish Speech は fishaudio の高品質 TTS です（80言語以上対応）。")
        print(f"モデル: {settings.fish_speech_model}")
        print("voice パラメータは現在 'default'（ランダム音声）のみ対応です。")
        print("日本語は Tier 1 サポート（最高品質）。")
        print("注意: A100/L4 GPU 推奨（VRAM 24GB以上必要）。ライセンス: Apache-2.0")
    elif settings.engine == "F5-TTS":
        print("F5-TTS はゼロショット音声クローニング TTS です（英語・中国語）。")
        print(f"モデル: {settings.f5tts_model}")
        print("デフォルトの参照音声（英語女性）を使用します。")
        print("日本語モデルを使う場合は --f5tts-ckpt-file / --f5tts-vocab-file を指定してください。")
        print("注意: GPU 推奨（VRAM ~2-4GB）。ライセンス: CC-BY-NC（モデル）")
    elif settings.engine == "Kokoro":
        print("Kokoro はフォームで音声を選択できます。")
        print("候補:", ", ".join(KOKORO_VOICE_PRESETS))
    elif settings.engine == "Kokoro-ONNX":
        print("Kokoro-ONNX は NVIDIA が最適化した Kokoro-82M の ONNX 版です（onnxruntime で実行、GPU/CPU 両対応）。")
        print(f"モデル: {settings.kokoro_onnx_hf_model}")
        print(f"provider: {settings.kokoro_onnx_provider}（auto/cuda は GPU 優先・CPU フォールバック、cpu は CPU 強制）")
        print(f"デフォルト voice: {settings.kokoro_onnx_default_voice}")
        print("候補(代表例):", ", ".join(KOKORO_ONNX_VOICE_PRESETS))
        print("全 53 voice は起動後の /v1/voices を参照してください。")
        print("言語は voice 名の接頭辞から自動判定します:")
        print("  a/b=英語(米/英), e=西語, f=仏語, h=ヒンディー, i=伊語, j=日本語, p=ポルトガル語(ブラジル), z=中国語")
        print("音素化は misaki（j=ja, z=zh, a/b=en+espeak fallback, e/f/h/i/p=espeak）で行います。")
        print("注意: GPU 不要でも動作（82M と軽量）。Voice cloning は未対応（preset のみ）。")
        print("ライセンス: コード / 重みとも Apache-2.0（商用 OK）。ベースは hexgrad/Kokoro-82M。")
    elif settings.engine == "MeloTTS":
        print("MeloTTS はフォームで代表的な voice を選択できます。")
        print("候補:", ", ".join(MELO_VOICE_PRESETS))
        print(f"現在の language: {settings.melo_language}")
    elif settings.engine == "Style-Bert-VITS2":
        print("Style-Bert-VITS2 は speaker_id ベースです。起動後の /v1/voices を確認して speaker_id を選んでください。")
    elif settings.engine == "Piper":
        print("Piper は voice 名がそのままモデル指定です。必要なら PIPER_VOICE を変更してください。")
    elif settings.engine == "Piper-Plus":
        print("Piper-Plus は日本語対応の軽量 TTS です（MIT ライセンス）。")
        print(f"モデル: {settings.piper_plus_model}")
        print("デフォルトは tsukuyomi（日本語女性）。GPU 不要。")
    elif settings.engine == "Qwen3-TTS":
        print("Qwen3-TTS は多言語対応の高品質 TTS です。")
        print(f"モデル: {settings.qwen3_hf_model}")
        print(f"language: {settings.qwen3_language}")
        print(f"デフォルト speaker: {settings.qwen3_default_speaker}")
        print("候補: aiden, dylan, eric, ono_anna, ryan, serena, sohee, uncle_fu, vivian")
    elif settings.engine == "VoxCPM2":
        print("VoxCPM2 は OpenBMB の高品質 TTS です（30言語対応、言語自動検出）。")
        print(f"モデル: {settings.voxcpm_hf_model}")
        print("voice パラメータは現在 'default' のみ対応です。")
        print("注意: GPU 推奨（VRAM ~8GB）。ライセンス: Apache-2.0")
    elif settings.engine == "Voxtral-TTS":
        print("Voxtral-TTS は Mistral AI の高品質 TTS です（vLLM バックエンド）。")
        print(f"モデル: {settings.voxtral_hf_model}")
        print(f"デフォルト voice: {settings.voxtral_default_voice}")
        print("候補: ar_male, casual_female, casual_male, cheerful_female, de_female, de_male,")
        print("      es_female, es_male, fr_female, fr_male, hi_female, hi_male, it_female,")
        print("      it_male, neutral_female, neutral_male, nl_female, nl_male, pt_female, pt_male")
        print("対応言語: en, fr, es, pt, it, nl, de, ar, hi")
        print("注意: A100 GPU 推奨（T4 では VRAM 不足の可能性あり）。ライセンス: CC BY-NC 4.0")
    elif settings.engine == "NeuTTS":
        print("NeuTTS は Neuphonic のオンデバイス TTS です（インスタント voice cloning、CPU 動作可）。")
        print(f"backbone: {settings.neutts_backbone_repo}")
        print(f"codec   : {settings.neutts_codec_repo}")
        print(f"デフォルト voice: {settings.neutts_default_voice}")
        print("候補:", ", ".join(NEUTTS_VOICE_PRESETS), "(dave/jo=英語, mateo=西語, greta=独語, juliette=仏語)")
        print("対応言語: 英語 / 西語 / 独語 / 仏語（日本語非対応）。")
        print("注意: 非英語の voice を使う場合は、対応言語の Nano backbone を指定してください。")
        print("ライセンス: neutts-air=Apache-2.0 / neutts-nano=NeuTTS Open License 1.0")
    elif settings.engine == "Sarashina-TTS":
        print("Sarashina-TTS は SB Intuitions の日本語中心 TTS です（日本語＋英語、ゼロショット音声クローン対応）。")
        print(f"モデル: {settings.sarashina_hf_model}")
        print(f"vLLM backend: {'有効' if settings.sarashina_use_vllm else '無効（HuggingFace transformers）'}")
        print(f"デフォルト voice: {settings.sarashina_default_voice}")
        print("voice 候補: default（プロンプトなしの plain TTS）")
        if settings.sarashina_prompt_wav and settings.sarashina_prompt_text:
            print(f"             clone（参照音声: {settings.sarashina_prompt_wav}）")
        else:
            print("             clone は --sarashina-prompt-wav / --sarashina-prompt-text を指定すると有効になります")
        print("注意: GPU 推奨（VRAM ~6GB / vLLM はさらに必要）。日本語は MOS が高い tier 1 サポート。")
        print("ライセンス: Sarashina Model NonCommercial License Agreement（商用利用不可）")
        print("生成音声には SilentCipher による不可聴ウォーターマークが埋め込まれます（モデル規約により除去禁止）。")
    elif settings.engine == "MOSS-TTS-Nano":
        print("MOSS-TTS-Nano は OpenMOSS の軽量 TTS です（100M パラメータ、20言語対応、CPU 動作）。")
        print(f"モデル: {settings.moss_tts_nano_hf_model}")
        print(f"mode: {settings.moss_tts_nano_mode}")
        print("voice パラメータは現在 'default'（プロンプトなしの plain TTS）のみ対応です。")
        print("注意: GPU 不要。ライセンス: Apache-2.0")
    elif settings.engine == "MOSS-TTS-v1.5":
        print("MOSS-TTS-v1.5 は OpenMOSS の 8B 多言語 TTS です（31言語、ゼロショット voice cloning、Apache-2.0）。")
        print(f"モデル: {settings.moss_tts_v1_5_hf_model}")
        print(f"language: {settings.moss_tts_v1_5_language}")
        print(f"attn_impl: {settings.moss_tts_v1_5_attn_impl} / max_new_tokens: {settings.moss_tts_v1_5_max_new_tokens}")
        print(f"デフォルト voice: {settings.moss_tts_v1_5_default_voice}")
        print("voice 候補: default（参照音声なし、language タグのみ）")
        if settings.moss_tts_v1_5_prompt_wav:
            print(f"             clone（参照音声: {settings.moss_tts_v1_5_prompt_wav}）")
        else:
            print("             clone は --moss-tts-v1-5-prompt-wav を指定すると有効になります")
        print("対応言語: Chinese / Cantonese / English / Arabic / Czech / Danish / Dutch / Finnish / French /")
        print("           German / Greek / Hebrew / Hindi / Hungarian / Italian / Japanese / Korean / Macedonian /")
        print("           Malay / Persian / Polish / Portuguese / Romanian / Russian / Spanish / Swahili /")
        print("           Swedish / Tagalog / Thai / Turkish / Vietnamese（計 31 言語）")
        print("注意: A100 必須（VRAM ~22GB resident + audio tokenizer。L4 22GB は不足し OOM 確認済み）。")
        print("      Python 3.12 venv で torch 2.9.1+cu128 / transformers 5.0.0 / accelerate を導入します。")
        print("ライセンス: コード / 重みとも Apache 2.0（商用 OK）。")
    elif settings.engine == "TinyTTS":
        print("TinyTTS は超軽量（~3.4MB）の英語専用 TTS です（CPU 動作、GPU 不要）。")
        print("voice パラメータは現在 'default' のみ対応です。")
        print("注意: 英語のみ対応。日本語テキストは正しく発音されません。ライセンス: Apache-2.0")
    elif settings.engine == "Chatterbox":
        print("Chatterbox は Resemble AI の多言語 TTS です（23言語対応、ゼロショット voice cloning 対応）。")
        print(f"language: {settings.chatterbox_language}")
        print(f"デフォルト voice: {settings.chatterbox_default_voice}")
        print("voice 候補: default（プロンプトなしの plain TTS）")
        if settings.chatterbox_prompt_wav:
            print(f"             clone（参照音声: {settings.chatterbox_prompt_wav}）")
        else:
            print("             clone は --chatterbox-prompt-wav を指定すると有効になります")
        print("対応言語: ar, da, de, el, en, es, fi, fr, he, hi, it, ja, ko, ms, nl, no, pl, pt, ru, sv, sw, tr, zh")
        print("注意: GPU 推奨（VRAM ~2-4GB）。ライセンス: MIT（コードと重み）")
    elif settings.engine == "Zonos":
        print("Zonos は Zyphra の多言語 TTS です（5言語対応・日本語含む、ゼロショット voice cloning 対応）。")
        print(f"モデル: {settings.zonos_hf_model}")
        print(f"language: {settings.zonos_language}")
        print(f"デフォルト voice: {settings.zonos_default_voice}")
        print("voice 候補: default（同梱の参照音声 assets/exampleaudio.mp3 を使用）")
        if settings.zonos_prompt_wav:
            print(f"             clone（参照音声: {settings.zonos_prompt_wav}）")
        else:
            print("             clone は --zonos-prompt-wav を指定すると有効になります")
        print("対応言語: en, ja, zh, fr, de（espeak-ng で音素化）")
        print("注意: GPU 推奨（VRAM 6GB+）。ライセンス: Apache-2.0（コードと重み）")
    elif settings.engine == "OuteTTS":
        print("OuteTTS は OuteAI の軽量多言語 TTS です（日本語含む多言語対応、voice cloning 対応）。")
        print(f"モデルサイズ: {settings.outetts_model_size} / backend: {settings.outetts_backend}")
        print(f"デフォルト speaker: {settings.outetts_default_speaker}")
        print(f"デフォルト voice: {settings.outetts_default_voice}")
        print("voice 候補: default（OUTETTS_DEFAULT_SPEAKER に該当する内蔵プロファイルを使用）")
        if settings.outetts_prompt_wav:
            print(f"             clone（参照音声: {settings.outetts_prompt_wav}）")
        else:
            print("             clone は --outetts-prompt-wav を指定すると有効になります（必要なら --outetts-prompt-text も併用）")
        print("注意: 日本語を話させる場合は日本語話者プロファイルの作成（clone）を推奨します。CPU/GPU 両対応。")
        if settings.outetts_model_size.upper() == "1B":
            print("ライセンス警告: 1B (Llama-OuteTTS-1.0-1B) の重みは CC-BY-NC-SA-4.0 + Llama 3.2 Community License で、商用利用は不可です。")
            print("                商用利用したい場合は 0.6B (OuteTTS-1.0-0.6B, Apache-2.0) を選択してください。")
        else:
            print("ライセンス: 0.6B はコード / 重みとも Apache-2.0（商用 OK）。1B に切り替えると重みは CC-BY-NC-SA-4.0 で非商用となります。")
    elif settings.engine == "Dia":
        print("Dia は Nari Labs の対話 TTS です（[S1]/[S2] 話者タグでマルチスピーカー一括生成、英語のみ）。")
        print(f"モデル: {settings.dia_hf_model}")
        print(f"compute_dtype: {settings.dia_compute_dtype}")
        print(f"デフォルト voice: {settings.dia_default_voice}")
        print("voice 候補: default（プロンプトなし。入力に [S1]/[S2] タグが無い場合は先頭に [S1] を自動挿入）")
        if settings.dia_prompt_wav and settings.dia_prompt_text:
            print(f"             clone（参照音声: {settings.dia_prompt_wav}, 書き起こし: {settings.dia_prompt_text}）")
        else:
            print("             clone は --dia-prompt-wav と --dia-prompt-text の両方を指定すると有効になります")
        print("対応言語: 英語のみ。日本語テキストは正しく発音されません。")
        print("注意: GPU 推奨（bf16/float16 で VRAM ~4.4GB / float32 で ~7.9GB）。ライセンス: Apache 2.0（コードと重み）")
    elif settings.engine == "OpenVoice-V2":
        print("OpenVoice V2 は MyShell の voice cloning TTS です（MeloTTS をベースに ToneColorConverter で声色変換）。")
        print(f"language: {settings.openvoice_language}")
        print(f"デフォルト voice: {settings.openvoice_default_voice}")
        print("voice 候補: default（OpenVoice 同梱の resources/example_reference.mp3 を参照音声として使用）")
        if settings.openvoice_prompt_wav:
            print(f"             clone（参照音声: {settings.openvoice_prompt_wav}）")
        else:
            print("             clone は --openvoice-prompt-wav を指定すると有効になります")
        print("対応言語: EN / ES / FR / ZH / JP / KR")
        print("注意: ベース TTS は MeloTTS のため、本リポの MeloTTS 単体エンジンと同じ依存解決問題（tokenizers Rust ビルド失敗）に当たる可能性があります。")
        print("ライセンス: コードと重みとも MIT（2024-04 以降、商用 OK）")
    elif settings.engine == "Kyutai-TTS":
        print("Kyutai TTS は Kyutai の英語 / フランス語 TTS です（DSM ベース、ストリーミング対応）。")
        print(f"モデル: {settings.kyutai_hf_repo}")
        print(f"voice repo: {settings.kyutai_voice_repo}")
        print(f"デフォルト voice path: {settings.kyutai_voice}")
        print(f"デフォルト voice: {settings.kyutai_default_voice}")
        print("voice 候補: default（KYUTAI_VOICE で指定した voice repo 内のパスを使用）")
        if settings.kyutai_prompt_wav:
            print(f"             clone（参照: {settings.kyutai_prompt_wav}）")
        else:
            print("             clone は --kyutai-prompt-wav を指定すると有効になります（.wav または .safetensors）")
        print("             任意の voice repo 内パス（例: 'expresso/ex03-...wav'）を voice に直接指定することも可能です。")
        print("対応言語: 英語 / フランス語のみ（日本語非対応）。")
        print("注意: GPU 推奨（VRAM ~6GB）。Python 3.10+。")
        print("ライセンス: コードは MIT (Python) / Apache 2.0 (Rust)、重みは CC-BY-4.0。")
    elif settings.engine == "Pocket-TTS":
        print("Pocket TTS は Kyutai の超軽量 CPU TTS です（100M パラメータ、~6x realtime on M4）。")
        print(f"language: {settings.pocket_language}")
        print(f"デフォルト speaker: {settings.pocket_default_speaker}")
        print(f"デフォルト voice: {settings.pocket_default_voice}")
        print("voice 候補: default（POCKET_DEFAULT_SPEAKER に指定した内蔵プリセットを使用）")
        if settings.pocket_prompt_wav:
            print(f"             clone（参照音声: {settings.pocket_prompt_wav}）")
        else:
            print("             clone は --pocket-prompt-wav を指定すると有効になります（.wav または .safetensors）")
        print("             内蔵プリセット名（alba, anna, charles, ...）を voice に直接渡すことも可能です。")
        print("対応言語: english / french_24l / german_24l / italian / portuguese / spanish_24l")
        print("注意: GPU 不要（CPU で十分高速）。Python 3.10+。")
        print("ライセンス: コードは MIT、重みは CC-BY-4.0。voice ごとに個別ライセンス（kyutai/tts-voices を参照）。")
        print("Prohibited use: 合意のない voice impersonation や偽情報の生成は禁止です。")
    elif settings.engine == "Spark-TTS":
        print("Spark-TTS は SparkAudio の Qwen2.5 ベース LLM-TTS です（中国語 / 英語、ゼロショット voice cloning）。")
        print(f"モデル: {settings.spark_hf_model}")
        print(f"デフォルト voice: {settings.spark_default_voice}")
        print(f"デフォルト gender / pitch / speed: {settings.spark_default_gender} / {settings.spark_default_pitch} / {settings.spark_default_speed}")
        print("voice 候補: default（プロンプト無し、gender/pitch/speed 制御モード）")
        if settings.spark_prompt_wav:
            print(f"             clone（参照音声: {settings.spark_prompt_wav}）")
            if settings.spark_prompt_text:
                print(f"             prompt_text: {settings.spark_prompt_text}")
        else:
            print("             clone は --spark-prompt-wav を指定すると有効になります（任意で --spark-prompt-text）")
        print("対応言語: 中国語 / 英語のみ（日本語非対応）。")
        print("注意: GPU 推奨（VRAM ~4GB）。出力は 16 kHz mono。")
        print("ライセンス警告: コードは Apache 2.0 ですが、重み（Spark-TTS-0.5B）は **CC BY-NC-SA 4.0** で")
        print("                 学習データのライセンス制約のため非商用のみです。商用利用は不可。")
    elif settings.engine == "CosyVoice2":
        print("CosyVoice2 は Alibaba FunAudioLLM のゼロショット voice cloning TTS です（多言語、日本語対応）。")
        print(f"モデル: {settings.cosyvoice_hf_model}")
        print(f"デフォルト voice: {settings.cosyvoice_default_voice}")
        print("voice 候補: default（同梱の asset/zero_shot_prompt.wav を参照音声として cross_lingual 推論）")
        if settings.cosyvoice_prompt_wav:
            print(f"             clone（参照音声: {settings.cosyvoice_prompt_wav}）")
            if settings.cosyvoice_prompt_text:
                print(f"             prompt_text: {settings.cosyvoice_prompt_text}（zero_shot 推論）")
            else:
                print("             prompt_text 未指定: cross_lingual 推論（参照と入力の言語が違っても OK）")
        else:
            print("             clone は --cosyvoice-prompt-wav を指定すると有効になります")
        print("対応言語: 中国語 / 英語 / 日本語 / 韓国語 / 独語 / 西語 / 仏語 / 伊語 / 露語 + 中国方言 18種類")
        print("注意: 上流要件により Python 3.10 専用 venv を作成します（uv venv --python 3.10）。")
        print("      GPU 推奨（VRAM ~4GB）。")
        print("ライセンス: コードは Apache 2.0、重み（CosyVoice2-0.5B）も Apache 2.0（HF モデルカード）。")
    elif settings.engine == "Orpheus-TTS":
        print("Orpheus-TTS は Canopy Labs の英語 LLM-TTS です（Llama-3.2-3B ベース、vLLM バックエンド）。")
        print(f"モデル: {settings.orpheus_hf_model}")
        print(f"デフォルト voice: {settings.orpheus_default_voice}")
        print(f"max_model_len: {settings.orpheus_max_model_len}")
        print("voice 候補:", ", ".join(["tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"]))
        print("対応言語: 英語のみ（日本語非対応）。")
        print("注意: GPU 必須（VRAM ~10-12GB、L4/A100 推奨）。Python 3.10+。")
        print("ライセンス: コードは Apache 2.0、重みは Apache 2.0 表記だがベースは Llama-3.2-3B-Instruct のため")
        print("           実質的に Llama 3.2 Community License も適用されます。")
    elif settings.engine == "Higgs-Audio-v2":
        print("Higgs Audio v2 は Boson AI の LLM ベース音声基盤モデルです（3B-base、表現力高、voice cloning）。")
        print(f"モデル: {settings.higgs_hf_model}")
        print(f"audio tokenizer: {settings.higgs_hf_tokenizer}")
        print(f"デフォルト voice: {settings.higgs_default_voice} / 参照プリセット: {settings.higgs_default_ref_voice}")
        print(f"max_new_tokens: {settings.higgs_max_new_tokens} / temperature: {settings.higgs_temperature}")
        print("voice 候補: default（examples/voice_prompts 内のプリセット名 = HIGGS_DEFAULT_REF_VOICE）")
        print("             プリセット名（belinda, broom_salesman 等）を voice に直接指定することも可能です。")
        if settings.higgs_prompt_wav and settings.higgs_prompt_text:
            print(f"             clone（参照音声: {settings.higgs_prompt_wav}）")
        else:
            print("             clone は --higgs-prompt-wav と --higgs-prompt-text を指定すると有効になります")
        print("対応言語: 英語中心（多言語対応の表記もあり）。")
        print("注意: A100 / L4 など 24GB+ VRAM 必須（T4 では起動不可）。Python 3.10 venv。")
        print("ライセンス警告: コードは Apache-2.0 ですが、重み（bosonai/higgs-audio-v2-generation-3B-base）は")
        print("                 **Boson Higgs Audio 2 Community License**（Llama 系派生）で、")
        print("                 商用利用には MAU 10万人以下の制限と、出力で他 LLM を学習させる用途の禁止があります。")
    elif settings.engine == "GPT-SoVITS":
        print("GPT-SoVITS は RVC-Boss の few-shot voice cloning TTS です（5秒の参照音声で zero-shot 推論）。")
        print(f"version: {settings.gpt_sovits_version}")
        print(f"prompt_lang: {settings.gpt_sovits_prompt_lang} / target_lang: {settings.gpt_sovits_target_lang}")
        if settings.gpt_sovits_prompt_wav and settings.gpt_sovits_prompt_text:
            print(f"参照音声: {settings.gpt_sovits_prompt_wav}")
            print(f"prompt_text: {settings.gpt_sovits_prompt_text}")
        else:
            print("参照音声未設定: --gpt-sovits-prompt-wav と --gpt-sovits-prompt-text が必須です。")
            print("(GPT-SoVITS は本質的に few-shot cloning モデルで、参照なしの推論はサポートしていません。)")
        print("対応言語: zh / en / ja / ko / yue (Cantonese)")
        print("注意: GPU 推奨（VRAM ~4-6GB）。Python 3.11 venv を作成し torch>=2.5.1 をインストールします。")
        print("      初回起動時に lj1995/GPT-SoVITS から v2 重み（~1.2GB）と BERT/HuBERT 基盤をダウンロードします。")
        print("ライセンス: コードは MIT、重み（lj1995/GPT-SoVITS）も MIT（商用 OK）。")
    elif settings.engine == "MaskGCT":
        print("MaskGCT は Amphion の Masked Generative Codec TTS です（ゼロショット voice cloning）。")
        print(f"デフォルト voice: {settings.maskgct_default_voice}")
        print(f"prompt_lang: {settings.maskgct_prompt_lang} / target_lang: {settings.maskgct_target_lang}")
        print("voice 候補: default（同梱の models/tts/maskgct/wav/prompt.wav を参照音声として使用）")
        if settings.maskgct_prompt_wav and settings.maskgct_prompt_text:
            print(f"             clone（参照音声: {settings.maskgct_prompt_wav}）")
        else:
            print("             clone は --maskgct-prompt-wav と --maskgct-prompt-text の両方を指定すると有効になります")
        print("対応言語: en / zh（およびその他、prompt_lang / target_lang で指定）。")
        print("注意: GPU 推奨（VRAM ~10-12GB）。Python 3.10 venv で torch==2.0.1 をインストールします。")
        print("ライセンス警告: コードは MIT、重み（amphion/MaskGCT）は **CC-BY-NC-4.0** で")
        print("                 商用利用は不可です。")
    elif settings.engine == "StyleTTS2":
        print("StyleTTS 2 は yl4579 の高品質 TTS です（拡散 + SLM 敵対学習、英語）。")
        print(f"デフォルト voice: {settings.styletts2_default_voice}")
        print(f"alpha: {settings.styletts2_alpha} / beta: {settings.styletts2_beta}")
        print(f"diffusion_steps: {settings.styletts2_diffusion_steps} / embedding_scale: {settings.styletts2_embedding_scale}")
        print("voice 候補: default（プロンプトなし、ランダム話者をサンプリング）")
        if settings.styletts2_prompt_wav:
            print(f"             clone（参照音声: {settings.styletts2_prompt_wav}）")
        else:
            print("             clone は --styletts2-prompt-wav を指定すると有効になります")
        print("対応言語: 英語のみ（gruut フォニマイザー）。日本語テキストは正しく発音されません。")
        print("注意: GPU 推奨（VRAM ~2-4GB）。Python 3.11 venv で legacy 依存を分離します。")
        print("ライセンス: コードは MIT（pip 版 styletts2 / 上流とも）。重み（LibriTTS）は")
        print("           上流の Custom License で、合成であることの開示が要求されます。")
    elif settings.engine == "CSM-1B":
        print("CSM-1B は Sesame AI の対話特化 TTS です（Llama 3.2 backbone + Mimi codec）。")
        print(f"モデル: {settings.csm_hf_model} + {settings.csm_llama_model}")
        print(f"デフォルト voice: {settings.csm_default_voice} / speaker_id: {settings.csm_default_speaker}")
        print(f"max_audio_length_ms: {settings.csm_max_audio_length_ms} / temperature: {settings.csm_temperature}")
        print("voice 候補: default, speaker_0, speaker_1, ...（任意の整数 speaker_id）")
        print("対応言語: 英語のみ。日本語テキストは正しく発音されません。")
        print("注意: GPU 推奨（VRAM ~6GB）。Python 3.11 venv を作成し torch==2.4.0 をインストールします。")
        print("HF gated: 初回利用時に sesame/csm-1b と meta-llama/Llama-3.2-1B 双方の同意が必要です。")
        print("          Colab Secrets で HF_TOKEN を設定してください。")
        print("ライセンス: コードと重みとも Apache 2.0（商用 OK）。Llama-3.2-1B は Llama 3.2 Community License。")
    elif settings.engine == "ChatTTS":
        print("ChatTTS は 2noise の対話特化 TTS です（笑い声 / ためらい / ポーズなどを表現）。")
        print(f"デフォルト voice: {settings.chattts_default_voice}")
        print(f"seed: {settings.chattts_seed} / temperature: {settings.chattts_temperature}")
        print("voice 候補: default（CHATTTS_SEED から再現可能な話者を生成）, random（毎回ランダム話者）")
        print("対応言語: 英語 / 中国語のみ。日本語テキストは正しく発音されません。")
        print("注意: 重みには高周波ノイズが意図的に挿入されており（乱用防止）、出力品質はやや劣化します。")
        print("ライセンス警告: コードは AGPL-3.0+、重みは CC-BY-NC-4.0 で **商用利用は不可**（教育 / 研究用途のみ）。")
    elif settings.engine == "Bark":
        print("Bark は Suno AI の生成的 TTS です（13言語対応、ノンバーバル音/効果音も生成可能）。")
        print(f"デフォルト voice: {settings.bark_default_voice}")
        print(f"small models: {'有効（VRAM ~8GB）' if settings.bark_use_small_models else '無効（フル版、VRAM ~12GB）'}")
        print("voice 候補は Bark 公式 Speaker Library 名（例: v2/en_speaker_0..9, v2/ja_speaker_0..9 など13言語×10話者）。")
        print("対応言語: en, de, es, fr, hi, it, ja, ko, pl, pt, ru, tr, zh")
        print("注意: GPU 推奨。生成プロセスはランダム性があり、同じ入力でも結果が変わります。")
        print("ライセンス: コードと重みとも MIT（商用 OK、ただし研究目的での提供を上流が明記）。")
    elif settings.engine == "VibeVoice":
        print("VibeVoice は Microsoft の長尺マルチスピーカー TTS です（最大 90 分・4 話者の一括生成）。")
        print(f"モデル: {settings.vibevoice_hf_model}")
        print(f"デフォルト speaker: {settings.vibevoice_default_speaker}（demo/voices/<speaker>.wav）")
        print(f"デフォルト voice: {settings.vibevoice_default_voice}")
        print(f"DDPM steps: {settings.vibevoice_ddpm_steps} / cfg_scale: {settings.vibevoice_cfg_scale}")
        print("voice 候補: default（VIBEVOICE_DEFAULT_SPEAKER 名で demo/voices から参照音声を選択）")
        if settings.vibevoice_prompt_wav:
            print(f"             clone（参照音声: {settings.vibevoice_prompt_wav}）")
        else:
            print("             clone は --vibevoice-prompt-wav を指定すると有効になります")
        print("対応言語: 英語 / 中国語のみ（モデル規約上、それ以外の言語は禁止）")
        print("注意: ライセンスは MIT ですが、Microsoft 公式に「research purpose only」と明記されており、")
        print("      なりすまし・ディスインフォ・実時間音声変換などは禁止です。商用 / 実運用での利用は推奨されていません。")
    elif settings.engine == "DramaBox":
        print("DramaBox は Resemble AI の表現力豊か（directable）な TTS です（LTX-2.3 + IC-LoRA、英語中心）。")
        print(f"モデル: {settings.dramabox_hf_model} + Gemma snapshot: {settings.dramabox_gemma_repo}")
        print(f"デフォルト voice: {settings.dramabox_default_voice} / 参照プリセット: {settings.dramabox_default_ref_voice}")
        print(f"cfg_scale: {settings.dramabox_cfg_scale} / stg_scale: {settings.dramabox_stg_scale} / duration_multiplier: {settings.dramabox_duration_multiplier}")
        print(f"dtype: {settings.dramabox_dtype} / compile: {settings.dramabox_compile} / bnb_4bit: {settings.dramabox_bnb_4bit}")
        print("voice 候補: default（assets/voices/<DRAMABOX_DEFAULT_REF_VOICE> を参照音声として使用）")
        if settings.dramabox_prompt_wav:
            print(f"             clone（参照音声: {settings.dramabox_prompt_wav}）")
        else:
            print("             clone は --dramabox-prompt-wav を指定すると有効になります（10秒以上の参照音声推奨）")
        print("             同梱プリセット（female_american, female_shadowheart, male_arnie, male_conan,")
        print("                          male_harvey_keitel, male_old_movie, male_petergriffin, male_samuel_j）も voice に直接指定可能です。")
        print("プロンプト記法: ディレクター調の英語プロンプトを推奨。例:")
        print("  'A woman speaks warmly, \"Hello, how are you today?\" She laughs, \"Hahaha, it is so good to see you!\"'")
        print("対応言語: 英語中心（Gemma 3 12B エンベディング）。日本語等の非英語テキストは正しく発音されません。")
        print("注意: A100 GPU 必須（VRAM ~24GB ピーク、T4/V100 では起動不可）。初回起動時に約 8.5GB のモデル + Gemma snapshot をダウンロードします。")
        print("生成音声には Resemble Perth による不可聴ウォーターマークが常時付与されます（除去不可）。")
        print("ライセンス警告: コードと重みは **LTX-2 Community License Agreement**（Lightricks）。")
        print("                年商 $10M+ の組織は商用ライセンス必須。非競合条項・配布時の同ライセンス継承条項あり。")
    elif settings.engine == "Scenema":
        print("Scenema Audio は Scenema AI の表現力豊か / 演技指向 TTS です（LTX-2.3 派生 + Gemma 3 12B、英語中心、多言語可）。")
        print(f"デフォルト voice: {settings.scenema_default_voice} / gender: {settings.scenema_default_gender}")
        print(f"seed: {settings.scenema_seed} (-1=ランダム) / pace: {settings.scenema_pace} / validate: {settings.scenema_validate}")
        print(f"skip_vc: {settings.scenema_skip_vc} / vc_steps: {settings.scenema_vc_steps} / vc_cfg_rate: {settings.scenema_vc_cfg_rate}")
        print(f"background_sfx: {settings.scenema_background_sfx} / Gemma quantize: {settings.scenema_gemma_quantize}")
        print("voice 候補: default, warm_male, smoky_female, child_girl, elderly_male, elderly_female")
        print("            voice に上記プリセット名以外を渡すと、その文字列を voice description として直接使用します。")
        if settings.scenema_prompt_wav:
            print(f"             clone（参照音声: {settings.scenema_prompt_wav}）")
        else:
            print("             clone は --scenema-prompt-wav を指定すると有効になります（10〜20秒の参照音声推奨）")
        print("入力テキストはそのまま <speak> でラップされ Scenema に渡されます。")
        print("上級者は input に直接 <speak voice=\"...\" gender=\"...\"><action>...</action>... </speak> の XML を書くと、")
        print("pass-through され action / sound タグなどの演技指示がそのまま効きます。")
        print("対応言語: 英語中心。Scenema は主要世界言語にも対応（モデルカード参照）。")
        print("注意: A100 GPU 必須（40GB VRAM）。初回起動時に約 38GB のチェックポイントをダウンロードします。")
        print("      Gemma 3 12B IT は HF gated。https://huggingface.co/google/gemma-3-12b-it で同意し、")
        print("      Colab Secrets で HF_TOKEN を設定してから起動してください。")
        print("ライセンス: コードは MIT。Scenema Audio の重み（ScenemaAI/scenema-audio）は LTX-2.3 から派生のため、")
        print("            **LTX-2 Community License Agreement**（Lightricks）が適用されます（DramaBox と同じ）。")
        print("            Gemma 3 12B IT は Gemma Terms of Use（Google）。")
        print("            年商 $10M+ の組織は LTX-2 商用ライセンス、Gemma も商用利用は Gemma 規約遵守が必要です。")
    elif settings.engine == "Supertonic":
        print("Supertonic は Supertone Inc. の超軽量オンデバイス TTS です（ONNX、~99M params、CPU 動作可）。")
        print(f"モデル: {settings.supertonic_model}")
        print(f"デフォルト voice: {settings.supertonic_default_voice}")
        print(f"デフォルト language: {settings.supertonic_default_lang}")
        print(f"total_steps: {settings.supertonic_total_steps}")
        print("voice 候補: M1, M2, M3, M4, M5（男性）/ F1, F2, F3, F4, F5（女性）")
        print("対応言語 (supertonic-3, 31言語 + na fallback):")
        print("  en, ko, ja, ar, bg, cs, da, de, el, es, et, fi, fr, hi, hr, hu,")
        print("  id, it, lt, lv, nl, pl, pt, ro, ru, sk, sl, sv, tr, uk, vi, na")
        print("注意: GPU 不要（ONNX Runtime で CPU 動作）。Voice cloning は未対応（preset のみ）。")
        print("ライセンス: コードは MIT、重みは OpenRAIL-M（商用 OK、ディープフェイク等の use-based 制限あり）。")
    elif settings.engine == "Irodori-TTS-Lite":
        print("Irodori-TTS-Lite は Irodori-TTS の int4 量子化ランタイムです（~1GB VRAM、音質ほぼ無劣化、MIT）。")
        print(f"モデル: {settings.irodori_lite_hf_checkpoint} / file: {settings.irodori_lite_checkpoint_file}")
        print(f"コーデック: {settings.irodori_lite_codec_repo}")
        print(f"codec int4: {'有効（VRAM さらに節約）' if settings.irodori_lite_codec_int4 else '無効（fp16 codec）'}")
        print("voice パラメータは現在 'default'（voice-design に焼き込まれた話者）のみ対応です。")
        if "v3" in settings.irodori_lite_hf_checkpoint.lower():
            print("v3 int4: Duration Predictor を使用するため seconds は自動推定されます。")
        else:
            print("voice-design int4: Duration Predictor を持たないため pyopenjtalk の音素数から seconds を導出します。")
        print("注意: GPU 推奨（VRAM ~1GB）。Triton カーネル使用のため Linux + CUDA 必須。")
        print("ライセンス: コード（runtime / patch）と重み（kizuna-intelligence/*-int4）はいずれも MIT。")
        print("           ベースの Aratako/Irodori-TTS と DACVAE コーデックも MIT。")
    else:
        print("Irodori-TTS は現状 voice 切り替えを持たない想定です。")


def launch_cloudflared(settings: Settings) -> str | None:
    ensure_cloudflared(settings.cloudflared_path)
    log_path = settings.log_dir / "cloudflared.log"
    log_file = open(log_path, "w")  # noqa: SIM115
    subprocess.Popen(
        [
            str(settings.cloudflared_path),
            "tunnel",
            "--url",
            f"http://127.0.0.1:{settings.app_port}",
            "--no-autoupdate",
        ],
        stdout=log_file,
        stderr=subprocess.STDOUT,
        cwd=str(settings.root_dir),
        start_new_session=True,
    )
    public_url = None
    read_pos = 0
    start = time.time()
    while time.time() - start < 60:
        time.sleep(0.5)
        with open(log_path) as f:
            f.seek(read_pos)
            new_text = f.read()
            read_pos = f.tell()
        if new_text:
            print(new_text, end="", flush=True)
            match = re.search(r"https://[-a-zA-Z0-9]+\.trycloudflare\.com", new_text)
            if match:
                public_url = match.group(0)
                break
    return public_url


def synth_test_wav(settings: Settings, base_url: str):
    output_path = settings.output_dir / f"{settings.engine.lower().replace(' ', '-').replace('/', '-')}.wav"
    selected_voice = resolve_selected_voice(settings)
    payload = {
        "model": settings.openai_model_id or settings.engine,
        "input": settings.test_text,
        "speed": settings.test_speed,
        "response_format": "wav",
    }
    if selected_voice:
        payload["voice"] = selected_voice
    run(
        [
            "curl",
            "-sS",
            "-X",
            "POST",
            f"{base_url}/v1/audio/speech",
            "-H",
            "Content-Type: application/json",
            "-d",
            json.dumps(payload, ensure_ascii=False),
            "--output",
            str(output_path),
        ]
    )
    print(f"test wav: {output_path}")


def print_dry_run_summary(settings: Settings):
    selected_voice = resolve_selected_voice(settings)
    print("\n=== Dry Run ===")
    print("repo dir  :", settings.repo_dir)
    print("root dir  :", settings.root_dir)
    print("installer :", INSTALLERS[settings.engine].__module__)
    print("base url  :", f"http://127.0.0.1:{settings.app_port}/v1")
    print("public url:", "enabled" if settings.expose_public_url else "disabled")
    print("voice     :", selected_voice or "(default)")
    print("input     :", settings.test_text)
    print("speed     :", settings.test_speed)
    print("dry-run のため、依存導入・clone・サーバ起動は実行していません。")


def launch(settings: Settings):
    settings.ensure_directories()
    print(f"engine: {settings.engine}")
    if settings.dry_run:
        print_engine_voice_hints(settings)
        print_dry_run_summary(settings)
        return
    ensure_uv(sys.executable)
    kill_old_processes(settings.app_port, settings.piper_backend_port)
    print_engine_voice_hints(settings)

    state = INSTALLERS[settings.engine](settings)

    if not wait_http(f"http://127.0.0.1:{settings.app_port}/", timeout=180):
        tail_log(state["log_path"])
        if "backend_log_path" in state:
            tail_log(state["backend_log_path"])
        raise RuntimeError(f"{settings.engine} OpenAI wrapper did not become ready.")

    local_base_url = f"http://127.0.0.1:{settings.app_port}"
    print("\n=== Local Ready ===")
    print("Base URL :", local_base_url + "/v1")
    print("Speech   :", local_base_url + "/v1/audio/speech")
    print("Models   :", local_base_url + "/v1/models")
    print("Voices   :", local_base_url + "/v1/voices")

    pretty_print_json_url(local_base_url + "/", "Root")
    pretty_print_json_url(local_base_url + "/v1/models", "Models")
    pretty_print_json_url(local_base_url + "/v1/voices", "Voices")
    synth_test_wav(settings, local_base_url)

    public_url = None
    if settings.expose_public_url:
        public_url = launch_cloudflared(settings)

    print("\n=== Log Tail ===")
    tail_log(state["log_path"], lines=60)
    if "backend_log_path" in state:
        tail_log(state["backend_log_path"], lines=40)

    if public_url:
        print("\n=== Public Ready ===")
        print("Base URL :", public_url + "/v1")
        print("Speech   :", public_url + "/v1/audio/speech")
        print("\nTest curl:")
        print(
            "curl -X POST "
            + shlex.quote(public_url + "/v1/audio/speech")
            + " -H 'Content-Type: application/json' "
            + "-d "
            + shlex.quote(
                json.dumps(
                    {
                        "model": settings.openai_model_id or settings.engine,
                        "input": settings.test_text,
                        "speed": settings.test_speed,
                        "response_format": "wav",
                        **({"voice": resolve_selected_voice(settings)} if resolve_selected_voice(settings) else {}),
                    },
                    ensure_ascii=False,
                )
            )
            + " --output out.wav"
        )
    elif settings.expose_public_url:
        print("cloudflared のURL取得に失敗しました。ログを確認してください。")
