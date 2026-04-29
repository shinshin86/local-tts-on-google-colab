from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
import time

from src.config import KOKORO_VOICE_PRESETS, MELO_VOICE_PRESETS, NEUTTS_VOICE_PRESETS, Settings
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
