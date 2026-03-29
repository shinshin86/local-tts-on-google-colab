from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
import time

from src.config import KOKORO_VOICE_PRESETS, MELO_VOICE_PRESETS, Settings
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
    return ""


def print_engine_voice_hints(settings: Settings):
    print("\n=== Voice Selection Hint ===")
    if settings.engine == "Kokoro":
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
    elif settings.engine == "Qwen3-TTS":
        print("Qwen3-TTS は多言語対応の高品質 TTS です。")
        print(f"モデル: {settings.qwen3_hf_model}")
        print(f"language: {settings.qwen3_language}")
        print(f"デフォルト speaker: {settings.qwen3_default_speaker}")
        print("候補: aiden, dylan, eric, ono_anna, ryan, serena, sohee, uncle_fu, vivian")
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
