# local-tts-on-google-colab

[English](README.md) | **日本語**

Google Colab 上で選択したローカル TTS を一時的に OpenAI 互換 `/v1/audio/speech` として起動し、動作確認できるようにするためのサンプルです。

対象エンジン:

| エンジン | Colab 動作確認 | 言語 |
|---|---|---|
| Kokoro | 動作OK | 日本語 / 英語 / 中国語 他 |
| Irodori-TTS | 動作OK | 日本語 |
| Piper | 動作OK | 英語（デフォルト）/ 多言語 |
| Piper-Plus | 動作OK | 日本語 / 英語 / 中国語 他 6言語 |
| Qwen3-TTS | 動作OK (GPU必須) | 日本語 / 英語 / 中国語 他 10言語 |
| VoxCPM2 | 動作OK (GPU必須) | 日本語 / 英語 / 中国語 他 30言語 |
| MOSS-TTS-Nano | 動作（出力が約2秒で切れる） | 日本語 / 英語 / 中国語 他 20言語 |
| TinyTTS | 動作OK | 英語 |
| Voxtral-TTS | 動作OK (GPU必須・VRAM 16GB+) | 英語 / フランス語 / スペイン語 他 9言語 |
| Sarashina-TTS | 動作OK (GPU必須・VRAM ~6GB) | 日本語 / 英語 |
| F5-TTS | 動作OK (GPU必須) | 英語 / 中国語（日本語は別モデル） |
| Chatterbox | 動作OK (GPU推奨) | 日本語 / 英語 / 中国語 他 23言語 |
| Zonos | 動作OK (GPU必須・VRAM ~6GB) | 日本語 / 英語 / 中国語 / フランス語 / ドイツ語 |
| OuteTTS | 動作OK (CPU可) | 日本語 / 英語 / 中国語 他 多言語 |
| Fish-Speech | 動作不可 | 日本語 / 英語 / 中国語 他 80言語以上 |
| MeloTTS | 動作不可 | - |
| Style-Bert-VITS2 | 動作不可 | - |
| CosyVoice2 | 動作不可 | - |

`MeloTTS`、`Style-Bert-VITS2`、`CosyVoice2` は Colab の uv + venv 環境で依存解決に問題があり、現時点では動作しません。

`Fish-Speech` は VRAM 24GB 以上が必要で A100/L4 GPU を想定していますが、Colab 環境ではモデルロード時に OOM（メモリ不足）でランタイムがクラッシュするため、現時点では動作しません。

`VOICEVOX` は含めていません。

## 使い方

### 最短手順

Colab では、以下のコードを 1 つのコードセルにそのまま貼り付けて実行するのを推奨します。

このセルは以下を自動で行います。

- 指定した `REPO_URL` / `REPO_REF` を clone / checkout
- `colab/bootstrap.py` を呼び出して選択した TTS を起動
- 必要なら `trycloudflare` の公開 URL も作成

`REPO_REF` には `main`、タグ、commit SHA を指定できます。再現性のため、常用時はタグか commit SHA を推奨します。

要点:

- まずは `ENGINE` と `REPO_REF` だけ触れば十分です
- 細かい engine 別パラメータは必要になったときだけ変更します
- 同内容のセルは [multi_tts_openai_colab.py](multi_tts_openai_colab.py) にあります

```python
#@title Local TTS on Google Colab -> OpenAI Compatible `/v1/audio/speech`
REPO_URL = "https://github.com/shinshin86/local-tts-on-google-colab.git"  #@param {type:"string"}
REPO_REF = "main"  #@param {type:"string"}
WORKDIR = "/content/local-tts-on-google-colab"  #@param {type:"string"}

ENGINE = "Kokoro"  #@param ["Irodori-TTS", "Kokoro", "MeloTTS", "MOSS-TTS-Nano", "NeuTTS", "Piper", "Piper-Plus", "Qwen3-TTS", "Sarashina-TTS", "Style-Bert-VITS2", "TinyTTS", "Voxtral-TTS"]
EXPOSE_PUBLIC_URL = True  #@param {type:"boolean"}
TEST_TEXT = "こんにちは。これは OpenAI 互換 TTS の動作確認です。"  #@param {type:"string"}
TEST_SPEED = 1.0  #@param {type:"number"}
TEST_VOICE = ""  #@param {type:"string"}
OPENAI_MODEL_ID = ""  #@param {type:"string"}

#@markdown ---
#@markdown Irodori-TTS
# V1を利用する場合: checkpoint="Aratako/Irodori-TTS-500M", codec_repo="facebook/dacvae-watermarked"
IRODORI_HF_CHECKPOINT = "Aratako/Irodori-TTS-500M-v2"  #@param {type:"string"}
IRODORI_CODEC_REPO = "Aratako/Semantic-DACVAE-Japanese-32dim"  #@param {type:"string"}
IRODORI_MODEL_PRECISION = "fp32"  #@param ["fp32", "bf16", "fp16"]
IRODORI_CODEC_PRECISION = "fp32"  #@param ["fp32", "bf16", "fp16"]

#@markdown ---
#@markdown Kokoro
KOKORO_DEFAULT_VOICE = "jf_alpha"  #@param ["jf_alpha", "jf_gongitsune", "jm_kumo", "af_heart", "af_bella", "am_adam", "bf_emma", "bm_george", "zf_xiaobei"]
KOKORO_DEFAULT_LANG_CODE = "j"  #@param ["j", "a", "b", "e", "f", "h", "i", "p", "z"]

#@markdown ---
#@markdown MeloTTS
MELO_LANGUAGE = "JP"  #@param ["JP", "EN", "ZH", "ES", "FR", "KR"]
MELO_DEFAULT_VOICE = "JP"  #@param ["JP", "EN-Default", "EN-US", "EN-BR", "EN_INDIA", "EN-AU", "ZH", "ES", "FR", "KR"]

#@markdown ---
#@markdown Style-Bert-VITS2
STYLE_BERT_MODEL_REPO = "litagin/style_bert_vits2_jvnv"  #@param {type:"string"}
STYLE_BERT_MODEL_SUBDIR = "jvnv-F2-jp"  #@param {type:"string"}
STYLE_BERT_MODEL_NAME = "jvnv-F2-jp"  #@param {type:"string"}
STYLE_BERT_SPEAKER_ID = 0  #@param {type:"integer"}
STYLE_BERT_STYLE = "Neutral"  #@param {type:"string"}

#@markdown ---
#@markdown Piper
PIPER_VOICE = "en_US-lessac-medium"  #@param {type:"string"}
PIPER_SPEAKER_ID = -1  #@param {type:"integer"}

#@markdown ---
#@markdown Piper-Plus
PIPER_PLUS_MODEL = "tsukuyomi"  #@param {type:"string"}

#@markdown ---
#@markdown Qwen3-TTS (GPU required)
QWEN3_HF_MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"  #@param ["Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"]
QWEN3_LANGUAGE = "Japanese"  #@param ["Chinese", "English", "Japanese", "Korean", "German", "French", "Russian", "Portuguese", "Spanish", "Italian"]
QWEN3_DEFAULT_SPEAKER = "ono_anna"  #@param ["aiden", "dylan", "eric", "ono_anna", "ryan", "serena", "sohee", "uncle_fu", "vivian"]

import shlex
import subprocess
from pathlib import Path


def run(cmd, *, cwd=None):
    print("$", shlex.join(cmd))
    proc = subprocess.Popen(
        cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    for line in proc.stdout:
        print(line, end="", flush=True)
    ret = proc.wait()
    if ret != 0:
        raise subprocess.CalledProcessError(ret, cmd)


def ensure_repo(repo_url: str, repo_ref: str, workdir: Path):
    if not workdir.exists():
        run(["git", "clone", repo_url, str(workdir)])
    else:
        print(f"reuse: {workdir}")

    run(["git", "fetch", "--all", "--tags", "--prune"], cwd=str(workdir))
    run(["git", "checkout", repo_ref], cwd=str(workdir))

    if repo_ref in {"main", "master"}:
        run(["git", "pull", "--ff-only", "origin", repo_ref], cwd=str(workdir))


def build_bootstrap_command(workdir: Path) -> list[str]:
    cmd = [
        "python",
        "colab/bootstrap.py",
        "--engine",
        ENGINE,
        "--root-dir",
        "/content/openai-compatible-local-tts",
        "--test-text",
        TEST_TEXT,
        "--test-speed",
        str(TEST_SPEED),
        "--test-voice",
        TEST_VOICE,
        "--openai-model-id",
        OPENAI_MODEL_ID,
        "--irodori-hf-checkpoint",
        IRODORI_HF_CHECKPOINT,
        "--irodori-model-precision",
        IRODORI_MODEL_PRECISION,
        "--irodori-codec-precision",
        IRODORI_CODEC_PRECISION,
        "--kokoro-default-voice",
        KOKORO_DEFAULT_VOICE,
        "--kokoro-default-lang-code",
        KOKORO_DEFAULT_LANG_CODE,
        "--melo-language",
        MELO_LANGUAGE,
        "--melo-default-voice",
        MELO_DEFAULT_VOICE,
        "--style-bert-model-repo",
        STYLE_BERT_MODEL_REPO,
        "--style-bert-model-subdir",
        STYLE_BERT_MODEL_SUBDIR,
        "--style-bert-model-name",
        STYLE_BERT_MODEL_NAME,
        "--style-bert-speaker-id",
        str(STYLE_BERT_SPEAKER_ID),
        "--style-bert-style",
        STYLE_BERT_STYLE,
        "--piper-voice",
        PIPER_VOICE,
        "--piper-speaker-id",
        str(PIPER_SPEAKER_ID),
        "--piper-plus-model",
        PIPER_PLUS_MODEL,
        "--qwen3-hf-model",
        QWEN3_HF_MODEL,
        "--qwen3-language",
        QWEN3_LANGUAGE,
        "--qwen3-default-speaker",
        QWEN3_DEFAULT_SPEAKER,
    ]
    cmd.append("--expose-public-url" if EXPOSE_PUBLIC_URL else "--no-expose-public-url")
    return cmd


def main():
    workdir = Path(WORKDIR)
    ensure_repo(REPO_URL, REPO_REF, workdir)
    run(build_bootstrap_command(workdir), cwd=str(workdir))


main()
```

### 実行後の確認

成功すると、以下が順に表示されます。

- ローカル URL
- `/v1/models`
- `/v1/voices`
- テスト WAV の出力先
- 必要なら `trycloudflare` の公開 URL

最初に確認するなら `Kokoro` を推奨します。

この実装は「1ランタイムで1エンジンずつ」の運用を前提にしています。別エンジンを試すときは、ランタイムを再起動してから再実行する想定です。

### 上級者向け

すでに clone 済みのリポジトリ上で直接起動したい場合は `colab/bootstrap.py` を呼べます。

```python
!python colab/bootstrap.py --engine Kokoro --expose-public-url
```

依存導入やサーバ起動を行わずに設定だけ確認したい場合は `--dry-run` を使います。

```python
!python colab/bootstrap.py --engine Kokoro --dry-run
```

## OpenAI 互換の範囲

対応エンドポイント:

- `GET /`
- `GET /v1/models`
- `GET /v1/voices`
- `POST /v1/audio/speech`

互換対象の主な入力:

- `model`
- `input`
- `voice`
- `speed`
- `response_format`

このサンプルは `wav` 固定です。`mp3` などへの変換は行っていません。

## エンジンごとの補足

### Kokoro

[hexgrad/kokoro](https://github.com/hexgrad/kokoro) を使った日本語・英語・中国語対応の軽量 TTS です。デフォルト voice は日本語の `jf_alpha` で、フォームから 9 種類の voice を選べます。

### Irodori-TTS

[Aratako/Irodori-TTS](https://github.com/Aratako/Irodori-TTS) を使った日本語 TTS です。デフォルトで Hugging Face の `Aratako/Irodori-TTS-500M-v2` モデルを使用します（V1 を利用する場合は `Aratako/Irodori-TTS-500M` に変更してください）。出力は 48kHz で高音質ですが、voice の切り替え機能はありません。

### Piper

[piper-tts](https://github.com/OHF-Voice/piper1-gpl) の内蔵 HTTP サーバーをバックエンドとして起動し、その前段に OpenAI 互換ラッパーを載せています。デフォルトは英語の `en_US-lessac-medium` です。依存が軽く、セットアップが安定しています。

### Piper-Plus

[ayutaz/piper-plus](https://github.com/ayutaz/piper-plus) をベースにした日本語対応の軽量 TTS です。元の Piper から日本語品質（OpenJTalk + プロソディ）と GPL フリー（MIT ライセンス）の方向で強化されています。GPU 不要で、CPU でも高速に動作します。デフォルトモデルは `tsukuyomi`（日本語女性）です。

### Qwen3-TTS

[QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) を使った多言語高品質 TTS です。9 種類の話者を内蔵し、日本語を含む 10 言語に対応しています。GPU ランタイム（T4 以上）が必要です。デフォルトは 0.6B モデル（軽量）で、フォームから 1.7B モデルも選べます。Apache 2.0 ライセンスです。

### VoxCPM2

[OpenBMB/VoxCPM](https://github.com/OpenBMB/VoxCPM) を使った高品質 TTS です。2B パラメータのモデルで、日本語を含む 30 言語に対応しており、言語を自動検出します。ゼロショット TTS、声デザイン（テキスト記述から声生成）、音声クローニングなどの機能を持ちます。GPU ランタイム（T4 以上、VRAM ~8GB）が必要です。ライセンス: Apache 2.0。

### MOSS-TTS-Nano

[OpenMOSS/MOSS-TTS-Nano](https://github.com/OpenMOSS/MOSS-TTS-Nano) を使った軽量多言語 TTS です。わずか 0.1B（100M）パラメータで、日本語・英語・中国語を含む 20 言語に対応し、GPU 不要・CPU のみで動作します。デフォルトの Hugging Face モデルは `OpenMOSS-Team/MOSS-TTS-Nano-100M`。`continuation` モード（プロンプト音声なしの plain TTS）で起動します。出力は 48 kHz ステレオ。ライセンス: Apache-2.0。注意: 音声自体は正常に生成されますが、現状では入力テキストの長さに関わらず出力が先頭 2 秒程度で切れてしまいます。ラッパーは MOSS-TTS-Nano の `model.inference()` に生成を委譲しているだけなので、修正には上流 `inference()` API 側で生成長パラメータを露出させる必要がありそうです。

### NeuTTS

[neuphonic/neutts](https://github.com/neuphonic/neutts) を使ったオンデバイス TTS です。**インスタント voice cloning** を採用しており、リクエストごとに参照音声の声色で合成します（プリセット話者という概念はありません）。upstream リポジトリに同梱されている 5 つの参照音声を OpenAI 互換 API の `voice` パラメータから指定できます:

| voice | 言語 | 性別 |
|---|---|---|
| `dave`     | 英語 | 男性 |
| `jo`       | 英語 | 女性 |
| `mateo`    | スペイン語 | 男性 |
| `greta`    | ドイツ語 | 女性 |
| `juliette` | フランス語 | 女性 |

デフォルト backbone は `neuphonic/neutts-air`（約 360M パラメータ、英語のみ、Apache 2.0）。他言語には Nano 系の言語別 backbone（`neuphonic/neutts-nano-french` / `-german` / `-spanish`、NeuTTS Open License 1.0）が用意されています。**参照音声の言語と backbone の言語は揃える必要があります** — 揃えないと不自然なアクセントや崩れた音声になります。ラッパーは初回利用時に参照音声を遅延エンコードしてメモリにキャッシュします。日本語は **非対応**。ライセンス: コードは Apache-2.0、モデル重みは backbone により異なります（下記参照）。独自の参照音声を追加することも技術的には可能ですが、必ず権利を持っている音声（本人の同意がある音声）でのみ行ってください。

### TinyTTS

[ecyht2/tiny-tts](https://github.com/ecyht2/tiny-tts) を使った超軽量の英語 TTS です。モデルはわずか 1.6M パラメータ（約 3.4MB）で、GPU 不要・CPU のみで 53 倍速のリアルタイム合成が可能です。音声は 44.1kHz で出力されます。voice の切り替え機能はありません。ライセンス: Apache 2.0。

### Voxtral-TTS

[mistralai/Voxtral-4B-TTS-2603](https://huggingface.co/mistralai/Voxtral-4B-TTS-2603) を使った多言語 TTS です。4B パラメータのモデルで、英語・フランス語・スペイン語・ドイツ語・イタリア語・ポルトガル語・オランダ語・アラビア語・ヒンディー語の 9 言語に対応しています。20 種類のプリセットボイスを内蔵し、wav / mp3 / flac / aac / opus など複数フォーマットに対応しています。バックエンドに vLLM + vllm-omni を使用します。GPU ランタイム（VRAM 16GB 以上）が必要です。Colab A100（VRAM 40GB）で動作確認済みですが、無料枠の T4（15GB）では VRAM 不足のため動作しない可能性があります。ライセンス: CC BY-NC 4.0（非商用のみ）。

### Sarashina-TTS

SB Intuitions の [sbintuitions/sarashina2.2-tts](https://huggingface.co/sbintuitions/sarashina2.2-tts) を使った日本語中心の TTS です。0.8B パラメータの LLM ベース TTS で、日本語（メイン）と英語に対応し、ゼロショット音声クローン機能を備えています。デフォルトの Hugging Face モデルは `sbintuitions/sarashina2.2-tts`。HuggingFace transformers バックエンドで VRAM ~6GB（Colab T4 で動作可能）、`--sarashina-use-vllm` を有効にすると vLLM バックエンドが使われ、より多くの VRAM を消費する代わりに高速になります。出力は 24kHz で、デフォルトでは SilentCipher の不可聴ウォーターマークが埋め込まれます — 上流モデル規約により除去・無効化は禁止されているのでそのまま利用してください。**ライセンス: Sarashina Model NonCommercial License Agreement（商用利用不可）。**

`voice` パラメータには次の値を指定できます。

| voice | 説明 |
|---|---|
| `default` | 参照音声なしの plain TTS（ゼロショットクローンなし） |
| `clone` | ゼロショット音声クローン。`--sarashina-prompt-wav` と `--sarashina-prompt-text` の両方を指定したときのみ有効。テキストは参照音声の書き起こしを正確に渡してください |

音声クローンを使う場合は、必ず権利を持っている音声（本人の同意がある音声）でのみ行ってください。

### F5-TTS

[SWivid/F5-TTS](https://github.com/SWivid/F5-TTS) を使ったゼロショット音声クローニング TTS です。参照音声の声質を模倣して音声を生成します。パッケージ同梱のデフォルト参照音声（英語女性）を使用します。日本語モデルを使う場合は `--f5tts-ckpt-file` / `--f5tts-vocab-file` でコミュニティ提供の日本語チェックポイントを指定してください。GPU ランタイム（T4 以上）が必要です。ライセンス: コード MIT / モデル CC-BY-NC。

### Chatterbox

Resemble AI の [resemble-ai/chatterbox](https://github.com/resemble-ai/chatterbox) を使った多言語 TTS です。Chatterbox Multilingual モデルは日本語・英語・中国語・フランス語・ドイツ語・スペイン語・韓国語など 23 言語に対応し、ゼロショット音声クローンを備えています。デフォルト言語は `ja`（日本語）。`--chatterbox-prompt-wav` を指定すると `clone` voice が有効になり、参照音声の声色で合成されます。GPU 推奨（VRAM ~2-4GB）。ライセンス: MIT（コードと重みの両方）。

`voice` パラメータには次の値を指定できます。

| voice | 説明 |
|---|---|
| `default` | 参照音声なしの plain TTS |
| `clone` | ゼロショット音声クローン。`--chatterbox-prompt-wav` を指定したときのみ有効 |

音声クローンを使う場合は、必ず権利を持っている音声（本人の同意がある音声）でのみ行ってください。

### Zonos

[Zyphra/Zonos](https://github.com/Zyphra/Zonos) を使った多言語 TTS です。英語・日本語・中国語・フランス語・ドイツ語に対応し、ゼロショット音声クローニングを備えています。デフォルトモデルは `Zyphra/Zonos-v0.1-transformer`（Apache 2.0）。音素化に `espeak-ng` を利用するため、インストーラが自動で `apt-get install espeak-ng` を実行します。デフォルト voice では upstream に同梱の `assets/exampleaudio.mp3` を参照音声として使用し、`--zonos-prompt-wav` を指定すると独自参照の `clone` voice が有効になります。GPU 必須（VRAM 6GB+、T4 動作可）。Hybrid backbone は Ampere 世代以降の GPU と `mamba-ssm` 依存を要求するため、ポータビリティのためデフォルトでは Transformer backbone を使用します。ライセンス: Apache 2.0（コードと重み）。

`voice` パラメータには次の値を指定できます。

| voice | 説明 |
|---|---|
| `default` | upstream に同梱の参照音声をそのまま使用 |
| `clone` | ゼロショット音声クローン。`--zonos-prompt-wav` を指定したときのみ有効 |

音声クローンを使う場合は、必ず権利を持っている音声（本人の同意がある音声）でのみ行ってください。

### OuteTTS

[edwko/OuteTTS](https://github.com/edwko/OuteTTS) を使った軽量多言語 TTS です。日本語を含む多言語に対応し、モデルサイズ（`0.6B` / `1B`）と backend（`HF` = transformers / `LLAMACPP` = GGUF）を選択できます。`--outetts-prompt-wav`（必要なら `--outetts-prompt-text` も）で voice cloning を有効にできます。デフォルト voice は `--outetts-default-speaker`（例: `EN-FEMALE-1-NEUTRAL`）で内蔵 speaker プロファイルを切り替えられます。日本語を発話させる場合は、日本語の参照音声から `clone` で speaker プロファイルを作るのが推奨です。CPU / GPU の両方で動作します。

**ライセンス（モデルサイズで異なります）:**

| モデル | コード | モデル重み | 商用利用 |
|---|---|---|---|
| `OuteAI/OuteTTS-1.0-0.6B` | Apache 2.0 | Apache 2.0 | OK |
| `OuteAI/Llama-OuteTTS-1.0-1B` | Apache 2.0 | CC-BY-NC-SA-4.0 + Llama 3.2 Community License | **不可** |

このラッパーのデフォルトサイズは `0.6B`（Apache 2.0）です。`1B` に切り替えると重みが非商用ライセンスになります。

`voice` パラメータには次の値を指定できます。

| voice | 説明 |
|---|---|
| `default` | `--outetts-default-speaker` で選んだ内蔵 speaker プロファイルで合成 |
| `clone` | 音声クローン。`--outetts-prompt-wav` を指定したときのみ有効 |

音声クローンを使う場合は、必ず権利を持っている音声（本人の同意がある音声）でのみ行ってください。

### Fish-Speech (現在動作不可)

[fishaudio/fish-speech](https://github.com/fishaudio/fish-speech) を使った高品質 TTS です。日本語は Tier 1 サポート（最高品質）で、80 言語以上に対応しています。VRAM 24GB 以上が必要で A100/L4 GPU を想定していますが、Colab 環境ではモデルロード時に OOM（メモリ不足）でランタイムがクラッシュするため、現時点では動作しません。ライセンス: Apache 2.0。

### CosyVoice2 (現在動作不可)

[FunAudioLLM/CosyVoice](https://github.com/FunAudioLLM/CosyVoice) を使う構成ですが、依存パッケージ（openai-whisper、onnxruntime-gpu、grpcio、deepspeed、lightning 等）が Python 3.12+ に対応しておらず、Colab の uv + venv 環境ではセットアップが完了しません。

### MeloTTS (現在動作不可)

[myshell-ai/MeloTTS](https://github.com/myshell-ai/MeloTTS) を使う構成ですが、依存パッケージ `tokenizers` のビルドに Rust コンパイラが必要なため、現在の Colab 環境ではインストールに失敗します。

### Style-Bert-VITS2 (現在動作不可)

[litagin02/Style-Bert-VITS2](https://github.com/litagin02/Style-Bert-VITS2) を使う構成ですが、`setuptools` / `torch` / `scipy` の依存整合性が取れず、現在の Colab 環境では音声合成まで到達できません。

## ライセンス

各エンジンのライセンスは以下の通りです。利用時は各プロジェクトの最新のライセンス条件を必ず確認してください。

| エンジン | コード | モデル重み | 商用利用 | 備考 |
|---|---|---|---|---|
| Kokoro | Apache 2.0 | Apache 2.0 | OK | |
| Irodori-TTS | MIT | MIT | OK | なりすまし・ディープフェイク生成を禁止する倫理規定あり |
| Piper | GPL-3.0 | MIT | 要注意 | デフォルト音声 `en_US-lessac-medium` の学習データ（Blizzard 2013）は研究目的限定・商用利用不可 |
| Piper-Plus | MIT | MIT | OK | |
| Qwen3-TTS | Apache 2.0 | Apache 2.0 | OK | |
| VoxCPM2 | Apache 2.0 | Apache 2.0 | OK | |
| MOSS-TTS-Nano | Apache 2.0 | Apache 2.0 | OK | 100M パラメータ、CPU 動作可 |
| NeuTTS | Apache 2.0 | Apache 2.0 (Air) / NeuTTS Open License 1.0 (Nano) | OK (Air) / 規約要確認 (Nano) | ボイスクローン。英 / 西 / 独 / 仏 |
| TinyTTS | Apache 2.0 | Apache 2.0 | OK | |
| Voxtral-TTS | — | CC BY-NC 4.0 | 不可 | vLLM + vllm-omni 経由。音声データセットのライセンス制約により非商用 |
| Sarashina-TTS | — | Sarashina Model NonCommercial License | 不可 | 日本語 / 英語。ゼロショット音声クローン対応。出力には SilentCipher のウォーターマークが付与される（除去禁止） |
| F5-TTS | MIT | CC-BY-NC | 不可（モデル） | モデル重みは Emilia データセットの制約により非商用 |
| Chatterbox | MIT | MIT | OK | 多言語（23言語、日本語含む）。ゼロショット voice cloning |
| Zonos | Apache 2.0 | Apache 2.0 | OK | 英 / 日 / 中 / 仏 / 独。ゼロショット voice cloning。`espeak-ng` 必須 |
| OuteTTS (0.6B) | Apache 2.0 | Apache 2.0 | OK | 日本語含む多言語、CPU 動作可、voice cloning |
| OuteTTS (1B)   | Apache 2.0 | CC-BY-NC-SA-4.0 + Llama 3.2 Community License | 不可 | Llama-3.2 ベース。重みは非商用 |
| Fish-Speech | Apache 2.0 | Apache 2.0 | OK | A100/L4 GPU 必須（VRAM 24GB+） |

**Piper について**: `piper-tts` パッケージは GPL-3.0 です。また、デフォルトの `en_US-lessac-medium` 音声は Lessac Technologies 提供の Blizzard 2013 データセットで学習されており、このデータセットのライセンスは商用利用を禁止しています。商用利用が必要な場合は、許容的なライセンスで学習された別の voice モデルを選択してください。

このリポジトリ自体は短時間の動作確認・技術検証を目的としています。

## 注意点

- Colab の管理ランタイムでは、外部公開やプロキシ利用は恒常運用向きではありません。このリポジトリは短時間の動作確認用です。
- エンジンごとに依存が重いため、別エンジンへの切り替えはランタイム再起動前提にしています。
- 各エンジン・音声モデルのライセンスは上記「ライセンス」セクションおよび各プロジェクトの公式情報を確認してください。

## 参考

- OpenAI Audio Speech API
  https://developers.openai.com/api/reference/resources/audio/subresources/speech/methods/create
- Irodori-TTS
  https://github.com/Aratako/Irodori-TTS
- Kokoro
  https://github.com/hexgrad/kokoro
- MeloTTS
  https://github.com/myshell-ai/MeloTTS
- Style-Bert-VITS2
  https://github.com/litagin02/Style-Bert-VITS2
- Piper
  https://github.com/OHF-Voice/piper1-gpl
- Piper-Plus
  https://github.com/ayutaz/piper-plus
- Qwen3-TTS
  https://github.com/QwenLM/Qwen3-TTS
- VoxCPM2
  https://github.com/OpenBMB/VoxCPM
- TinyTTS
  https://github.com/ecyht2/tiny-tts
- MOSS-TTS-Nano
  https://github.com/OpenMOSS/MOSS-TTS-Nano
- NeuTTS
  https://github.com/neuphonic/neutts
- Voxtral-TTS
  https://huggingface.co/mistralai/Voxtral-4B-TTS-2603
- Sarashina-TTS
  https://github.com/sbintuitions/sarashina2.2-tts
- F5-TTS
  https://github.com/SWivid/F5-TTS
- Chatterbox
  https://github.com/resemble-ai/chatterbox
- Zonos
  https://github.com/Zyphra/Zonos
- OuteTTS
  https://github.com/edwko/OuteTTS
- Fish Speech
  https://github.com/fishaudio/fish-speech
- CosyVoice
  https://github.com/FunAudioLLM/CosyVoice
