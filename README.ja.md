# local-tts-on-google-colab

![Logo](./images/logo.png)

[English](README.md) | **日本語**

Google Colab 上で選択したローカル TTS を一時的に OpenAI 互換 `/v1/audio/speech` として起動し、動作確認できるようにするためのサンプルです。

対象エンジン:

| エンジン | Colab 動作確認 | 言語 |
|---|---|---|
| Kokoro | 動作OK | 日本語 / 英語 / 中国語 他 |
| Kokoro-ONNX | 動作OK | 日本語 / 英語 / 中国語 他 |
| Irodori-TTS | 動作OK | 日本語 |
| Irodori-TTS-Lite | 動作OK（GPU 必須、VRAM ~1GB、int4 量子化） | 日本語 |
| Piper | 動作OK | 英語（デフォルト）/ 多言語 |
| Piper-Plus | 動作OK | 日本語 / 英語 / 中国語 他 6言語 |
| Qwen3-TTS | 動作OK (GPU必須) | 日本語 / 英語 / 中国語 他 10言語 |
| VoxCPM2 | 動作OK (GPU必須) | 日本語 / 英語 / 中国語 他 30言語 |
| MOSS-TTS-Nano | 動作（出力が約2秒で切れる） | 日本語 / 英語 / 中国語 他 20言語 |
| MOSS-TTS-v1.5 | A100 で動作確認（L4 22GB は VRAM 不足：モデル+activation+音声トークナイザーで超過） | 日本語 / 英語 / 中国語 / 韓国語 他 31言語 |
| MOSS-TTS-Local-v1.5 | L4 で動作確認（~4B MossTTSLocal、~12.4GB VRAM。8B 版が OOM する L4 でも動作） | 日本語 / 英語 / 中国語 / 韓国語 他 31言語 |
| NeuTTS | 動作OK (CPU可・voice cloning) | 英語 / スペイン語 / ドイツ語 / フランス語 |
| TinyTTS | 動作OK | 英語 |
| Supertonic | 動作OK (CPU可・ONNX・~99M params) | 英語 / 日本語 / 韓国語 他 31言語 |
| Voxtral-TTS | 動作OK (GPU必須・VRAM 16GB+) | 英語 / フランス語 / スペイン語 他 9言語 |
| Sarashina-TTS | 動作OK (GPU必須・VRAM ~6GB) | 日本語 / 英語 |
| F5-TTS | 動作OK (GPU必須) | 英語 / 中国語（日本語は別モデル） |
| Chatterbox | 動作OK (GPU推奨) | 日本語 / 英語 / 中国語 他 23言語 |
| Zonos | 動作OK (GPU必須・VRAM ~6GB) | 日本語 / 英語 / 中国語 / フランス語 / ドイツ語 |
| ZONOS2 | 動作OK (L4検証済・sm_80+必須) | 41言語 (tier-1: 日本語 / 英語 / 中国語) |
| OuteTTS | 動作OK (CPU可) | 日本語 / 英語 / 中国語 他 多言語 |
| Dia | 動作OK (GPU推奨) | 英語（マルチスピーカー対話） |
| Kyutai-TTS | 動作OK (GPU推奨) | 英語 / フランス語 |
| Pocket-TTS | 動作OK (CPU可・~6x realtime) | 英語 / 仏 / 独 / 伊 / 葡 / 西 |
| Orpheus-TTS | 動作不可（HF gated 重み・Llama 3.2 ライセンス同意 + `HF_TOKEN` 必須） | 英語（Llama-3.2-3B ベース、vLLM） |
| CosyVoice2 | 動作OK (GPU推奨・Python 3.10 venv) | 日本語 / 英語 / 中 / 韓 / 独 他 9言語 |
| Spark-TTS | 動作OK (GPU推奨) | 英語 / 中国語（重みは非商用） |
| OpenVoice-V2 | 動作不可（Python 3.13 で `av==10` がビルドできない） | 日本語 / 英語 / 西 / 仏 / 中 / 韓 |
| VibeVoice | 動作不可（upstream API 移行中） | 英語 / 中国語（長尺・最大 4 話者） |
| Fish-Speech | 動作不可 | 日本語 / 英語 / 中国語 他 80言語以上 |
| MeloTTS | 動作不可 | - |
| Style-Bert-VITS2 | 動作不可 | - |
| Bark | Colab 動作確認済み (GPU推奨・~12GB / small=8GB) | 英語 / 日本語 / 中国語 他 13言語 |
| ChatTTS | Colab 動作確認済み (GPU推奨・**商用不可**) | 英語 / 中国語 |
| CSM-1B | デフォルトで動作不可（`sesame/csm-1b` と `meta-llama/Llama-3.2-1B` の HF gated。両方のライセンス同意 + `HF_TOKEN` が必要） | 英語（Llama-3.2-1B ベース + Mimi codec） |
| MisoTTS | A100 で動作・`HF_TOKEN` 不要（Llama 3.2 トークナイザは ungated な `unsloth/Llama-3.2-1B` ミラーから取得。8B の Sesame-CSM フォーク、~32GB F32 ckpt → GPU 上 bf16 ~16GB。T4/L4 は OOM 想定） | 英語中心（Llama 8B ベース + Mimi codec。その他言語は upstream 未記載） |
| StyleTTS2 | Colab 動作確認済み (GPU推奨・Python 3.11 venv) | 英語 |
| MaskGCT | Colab 動作確認済み (GPU必須・~10-12GB・**商用不可**) | 英語 / 中国語 |
| GPT-SoVITS | Colab でエンジン起動確認済み（synthesis には参照音声必須・default speaker モード非対応・`--gpt-sovits-prompt-wav` と `--gpt-sovits-prompt-text` を指定） | 中 / 英 / 日 / 韓 / 粤 |
| Higgs-Audio-v2 | デフォルトで動作不可（HF 上の checkpoint が未リリースの `boson_multimodal` / transformers 5.x を要求。エンジンは起動するが audio tokenizer ロード時に上流コードと config schema が一致せず推論失敗） | 英語 |
| Higgs-Audio-v3 | Colab L4 / A100 動作確認済み（GPU 必須、ロード時 ~19.9GB。T4 非対応、SGLang-Omni 経由で初回起動が遅い ~10-12 分、**非商用の重み — hosted API 不可**） | 100+ 言語（日本語含む） |
| dots.tts | Colab L4 動作確認済み（GPU 必須、bf16 常駐 ~5.4GB、出力 48 kHz。重みは ungated で `HF_TOKEN` 不要。ゼロショット cloning モデルのため `default` は話者がランダム、安定した声色は `clone` を使用） | 24 言語（日本語含む） |
| LFM2.5-Audio-JP | Colab L4 動作確認済み（GPU 必須、VRAM 常駐 ~6.3GB。重みは ungated で `HF_TOKEN` 不要。内蔵の日本語ボイス1種のみ、cloning 非対応。出力 24 kHz） | 日本語 |
| Ming-omni-TTS | Colab A100 動作確認済み（**A100 40GB 必須** — 16.8B-A3B MoE、ロード時 ~35GB VRAM。40GB に収めるため `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` が必要で L4 24GB では動かない。重みは ungated で `HF_TOKEN` 不要。zero-shot cloning。出力 44.1 kHz） | 中国語 / 英語が中心（広東語などの方言制御あり） |
| DramaBox | Colab A100 動作確認済み（GPU 必須、VRAM ~24GB ピーク、**LTX-2 Community License — 非競合条項あり**） | 英語 |
| Scenema | **Colab A100（40GB VRAM）必須**。初回起動時に約 38GB ダウンロード。音声モデルは LTX-2.3 派生のため **LTX-2 Community License**（DramaBox と同じ）。Gemma 3 12B IT 利用（HF gated、`HF_TOKEN` 必須） | 英語中心の多言語 |

`MeloTTS`、`Style-Bert-VITS2` は Colab の uv + venv 環境で依存解決に問題があり、現時点では動作しません。

`Fish-Speech` は VRAM 24GB 以上が必要で A100/L4 GPU を想定していますが、Colab 環境ではモデルロード時に OOM（メモリ不足）でランタイムがクラッシュするため、現時点では動作しません。

`VOICEVOX` は含めていません。

## AIエージェントを用いた Colab 接続方法について

このリポジトリではAIエージェントからの利用を想定して [.mcp.json](.mcp.json) を含めています。デフォルトでは [shinshin86/colab-mcp-go](https://github.com/shinshin86/colab-mcp-go) を使う設定です。

`colab-mcp-go` は [googlecolab/colab-mcp](https://github.com/googlecolab/colab-mcp) のローカルブリッジを Go で移植した非公式プロジェクトで、このリポジトリでは Codex や Claude Code などのエージェントから Colab を扱う用途を想定しています。  
（公式 Colab MCP の Codex 対応状況は変わる可能性がありますが、このリポジトリでは Codex でも扱いやすい選択肢として Go 版をデフォルトにしています）

ただし、このリポジトリ自体は特定の Colab 接続手段に依存していません。

公式 `colab-mcp` をはじめ、別の手段に置き換えることも可能です。その際は `.mcp.json` を書き換えてご利用ください。

## 使い方

### 最短手順 — WebUI コマンドジェネレーター

`#@param` フォームを手で触りたくない場合は、GitHub Pages の
**Colab セルジェネレーター**を使ってください:

👉 **<https://shinshin86.github.io/local-tts-on-google-colab/>**

エンジンを選択し、必要なオプションをフォームで設定（選んだエンジンに関係する
項目だけが表示されます）し、**Copy cell** を押します。
[Colab スクラッチパッド](https://colab.research.google.com/notebooks/empty.ipynb)
（ファイルは Drive に自動保存されない一時ノートブック）を開いて
貼り付け・実行するだけです。各エンジンの Colab 動作状況、対応言語、
ライセンス上の注意点も画面に表示されるので、起動前に確認できます。

WebUI は静的サイト (`docs/`) で、
[multi_tts_openai_colab.py](multi_tts_openai_colab.py) から
`tools/sync_webui.py` 経由で生成されます。生成されるセルは下の正準セルと
同じ `colab/bootstrap.py` を呼び出します。

### 手動セル（正準フォーム）

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

ENGINE = "Kokoro"  #@param ["Bark", "ChatTTS", "Chatterbox", "CosyVoice2", "CSM-1B", "Dia", "dots.tts", "DramaBox", "F5-TTS", "Fish-Speech", "GPT-SoVITS", "Higgs-Audio-v2", "Higgs-Audio-v3", "Irodori-TTS", "Irodori-TTS-Lite", "Kokoro", "Kokoro-ONNX", "Kyutai-TTS", "LFM2.5-Audio-JP", "MaskGCT", "MeloTTS", "Ming-omni-TTS", "MisoTTS", "MOSS-TTS-Nano", "MOSS-TTS-v1.5", "MOSS-TTS-Local-v1.5", "NeuTTS", "OpenVoice-V2", "Orpheus-TTS", "OuteTTS", "Piper", "Piper-Plus", "Pocket-TTS", "Qwen3-TTS", "Sarashina-TTS", "Scenema", "Spark-TTS", "Style-Bert-VITS2", "StyleTTS2", "Supertonic", "TinyTTS", "VibeVoice", "VoxCPM2", "Voxtral-TTS", "Zonos", "ZONOS2"]
EXPOSE_PUBLIC_URL = True  #@param {type:"boolean"}
TEST_TEXT = "こんにちは。これは OpenAI 互換 TTS の動作確認です。"  #@param {type:"string"}
TEST_SPEED = 1.0  #@param {type:"number"}
TEST_VOICE = ""  #@param {type:"string"}
OPENAI_MODEL_ID = ""  #@param {type:"string"}

#@markdown ---
#@markdown F5-TTS (GPU required)
F5TTS_MODEL = "F5TTS_v1_Base"  #@param {type:"string"}
F5TTS_CKPT_FILE = ""  #@param {type:"string"}
F5TTS_VOCAB_FILE = ""  #@param {type:"string"}

#@markdown ---
#@markdown Fish-Speech (A100/L4 GPU required, VRAM 24GB+)
FISH_SPEECH_MODEL = "fishaudio/s2-pro"  #@param {type:"string"}

#@markdown ---
#@markdown Irodori-TTS
#@markdown - Default: v3 (Rectified Flow DiT, Duration Predictor + always-on SilentCipher watermark).
#@markdown - Older variants: "Aratako/Irodori-TTS-500M-v2" or v1 ("Aratako/Irodori-TTS-500M" + codec_repo="facebook/dacvae-watermarked").
#@markdown - License: MIT for code ([Aratako/Irodori-TTS](https://github.com/Aratako/Irodori-TTS)), all weight variants (v1/v2/v3), and the Aratako/Semantic-DACVAE-Japanese-32dim codec. Commercial use OK. The author requests ethical use (no impersonation/deepfake).
IRODORI_HF_CHECKPOINT = "Aratako/Irodori-TTS-500M-v3"  #@param ["Aratako/Irodori-TTS-500M-v3", "Aratako/Irodori-TTS-500M-v2", "Aratako/Irodori-TTS-500M"]
IRODORI_CODEC_REPO = "Aratako/Semantic-DACVAE-Japanese-32dim"  #@param {type:"string"}
IRODORI_MODEL_PRECISION = "fp32"  #@param ["fp32", "bf16", "fp16"]
IRODORI_CODEC_PRECISION = "fp32"  #@param ["fp32", "bf16", "fp16"]

#@markdown ---
#@markdown Irodori-TTS-Lite (int4-quantized Irodori-TTS, ~1GB VRAM, MIT)
#@markdown - Default checkpoint is voice-design int4 (speaker baked in, seconds derived from text via pyopenjtalk).
#@markdown - For the v3-derived int4 with built-in Duration Predictor, switch to
#@markdown   "kizuna-intelligence/Irodori-TTS-500M-v3-int4" AND set IRODORI_LITE_CHECKPOINT_FILE="model.safetensors".
#@markdown - License: MIT for the default `kizuna-intelligence/Irodori-TTS-Lite-int4` weights, inherited from upstream `Aratako/Irodori-TTS` (MIT). The alternate `kizuna-intelligence/Irodori-TTS-500M-v3-int4` HF card declares no license — confirm with upstream before commercial use.
IRODORI_LITE_HF_CHECKPOINT = "kizuna-intelligence/Irodori-TTS-Lite-int4"  #@param ["kizuna-intelligence/Irodori-TTS-Lite-int4", "kizuna-intelligence/Irodori-TTS-500M-v3-int4"]
IRODORI_LITE_CHECKPOINT_FILE = "dit_int4.safetensors"  #@param ["dit_int4.safetensors", "model.safetensors"]
IRODORI_LITE_CODEC_REPO = "Aratako/Semantic-DACVAE-Japanese-32dim"  #@param {type:"string"}
IRODORI_LITE_CODEC_INT4 = False  #@param {type:"boolean"}

#@markdown ---
#@markdown Kokoro
#@markdown - License: Apache 2.0 for both code ([hexgrad/kokoro](https://github.com/hexgrad/kokoro)) and weights ([hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)). Commercial use OK.
KOKORO_DEFAULT_VOICE = "jf_alpha"  #@param ["jf_alpha", "jf_gongitsune", "jm_kumo", "af_heart", "af_bella", "am_adam", "bf_emma", "bm_george", "zf_xiaobei"]
KOKORO_DEFAULT_LANG_CODE = "j"  #@param ["j", "a", "b", "e", "f", "h", "i", "p", "z"]

#@markdown ---
#@markdown Kokoro-ONNX (NVIDIA-optimized Kokoro-82M, onnxruntime, GPU/CPU)
#@markdown - NVIDIA's ONNX build of hexgrad/Kokoro-82M, run via onnxruntime with misaki G2P. 53 preset voices, 9 languages.
#@markdown - provider: auto/cuda prefer GPU and fall back to CPU; cpu forces CPU-only.
#@markdown - License: Apache 2.0 for both code and weights ([nvidia/kokoro-82M-onnx-opt](https://huggingface.co/nvidia/kokoro-82M-onnx-opt), base [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)). Commercial use OK.
KOKORO_ONNX_HF_MODEL = "nvidia/kokoro-82M-onnx-opt"  #@param {type:"string"}
KOKORO_ONNX_DEFAULT_VOICE = "jf_alpha"  #@param ["jf_alpha", "jf_gongitsune", "jm_kumo", "af_heart", "af_bella", "am_adam", "am_michael", "bf_emma", "bm_george", "zf_xiaobei", "zm_yunjian", "ef_dora", "ff_siwis", "hf_alpha", "if_sara", "pf_dora"]
KOKORO_ONNX_DEFAULT_LANG_CODE = "j"  #@param ["j", "a", "b", "e", "f", "h", "i", "p", "z"]
KOKORO_ONNX_PROVIDER = "auto"  #@param ["auto", "cuda", "cpu"]

#@markdown ---
#@markdown Kyutai-TTS (GPU recommended, English/French only, CC-BY-4.0 weights)
KYUTAI_HF_REPO = "kyutai/tts-1.6b-en_fr"  #@param {type:"string"}
KYUTAI_VOICE_REPO = "kyutai/tts-voices"  #@param {type:"string"}
KYUTAI_VOICE = "expresso/ex03-ex01_happy_001_channel1_334s.wav"  #@param {type:"string"}
KYUTAI_PROMPT_WAV = ""  #@param {type:"string"}
KYUTAI_DEFAULT_VOICE = "default"  #@param ["default", "clone"]

#@markdown ---
#@markdown Pocket-TTS (CPU-only, EN/FR/DE/IT/PT/ES, MIT code, CC-BY-4.0 weights)
POCKET_LANGUAGE = "english"  #@param ["english", "english_2026-01", "english_2026-04", "french_24l", "german_24l", "italian", "portuguese", "spanish_24l"]
POCKET_DEFAULT_SPEAKER = "alba"  #@param ["alba", "anna", "azelma", "bill_boerst", "caro_davy", "charles", "cosette", "eponine", "eve", "fantine", "george", "jane", "jean", "javert", "marius", "mary", "michael", "paul", "peter_yearsley", "stuart_bell", "vera"]
POCKET_PROMPT_WAV = ""  #@param {type:"string"}
POCKET_DEFAULT_VOICE = "default"  #@param ["default", "clone"]

#@markdown ---
#@markdown MeloTTS
#@markdown - License: MIT for both code ([myshell-ai/MeloTTS](https://github.com/myshell-ai/MeloTTS)) and the per-language weight repos. Commercial use OK.
MELO_LANGUAGE = "JP"  #@param ["JP", "EN", "ZH", "ES", "FR", "KR"]
MELO_DEFAULT_VOICE = "JP"  #@param ["JP", "EN-Default", "EN-US", "EN-BR", "EN_INDIA", "EN-AU", "ZH", "ES", "FR", "KR"]

#@markdown ---
#@markdown Style-Bert-VITS2
#@markdown - License: code is **AGPL-3.0** ([litagin02/Style-Bert-VITS2](https://github.com/litagin02/Style-Bert-VITS2)); default weights are **CC-BY-SA-4.0** (`litagin/style_bert_vits2_jvnv`, inherited from the JVNV corpus). Commercial use is permitted only under copyleft obligations: AGPL §13 requires source disclosure to network users; CC-BY-SA requires attribution + share-alike on any derivative.
STYLE_BERT_MODEL_REPO = "litagin/style_bert_vits2_jvnv"  #@param {type:"string"}
STYLE_BERT_MODEL_SUBDIR = "jvnv-F2-jp"  #@param {type:"string"}
STYLE_BERT_MODEL_NAME = "jvnv-F2-jp"  #@param {type:"string"}
STYLE_BERT_SPEAKER_ID = 0  #@param {type:"integer"}
STYLE_BERT_STYLE = "Neutral"  #@param {type:"string"}

#@markdown ---
#@markdown Piper
#@markdown - License: code is MIT ([rhasspy/piper](https://github.com/rhasspy/piper)). The default `rhasspy/piper-voices` repo is MIT at the repo level, but **each individual voice has its own license** (MIT / CC0 / CC-BY / CC-BY-SA / etc.) depending on dataset provenance. Before commercial deployment, check the specific voice's MODEL_CARD inside the voice repo.
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

#@markdown ---
#@markdown VoxCPM2 (GPU required)
VOXCPM_HF_MODEL = "openbmb/VoxCPM2"  #@param {type:"string"}
VOXCPM_CFG_VALUE = 2.0  #@param {type:"number"}
VOXCPM_INFERENCE_TIMESTEPS = 10  #@param {type:"integer"}

#@markdown ---
#@markdown MOSS-TTS-Nano (CPU OK)
MOSS_TTS_NANO_HF_MODEL = "OpenMOSS-Team/MOSS-TTS-Nano-100M"  #@param {type:"string"}
MOSS_TTS_NANO_MODE = "continuation"  #@param ["continuation", "voice_clone"]

#@markdown ---
#@markdown MOSS-TTS-v1.5 (A100 required — L4 22GB is insufficient, 31 languages, Apache 2.0)
#@markdown - 8B-parameter LLM-based TTS from [OpenMOSS/MOSS-TTS](https://github.com/OpenMOSS/MOSS-TTS) with zero-shot voice cloning.
#@markdown - Verified on Colab A100; OOM-confirmed on Colab L4 (22GB) — transformers device_map pre-allocates ~22GB at load before the audio tokenizer is moved to GPU.
#@markdown - Installs with the upstream `[torch-runtime]` extra (`torch==2.9.1+cu128`, `transformers==5.0.0`) plus `accelerate` under a dedicated Python 3.12 venv.
#@markdown - License: code and weights are both Apache 2.0. Commercial use OK.
MOSS_TTS_V1_5_HF_MODEL = "OpenMOSS-Team/MOSS-TTS-v1.5"  #@param {type:"string"}
MOSS_TTS_V1_5_LANGUAGE = "Japanese"  #@param ["Chinese", "Cantonese", "English", "Arabic", "Czech", "Danish", "Dutch", "Finnish", "French", "German", "Greek", "Hebrew", "Hindi", "Hungarian", "Italian", "Japanese", "Korean", "Macedonian", "Malay", "Persian", "Polish", "Portuguese", "Romanian", "Russian", "Spanish", "Swahili", "Swedish", "Tagalog", "Thai", "Turkish", "Vietnamese"]
MOSS_TTS_V1_5_PROMPT_WAV = ""  #@param {type:"string"}
MOSS_TTS_V1_5_DEFAULT_VOICE = "default"  #@param ["default", "clone"]
MOSS_TTS_V1_5_ATTN_IMPL = "sdpa"  #@param ["sdpa", "eager", "flash_attention_2"]
MOSS_TTS_V1_5_MAX_NEW_TOKENS = 4096  #@param {type:"integer"}

#@markdown ---
#@markdown MOSS-TTS-Local-v1.5 (L4 OK — ~4B MossTTSLocal, 31 languages, 48kHz stereo, Apache 2.0)
#@markdown - ~4B-parameter MossTTSLocal checkpoint from [OpenMOSS/MOSS-TTS](https://github.com/OpenMOSS/MOSS-TTS) with zero-shot voice cloning.
#@markdown - Lighter than the 8B MOSS-TTS-v1.5 (MossTTSDelay), so it fits on Colab L4; uses MOSS-Audio-Tokenizer-v2 for native 48kHz stereo output.
#@markdown - Installs with the upstream `[torch-runtime]` extra (`torch==2.9.1+cu128`, `transformers==5.0.0`) plus `accelerate` under a dedicated Python 3.12 venv.
#@markdown - License: code and weights are both Apache 2.0. Commercial use OK.
MOSS_LOCAL_V1_5_HF_MODEL = "OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5"  #@param {type:"string"}
MOSS_LOCAL_V1_5_LANGUAGE = "Japanese"  #@param ["Chinese", "Cantonese", "English", "Arabic", "Czech", "Danish", "Dutch", "Finnish", "French", "German", "Greek", "Hebrew", "Hindi", "Hungarian", "Italian", "Japanese", "Korean", "Macedonian", "Malay", "Persian", "Polish", "Portuguese", "Romanian", "Russian", "Spanish", "Swahili", "Swedish", "Tagalog", "Thai", "Turkish", "Vietnamese"]
MOSS_LOCAL_V1_5_PROMPT_WAV = ""  #@param {type:"string"}
MOSS_LOCAL_V1_5_DEFAULT_VOICE = "default"  #@param ["default", "clone"]
MOSS_LOCAL_V1_5_ATTN_IMPL = "sdpa"  #@param ["sdpa", "eager", "flash_attention_2"]
MOSS_LOCAL_V1_5_MAX_NEW_TOKENS = 4096  #@param {type:"integer"}

#@markdown ---
#@markdown NeuTTS (CPU OK, EN/ES/DE/FR, voice cloning)
#@markdown - License is **split** across backbones:
#@markdown   - Code [neuphonic/neutts](https://github.com/neuphonic/neutts): Apache 2.0.
#@markdown   - Weights `neuphonic/neutts-air` and codec `neuphonic/neucodec`: Apache 2.0 (commercial use OK).
#@markdown   - Weights `neuphonic/neutts-nano` and the per-language Nano variants (`-french` / `-german` / `-spanish`): **NeuTTS Open License v1.0** — commercial use OK only if your annual revenue is below USD $5M; above that, a paid license from Neuphonic is required.
#@markdown - All outputs include a Resemble Perth (imperceptible) watermark.
NEUTTS_BACKBONE_REPO = "neuphonic/neutts-air"  #@param ["neuphonic/neutts-air", "neuphonic/neutts-nano", "neuphonic/neutts-nano-french", "neuphonic/neutts-nano-german", "neuphonic/neutts-nano-spanish"]
NEUTTS_CODEC_REPO = "neuphonic/neucodec"  #@param {type:"string"}
NEUTTS_DEFAULT_VOICE = "jo"  #@param ["dave", "jo", "greta", "juliette", "mateo"]

#@markdown ---
#@markdown Sarashina-TTS (GPU required, ~6GB VRAM, JP/EN, NonCommercial)
SARASHINA_HF_MODEL = "sbintuitions/sarashina2.2-tts"  #@param {type:"string"}
SARASHINA_USE_VLLM = False  #@param {type:"boolean"}
SARASHINA_PROMPT_WAV = ""  #@param {type:"string"}
SARASHINA_PROMPT_TEXT = ""  #@param {type:"string"}
SARASHINA_DEFAULT_VOICE = "default"  #@param ["default", "clone"]

#@markdown ---
#@markdown Chatterbox (GPU recommended, multilingual incl JP, voice cloning)
CHATTERBOX_LANGUAGE = "ja"  #@param ["ar", "da", "de", "el", "en", "es", "fi", "fr", "he", "hi", "it", "ja", "ko", "ms", "nl", "no", "pl", "pt", "ru", "sv", "sw", "tr", "zh"]
CHATTERBOX_PROMPT_WAV = ""  #@param {type:"string"}
CHATTERBOX_DEFAULT_VOICE = "default"  #@param ["default", "clone"]

#@markdown ---
#@markdown Zonos (GPU recommended, JP/EN/ZH/FR/DE, voice cloning, Apache 2.0)
ZONOS_HF_MODEL = "Zyphra/Zonos-v0.1-transformer"  #@param {type:"string"}
ZONOS_LANGUAGE = "ja"  #@param ["en", "ja", "zh", "fr", "de"]
ZONOS_PROMPT_WAV = ""  #@param {type:"string"}
ZONOS_DEFAULT_VOICE = "default"  #@param ["default", "clone"]

#@markdown ---
#@markdown ZONOS2 (L4 required, 41 languages incl. Japanese, voice cloning)
#@markdown - Zyphra's latest TTS: MoE backbone + DAC tokens + ECAPA-TDNN embedding, served by the bundled Mini-SGLang server. We launch `uv run python -m zonos2` as a backend and proxy its `/tts/generate` (44.1 kHz float32 PCM) as OpenAI-compatible.
#@markdown - **GPU sm_80+ required** (L4 / A100): the backbone uses flashinfer / sgl_kernel / cutlass kernels, so T4 does not work. First launch is slow (`uv sync` fetches GPU kernels, then the weights download).
#@markdown - `default` = a shipped reference voice (`default_voices/<ZONOS2_DEFAULT_REF>`); set `ZONOS2_PROMPT_WAV` and use `voice="clone"` for your own reference. `accurate_mode` on = closer voice match, off = more expressive.
#@markdown - License: code is **MIT** (pyproject), weights are **Apache-2.0** ([HF model card](https://huggingface.co/Zyphra/ZONOS2)). Both allow commercial use.
ZONOS2_HF_MODEL = "Zyphra/ZONOS2"  #@param {type:"string"}
ZONOS2_LANGUAGE = "ja"  #@param ["en_us", "en_gb", "fr_fr", "de", "es", "it", "pt_br", "ja", "cmn", "ko"]
ZONOS2_PROMPT_WAV = ""  #@param {type:"string"}
ZONOS2_DEFAULT_VOICE = "default"  #@param ["default", "clone"]
ZONOS2_DEFAULT_REF = "AmericanFemale.mp3"  #@param {type:"string"}
ZONOS2_ACCURATE_MODE = True  #@param {type:"boolean"}
ZONOS2_SEED = -1  #@param {type:"integer"}

#@markdown ---
#@markdown OuteTTS (CPU OK, multilingual incl JP, voice cloning)
#@markdown - 0.6B: code/weights both Apache 2.0 (commercial use OK).
#@markdown - 1B: weights are CC-BY-NC-SA-4.0 + Llama 3.2 Community License (non-commercial only).
OUTETTS_MODEL_SIZE = "0.6B"  #@param ["0.6B", "1B"]
OUTETTS_BACKEND = "HF"  #@param ["HF", "LLAMACPP"]
OUTETTS_DEFAULT_SPEAKER = "EN-FEMALE-1-NEUTRAL"  #@param {type:"string"}
OUTETTS_PROMPT_WAV = ""  #@param {type:"string"}
OUTETTS_PROMPT_TEXT = ""  #@param {type:"string"}
OUTETTS_DEFAULT_VOICE = "default"  #@param ["default", "clone"]

#@markdown ---
#@markdown Dia (GPU recommended, English-only, [S1]/[S2] dialogue, Apache 2.0)
DIA_HF_MODEL = "nari-labs/Dia-1.6B-0626"  #@param {type:"string"}
DIA_COMPUTE_DTYPE = "float16"  #@param ["float16", "bfloat16", "float32"]
DIA_PROMPT_WAV = ""  #@param {type:"string"}
DIA_PROMPT_TEXT = ""  #@param {type:"string"}
DIA_DEFAULT_VOICE = "default"  #@param ["default", "clone"]

#@markdown ---
#@markdown OpenVoice V2 (GPU recommended, multilingual incl JP, voice cloning, MIT)
#@markdown - Pipeline: MeloTTS base TTS -> ToneColorConverter (V2 checkpoints).
#@markdown - May hit the same MeloTTS dependency issue that breaks the standalone MeloTTS engine.
OPENVOICE_LANGUAGE = "JP"  #@param ["EN", "ES", "FR", "ZH", "JP", "KR"]
OPENVOICE_PROMPT_WAV = ""  #@param {type:"string"}
OPENVOICE_DEFAULT_VOICE = "default"  #@param ["default", "clone"]

#@markdown ---
#@markdown CosyVoice2 (GPU recommended, multilingual incl JP, Apache 2.0)
#@markdown - Forces a Python 3.10 venv because upstream pins (torch 2.3.1, openai-whisper 20231117, etc.) do not resolve under Python 3.12.
#@markdown - License: Apache 2.0 for both code ([FunAudioLLM/CosyVoice](https://github.com/FunAudioLLM/CosyVoice)) and weights ([FunAudioLLM/CosyVoice2-0.5B](https://huggingface.co/FunAudioLLM/CosyVoice2-0.5B)). Commercial use OK. The model card adds a non-binding academic-purpose / takedown request.
COSYVOICE_HF_MODEL = "FunAudioLLM/CosyVoice2-0.5B"  #@param {type:"string"}
COSYVOICE_PROMPT_WAV = ""  #@param {type:"string"}
COSYVOICE_PROMPT_TEXT = ""  #@param {type:"string"}
COSYVOICE_DEFAULT_VOICE = "default"  #@param ["default", "clone"]

#@markdown ---
#@markdown Spark-TTS (GPU recommended, EN/ZH only, voice cloning + gender/pitch/speed control)
#@markdown - Code: Apache 2.0. Weights: CC BY-NC-SA 4.0 (non-commercial only) due to training data license.
SPARK_HF_MODEL = "SparkAudio/Spark-TTS-0.5B"  #@param {type:"string"}
SPARK_DEFAULT_VOICE = "default"  #@param ["default", "clone"]
SPARK_DEFAULT_GENDER = "female"  #@param ["male", "female"]
SPARK_DEFAULT_PITCH = "moderate"  #@param ["very_low", "low", "moderate", "high", "very_high"]
SPARK_DEFAULT_SPEED = "moderate"  #@param ["very_low", "low", "moderate", "high", "very_high"]
SPARK_PROMPT_WAV = ""  #@param {type:"string"}
SPARK_PROMPT_TEXT = ""  #@param {type:"string"}

#@markdown ---
#@markdown Orpheus-TTS (currently not working — HF-gated weights)
#@markdown - Code: Apache 2.0. Weights: Apache 2.0 + Llama 3.2 Community License (base model).
#@markdown - Pinned to vllm==0.7.3 due to a known regression in newer vLLM 0.7.x.
#@markdown - **Before running**: request access to canopylabs/orpheus-3b-0.1-ft AND
#@markdown   meta-llama/Llama-3.2-3B-Instruct on HF, accept the Llama 3.2 license,
#@markdown   then set `HF_TOKEN` (Colab Secrets → New secret with notebook access).
#@markdown   See the README "Orpheus-TTS" section for the full setup.
ORPHEUS_HF_MODEL = "canopylabs/orpheus-tts-0.1-finetune-prod"  #@param {type:"string"}
ORPHEUS_DEFAULT_VOICE = "tara"  #@param ["tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"]
ORPHEUS_MAX_MODEL_LEN = 2048  #@param {type:"integer"}

#@markdown ---
#@markdown VibeVoice (GPU required, English/Chinese, long-form multi-speaker)
#@markdown - License: MIT, but Microsoft tags this as "research purpose only".
#@markdown - Non-EN/ZH languages, voice impersonation, and disinformation use are prohibited.
VIBEVOICE_HF_MODEL = "microsoft/VibeVoice-1.5B"  #@param {type:"string"}
VIBEVOICE_DEFAULT_SPEAKER = "en-Alice_woman"  #@param {type:"string"}
VIBEVOICE_PROMPT_WAV = ""  #@param {type:"string"}
VIBEVOICE_DEFAULT_VOICE = "default"  #@param ["default", "clone"]
VIBEVOICE_DDPM_STEPS = 10  #@param {type:"integer"}
VIBEVOICE_CFG_SCALE = 1.3  #@param {type:"number"}

#@markdown ---
#@markdown Bark (GPU recommended, 13 languages, MIT)
#@markdown - License: code & weights both MIT. Generative audio (laughter, sound effects).
BARK_DEFAULT_VOICE = "v2/en_speaker_6"  #@param ["v2/en_speaker_0", "v2/en_speaker_6", "v2/en_speaker_9", "v2/ja_speaker_0", "v2/ja_speaker_6", "v2/ja_speaker_9", "v2/zh_speaker_0", "v2/zh_speaker_6", "v2/de_speaker_0", "v2/es_speaker_0", "v2/fr_speaker_0", "v2/hi_speaker_0", "v2/it_speaker_0", "v2/ko_speaker_0", "v2/pt_speaker_0", "v2/ru_speaker_0"]
BARK_USE_SMALL_MODELS = False  #@param {type:"boolean"}

#@markdown ---
#@markdown ChatTTS (GPU recommended, EN/ZH, AGPL-3.0+ code / CC-BY-NC-4.0 weights)
#@markdown - **Non-commercial only.** Weights contain intentional high-frequency noise to deter misuse.
CHATTTS_DEFAULT_VOICE = "default"  #@param ["default", "random"]
CHATTTS_SEED = 2  #@param {type:"integer"}
CHATTTS_TEMPERATURE = 0.3  #@param {type:"number"}

#@markdown ---
#@markdown Sesame CSM-1B (GPU required, English-only, Apache 2.0)
#@markdown - **HF gated**: accept terms for `sesame/csm-1b` AND `meta-llama/Llama-3.2-1B`, then set `HF_TOKEN`.
#@markdown - License: CSM weights ([sesame/csm-1b](https://huggingface.co/sesame/csm-1b)) and code ([SesameAILabs/csm](https://github.com/SesameAILabs/csm)) are Apache 2.0, but the **effective stack is governed by the Llama 3.2 Community License** because of the meta-llama/Llama-3.2-1B backbone. Commercial use is OK below ~700M MAU and requires "Built with Llama" attribution + Meta's Acceptable Use Policy compliance.
CSM_HF_MODEL = "sesame/csm-1b"  #@param {type:"string"}
CSM_LLAMA_MODEL = "meta-llama/Llama-3.2-1B"  #@param {type:"string"}
CSM_DEFAULT_VOICE = "default"  #@param {type:"string"}
CSM_DEFAULT_SPEAKER = 0  #@param {type:"integer"}
CSM_MAX_AUDIO_LENGTH_MS = 10000  #@param {type:"integer"}
CSM_TEMPERATURE = 0.9  #@param {type:"number"}

#@markdown ---
#@markdown MisoTTS (A100 required — Sesame CSM fork, 8B, English-centric, Modified MIT)
#@markdown - **No HF_TOKEN needed**: `generator.py` hardcodes the gated `meta-llama/Llama-3.2-1B` tokenizer, so the wrapper redirects it to the ungated, byte-identical `MISOTTS_TOKENIZER_REPO` (default `unsloth/Llama-3.2-1B`). Set it to `meta-llama/Llama-3.2-1B` (with `HF_TOKEN` + license acceptance) to use the official source instead.
#@markdown - License: MisoTTS code & weights are **Modified MIT** ([MisoLabsAI/MisoTTS](https://github.com/MisoLabsAI/MisoTTS), [MisoLabs/MisoTTS](https://huggingface.co/MisoLabs/MisoTTS)) — commercial use OK, but products with >50M MAU or >$10M/month revenue must display "Miso Labs" in the UI. The Llama 3.2 tokenizer (ungated mirror or official) is governed by the **Llama 3.2 Community License**. Output carries an inaudible SilentCipher watermark applied inside `generate()` (do not remove).
#@markdown - The ~32GB F32 checkpoint loads as bf16 (~16GB on GPU). `voice="clone"` needs `MISOTTS_PROMPT_WAV` (optionally `MISOTTS_PROMPT_TEXT`); otherwise use `voice="default"` / `speaker_<int>`.
MISOTTS_HF_MODEL = "MisoLabs/MisoTTS"  #@param {type:"string"}
MISOTTS_DEFAULT_VOICE = "default"  #@param ["default", "clone"]
MISOTTS_DEFAULT_SPEAKER = 0  #@param {type:"integer"}
MISOTTS_PROMPT_WAV = ""  #@param {type:"string"}
MISOTTS_PROMPT_TEXT = ""  #@param {type:"string"}
MISOTTS_MAX_AUDIO_LENGTH_MS = 30000  #@param {type:"integer"}
MISOTTS_TEMPERATURE = 0.9  #@param {type:"number"}
MISOTTS_TOPK = 50  #@param {type:"integer"}
MISOTTS_TOKENIZER_REPO = "unsloth/Llama-3.2-1B"  #@param {type:"string"}

#@markdown ---
#@markdown StyleTTS 2 (GPU recommended, English-only, MIT code / Custom weights)
#@markdown - Uses sidharthrajaram/StyleTTS2 (MIT, gruut-based — GPL-free).
#@markdown - Weights from yl4579/StyleTTS2-LibriTTS require disclosing that audio is synthesized.
STYLETTS2_DEFAULT_VOICE = "default"  #@param ["default", "clone"]
STYLETTS2_PROMPT_WAV = ""  #@param {type:"string"}
STYLETTS2_ALPHA = 0.3  #@param {type:"number"}
STYLETTS2_BETA = 0.7  #@param {type:"number"}
STYLETTS2_DIFFUSION_STEPS = 5  #@param {type:"integer"}
STYLETTS2_EMBEDDING_SCALE = 1.0  #@param {type:"number"}

#@markdown ---
#@markdown MaskGCT (GPU required, EN/ZH, MIT code / CC-BY-NC-4.0 weights)
#@markdown - **Non-commercial only.** Zero-shot voice cloning — always uses a reference audio.
MASKGCT_DEFAULT_VOICE = "default"  #@param ["default", "clone"]
MASKGCT_PROMPT_WAV = ""  #@param {type:"string"}
MASKGCT_PROMPT_TEXT = ""  #@param {type:"string"}
MASKGCT_PROMPT_LANG = "en"  #@param ["en", "zh", "ja", "ko", "fr", "de"]
MASKGCT_TARGET_LANG = "en"  #@param ["en", "zh", "ja", "ko", "fr", "de"]

#@markdown ---
#@markdown GPT-SoVITS (GPU recommended, ZH/EN/JA/KO/YUE, MIT)
#@markdown - Few-shot voice cloning (5-second reference). Reference audio + transcript required.
#@markdown - License: MIT for both code ([RVC-Boss/GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)) and pretrained weights (`lj1995/GPT-SoVITS`). Commercial use OK. Per-version (v1/v2/v3/v4) licenses are not individually attested upstream — verify before shipping a commercial product.
GPT_SOVITS_VERSION = "v2"  #@param ["v1", "v2", "v2Pro", "v2ProPlus", "v3", "v4"]
GPT_SOVITS_DEFAULT_VOICE = "default"  #@param ["default", "clone"]
GPT_SOVITS_PROMPT_WAV = ""  #@param {type:"string"}
GPT_SOVITS_PROMPT_TEXT = ""  #@param {type:"string"}
GPT_SOVITS_PROMPT_LANG = "en"  #@param ["en", "zh", "ja", "ko", "yue", "auto"]
GPT_SOVITS_TARGET_LANG = "en"  #@param ["en", "zh", "ja", "ko", "yue", "auto"]

#@markdown ---
#@markdown Higgs Audio v2 (A100/L4 required, VRAM 24GB+, voice cloning)
#@markdown - Code: Apache-2.0. **Weights: Boson Higgs Audio 2 Community License** (Llama-derived).
#@markdown - Restrictions: >100k MAU requires extra license; outputs cannot be used to train other LLMs.
HIGGS_HF_MODEL = "bosonai/higgs-audio-v2-generation-3B-base"  #@param {type:"string"}
HIGGS_HF_TOKENIZER = "bosonai/higgs-audio-v2-tokenizer"  #@param {type:"string"}
HIGGS_DEFAULT_VOICE = "default"  #@param {type:"string"}
HIGGS_DEFAULT_REF_VOICE = "belinda"  #@param {type:"string"}
HIGGS_PROMPT_WAV = ""  #@param {type:"string"}
HIGGS_PROMPT_TEXT = ""  #@param {type:"string"}
HIGGS_MAX_NEW_TOKENS = 1024  #@param {type:"integer"}
HIGGS_TEMPERATURE = 0.7  #@param {type:"number"}

#@markdown ---
#@markdown Higgs Audio v3 (A100/L4 required, 100+ languages incl. Japanese, voice cloning)
#@markdown - Separate 4B chat-native TTS (Qwen3-4B backbone) served by **SGLang-Omni**, which natively exposes `/v1/audio/speech`. Distinct from Higgs Audio v2.
#@markdown - **No HF_TOKEN needed**: weights ([bosonai/higgs-audio-v3-tts-4b](https://huggingface.co/bosonai/higgs-audio-v3-tts-4b)) are ungated.
#@markdown - L4 (24GB) verified (~19.9GB at load). T4 unsupported (sgl-kernel/flash-attn need sm_80+). First launch is slow (~10-12 min: download + torch.compile / CUDA-graph capture). Python 3.12 venv builds sglang-omni from source.
#@markdown - License: GitHub code is Apache-2.0, but **weights are under the Boson Higgs Audio v3 Research and Non-Commercial License** ([LICENSE](https://huggingface.co/bosonai/higgs-audio-v3-tts-4b/blob/main/LICENSE)). Personal use / short-term evaluation is permitted; hosted API, production, or revenue-generating use needs a separate commercial license. `voice="clone"` needs `HIGGS_V3_PROMPT_WAV` (optionally `HIGGS_V3_PROMPT_TEXT`); otherwise use `voice="default"`.
HIGGS_V3_HF_MODEL = "bosonai/higgs-audio-v3-tts-4b"  #@param {type:"string"}
HIGGS_V3_DEFAULT_VOICE = "default"  #@param ["default", "clone"]
HIGGS_V3_PROMPT_WAV = ""  #@param {type:"string"}
HIGGS_V3_PROMPT_TEXT = ""  #@param {type:"string"}
HIGGS_V3_TEMPERATURE = 0.7  #@param {type:"number"}
HIGGS_V3_TOP_K = 50  #@param {type:"integer"}
HIGGS_V3_MAX_NEW_TOKENS = 2048  #@param {type:"integer"}

#@markdown ---
#@markdown dots.tts (L4 required, 24 languages incl. Japanese, voice cloning)
#@markdown - rednote-hilab's 2B fully continuous, end-to-end autoregressive TTS (Qwen2.5-1.5B backbone + flow-matching head over a 48 kHz AudioVAE). Runs in-process (no separate backend).
#@markdown - **No HF_TOKEN needed**: weights ([rednote-hilab/dots.tts-base](https://huggingface.co/rednote-hilab/dots.tts-base), ~9.5GB) are ungated. Python 3.12 venv installs torch==2.8.0.
#@markdown - Fundamentally a zero-shot cloning model: `default` = no reference (random-voice sampling; a stable single speaker is only meaningful on a fine-tuned checkpoint). Set `DOTS_TTS_PROMPT_WAV` (optionally `DOTS_TTS_PROMPT_TEXT` for continuation cloning) and use `voice="clone"` for a stable voice.
#@markdown - Checkpoints (all 2B / Apache-2.0): `rednote-hilab/dots.tts-base` (Pretrain), `rednote-hilab/dots.tts-soar` (Self-Corrective Alignment, higher SIM), `rednote-hilab/dots.tts-mf` (MeanFlow distilled, NFE=4, fastest).
#@markdown - License: code and weights are both **Apache-2.0** (commercial use OK). Misuse for impersonation/fraud/disinformation is prohibited by the upstream terms.
DOTS_TTS_HF_MODEL = "rednote-hilab/dots.tts-base"  #@param ["rednote-hilab/dots.tts-base", "rednote-hilab/dots.tts-soar", "rednote-hilab/dots.tts-mf"]
DOTS_TTS_DEFAULT_VOICE = "default"  #@param ["default", "clone"]
DOTS_TTS_PROMPT_WAV = ""  #@param {type:"string"}
DOTS_TTS_PROMPT_TEXT = ""  #@param {type:"string"}
DOTS_TTS_LANGUAGE = "auto_detect"  #@param {type:"string"}
DOTS_TTS_NUM_STEPS = 10  #@param {type:"integer"}
DOTS_TTS_GUIDANCE_SCALE = 1.2  #@param {type:"number"}
DOTS_TTS_SPEAKER_SCALE = 1.5  #@param {type:"number"}

#@markdown ---
#@markdown LFM2.5-Audio-JP (L4 required, Japanese-only, no voice cloning)
#@markdown - Liquid AI's end-to-end speech-text model (1.5B): speech-to-speech / ASR / TTS. This JP checkpoint is Japanese-only with a single built-in voice (no reference / cloning). Runs in-process via the `liquid-audio` library; output is 24 kHz.
#@markdown - **No HF_TOKEN needed**: weights ([LiquidAI/LFM2.5-Audio-1.5B-JP](https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B-JP)) are ungated. Python 3.12 venv installs liquid-audio + torch>=2.8 (flash-attn optional, falls back to torch SDPA).
#@markdown - TTS uses sequential generation with the system prompt below. Tune length with `LFM2_AUDIO_JP_MAX_NEW_TOKENS` (counts text+audio tokens; audio is ~12.5 frames/sec).
#@markdown - License: code and weights are **LFM Open License v1.0** (commercial OK for orgs under $10M annual revenue; above that needs a separate commercial license). Audio encoder is Apache-2.0 (NVIDIA NeMo); audio codec (Mimi) is CC-BY-4.0 (Kyutai).
LFM2_AUDIO_JP_HF_MODEL = "LiquidAI/LFM2.5-Audio-1.5B-JP"  #@param {type:"string"}
LFM2_AUDIO_JP_SYSTEM_PROMPT = "Perform TTS in japanese."  #@param {type:"string"}
LFM2_AUDIO_JP_MAX_NEW_TOKENS = 1024  #@param {type:"integer"}
LFM2_AUDIO_JP_AUDIO_TEMPERATURE = 0.8  #@param {type:"number"}
LFM2_AUDIO_JP_AUDIO_TOP_K = 64  #@param {type:"integer"}

#@markdown ---
#@markdown Ming-omni-TTS (A100 required, ~34GB weights, Chinese/English-centric, voice cloning)
#@markdown - inclusionAI's 16.8B-A3B MoE audio LM (~3B active params) with a 12.5 Hz continuous tokenizer + DiT head. Runs in-process (no separate backend).
#@markdown - **A100 40GB required**: the 16.8B checkpoint is ~34GB in bf16 and will not fit on an L4 (24GB). Python 3.10 venv installs torch==2.6.0 + grouped_gemm (MoE kernel, built from source) + a FlashAttention wheel.
#@markdown - **No HF_TOKEN needed**: weights ([inclusionAI/Ming-omni-tts-16.8B-A3B](https://huggingface.co/inclusionAI/Ming-omni-tts-16.8B-A3B), ~34GB) are ungated.
#@markdown - `default` = the built-in voice (zero speaker-embedding, no reference). Set `MING_OMNI_TTS_PROMPT_WAV` (optionally `MING_OMNI_TTS_PROMPT_TEXT`) and use `voice="clone"` for zero-shot cloning. Output is 44.1 kHz.
#@markdown - License: code is **MIT** ([GitHub](https://github.com/inclusionAI/Ming-omni-tts)), weights are **Apache-2.0** (HF model card). Both allow commercial use. Misuse for impersonation/fraud/disinformation is prohibited by the upstream terms.
#@markdown - **Prompt-driven control**: `MING_OMNI_TTS_TASK` switches what to generate — `speech`, `music`, or `tta` (sound events); for `music`/`tta` the input text is a description. `MING_OMNI_TTS_STYLE` / `_EMOTION` / `_DIALECT` are natural-language voice design (mapped to the Chinese instruction keys 风格 / 情感 / 方言; best written in Chinese, e.g. style=`温柔自然的年轻女性声音` for a gentle female voice). All optional — empty keeps the plain `default`/`clone` behavior. They can also be overridden per request via the `task`/`style`/`emotion`/`dialect` body fields.
MING_OMNI_TTS_HF_MODEL = "inclusionAI/Ming-omni-tts-16.8B-A3B"  #@param {type:"string"}
MING_OMNI_TTS_DEFAULT_VOICE = "default"  #@param ["default", "clone"]
MING_OMNI_TTS_PROMPT_WAV = ""  #@param {type:"string"}
MING_OMNI_TTS_PROMPT_TEXT = ""  #@param {type:"string"}
MING_OMNI_TTS_TASK = "speech"  #@param ["speech", "music", "tta"]
MING_OMNI_TTS_STYLE = ""  #@param {type:"string"}
MING_OMNI_TTS_EMOTION = ""  #@param {type:"string"}
MING_OMNI_TTS_DIALECT = ""  #@param {type:"string"}
MING_OMNI_TTS_MAX_DECODE_STEPS = 200  #@param {type:"integer"}
MING_OMNI_TTS_CFG = 2.0  #@param {type:"number"}
MING_OMNI_TTS_SIGMA = 0.25  #@param {type:"number"}
MING_OMNI_TTS_TEMPERATURE = 0.0  #@param {type:"number"}

#@markdown ---
#@markdown DramaBox (A100 required, VRAM ~24GB, English-only, voice cloning)
#@markdown - Resemble AI's directable / expressive TTS (IC-LoRA fine-tune of LTX-2.3, paralinguistic cues like laughs/sighs).
#@markdown - **License: LTX-2 Community License** (Lightricks). Non-compete clause; commercial license required for org revenue $10M+.
#@markdown - Generated audio is **always watermarked** with Resemble Perth (imperceptible, non-removable per upstream).
#@markdown - First-run downloads ~8.5GB (DramaBox) + Gemma 3 12B snapshot.
DRAMABOX_HF_MODEL = "ResembleAI/Dramabox"  #@param {type:"string"}
DRAMABOX_GEMMA_REPO = "unsloth/gemma-3-12b-it-bnb-4bit"  #@param {type:"string"}
DRAMABOX_DEFAULT_VOICE = "default"  #@param ["default", "clone"]
DRAMABOX_DEFAULT_REF_VOICE = "female_american"  #@param ["female_american", "female_shadowheart", "male_arnie", "male_conan", "male_harvey_keitel", "male_old_movie", "male_petergriffin", "male_samuel_j"]
DRAMABOX_PROMPT_WAV = ""  #@param {type:"string"}
DRAMABOX_DTYPE = "bf16"  #@param ["bf16", "fp16"]
DRAMABOX_CFG_SCALE = 2.5  #@param {type:"number"}
DRAMABOX_STG_SCALE = 1.5  #@param {type:"number"}
DRAMABOX_DURATION_MULTIPLIER = 1.1  #@param {type:"number"}
DRAMABOX_SEED = 42  #@param {type:"integer"}
DRAMABOX_COMPILE = False  #@param {type:"boolean"}
DRAMABOX_NO_BNB_4BIT = False  #@param {type:"boolean"}

#@markdown ---
#@markdown Scenema (A100 required, 40GB VRAM, English-centric multilingual, voice cloning)
#@markdown - Zero-shot expressive voice cloning + speech generation (Scenema AI).
#@markdown - Audio model is derived from LTX-2.3 → **LTX-2 Community License** (Lightricks, same as DramaBox).
#@markdown - Uses Gemma 3 12B IT (gated): accept https://huggingface.co/google/gemma-3-12b-it and set `HF_TOKEN`.
#@markdown - First-run downloads ~38GB (audio transformer + pipeline + Gemma 3 12B + SeedVC + BigVGAN + Whisper).
#@markdown - Pass plain text → wrapped in `<speak voice="..." gender="...">` automatically.
#@markdown   Or write the full `<speak>...<action>...</action>...</speak>` XML yourself for emotion control.
SCENEMA_DEFAULT_VOICE = "default"  #@param ["default", "warm_male", "smoky_female", "child_girl", "elderly_male", "elderly_female", "clone"]
SCENEMA_DEFAULT_GENDER = "male"  #@param ["male", "female"]
SCENEMA_PROMPT_WAV = ""  #@param {type:"string"}
SCENEMA_GEMMA_QUANTIZE = "nf4"  #@param ["nf4", ""]
SCENEMA_SEED = -1  #@param {type:"integer"}
SCENEMA_PACE = 1.5  #@param {type:"number"}
SCENEMA_NO_VALIDATE = False  #@param {type:"boolean"}
SCENEMA_MIN_MATCH_RATIO = 0.90  #@param {type:"number"}
SCENEMA_SKIP_VC = False  #@param {type:"boolean"}
SCENEMA_VC_STEPS = 25  #@param {type:"integer"}
SCENEMA_VC_CFG_RATE = 0.5  #@param {type:"number"}
SCENEMA_BACKGROUND_SFX = False  #@param {type:"boolean"}

#@markdown ---
#@markdown Supertonic (CPU OK, 31 languages incl JP/KO/EN, ONNX)
#@markdown - Code: MIT. Weights: OpenRAIL-M (commercial OK, use-based ethical restrictions).
#@markdown - Voice presets only (no voice cloning). Language is auto-detected for CJK text.
SUPERTONIC_MODEL = "supertonic-3"  #@param ["supertonic-3", "supertonic-2", "supertonic"]
SUPERTONIC_DEFAULT_VOICE = "M1"  #@param ["M1", "M2", "M3", "M4", "M5", "F1", "F2", "F3", "F4", "F5"]
SUPERTONIC_DEFAULT_LANG = "en"  #@param ["en", "ja", "ko", "ar", "bg", "cs", "da", "de", "el", "es", "et", "fi", "fr", "hi", "hr", "hu", "id", "it", "lt", "lv", "nl", "pl", "pt", "ro", "ru", "sk", "sl", "sv", "tr", "uk", "vi", "na"]
SUPERTONIC_TOTAL_STEPS = 5  #@param {type:"integer"}

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
        "--f5tts-model",
        F5TTS_MODEL,
        "--f5tts-ckpt-file",
        F5TTS_CKPT_FILE,
        "--f5tts-vocab-file",
        F5TTS_VOCAB_FILE,
        "--fish-speech-model",
        FISH_SPEECH_MODEL,
        "--irodori-hf-checkpoint",
        IRODORI_HF_CHECKPOINT,
        "--irodori-codec-repo",
        IRODORI_CODEC_REPO,
        "--irodori-model-precision",
        IRODORI_MODEL_PRECISION,
        "--irodori-codec-precision",
        IRODORI_CODEC_PRECISION,
        "--irodori-lite-hf-checkpoint",
        IRODORI_LITE_HF_CHECKPOINT,
        "--irodori-lite-checkpoint-file",
        IRODORI_LITE_CHECKPOINT_FILE,
        "--irodori-lite-codec-repo",
        IRODORI_LITE_CODEC_REPO,
        "--kokoro-default-voice",
        KOKORO_DEFAULT_VOICE,
        "--kokoro-default-lang-code",
        KOKORO_DEFAULT_LANG_CODE,
        "--kokoro-onnx-hf-model",
        KOKORO_ONNX_HF_MODEL,
        "--kokoro-onnx-default-voice",
        KOKORO_ONNX_DEFAULT_VOICE,
        "--kokoro-onnx-default-lang-code",
        KOKORO_ONNX_DEFAULT_LANG_CODE,
        "--kokoro-onnx-provider",
        KOKORO_ONNX_PROVIDER,
        "--kyutai-hf-repo",
        KYUTAI_HF_REPO,
        "--kyutai-voice-repo",
        KYUTAI_VOICE_REPO,
        "--kyutai-voice",
        KYUTAI_VOICE,
        "--kyutai-prompt-wav",
        KYUTAI_PROMPT_WAV,
        "--kyutai-default-voice",
        KYUTAI_DEFAULT_VOICE,
        "--pocket-language",
        POCKET_LANGUAGE,
        "--pocket-default-speaker",
        POCKET_DEFAULT_SPEAKER,
        "--pocket-prompt-wav",
        POCKET_PROMPT_WAV,
        "--pocket-default-voice",
        POCKET_DEFAULT_VOICE,
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
        "--voxcpm-hf-model",
        VOXCPM_HF_MODEL,
        "--voxcpm-cfg-value",
        str(VOXCPM_CFG_VALUE),
        "--voxcpm-inference-timesteps",
        str(VOXCPM_INFERENCE_TIMESTEPS),
        "--moss-tts-nano-hf-model",
        MOSS_TTS_NANO_HF_MODEL,
        "--moss-tts-nano-mode",
        MOSS_TTS_NANO_MODE,
        "--moss-tts-v1-5-hf-model",
        MOSS_TTS_V1_5_HF_MODEL,
        "--moss-tts-v1-5-language",
        MOSS_TTS_V1_5_LANGUAGE,
        "--moss-tts-v1-5-prompt-wav",
        MOSS_TTS_V1_5_PROMPT_WAV,
        "--moss-tts-v1-5-default-voice",
        MOSS_TTS_V1_5_DEFAULT_VOICE,
        "--moss-tts-v1-5-attn-impl",
        MOSS_TTS_V1_5_ATTN_IMPL,
        "--moss-tts-v1-5-max-new-tokens",
        str(MOSS_TTS_V1_5_MAX_NEW_TOKENS),
        "--moss-local-v1-5-hf-model",
        MOSS_LOCAL_V1_5_HF_MODEL,
        "--moss-local-v1-5-language",
        MOSS_LOCAL_V1_5_LANGUAGE,
        "--moss-local-v1-5-prompt-wav",
        MOSS_LOCAL_V1_5_PROMPT_WAV,
        "--moss-local-v1-5-default-voice",
        MOSS_LOCAL_V1_5_DEFAULT_VOICE,
        "--moss-local-v1-5-attn-impl",
        MOSS_LOCAL_V1_5_ATTN_IMPL,
        "--moss-local-v1-5-max-new-tokens",
        str(MOSS_LOCAL_V1_5_MAX_NEW_TOKENS),
        "--neutts-backbone-repo",
        NEUTTS_BACKBONE_REPO,
        "--neutts-codec-repo",
        NEUTTS_CODEC_REPO,
        "--neutts-default-voice",
        NEUTTS_DEFAULT_VOICE,
        "--sarashina-hf-model",
        SARASHINA_HF_MODEL,
        "--sarashina-prompt-wav",
        SARASHINA_PROMPT_WAV,
        "--sarashina-prompt-text",
        SARASHINA_PROMPT_TEXT,
        "--sarashina-default-voice",
        SARASHINA_DEFAULT_VOICE,
        "--chatterbox-language",
        CHATTERBOX_LANGUAGE,
        "--chatterbox-prompt-wav",
        CHATTERBOX_PROMPT_WAV,
        "--chatterbox-default-voice",
        CHATTERBOX_DEFAULT_VOICE,
        "--zonos-hf-model",
        ZONOS_HF_MODEL,
        "--zonos-language",
        ZONOS_LANGUAGE,
        "--zonos-prompt-wav",
        ZONOS_PROMPT_WAV,
        "--zonos-default-voice",
        ZONOS_DEFAULT_VOICE,
        "--zonos2-hf-model",
        ZONOS2_HF_MODEL,
        "--zonos2-language",
        ZONOS2_LANGUAGE,
        "--zonos2-prompt-wav",
        ZONOS2_PROMPT_WAV,
        "--zonos2-default-voice",
        ZONOS2_DEFAULT_VOICE,
        "--zonos2-default-ref",
        ZONOS2_DEFAULT_REF,
        "--zonos2-seed",
        str(ZONOS2_SEED),
        "--outetts-model-size",
        OUTETTS_MODEL_SIZE,
        "--outetts-backend",
        OUTETTS_BACKEND,
        "--outetts-default-speaker",
        OUTETTS_DEFAULT_SPEAKER,
        "--outetts-prompt-wav",
        OUTETTS_PROMPT_WAV,
        "--outetts-prompt-text",
        OUTETTS_PROMPT_TEXT,
        "--outetts-default-voice",
        OUTETTS_DEFAULT_VOICE,
        "--dia-hf-model",
        DIA_HF_MODEL,
        "--dia-compute-dtype",
        DIA_COMPUTE_DTYPE,
        "--dia-prompt-wav",
        DIA_PROMPT_WAV,
        "--dia-prompt-text",
        DIA_PROMPT_TEXT,
        "--dia-default-voice",
        DIA_DEFAULT_VOICE,
        "--openvoice-language",
        OPENVOICE_LANGUAGE,
        "--openvoice-prompt-wav",
        OPENVOICE_PROMPT_WAV,
        "--openvoice-default-voice",
        OPENVOICE_DEFAULT_VOICE,
        "--cosyvoice-hf-model",
        COSYVOICE_HF_MODEL,
        "--cosyvoice-prompt-wav",
        COSYVOICE_PROMPT_WAV,
        "--cosyvoice-prompt-text",
        COSYVOICE_PROMPT_TEXT,
        "--cosyvoice-default-voice",
        COSYVOICE_DEFAULT_VOICE,
        "--spark-hf-model",
        SPARK_HF_MODEL,
        "--spark-default-voice",
        SPARK_DEFAULT_VOICE,
        "--spark-default-gender",
        SPARK_DEFAULT_GENDER,
        "--spark-default-pitch",
        SPARK_DEFAULT_PITCH,
        "--spark-default-speed",
        SPARK_DEFAULT_SPEED,
        "--spark-prompt-wav",
        SPARK_PROMPT_WAV,
        "--spark-prompt-text",
        SPARK_PROMPT_TEXT,
        "--orpheus-hf-model",
        ORPHEUS_HF_MODEL,
        "--orpheus-default-voice",
        ORPHEUS_DEFAULT_VOICE,
        "--orpheus-max-model-len",
        str(ORPHEUS_MAX_MODEL_LEN),
        "--vibevoice-hf-model",
        VIBEVOICE_HF_MODEL,
        "--vibevoice-default-speaker",
        VIBEVOICE_DEFAULT_SPEAKER,
        "--vibevoice-prompt-wav",
        VIBEVOICE_PROMPT_WAV,
        "--vibevoice-default-voice",
        VIBEVOICE_DEFAULT_VOICE,
        "--vibevoice-ddpm-steps",
        str(VIBEVOICE_DDPM_STEPS),
        "--vibevoice-cfg-scale",
        str(VIBEVOICE_CFG_SCALE),
        "--bark-default-voice",
        BARK_DEFAULT_VOICE,
        "--chattts-default-voice",
        CHATTTS_DEFAULT_VOICE,
        "--chattts-seed",
        str(CHATTTS_SEED),
        "--chattts-temperature",
        str(CHATTTS_TEMPERATURE),
        "--csm-hf-model",
        CSM_HF_MODEL,
        "--csm-llama-model",
        CSM_LLAMA_MODEL,
        "--csm-default-voice",
        CSM_DEFAULT_VOICE,
        "--csm-default-speaker",
        str(CSM_DEFAULT_SPEAKER),
        "--csm-max-audio-length-ms",
        str(CSM_MAX_AUDIO_LENGTH_MS),
        "--csm-temperature",
        str(CSM_TEMPERATURE),
        "--misotts-hf-model",
        MISOTTS_HF_MODEL,
        "--misotts-default-voice",
        MISOTTS_DEFAULT_VOICE,
        "--misotts-default-speaker",
        str(MISOTTS_DEFAULT_SPEAKER),
        "--misotts-prompt-wav",
        MISOTTS_PROMPT_WAV,
        "--misotts-prompt-text",
        MISOTTS_PROMPT_TEXT,
        "--misotts-max-audio-length-ms",
        str(MISOTTS_MAX_AUDIO_LENGTH_MS),
        "--misotts-temperature",
        str(MISOTTS_TEMPERATURE),
        "--misotts-topk",
        str(MISOTTS_TOPK),
        "--misotts-tokenizer-repo",
        MISOTTS_TOKENIZER_REPO,
        "--styletts2-default-voice",
        STYLETTS2_DEFAULT_VOICE,
        "--styletts2-prompt-wav",
        STYLETTS2_PROMPT_WAV,
        "--styletts2-alpha",
        str(STYLETTS2_ALPHA),
        "--styletts2-beta",
        str(STYLETTS2_BETA),
        "--styletts2-diffusion-steps",
        str(STYLETTS2_DIFFUSION_STEPS),
        "--styletts2-embedding-scale",
        str(STYLETTS2_EMBEDDING_SCALE),
        "--maskgct-default-voice",
        MASKGCT_DEFAULT_VOICE,
        "--maskgct-prompt-wav",
        MASKGCT_PROMPT_WAV,
        "--maskgct-prompt-text",
        MASKGCT_PROMPT_TEXT,
        "--maskgct-prompt-lang",
        MASKGCT_PROMPT_LANG,
        "--maskgct-target-lang",
        MASKGCT_TARGET_LANG,
        "--gpt-sovits-version",
        GPT_SOVITS_VERSION,
        "--gpt-sovits-default-voice",
        GPT_SOVITS_DEFAULT_VOICE,
        "--gpt-sovits-prompt-wav",
        GPT_SOVITS_PROMPT_WAV,
        "--gpt-sovits-prompt-text",
        GPT_SOVITS_PROMPT_TEXT,
        "--gpt-sovits-prompt-lang",
        GPT_SOVITS_PROMPT_LANG,
        "--gpt-sovits-target-lang",
        GPT_SOVITS_TARGET_LANG,
        "--higgs-hf-model",
        HIGGS_HF_MODEL,
        "--higgs-hf-tokenizer",
        HIGGS_HF_TOKENIZER,
        "--higgs-default-voice",
        HIGGS_DEFAULT_VOICE,
        "--higgs-default-ref-voice",
        HIGGS_DEFAULT_REF_VOICE,
        "--higgs-prompt-wav",
        HIGGS_PROMPT_WAV,
        "--higgs-prompt-text",
        HIGGS_PROMPT_TEXT,
        "--higgs-max-new-tokens",
        str(HIGGS_MAX_NEW_TOKENS),
        "--higgs-temperature",
        str(HIGGS_TEMPERATURE),
        "--higgs-v3-hf-model",
        HIGGS_V3_HF_MODEL,
        "--higgs-v3-default-voice",
        HIGGS_V3_DEFAULT_VOICE,
        "--higgs-v3-prompt-wav",
        HIGGS_V3_PROMPT_WAV,
        "--higgs-v3-prompt-text",
        HIGGS_V3_PROMPT_TEXT,
        "--higgs-v3-temperature",
        str(HIGGS_V3_TEMPERATURE),
        "--higgs-v3-top-k",
        str(HIGGS_V3_TOP_K),
        "--higgs-v3-max-new-tokens",
        str(HIGGS_V3_MAX_NEW_TOKENS),
        "--dots-tts-hf-model",
        DOTS_TTS_HF_MODEL,
        "--dots-tts-default-voice",
        DOTS_TTS_DEFAULT_VOICE,
        "--dots-tts-prompt-wav",
        DOTS_TTS_PROMPT_WAV,
        "--dots-tts-prompt-text",
        DOTS_TTS_PROMPT_TEXT,
        "--dots-tts-language",
        DOTS_TTS_LANGUAGE,
        "--dots-tts-num-steps",
        str(DOTS_TTS_NUM_STEPS),
        "--dots-tts-guidance-scale",
        str(DOTS_TTS_GUIDANCE_SCALE),
        "--dots-tts-speaker-scale",
        str(DOTS_TTS_SPEAKER_SCALE),
        "--lfm2-audio-jp-hf-model",
        LFM2_AUDIO_JP_HF_MODEL,
        "--lfm2-audio-jp-system-prompt",
        LFM2_AUDIO_JP_SYSTEM_PROMPT,
        "--lfm2-audio-jp-max-new-tokens",
        str(LFM2_AUDIO_JP_MAX_NEW_TOKENS),
        "--lfm2-audio-jp-audio-temperature",
        str(LFM2_AUDIO_JP_AUDIO_TEMPERATURE),
        "--lfm2-audio-jp-audio-top-k",
        str(LFM2_AUDIO_JP_AUDIO_TOP_K),
        "--ming-omni-tts-hf-model",
        MING_OMNI_TTS_HF_MODEL,
        "--ming-omni-tts-default-voice",
        MING_OMNI_TTS_DEFAULT_VOICE,
        "--ming-omni-tts-prompt-wav",
        MING_OMNI_TTS_PROMPT_WAV,
        "--ming-omni-tts-prompt-text",
        MING_OMNI_TTS_PROMPT_TEXT,
        "--ming-omni-tts-task",
        MING_OMNI_TTS_TASK,
        "--ming-omni-tts-style",
        MING_OMNI_TTS_STYLE,
        "--ming-omni-tts-emotion",
        MING_OMNI_TTS_EMOTION,
        "--ming-omni-tts-dialect",
        MING_OMNI_TTS_DIALECT,
        "--ming-omni-tts-max-decode-steps",
        str(MING_OMNI_TTS_MAX_DECODE_STEPS),
        "--ming-omni-tts-cfg",
        str(MING_OMNI_TTS_CFG),
        "--ming-omni-tts-sigma",
        str(MING_OMNI_TTS_SIGMA),
        "--ming-omni-tts-temperature",
        str(MING_OMNI_TTS_TEMPERATURE),
        "--supertonic-model",
        SUPERTONIC_MODEL,
        "--supertonic-default-voice",
        SUPERTONIC_DEFAULT_VOICE,
        "--supertonic-default-lang",
        SUPERTONIC_DEFAULT_LANG,
        "--supertonic-total-steps",
        str(SUPERTONIC_TOTAL_STEPS),
        "--dramabox-hf-model",
        DRAMABOX_HF_MODEL,
        "--dramabox-gemma-repo",
        DRAMABOX_GEMMA_REPO,
        "--dramabox-default-voice",
        DRAMABOX_DEFAULT_VOICE,
        "--dramabox-default-ref-voice",
        DRAMABOX_DEFAULT_REF_VOICE,
        "--dramabox-prompt-wav",
        DRAMABOX_PROMPT_WAV,
        "--dramabox-dtype",
        DRAMABOX_DTYPE,
        "--dramabox-cfg-scale",
        str(DRAMABOX_CFG_SCALE),
        "--dramabox-stg-scale",
        str(DRAMABOX_STG_SCALE),
        "--dramabox-duration-multiplier",
        str(DRAMABOX_DURATION_MULTIPLIER),
        "--dramabox-seed",
        str(DRAMABOX_SEED),
        "--scenema-default-voice",
        SCENEMA_DEFAULT_VOICE,
        "--scenema-default-gender",
        SCENEMA_DEFAULT_GENDER,
        "--scenema-prompt-wav",
        SCENEMA_PROMPT_WAV,
        "--scenema-gemma-quantize",
        SCENEMA_GEMMA_QUANTIZE,
        "--scenema-seed",
        str(SCENEMA_SEED),
        "--scenema-pace",
        str(SCENEMA_PACE),
        "--scenema-min-match-ratio",
        str(SCENEMA_MIN_MATCH_RATIO),
        "--scenema-vc-steps",
        str(SCENEMA_VC_STEPS),
        "--scenema-vc-cfg-rate",
        str(SCENEMA_VC_CFG_RATE),
    ]
    if IRODORI_LITE_CODEC_INT4:
        cmd.append("--irodori-lite-codec-int4")
    if SARASHINA_USE_VLLM:
        cmd.append("--sarashina-use-vllm")
    if BARK_USE_SMALL_MODELS:
        cmd.append("--bark-use-small-models")
    if DRAMABOX_COMPILE:
        cmd.append("--dramabox-compile")
    if DRAMABOX_NO_BNB_4BIT:
        cmd.append("--dramabox-no-bnb-4bit")
    if SCENEMA_NO_VALIDATE:
        cmd.append("--scenema-no-validate")
    if SCENEMA_SKIP_VC:
        cmd.append("--scenema-skip-vc")
    if SCENEMA_BACKGROUND_SFX:
        cmd.append("--scenema-background-sfx")
    if not ZONOS2_ACCURATE_MODE:
        cmd.append("--zonos2-no-accurate-mode")
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

### Kokoro-ONNX

NVIDIA が Kokoro-82M を ONNX 化したモデル（[nvidia/kokoro-82M-onnx-opt](https://huggingface.co/nvidia/kokoro-82M-onnx-opt)）を、PyTorch ではなく `onnxruntime` で実行するエンジンです。モデル同梱の `voices.bin` から 53 種類すべての preset voice を公開し、Kokoro と同じ 9 言語（米/英英語・スペイン語・フランス語・ヒンディー語・イタリア語・日本語・ブラジルポルトガル語・中国語）に対応します。言語は voice 名の接頭辞（`a`/`b`=英語, `e`=西語, `f`=仏語, `h`=ヒンディー, `i`=伊語, `j`=日本語, `p`=ポルトガル語, `z`=中国語）から自動判定します。

音素化には Kokoro 公式の G2P である [misaki](https://github.com/hexgrad/misaki) を使用します（日本語=`ja`、中国語=`zh`、英語=`en`+espeak-ng フォールバック、スペイン/フランス/ヒンディー/イタリア/ポルトガル=espeak-ng）。NVIDIA は本モデルを Windows/ONNXRuntime-EP 向けの独自フォニマイザ資産付きで配布しているため、本ラッパーは ONNX グラフ（`tokens` / `style` / `speed` → `audio`）を直接駆動し、音素は misaki から供給することで Colab(Linux) で動く形にしています。

`KOKORO_ONNX_PROVIDER` で実行プロバイダを選べます。`auto`（デフォルト）と `cuda` は GPU 優先・CPU フォールバック、`cpu` は CPU 強制です。82M と軽量なため GPU でも CPU でも快適に動作します。デフォルト voice は日本語の `jf_alpha` で、全 voice は `/v1/voices` で確認できます。Voice cloning は非対応（preset のみ）です。PyTorch 版の `Kokoro` とは別エンジンなので、両方を個別に選択できます。

### Irodori-TTS

[Aratako/Irodori-TTS](https://github.com/Aratako/Irodori-TTS) を使った日本語 TTS です。デフォルトで Hugging Face の `Aratako/Irodori-TTS-500M-v3`（スクラッチ学習の Rectified Flow DiT）を使用します。V2 にフォールバックする場合は `IRODORI_HF_CHECKPOINT=Aratako/Irodori-TTS-500M-v2`、V1 を利用する場合は `Aratako/Irodori-TTS-500M` を指定してください。出力は 48kHz で高音質ですが、voice の切り替え機能はありません。

V3 では上流の以下の変更にラッパー側で自動追従します:

- **Duration Predictor**: V3 では `seconds=None` を渡し、入力テキストから出力長を自動推定させます（V2 / V1 は従来通り 30 秒固定）。
- **SilentCipher ウォーターマーク統合**: V3 重みには [SilentCipher](https://github.com/sony/silentcipher) が同梱されており、上流の `InferenceRuntime` 内で常時初期化されます（`RuntimeKey` から `enable_watermark` 引数も削除されており、ユーザー側からの無効化スイッチは公開されていません）。SilentCipher の重みが読み込める限り、生成音声には常にウォーターマークが入ります。**ウォーターマークの除去は禁止**です（モデルリリースの一部として配布されています）。

### Irodori-TTS-Lite

[kizuna-intelligence/Irodori-TTS-Lite](https://github.com/kizuna-intelligence/Irodori-TTS-Lite) を使った Irodori-TTS の int4 量子化推論ランタイムです。`irodori_tts.inference_runtime.InferenceRuntime.from_key` をモンキーパッチして、4-bit safetensors を Triton の `FusedInt4Linear` カーネルで直接読み込めるようにします。エンドツーエンドのピーク VRAM は約 1 GB（fp32 パスの ~2 GB から削減）で、音質はほぼ劣化しません。

利用可能な int4 チェックポイントは 2 種類:

- **`kizuna-intelligence/Irodori-TTS-Lite-int4`**（デフォルト）: voice-design int4（話者は重みに焼き込み、Duration Predictor なし）。ラッパー側で `pyopenjtalk` の音素数から `seconds` を導出します。
- **`kizuna-intelligence/Irodori-TTS-500M-v3-int4`**: v3 ベースの int4（Duration Predictor 付き）。利用するには `IRODORI_LITE_HF_CHECKPOINT=kizuna-intelligence/Irodori-TTS-500M-v3-int4` **かつ** `IRODORI_LITE_CHECKPOINT_FILE=model.safetensors` を指定してください。

voice 切り替えは未対応（話者は重みに焼き込まれているため、Irodori-TTS 本体と同じ挙動）。DACVAE コーデックは `Aratako/Semantic-DACVAE-Japanese-32dim`（fp16）をデフォルトで使用します。`IRODORI_LITE_CODEC_INT4=1` を指定するとコーデックも int4 化され、デコード遅延が約 150 ms 増える代わりにピーク VRAM が約 500 MB 削減されます。

GPU 必須: int4 パスは Triton カーネルを使用するため、Linux + CUDA（= Colab GPU ランタイム）が必要です。

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

### MOSS-TTS-v1.5

[OpenMOSS/MOSS-TTS](https://github.com/OpenMOSS/MOSS-TTS) の 8B パラメータ LLM ベース多言語 TTS で、Hugging Face の `OpenMOSS-Team/MOSS-TTS-v1.5` を使用します。中国語、広東語、英語、アラビア語、チェコ語、デンマーク語、オランダ語、フィンランド語、フランス語、ドイツ語、ギリシャ語、ヘブライ語、ヒンディー語、ハンガリー語、イタリア語、**日本語**、韓国語、マケドニア語、マレー語、ペルシャ語、ポーランド語、ポルトガル語、ルーマニア語、ロシア語、スペイン語、スワヒリ語、スウェーデン語、タガログ語、タイ語、トルコ語、ベトナム語の **31 言語**に対応します。フォームの `MOSS_TTS_V1_5_LANGUAGE` で言語タグを明示できます。`MOSS_TTS_V1_5_PROMPT_WAV` に参照音声を指定して `voice="clone"` を選ぶことでゼロショット voice cloning も使えます。**Colab A100 が必須**です — bf16 重みは公称 16 GB ですが、transformers の `device_map=` 経由ロードで KV cache / attention buffer まで先取り確保される結果、音声トークナイザーを GPU に移す前で既に約 22 GB を占有し、L4 (22 GB) ではこの段階で OOM します（Colab A100 で end-to-end 確認済み、L4 で OOM 再現済み）。A100 では合成は 1 リクエストあたり約 4 秒（24 kHz mono）で完了します。インストーラは専用の Python 3.12 venv を作成し、上流の `[torch-runtime]` extra（`torch==2.9.1+cu128` / `transformers==5.0.0`）と `accelerate`（`device_map=` に必要）を導入します。`attn_implementation` のデフォルトは `sdpa` で、`flash_attention_2` に切り替える場合は別途 `flash-attn` のインストールが必要です。ライセンス: コード・重みとも **Apache 2.0**（商用利用 OK）。

### MOSS-TTS-Local-v1.5

[OpenMOSS/MOSS-TTS](https://github.com/OpenMOSS/MOSS-TTS) の ~4B パラメータ `MossTTSLocal` 系チェックポイントで、Hugging Face の `OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5` を使用します。OpenAI 互換ラッパーと **31 言語**（中国語、広東語、英語、アラビア語、チェコ語、デンマーク語、オランダ語、フィンランド語、フランス語、ドイツ語、ギリシャ語、ヘブライ語、ヒンディー語、ハンガリー語、イタリア語、**日本語**、韓国語、マケドニア語、マレー語、ペルシャ語、ポーランド語、ポルトガル語、ルーマニア語、ロシア語、スペイン語、スワヒリ語、スウェーデン語、タガログ語、タイ語、トルコ語、ベトナム語）、ゼロショット voice cloning は 8B の `MOSS-TTS-v1.5` と共通で、`MOSS_LOCAL_V1_5_LANGUAGE` フォームフィールドおよび `MOSS_LOCAL_V1_5_PROMPT_WAV` + `voice="clone"` で制御します。大きな違いはアーキテクチャとサイズで、8B の `MossTTSDelay` ではなく時間同期型の `MossTTSLocal`（RVQ depth transformer）を ~4B パラメータで採用し、MOSS-Audio-Tokenizer-v2 を介してネイティブ 48 kHz ステレオで出力します。8B 版のおよそ半分のサイズなので、Colab L4 の 22 GB VRAM に余裕を持って収まります — Colab L4 で end-to-end 確認済み（resident ~12.4 GB。8B 版が OOM する L4 でも動作。出力はネイティブ 48 kHz ステレオ WAV、1 文で約 6 秒）。インストーラは v1.5 と同じ構成で、専用の Python 3.12 venv に上流の `[torch-runtime]` extra（`torch==2.9.1+cu128` / `transformers==5.0.0`）と `accelerate`（`device_map=` に必要）を導入します。`attn_implementation` のデフォルトは `sdpa` で、`flash_attention_2` に切り替える場合は別途 `flash-attn` のインストールが必要です。ライセンス: コード・重みとも **Apache 2.0**（商用利用 OK）。

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

### Supertonic

Supertone Inc. の [supertone-inc/supertonic](https://github.com/supertone-inc/supertonic) を使った超軽量オンデバイス TTS です。`supertonic-3` モデル（約 99M params、ONNX アセット約 305MB）は英語・日本語・韓国語を含む 31 言語に対応し、未対応テキスト向けに `na` フォールバックも持ちます。ONNX Runtime で完全に CPU 動作（GPU 不要）。本ラッパーは初回リクエストでモデルをロードし、voice style はリクエスト間でキャッシュします。

`voice` パラメータは内蔵 10 プリセットを公開します:

| voice | 説明 |
|---|---|
| `M1` – `M5` | 男性プリセット（M1: 明るく前向き / M2: 落ち着いた低音 / M3: 信頼感のあるビジネス / M4: 柔らかく親しみやすい / M5: 暖かみのあるナレーション） |
| `F1` – `F5` | 女性プリセット（F1: 落ち着き / F2: 明るく若々しい / F3: アナウンサー調 / F4: 自信のあるビジネス / F5: 優しいウェルネス） |
| `default` | `--supertonic-default-voice` のエイリアス（既定は `M1`） |

公式 Python SDK は voice cloning に対応していません（クローン音声は Supertone の Voice Builder サービスで作成する想定）。`voice=clone` を指定すると明示的に 4xx を返します。

Supertonic-3 は合成時に言語ヒントが必要ですが、OpenAI 互換の `/v1/audio/speech` には `language` フィールドが無いため、本ラッパーは JSON body に拡張フィールド `language` を受け付けます。`language` が省略された場合は入力テキストのスクリプトから簡易判定（ひらがな・カタカナ・CJK → `ja`、ハングル → `ko`）し、それ以外は `--supertonic-default-lang`（既定 `en`）にフォールバックします。言語が分からないテキストは `"na"` を指定してください。

ライセンス: **コードは MIT**（https://github.com/supertone-inc/supertonic）、**重みは OpenRAIL-M**（https://huggingface.co/Supertone/supertonic-3）。商用利用 OK。なりすまし・ディープフェイク・誹謗中傷など、OpenRAIL-M の use-based ethical restrictions は適用されます（詳細はライセンス本文を参照）。

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

### ZONOS2

Zyphra の最新 TTS、[Zyphra/ZONOS2](https://github.com/Zyphra/Zonos2) です。Mixture-of-Experts バックボーンが、NeMo で正規化した UTF-8 バイト列と ECAPA-TDNN 話者埋め込みから DAC トークンを生成します（600万時間以上の多言語音声で学習）。3 つの tier で計 41 言語に対応し（tier-1: 英語・中国語・日本語）、高忠実度のゼロショット音声クローンを備えています。デフォルトモデルは `Zyphra/ZONOS2`。

インプロセスで動く Zonos v0.1 と異なり、ZONOS2 は [Mini-SGLang](https://github.com/sgl-project/mini-sglang) ベースの推論サーバのみを提供します。インストーラはリポジトリを clone し、`uv sync` でプロジェクト環境を構築したうえで `uv run python -m zonos2` を `--zonos2-backend-port`（既定 5003）でバックエンド起動します。薄い OpenAI 互換プロキシが `/v1/audio/speech` をバックエンドの `/tts/generate` に転送し、返ってくる 44.1kHz float32 PCM を WAV に変換します。

**GPU は compute capability sm_80+（L4 / A100）が必須です。** バックボーンが `flashinfer` / `sgl_kernel` / `cutlass` カーネルに依存するため、T4（sm_75）では動作しません。初回起動は遅く、`uv sync` での GPU カーネル取得に加え、初回生成時に重みがダウンロードされます。

`voice` パラメータには次の値を指定できます。

| voice | 説明 |
|---|---|
| `default` | 同梱の参照音声（`default_voices/<--zonos2-default-ref>`、例: `AmericanFemale.mp3`）を使用。クロスリンガルにクローンするため、`--zonos2-language ja` で日本語も話せます |
| `clone` | ゼロショット音声クローン。`--zonos2-prompt-wav` を指定したときのみ有効 |

`--zonos2-accurate-mode`（既定 ON）は声質再現を重視し、`--zonos2-no-accurate-mode` で表現力重視になります。`--zonos2-seed` に 0 以上を指定すると出力が再現可能になります。音声クローンを使う場合は、必ず権利を持っている音声（本人の同意がある音声）でのみ行ってください。ライセンス: コードは MIT（pyproject）、重みは Apache 2.0（HF モデルカード）。いずれも商用利用可。

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

### Dia

[nari-labs/dia](https://github.com/nari-labs/dia) を使った対話特化 TTS です。1.6B パラメータのモデルで、`[S1]` / `[S2]` 話者タグをプロンプトに含めることで、マルチスピーカー対話を 1 パスで生成します。現状は英語のみ対応。入力に話者タグが無い場合はラッパーが先頭に `[S1]` を自動挿入するので、シングルスピーカーの平文 TTS としても利用できます。デフォルトモデル: `nari-labs/Dia-1.6B-0626`。`--dia-prompt-wav` と `--dia-prompt-text` の両方を指定すると `clone` voice が有効になり、参照音声で声色を条件付けられます。GPU 推奨（VRAM ~4.4GB at float16/bfloat16、~7.9GB at float32）。ライセンス: Apache 2.0（コードと重み）。

`voice` パラメータには次の値を指定できます。

| voice | 説明 |
|---|---|
| `default` | 参照音声なしの plain TTS。`input` に `[S1]` / `[S2]` を含めるとマルチスピーカー対話になります |
| `clone` | 音声クローン。`--dia-prompt-wav` と `--dia-prompt-text` の両方が必要 |

音声クローンを使う場合は、必ず権利を持っている音声（本人の同意がある音声）でのみ行ってください。

### Kyutai-TTS

[kyutai-labs/delayed-streams-modeling](https://github.com/kyutai-labs/delayed-streams-modeling) を使った Kyutai Labs の英語 / フランス語 TTS です。Delayed Streams Modeling (DSM) フレームワーク上に実装されており、ストリーミング推論に対応しています。デフォルトモデルは `kyutai/tts-1.6b-en_fr`（1.6B パラメータ、英語 + フランス語）。voice は別の Hugging Face リポ（デフォルト `kyutai/tts-voices`）から読み込まれ、`default` voice では `KYUTAI_VOICE`（デフォルト `expresso/ex03-ex01_happy_001_channel1_334s.wav`）が参照されます。`--kyutai-prompt-wav`（ローカルの `.wav` または事前計算済みの `.safetensors` voice cache）を指定すると `clone` voice が有効になります。任意の voice repo 内パスを `voice` パラメータに直接指定することも可能です。GPU 推奨（CUDA、VRAM ~6GB）。日本語は **非対応**。ライセンス: コードは MIT (Python) / Apache 2.0 (Rust)、モデル重みは CC-BY-4.0。

### Pocket-TTS

[kyutai-labs/pocket-tts](https://github.com/kyutai-labs/pocket-tts) を使った Kyutai Labs の超軽量 CPU TTS です。100M パラメータで、MacBook Air M4 の CPU 2 コアだけで ~6x realtime で動作します。**GPU 不要**。デフォルトの Hugging Face モデルは `kyutai/pocket-tts`、voice は `kyutai/tts-voices` から取得。言語別モデルが用意されており（`english` / `english_2026-01` / `english_2026-04` / `french_24l` / `german_24l` / `italian` / `portuguese` / `spanish_24l`）、`--pocket-language` で選択します。`default` voice は `POCKET_DEFAULT_SPEAKER`（デフォルト: `alba`）の内蔵プリセットを使用、`--pocket-prompt-wav` を指定すると独自音声からの `clone` voice が有効になります。21 種類の内蔵プリセット名（`alba`、`anna`、`charles` ...）を `voice` パラメータに直接渡すこともできます。ライセンス: コードは MIT、モデル重みは CC-BY-4.0、**voice ごとに個別ライセンス**（[kyutai/tts-voices](https://huggingface.co/kyutai/tts-voices) を参照）。**Prohibited use:** 上流規約により、合意のない voice impersonation や偽情報の生成は禁止されています。

### Spark-TTS

[SparkAudio/Spark-TTS](https://github.com/SparkAudio/Spark-TTS) を使った 0.5B パラメータの Qwen2.5 ベース LLM-TTS です。**英語 / 中国語のみ**（日本語は **非対応**）対応で、ゼロショット voice cloning と、参照音声なしでの gender / pitch / speed 制御生成の 2 モードを持ちます。出力は 16 kHz モノラル WAV。GPU 推奨（VRAM ~4GB）。

`voice` パラメータ:

| voice | 説明 |
|---|---|
| `default` | 参照音声なしのプレーン TTS。`--spark-default-gender`（`male` / `female`）、`--spark-default-pitch`（`very_low` / `low` / `moderate` / `high` / `very_high`）、`--spark-default-speed`（同 5 段階）で制御します。 |
| `clone` | ゼロショット voice cloning。`--spark-prompt-wav` を必須とし、`--spark-prompt-text`（参照音声の書き起こし、任意）を一緒に渡すと品質が安定します。 |

voice cloning では、必ず権利を持つ参照音声（話者本人の同意）のみを使用してください。

**ライセンス注意:** コードは Apache 2.0 ですが、**`Spark-TTS-0.5B` の重みは CC BY-NC-SA 4.0（非商用のみ）** に変更されています（学習データのライセンス制約のため、当初 Apache 2.0 だったところを上流で再ライセンス）。Sarashina-TTS / OuteTTS 1B / Voxtral-TTS と同じ扱いになります — 研究・個人利用は OK、商用利用は不可。上流モデルカードでも、合意のない voice cloning・なりすまし・詐欺・違法利用は禁止されています。

### Orpheus-TTS（現在動作不可 — HF gated 重み）

[canopyai/Orpheus-TTS](https://github.com/canopyai/Orpheus-TTS) を使う構成で、`meta-llama/Llama-3.2-3B-Instruct` をベースに `orpheus-speech` 経由で vLLM 上にホスティングする英語 LLM-TTS として実装しています。デフォルトチェックポイント `canopylabs/orpheus-tts-0.1-finetune-prod` には英語 voice が 8 種類同梱されています: `tara`、`leah`、`jess`、`leo`、`dan`、`mia`、`zac`、`zoe`。出力は 24 kHz モノラル WAV。日本語は **非対応**。

**Colab で素のままでは動かない理由:** 重みの実体である `canopylabs/orpheus-3b-0.1-ft` が **Hugging Face の gated リポジトリ**になっています（Meta の `Llama-3.2-3B-Instruct` をベースにした fine-tune モデルのため、Llama 3.2 Community License 同意が必須）。トークン未設定だと vLLM がモデルロード時に下記で失敗します:

```
OSError: You are trying to access a gated repo.
Access to model canopylabs/orpheus-3b-0.1-ft is restricted. You must have access to it and be authenticated to access it.
```

**動かすにはセル実行前に以下をすべて済ませる必要があります:**

1. Hugging Face にログインして、**両方**のリポジトリでアクセス申請を行う（フォーム送信後ほぼ即承認）:
   - <https://huggingface.co/canopylabs/orpheus-3b-0.1-ft>
   - <https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct>
2. Meta のリポジトリで **Llama 3.2 Community License に同意**する。Orpheus の重みリポは Apache 2.0 表記ですが、ベースが Llama-3.2 のため、Meta の Acceptable Use Policy など Llama 側のライセンスも実質的に適用されます。
3. <https://huggingface.co/settings/tokens> でアクセストークンを発行し、Colab ランタイムの環境変数 `HF_TOKEN` として渡す。最も簡単なのは *Tools → Secrets → New secret* で key=`HF_TOKEN` の値にトークンを貼り付け、「Notebook access」を有効化する手順。ラッパーは `os.environ` 経由で取得し、エンジンサブプロセスに引き継ぎます。

ラッパーは `vllm==0.7.3` をピン（新しい 0.7.x で Orpheus のストリーミング生成を壊す regression あり）、Python 3.12 の venv を強制します（`xgrammar==0.1.11` は cp313 wheel 未提供のため）。GPU 必須、L4 / A100 推奨（3B 重み + vLLM KV キャッシュで VRAM ~10–12GB）。

**ライセンス（アクセス取得後）:** コードは Apache 2.0。重みリポジトリも Apache 2.0 表記ですが、`meta-llama/Llama-3.2-3B-Instruct` から fine-tune されているため、**Llama 3.2 Community License も実質的に適用されます**（OuteTTS 1B と同じ状況）。商用利用前に両方の規約を必ず確認してください。

### OpenVoice-V2 (現在動作不可)

[myshell-ai/OpenVoice](https://github.com/myshell-ai/OpenVoice) V2 を使う構成で、2 段階の voice cloning TTS（MeloTTS でベース合成 → ToneColorConverter で声色変換）として実装しています。コード・重みともに MIT で商用利用可。

**Colab で動かない理由**: OpenVoice の `pyproject.toml` が `faster-whisper==0.9.0` をハードピンしており、その推移依存で `av>=10.dev0,<11.dev0` も固定されます。`av` の 10.x 系列は Python 3.13（現在の Colab の既定）向けに wheel が無く、Cython 3.x ではソースのビルドにも失敗します:

```
av/logging.pyx:216:22: Cannot assign type 'const char *(void *) except?
NULL nogil' to 'const char *(*)(void *) noexcept nogil'.
```

`faster-whisper>=1.0`（`av==17.x` で py3.13 wheel あり）を先に入れても、uv が OpenVoice 側のピンを優先して 0.9.0 に巻き戻すため改善しません。回避するには `--no-deps` で入れて OpenVoice + MeloTTS の依存ツリーを手動で再構成する必要があり、その過程で本リポの MeloTTS 単体エンジンも壊している Rust ツールチェーン問題まで巻き込みます。

upstream がピンを緩めた段階で再アクティベートできるよう、ラッパー側のコードはそのまま残しています。**ライセンス（仮に動いた場合）:** コード・重みとも MIT（2024-04 以降）。

### VibeVoice (現在動作不可)

[microsoft/VibeVoice](https://github.com/microsoft/VibeVoice) を使う構成で、1.5B パラメータの長尺・マルチスピーカー TTS（最大 4 話者・約 90 分を 1 パスで生成）として実装しています。Colab L4 GPU 上でモデルロードまでは到達しますが、upstream Microsoft リポジトリが現在 **破壊的な API 移行中** で、合成リクエストが完了しません:

- 推論クラスが `VibeVoiceForConditionalGenerationInference` → `VibeVoiceForConditionalGeneration` にリネーム（ラッパーで吸収済み）
- `model.set_ddpm_inference_steps(...)` が削除され、DDPM ステップは `model.model.noise_scheduler.set_timesteps(...)` 経由で設定する形に変更（ラッパーで吸収済み）
- 致命的なのが reference 配布形式の変更: upstream は **`.wav` 参照音声ファイルを `demo/voices/` から取り下げ**、代わりに事前計算済みの **`.pt` (prompt cache)** を `demo/voices/streaming_model/` で配布する形になりました（例: `en-Carter_man.pt` / `jp-Spk1_woman.pt`）。推奨パスも `processor.process_input_with_cached_prompt(cached_prompt=torch.load(...))` に変わっており、本ラッパーが使っている `processor(text=..., voice_samples=[wav_path])` API では使い物になる参照音声が無くなっています。

upstream の API が落ち着いた段階で再アクティベートできるよう、ラッパー側の実装はそのままツリーに残しています。**ライセンス注意（仮に動いた場合でも）:** モデルカードに **「research purpose only」** と明記されており、英語・中国語以外の言語、なりすまし、ディスインフォメーション、リアルタイム音声変換などは禁止です。動くようになっても商用 / 実運用には使わないでください。

### Fish-Speech (現在動作不可)

[fishaudio/fish-speech](https://github.com/fishaudio/fish-speech) を使った高品質 TTS です。日本語は Tier 1 サポート（最高品質）で、80 言語以上に対応しています。VRAM 24GB 以上が必要で A100/L4 GPU を想定していますが、Colab 環境ではモデルロード時に OOM（メモリ不足）でランタイムがクラッシュするため、現時点では動作しません。ライセンス: Apache 2.0。

### CosyVoice2

[FunAudioLLM/CosyVoice](https://github.com/FunAudioLLM/CosyVoice) を使った Alibaba FunAudioLLM の多言語ゼロショット voice cloning TTS です。0.5B パラメータの v2 チェックポイント（`FunAudioLLM/CosyVoice2-0.5B`）は **日本語**・英語・中国語・韓国語・独語・西語・仏語・伊語・露語の 9 言語に加えて中国方言 18 種類以上をサポートし、cross-lingual のゼロショットクローンが可能です。本ラッパーは上流の pin（`torch==2.3.1`、`openai-whisper==20231117`、`onnxruntime-gpu==1.18.0` 等）が Colab デフォルトの Python 3.12 で解決しないため、**Python 3.10 の venv を強制**します（`uv venv --python 3.10`）。`Matcha-TTS` サブモジュールが必要なので `--recursive` で clone します。GPU 推奨（VRAM ~4GB）。

`voice` パラメータ:

| voice | 説明 |
|---|---|
| `default` | 上流同梱の `asset/zero_shot_prompt.wav`（中国語女性）を参照音声として `inference_cross_lingual` を呼びます。入力言語と参照言語が違っても動作します。 |
| `clone` | `--cosyvoice-prompt-wav` で参照音声を指定したときに有効。`--cosyvoice-prompt-text` も併記すると `inference_zero_shot`（書き起こし一致のときに高品質）、未指定なら `inference_cross_lingual` にフォールバックします。 |

voice cloning では、必ず権利を持つ参照音声（話者本人の同意）のみを使用してください。

ライセンス: コード（CosyVoice リポジトリ）も重み（`CosyVoice2-0.5B`、HF モデルカード明記）も Apache 2.0。

### Bark

[suno-ai/bark](https://github.com/suno-ai/bark) を使った Suno の生成的 text-to-audio モデルです。13 言語対応（英 / 独 / 西 / 仏 / ヒンディー / 伊 / **日本語** / 韓 / ポーランド / 葡 / 露 / トルコ / 簡体中）で、笑い声・ため息などのノンバーバル音や簡単な効果音も生成できます。voice プリセットは upstream の Speaker Library 名 `v2/<lang>_speaker_<n>`（言語ごとに 10 話者）です。

フル版は VRAM ~12GB、`BARK_USE_SMALL_MODELS=True`（または `--bark-use-small-models`）を指定すると ~8GB に収まります。生成プロセスはランダム性があり、同じ入力でも結果が変わります。

ライセンス: コードと重みとも MIT（商用 OK）。著者は研究目的での提供を明記しており、悪用の可能性を認識しているため責任ある利用を推奨しています。

### ChatTTS

[2noise/ChatTTS](https://github.com/2noise/ChatTTS) を使った 2noise の対話特化 TTS です。日常会話用に設計され、笑い声 / ためらい / ポーズなどを表現します。英語 / 中国語のみ対応。

`voice` パラメータ:

| voice | 説明 |
|---|---|
| `default` | `--chattts-seed`（デフォルト 2）から再現可能な話者を生成。同じ seed なら同じ話者になります。 |
| `random` | リクエストのたびに `chat.sample_random_speaker()` でランダム話者を生成。 |

**ライセンス警告（重要）:** コードは **AGPL-3.0+**、**重みは CC BY-NC 4.0** で、本エンジンは **研究 / 教育目的のみ — 商用利用は不可** です。重みには乱用防止のため学習時に意図的な高周波ノイズが加えられており、音質はやや劣化します。

### CSM-1B (Sesame Conversational Speech Model)

[SesameAILabs/csm](https://github.com/SesameAILabs/csm) を使った Sesame の対話特化 TTS です。アーキテクチャは Llama-3.2-1B backbone + Mimi codec を出力する音声デコーダ（Kyutai-TTS と同じ codec 系統）。英語のみ対応。

本ラッパーは上流の pin（`torch==2.4.0`、`torchtune==0.4.0`、`torchao==0.9.0`）に合わせて **Python 3.11 の venv** を強制し、上流 README に従って `NO_TORCH_COMPILE=1` を設定します。GPU 推奨（VRAM ~6GB）。

**HF gated 重み** — モデルカードで条件への同意が必要な上、Llama-3.2-1B ベースモデルも Llama 3.2 Community License で gated です。Colab で利用する場合:

1. `https://huggingface.co/sesame/csm-1b` で条件に同意。
2. `https://huggingface.co/meta-llama/Llama-3.2-1B` で Llama 3.2 Community License に同意。
3. Colab Secrets で `HF_TOKEN` を設定（notebook access 有効化）。

`voice` パラメータ:

| voice | 説明 |
|---|---|
| `default` | `--csm-default-speaker`（デフォルト 0）の speaker_id で合成。 |
| `speaker_<int>` | speaker_id を直接指定（例: `speaker_1`）。 |

ライセンス: コードと CSM-1B の重みは Apache 2.0、Llama-3.2-1B ベースは Llama 3.2 Community License。

### MisoTTS

[MisoLabsAI/MisoTTS](https://github.com/MisoLabsAI/MisoTTS) を使った Miso Labs の対話特化 8B TTS で、Hugging Face の `MisoLabs/MisoTTS` 重みを利用します。**Sesame CSM のフォーク**であり、Llama 系の 8B backbone + ~300M の音声デコーダが 24kHz の Mimi codec 音声を出力します。そのため API は CSM と同型です（`load_miso_8b` / `Segment` / `generate(text, speaker, context, max_audio_length_ms, temperature, topk)`）。

本ラッパーは CSM と同じ上流 pin（`torch==2.4.0`、`torchtune==0.4.0`、`torchao==0.9.0`、加えて `moshi` / `silentcipher`）に合わせて **Python 3.11 の venv** を強制し、上流 run スクリプトに従って `NO_TORCH_COMPILE=1` を設定します。**Colab A100 必須** — チェックポイントはディスク上 ~32GB（F32 の `model.safetensors`）で、bf16（~16GB）としてロードされ、Mimi codec と活性化を合わせると T4/L4 の VRAM を超えます（小型 Colab GPU では OOM の想定）。

**トークナイザ（既定で HF gating 不要）** — `generator.py` はテキストトークナイザを `AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")` とハードコードしており、これは HF gated です（そのままだと初回リクエストで `OSError: gated repo ... 401`）。本ラッパーはこの読み込みを **バイト同一のトークナイザファイルを持つ ungated ミラー**にリダイレクトします（`MISOTTS_TOKENIZER_REPO`、既定は [`unsloth/Llama-3.2-1B`](https://huggingface.co/unsloth/Llama-3.2-1B) — 128k の Llama 3 語彙・特殊トークン ID とも同一なのでトークン ID は一致し、出力は変わりません）。そのため **`HF_TOKEN` もライセンス同意（クリック）も不要**で動きます。MisoTTS の重み自体も公開です。公式ソースを使いたい場合は、`https://huggingface.co/meta-llama/Llama-3.2-1B` で Llama 3.2 Community License に同意し `HF_TOKEN` を Colab Secrets に設定した上で `--misotts-tokenizer-repo meta-llama/Llama-3.2-1B` を指定してください。

`voice` パラメータ:

| voice | 説明 |
|---|---|
| `default` | `--misotts-default-speaker`（デフォルト 0）の speaker_id で合成。 |
| `speaker_<int>` | speaker_id を直接指定（例: `speaker_1`）。 |
| `clone` | ゼロショット音声クローン。`--misotts-prompt-wav`（任意で参照書き起こし `--misotts-prompt-text`）が必須。未指定の場合は 400 を返します。 |

Sesame CSM 同様、`generate()` は出力に **不可聴の SilentCipher ウォーターマーク**（`MISO_TTS_WATERMARK`）を埋め込み、AI 生成であることを示します。このウォーターマークは除去禁止です。

ライセンス: MisoTTS のコード / 重みとも **Modified MIT**（標準 MIT に 1 条項追加 — MAU 5,000万超 または 月商 $1,000万超の製品は UI に "Miso Labs" を明示する義務）。それ以外は商用利用可。なお実行時に読み込む Llama 3.2 トークナイザ（ungated な `unsloth/Llama-3.2-1B` ミラー、または公式の `meta-llama/Llama-3.2-1B`）は **Llama 3.2 Community License** の対象です。ungated ミラーで回避できるのは HF のアクセス申請ゲートのみで、ライセンス自体は引き続き実効スタックに適用されます。

### StyleTTS2

[yl4579/StyleTTS2](https://github.com/yl4579/StyleTTS2) の拡散 + SLM 敵対学習による高品質 TTS です。本ラッパーは [sidharthrajaram/StyleTTS2](https://github.com/sidharthrajaram/StyleTTS2) の pip パッケージを使用し、上流の **phonemizer (GPL-3.0)** ではなく **gruut (MIT)** に置き換えることで GPL の伝播を回避しています。

英語のみ対応（gruut は英語中心）。sidharthrajaram 版の legacy pin を分離するため **Python 3.11 venv** を使用します。

`voice` パラメータ:

| voice | 説明 |
|---|---|
| `default` | 拡散からランダムスタイルをサンプリング（参照音声なし）。 |
| `clone` | 音声クローン。`--styletts2-prompt-wav` を指定したときのみ有効。 |

`--styletts2-alpha` は参照音声への音色類似度（0=完全に参照、1=完全にサンプリング）、`--styletts2-beta` は韻律類似度を制御します。`--styletts2-diffusion-steps` と `--styletts2-embedding-scale` は変動性と感情強度の上流ノブです。

**ライセンス警告:** コード（upstream / pip 版とも）は MIT ですが、LibriTTS 重み（`yl4579/StyleTTS2-LibriTTS`）は **Custom ライセンスで合成であることの開示が要求されます**。voice cloning には話者の明示的な同意が必要です。

### MaskGCT

[open-mmlab/Amphion](https://github.com/open-mmlab/Amphion/tree/main/models/tts/maskgct) の Masked Generative Codec Transformer TTS です（Amphion）。完全な non-autoregressive で明示的なアラインメントが不要。ゼロショット voice cloning で、毎リクエストに参照音声が必要です。英 / 中（`prompt_lang` / `target_lang` で他言語も）。

本ラッパーは Amphion の `models/tts/maskgct` / `models/codec` / `utils` のみ sparse-checkout し（フルリポは巨大）、上流 pin（`torch==2.0.1` / `transformers==4.41.2` / `numpy==1.26.0`）に合わせて **Python 3.10 venv** を強制します。`espeak-ng` システムパッケージが必要で、CosyVoice2 と同じ `setuptools<70` プリインストール対応（依存推移の `openai-whisper==20231117` の legacy setup.py が `pkg_resources` を要求）が必要です。

`voice` パラメータ:

| voice | 説明 |
|---|---|
| `default` | 上流同梱の `models/tts/maskgct/wav/prompt.wav`（英語女性）を参照音声として使用。 |
| `clone` | 音声クローン。`--maskgct-prompt-wav` と `--maskgct-prompt-text` の両方を指定したときのみ有効。 |

GPU 必須（VRAM ~10-12GB）。

**ライセンス警告:** コードは MIT ですが、重み（`amphion/MaskGCT`）は **CC BY-NC 4.0** で **商用利用は不可** です。

### GPT-SoVITS

[RVC-Boss/GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) の few-shot voice cloning TTS です。5秒の参照音声でゼロショット推論可能、1分でファインチューニングも可能。**中国語 / 英語 / 日本語 / 韓国語 / 広東語** に対応。複数のモデルバージョン（v1, v2, v2Pro, v2ProPlus, v3, v4）を `--gpt-sovits-version`（デフォルト `v2`）で切り替え可能。

GPT-SoVITS は本質的に few-shot cloning モデルで、内蔵の「default speaker」モードはありません。本ラッパーは起動時に `--gpt-sovits-prompt-wav` と `--gpt-sovits-prompt-text` を要求し、未指定の場合はリクエストごとに 400 を返します。

ラッパーは **Python 3.11 venv** を使用します（上流が Python 3.10 / 3.11 対応を明記）。初回起動時は `lj1995/GPT-SoVITS` から v2 重み（`gsv-v2final-pretrained/` ~1.2GB）と `chinese-hubert-base/`、`chinese-roberta-wwm-ext-large/` のみを selective に snapshot_download し、5.3GB 全体は取得しません。GPU 推奨（VRAM ~4-6GB）。

`voice` パラメータ:

| voice | 説明 |
|---|---|
| `default` / `clone` | どちらも設定済みの prompt 音声 + 書き起こしを使用（GPT-SoVITS には plain-TTS のフォールバックがありません）。 |

ライセンス: コードは MIT、重み（`lj1995/GPT-SoVITS`）も MIT で商用 OK。

### Higgs-Audio-v2

[boson-ai/higgs-audio](https://github.com/boson-ai/higgs-audio) の Boson AI による LLM ベース音声基盤モデルです。3B-base 生成モデルと別モジュールの audio tokenizer（`bosonai/higgs-audio-v2-tokenizer`）の組み合わせ。voice cloning は同梱プリセット（`belinda`、`broom_salesman` 等が `examples/voice_prompts/` 以下）または任意の音声パスで動作します。

**ハードウェア**: 3B モデルで VRAM ~24GB が必要なため、**A100 / L4（Colab Pro）必須** です。T4（無料枠）ではモデルのホストができません。

ラッパーは **Python 3.10 venv**（上流の NVIDIA コンテナ仕様に対応）を使用し、`pip install -e .` で `boson_multimodal` パッケージを公開します。デフォルト参照は `belinda`、`--higgs-default-ref-voice <name>` で `examples/voice_prompts/` 以下の任意プリセットに切り替え可能。

`voice` パラメータ:

| voice | 説明 |
|---|---|
| `default` | プリセット `--higgs-default-ref-voice`（デフォルト `belinda`）を使用。 |
| `clone` | 音声クローン。`--higgs-prompt-wav` と `--higgs-prompt-text` の両方を指定したときのみ有効。 |
| `<preset_name>` | `examples/voice_prompts/` 以下の任意プリセット名（例: `broom_salesman`）。 |

**ライセンス警告（重要）:** コードは Apache-2.0 ですが、重み（`bosonai/higgs-audio-v2-generation-3B-base`）は **Boson Higgs Audio 2 Community License**（Meta Llama 3 Community License の派生）です。制約:

- 商用利用は可能ですが、月間アクティブユーザーが 10 万人を超える場合は Boson AI から拡張ライセンスを取得する必要があります。
- 出力を他の大規模言語モデルの学習に使用することは禁止されています。
- 再配布には attribution が必要です。
- Meta の Acceptable Use Policy への準拠が必要です。

**ステータス（現時点でデフォルト動作不可）:** HF 上の checkpoint は `boson-ai/higgs-audio` の未リリースブランチと transformers 5.x を要求しますが、PyPI の `boson_multimodal` は transformers 4.46.x ベースです。エンジンの組み込み（インストール / venv / app / voice list / cloudflared）は正しく動作し、Colab L4 で `/`、`/v1/models`、`/v1/voices` はすべて 200 を返しますが、`/v1/audio/speech` は audio tokenizer のロード（`load_higgs_audio_tokenizer`）で 500 になります。具体的には HF の flat config キー（`acoustic_model_config`、`semantic_model_config`）を `HiggsAudioTokenizer.__init__` がそのまま受け付けないためです。前段の 2 つの不一致（`text_config` のデフォルトが `padding_idx=128001 / num_embeddings=32000` を引き起こす問題、および未リリース transformers 5.x にしか存在しない `tokenizer_class=TokenizersBackend` 参照）はラッパー側で workaround 済みですが、audio tokenizer の schema drift は `boson-ai/higgs-audio` 上流の修正が必要です。上流が公開済み config に合わせてコードを更新すれば、本ラッパーはそのまま動作するはずです。

### Higgs-Audio-v3

Boson AI による別系統の新しいモデルで、表現力豊かな会話音声向けの 4B chat-native TTS（Qwen3-4B backbone、`HiggsMultimodalQwen3ForConditionalGeneration`）です。100+ 言語に対応し、ゼロショット voice cloning が可能です。上記の v2 とは**無関係**で、`boson-ai/higgs-audio` の GitHub リポジトリは使いません（同リポは v2 専用で「Higgs Audio v3 is a standalone release」と明記）。v3 は **SGLang-Omni**（[sgl-project/sglang-omni](https://github.com/sgl-project/sglang-omni)）が配信し、OpenAI 互換の `/v1/audio/speech` をネイティブに公開します。本ラッパーはそのバックエンドを `--higgs-v3-backend-port`（既定 5002）で起動し、プロキシします。

**重みは gated ではありません**（`bosonai/higgs-audio-v3-tts-4b`、約 9.3GB）。`HF_TOKEN`・ログイン・ライセンス同意クリックのいずれもなしでダウンロードできます。

**ハードウェア**: **Colab L4（24GB）で動作確認済み**。ロード時に約 19.9GB を使用し L4 の上限に近いため、余裕を見て **A100 推奨**。**T4（無料枠）は非対応**です（`sgl-kernel` / flash-attn が compute capability sm_80+ を要求、T4 は sm_75）。初回起動は遅く（~10-12 分）、モデル DL・重みロード・torch.compile / CUDA グラフキャプチャに時間がかかります。

ラッパーは **Python 3.12 venv で sglang-omni をソースからビルド**します（コミット固定）。インストールには素の `pip` ではなく **`uv` が必須**です。`descript-audiotools` と `grpcio` の間の protobuf バージョン衝突は、sglang-omni の `pyproject.toml` にある `[tool.uv] override-dependencies` でのみ解決できます。隔離 venv で動かすことで、Colab プリインストールの TensorFlow（同梱 protobuf が descriptor を二重登録しバックエンドを起動時にクラッシュさせる）も回避します。

`voice` パラメータ:

| voice | 説明 |
|---|---|
| `default` | 参照音声なしの内蔵スピーカー。 |
| `clone` | ゼロショット voice cloning。`--higgs-v3-prompt-wav`（任意で `--higgs-v3-prompt-text`）を指定した場合のみ有効。未指定なら 400 を返します。 |

**ライセンス警告（重要）:** GitHub コード（SGLang-Omni）は Apache-2.0 ですが、**重みは Boson Higgs Audio v3 Research and Non-Commercial License** です（[LICENSE](https://huggingface.co/bosonai/higgs-audio-v3-tts-4b/blob/main/LICENSE)）。

- **許可**: 研究、個人 / 趣味利用、短期の評価・テスト。本エンジンを Colab で自分の検証のために動かすのはこの範囲に該当します。
- **別途商用ライセンスが必要（禁止）**: hosted use（API・SaaS・プラグイン・組織外のエンドユーザーに提供する用途）、production 運用、収益化。
- 同意のない voice cloning・なりすまし・違法利用は禁止。
- 再配布時はライセンスと帰属表示の同梱が必要。

本リポジトリは重みをダウンロードするだけ（ダウンロード時にユーザーが Boson のライセンスに同意）で、重みの再配布は行いません。

### dots.tts

rednote-hilab による 2B パラメータの完全連続・end-to-end 自己回帰 TTS です（[rednote-hilab/dots.tts](https://github.com/rednote-hilab/dots.tts)）。バックボーンは semantic encoder と **Qwen2.5-1.5B-Base** LLM（BPE テキストを音素化せず直接入力）、48 kHz AudioVAE 上で動く自己回帰 flow-matching（DiT）音響ヘッドの組み合わせで、パイプライン中に離散コーデックトークンが一切ありません。Seed-TTS-Eval でオープンソース SOTA、24 言語の MiniMax 多言語ベンチで平均話者類似度トップを報告しています。

ラッパーは upstream の GitHub リポジトリを clone し、**Python 3.12 venv** を作成して `uv pip install -e . -c constraints/recommended.txt`（`torch==2.8.0` を固定）を実行します。通常厄介な依存である `WeTextProcessing` → `pynini==2.1.6` は `cp312` の prebuilt manylinux wheel で入るため、OpenFst をソースビルドしません。モデルは**インプロセス**でロードされ（別バックエンド不要）、初回リクエスト時に `rednote-hilab/dots.tts-base` から約 9.5GB をダウンロードします（ungated、`HF_TOKEN` 不要）。

dots.tts は本質的に**ゼロショット voice cloning** モデルで、公開チェックポイントには組み込みのデフォルト話者がありません。`voice` パラメータは以下を公開します:

| voice | 説明 |
|---|---|
| `default` | 参照なし＝ランダム話者サンプリング。安定した単一話者は fine-tune 済みチェックポイントでのみ意味を持ち、base/SOAR/MF では声色が毎回ランダムになります。 |
| `clone` | `--dots-tts-prompt-wav` からゼロショット cloning。`--dots-tts-prompt-text` も指定すると **continuation cloning**（参照音声＋書き起こし、推奨）、未指定なら **x-vector のみの cloning**（話者埋め込みから音色を推定）。参照 wav 未設定なら 4xx を返します。 |

`--dots-tts-language` はモデル側の言語タグ（`none` / `auto_detect` / `EN`・`JA` 等のコード）を制御します。既定の `auto_detect` は入力テキストから推定し、日英混在でも良好に動作します。チューニング: `--dots-tts-num-steps`（flow-matching ステップ数、既定 10）、`--dots-tts-guidance-scale`（CFG、既定 1.2）、`--dots-tts-speaker-scale`（既定 1.5）。出力は 48 kHz。

`--dots-tts-hf-model` で 3 つのチェックポイントを切り替えられます（いずれも 2B / Apache-2.0）:

| repo | バリアント | 備考 |
|---|---|---|
| `rednote-hilab/dots.tts-base` | Pretrain | 既定 |
| `rednote-hilab/dots.tts-soar` | Self-Corrective Alignment | 話者類似度・安定性が高い |
| `rednote-hilab/dots.tts-mf` | MeanFlow 蒸留 | NFE=4 で最速（CFG が student に融合済み） |

**ライセンス**: コード・重みともに **Apache-2.0**（商用 OK）。LLM backbone は Qwen2.5-1.5B-Base 由来。上流のモデルカードは、高品質なゼロショット cloning をなりすまし・詐欺・偽情報に使わないこと、AI 生成音声であることを明示することを求めています。

### LFM2.5-Audio-JP

Liquid AI の [LFM2.5-Audio](https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B-JP) の日本語特化版です。専用 TTS ではなく **end-to-end のマルチモーダル音声・テキストモデル**で、LFM2.5-1.2B backbone + FastConformer audio encoder + RQ-transformer（離散音声トークン生成）+ 軽量な LFM2 ベース audio detokenizer（Mimi codec、8 codebooks、24 kHz）で構成されます。音声対話・ASR・TTS が可能で、本ラッパーは **TTS パス**のみを使います。

ラッパーは [`liquid-audio`](https://github.com/Liquid4All/liquid-audio) ライブラリを **Python 3.12 venv** に導入し（`pip install liquid-audio`、`torch>=2.8` を固定）、モデルを**インプロセス**でロードします（`LFM2AudioModel` / `LFM2AudioProcessor`、`device="cuda"`、bf16）。`flash-attn` は**任意**で、未導入時は torch SDPA に自動フォールバックするため導入しません。初回リクエスト時に `LiquidAI/LFM2.5-Audio-1.5B-JP` から重みをダウンロードします（ungated、`HF_TOKEN` 不要）。

TTS は **sequential generation** を使います。system prompt（`Perform TTS in japanese.`、`--lfm2-audio-jp-system-prompt` で変更可）がタスクを選択し、user ターンにテキストを渡し、assistant ターンの audio フレーム（`numel == 8`、Mimi codebook ごとに 1 要素）を集めて 24 kHz 波形に detokenize します。終端の end-of-audio フレーム（コード `== 2048`）は decode 前に除去します（detokenizer は `[0, 2047]` のコードのみ受け付けるため）。

`voice` パラメータは `default`（内蔵の日本語ボイス1種）のみです。**参照音声 / voice cloning は非対応**です（英語 base モデル `LiquidAI/LFM2.5-Audio-1.5B` は US/UK の named voice を4種持ちますが、本 JP モデルは日本語1ボイス）。チューニング: `--lfm2-audio-jp-max-new-tokens`（既定 1024。text+audio トークン合計をカウント、audio は ~12.5 frames/sec）、`--lfm2-audio-jp-audio-temperature`（0.8）、`--lfm2-audio-jp-audio-top-k`（64）。

**ライセンス**: コード・重みは **LFM Open License v1.0** です。年商 **USD 10M 未満**の組織は商用利用可、超過する場合は Liquid AI の別途商用ライセンスが必要です（適格な非営利団体は研究目的で免除）。同梱の **audio encoder は Apache-2.0**（NVIDIA NeMo 由来）、**audio codec（Mimi）は CC-BY-4.0**（Kyutai）です。ライセンスは規約違反時に終了し、特許訴訟による終了条項も含みます。

### Ming-omni-TTS

inclusionAI の [Ming-omni-tts](https://github.com/inclusionAI/Ming-omni-tts) の **16.8B-A3B** チェックポイントです。Mixture-of-Experts の音声言語モデル（active ~3B パラメータ）で、独自の 12.5 Hz 連続トークナイザ + DiT（diffusion-transformer）音響ヘッド（44.1 kHz codec）で構成されます。zero-shot voice cloning、自然言語による voice design、感情・方言・話速の制御、さらに BGM 付き音声生成や TTA まで対応し、中国語（広東語含む）と英語に強みがあります。デフォルトモデル: `inclusionAI/Ming-omni-tts-16.8B-A3B`。

**ハードウェア**: 16.8B チェックポイントは bf16 で **~34GB** あるため、**Google Colab A100（40GB）必須**です。L4（24GB）には載りません。（上流には軽量な `inclusionAI/Ming-omni-tts-0.5B` もあり `--ming-omni-tts-hf-model` で選択できますが、本ラッパーは 16.8B-A3B をデフォルト・検証対象とします。）

ラッパーは上流の GitHub リポジトリを clone し、**Python 3.10 venv** を `torch==2.6.0`（上流 `requirements.txt` に準拠）で構築します。MoE backbone には **`grouped_gemm`** が必要で、これは導入時にその torch に対してソースからビルドされます。**FlashAttention** はソースビルドせず、`cu12torch2.6 cp310` のビルド済み wheel を導入します（上流はコメントアウトしています）。`campplus.onnx` の話者埋め込み抽出のため `onnxruntime` も追加します。モデルは**インプロセス**でロードし（別バックエンドなし）、MoE トークナイザ（`tokenization_bailing.py`）とモデリングコード（`modeling_bailingmm.py`）がリポジトリ直下にあるため、uvicorn の cwd をリポジトリルートに設定して起動します。初回リクエスト時に `inclusionAI/Ming-omni-tts-16.8B-A3B` から ~34GB をダウンロードします（ungated、`HF_TOKEN` 不要）。

`voice` パラメータ:

| voice | 挙動 |
| --- | --- |
| `default` | 内蔵ボイス（zero speaker-embedding、参照音声なし）。 |
| `clone` | `--ming-omni-tts-prompt-wav`（任意で `--ming-omni-tts-prompt-text` に参照音声の書き起こし）からの zero-shot cloning。prompt wav 未設定なら 4xx を返します。 |

チューニング: `--ming-omni-tts-max-decode-steps`（既定 200）、`--ming-omni-tts-cfg`（guidance scale、既定 2.0）、`--ming-omni-tts-sigma`（既定 0.25）、`--ming-omni-tts-temperature`（既定 0.0）。テキスト正規化はスキップします（上流は MoE チェックポイントでは非対応と明記）。出力は 44.1 kHz です。

**プロンプト駆動の制御。** Ming はマルチタスクモデルなので、ラッパーは prompt / instruction 制御も公開します（すべて任意。空のままなら従来の `default`/`clone` 挙動）:

| 制御 | フラグ | 効果 |
| --- | --- | --- |
| タスク | `--ming-omni-tts-task` | `speech`（既定）/ `music` / `tta`（効果音）。`music`/`tta` では `input` テキストは**説明文**になり、decode 値が上流推奨（cfg 4.5 / sigma 0.3 / temperature 2.5）に切り替わります。 |
| スタイル | `--ming-omni-tts-style` | 自然言語による声デザイン → instruction キー `风格`。例: `温柔自然的年轻女性声音`（優しい女性の声）。参照音声なしで男性/女性/ASMR 等を指定する手段。 |
| 感情 | `--ming-omni-tts-emotion` | instruction キー `情感`（例: `高兴`）。 |
| 方言 | `--ming-omni-tts-dialect` | instruction キー `方言`（例: 広東語 `广粤话`）。 |

instruction キーは中国語で、説明文も**中国語で書くのが最も安定**します。これらは `/v1/audio/speech` の body の `task` / `style` / `emotion` / `dialect` フィールドでリクエスト単位の上書きも可能（`null`/未指定なら起動時の既定値にフォールバック）なので、`input`/`voice` だけ送る標準的な OpenAI クライアントには影響しません。

**ライセンス**: コードは **MIT**（[GitHub リポジトリ](https://github.com/inclusionAI/Ming-omni-tts)）、重みは **Apache-2.0**（[HF モデルカード](https://huggingface.co/inclusionAI/Ming-omni-tts-16.8B-A3B)）。いずれも商用利用可です。高品質な zero-shot cloning が可能なモデルである以上、なりすまし・詐欺・偽情報への利用は上流の規約で禁止されています。

### DramaBox

[resemble-ai/DramaBox](https://github.com/resemble-ai/DramaBox) の Resemble AI による「ディレクション可能」な表現力豊か TTS です。Lightricks の LTX-2.3（音声専用 branch）を IC-LoRA で fine-tune し、テキストエンコーダに Gemma 3 12B を採用。英語プロンプトの中で感情・ペーシング・笑い声・ため息などのパラ言語的要素を直接ディレクション可能。ボイスクローンは 10 秒以上の参照音声で動作します。

**ハードウェア**: VRAM ~24GB ピークが必要なため、**Google Colab A100（40GB）必須** です。T4 / V100 ではモデルのホスト不可。

ラッパーは上流の GitHub repo を clone し、`requirements.txt`（`torch==2.8.0`、`pydantic==2.10.6`、`bitsandbytes`、`resemble-perth` など）をインストール、FastAPI app から `<repo>/src` と `<repo>/ltx2` を sys.path に挿入してモジュール解決します。初回起動時に `ResembleAI/Dramabox` から約 8.5GB、加えて `unsloth/gemma-3-12b-it-bnb-4bit` の snapshot をダウンロードします。

プロンプトはディレクター指示風の英語が推奨です。例:

```
A woman speaks warmly, "Hello, how are you today?" She laughs, "Hahaha, it is so good to see you!"
```

`voice` パラメータ:

| voice | 説明 |
|---|---|
| `default` | プリセット `--dramabox-default-ref-voice`（デフォルト `female_american`）を使用。 |
| `clone` | 音声クローン。`--dramabox-prompt-wav` を指定したときのみ有効（10秒以上推奨）。 |
| `<preset_name>` | `DramaBox/assets/voices/` 同梱プリセット: `female_american`、`female_shadowheart`、`male_arnie`、`male_conan`、`male_harvey_keitel`、`male_old_movie`、`male_petergriffin`、`male_samuel_j`。 |

**ウォーターマーク**: 生成された音声には常に Resemble の **Perth** 不可聴ウォーターマーカー（`perth.PerthImplicitWatermarker`）が埋め込まれます。これは上流の設計通り保持しています（除去禁止）。人間の聴覚では判別できませんが、Resemble のツールで AI 生成音声として検出できます。

**ライセンス警告（重要）:** コード・重みとも **LTX-2 Community License Agreement**（Lightricks）で配布されており、Apache 2.0 / MIT / Llama Community License よりも実質的に厳しいです:

- 非商用は自由、年商 1,000 万米ドル未満の組織も商用可ですが、それを超える組織は Lightricks から有償ライセンスの取得が必要。
- **非競合条項**: モデルや派生物を使って競合モデルを学習させたり、Lightricks の提供と直接競合する製品を構築することは禁止。
- 派生物を再配布する場合、同じライセンス（使用制限・acceptable use policy 含む）を継承する必要あり。
- Acceptable use 制限は広範: 同意のないディープフェイク・なりすまし禁止、AI 生成であることを開示しない使用禁止、誤情報禁止、医療助言禁止、自動法的判断禁止、軍事 / 武器 / マルウェア用途禁止 など。

本リポジトリは Colab での個人・研究用途の評価ラッパーです。LTX-2 Community License が自身の用途に適合するかは利用者の責任で確認してください。

### Scenema

Scenema AI による zero-shot な表現力豊か / 演技指向 TTS です（[ScenemaAI/scenema-audio](https://github.com/ScenemaAI/scenema-audio)）。音声 diffusion transformer は Lightricks の LTX-2（22B 音声 + 映像モデル）から抽出した派生で、テキスト条件付けに Gemma 3 12B IT、声色変換に SeedVC、クリーン化に MelBandRoFormer を組み合わせています。最大の特徴は「演技」で、英語の自由記述で声色を描写すると、意図・ペーシング・息遣い・感情アークまで含めて発話します（参照話者が録音した感情でなくても再現可能）。

**ステータス: Colab 未検証**。インストール / 配線 / API レイヤ（installer・app・voice プリセット・XML pass-through）は一通り実装済みですが、Scenema のテキストエンコーダは **HF gated な Gemma 3 12B IT** です。`HF_TOKEN` の設定と Google の Gemma Terms of Use 同意は本リポのデフォルト検証ワークフローの範囲外のため、Colab A100 での E2E 検証は保留しています。実際に Colab で動かしたい場合は、以下のセットアップを利用者側で行ってください。動作確認できた方からの trycloudflare ログ付き Issue / PR は歓迎です。

**起動前のセットアップ:**

1. Hugging Face にサインインし、[google/gemma-3-12b-it](https://huggingface.co/google/gemma-3-12b-it) モデルカードの「Acknowledge license」で Gemma Terms of Use に同意。
2. https://huggingface.co/settings/tokens で **Read** スコープのトークンを発行。
3. Colab ノートブック左サイドバーの **🔑 Secrets** を開き、**名前 `HF_TOKEN`** で 2 のトークン値を保存、**ノートブックからのアクセスをオン**。

未設定で起動すると lifespan の `snapshot_download("google/gemma-3-12b-it")` が `401 Unauthorized` で失敗し、`AudioProcessor` が ready にならず `/v1/audio/speech` は 503 を返します。

**ハードウェア**: **Colab A100（40GB）必須**。本ラッパーは INT8 音声 transformer + NF4 Gemma プロファイル（常駐 VRAM ~13GB）を使用し、24GB GPU でも動作・40GB なら余裕。T4 / V100 ではモデルのホスト不可。初回起動時に合計約 38GB（audio transformer 約 5GB + pipeline 約 7GB + Gemma 3 12B 約 24GB + SeedVC 約 1.6GB + BigVGAN v2 + Whisper Small）をダウンロードします。

**OpenAI API へのマッピング**: 本リポジトリは Scenema を標準の `/v1/audio/speech` エンドポイントとして公開します。Scenema のネイティブ入力は `<action>` 演技指示を含む構造化 `<speak>` XML なので、入力方法は 2 通り対応します:

1. **プレーンテキスト**（デフォルト）: `input` は `<speak voice="<プリセット description>" gender="<gender>">…</speak>` で自動的にラップされます。`voice` パラメータはプリセット名から description を引き、プリセット名でなければ文字列をそのまま voice description として使用します。OpenAI SDK クライアントはこの形式で利用します。
2. **XML pass-through**（上級）: `input` が `<speak` で始まる場合、文字列はそのまま Scenema に渡されます。`<action>angry shout</action>` / `<sound>thunder</sound>` / 複数セグメントによる演技指示を活用できます。

`voice` パラメータ:

| voice | 説明 |
|---|---|
| `default` | 落ち着いた英国訛りの男性ナレーター（ベースライン中性ボイス）。 |
| `warm_male` | 温かみのある中年男性ナレーター。深いがやわらかいトーン。 |
| `smoky_female` | スモーキーで低音域の女性ボイス。親密で告白的なトーン。 |
| `child_girl` | 明るく息せき切った 6 歳の女の子。 |
| `elderly_male` | 渋い高齢男性の語り手。深いバリトン。 |
| `elderly_female` | 70 代女性の柔らかいアルト。 |
| `clone` | ゼロショット音声クローン。`--scenema-prompt-wav` 指定時のみ有効（10〜20 秒、感情変化のある参照音声推奨）。 |
| `<任意の文字列>` | そのまま Scenema の voice description として使用（例: `"Male, mid 60s. Deep baritone with gravel. Slight Southern American inflection."`）。 |

**ライセンス警告（重要）**: コードは MIT ですが、**音声 transformer の重みは LTX-2.3 派生のため LTX-2 Community License Agreement**（Lightricks）が継承されます — DramaBox と同じ制約の強いライセンスです。Gemma 3 12B IT は別途 Gemma Terms of Use（Google）の対象です。両方を自身の用途に対して遵守できるかは利用者の責任で確認してください。Acceptable use 制限は広範（なりすまし / ディープフェイク / 偽情報 / 軍事 / 医療助言など禁止）。

### MeloTTS (現在動作不可)

[myshell-ai/MeloTTS](https://github.com/myshell-ai/MeloTTS) を使う構成ですが、依存パッケージ `tokenizers` のビルドに Rust コンパイラが必要なため、現在の Colab 環境ではインストールに失敗します。

### Style-Bert-VITS2 (現在動作不可)

[litagin02/Style-Bert-VITS2](https://github.com/litagin02/Style-Bert-VITS2) を使う構成ですが、`setuptools` / `torch` / `scipy` の依存整合性が取れず、現在の Colab 環境では音声合成まで到達できません。

## ライセンス

各エンジンのライセンスは以下の通りです。利用時は各プロジェクトの最新のライセンス条件を必ず確認してください。

| エンジン | コード | モデル重み | 商用利用 | 備考 |
|---|---|---|---|---|
| Kokoro | Apache 2.0 | Apache 2.0 | OK | |
| Kokoro-ONNX | Apache 2.0 | Apache 2.0 | OK | NVIDIA による hexgrad/Kokoro-82M の ONNX 再配布。コード・重みとも Apache 2.0 |
| Irodori-TTS | MIT | MIT (v1 / v2 / v3) | OK | なりすまし・ディープフェイク生成を禁止する倫理規定あり。V3 は SilentCipher ウォーターマーク同梱（除去禁止） |
| Irodori-TTS-Lite | MIT | MIT (`kizuna-intelligence/Irodori-TTS-Lite-int4`, `kizuna-intelligence/Irodori-TTS-500M-v3-int4`) | OK | Irodori-TTS 用の int4 量子化ランタイム。Triton カーネル使用のため Linux + CUDA 必須。`fused_int4_linear.py` は OneCompression（Fujitsu Ltd., MIT）からベンダリング |
| Piper | GPL-3.0 | MIT | 要注意 | デフォルト音声 `en_US-lessac-medium` の学習データ（Blizzard 2013）は研究目的限定・商用利用不可 |
| Piper-Plus | MIT | MIT | OK | |
| Qwen3-TTS | Apache 2.0 | Apache 2.0 | OK | |
| VoxCPM2 | Apache 2.0 | Apache 2.0 | OK | |
| MOSS-TTS-Nano | Apache 2.0 | Apache 2.0 | OK | 100M パラメータ、CPU 動作可 |
| MOSS-TTS-v1.5 | Apache 2.0 | Apache 2.0 | OK | 8B パラメータ、**A100 必須**（ロード時 ~22GB + 音声トークナイザー。L4 22GB は不足）。31言語（日本語含む）。ゼロショット voice cloning |
| MOSS-TTS-Local-v1.5 | Apache 2.0 | Apache 2.0 | OK | ~4B MossTTSLocal パラメータ、L4 で動作確認（~12.4GB。8B 版は L4 で OOM）。48kHz ステレオ。31言語（日本語含む）。ゼロショット voice cloning |
| NeuTTS | Apache 2.0 | Apache 2.0 (Air) / NeuTTS Open License 1.0 (Nano) | OK (Air) / 規約要確認 (Nano) | ボイスクローン。英 / 西 / 独 / 仏 |
| TinyTTS | Apache 2.0 | Apache 2.0 | OK | |
| Supertonic | MIT | OpenRAIL-M | OK | 31言語（日 / 韓 / 英含む）。CPU 動作（ONNX）。なりすまし・ディープフェイク等の use-based ethical restrictions あり |
| Voxtral-TTS | — | CC BY-NC 4.0 | 不可 | vLLM + vllm-omni 経由。音声データセットのライセンス制約により非商用 |
| Sarashina-TTS | — | Sarashina Model NonCommercial License | 不可 | 日本語 / 英語。ゼロショット音声クローン対応。出力には SilentCipher のウォーターマークが付与される（除去禁止） |
| F5-TTS | MIT | CC-BY-NC | 不可（モデル） | モデル重みは Emilia データセットの制約により非商用 |
| Chatterbox | MIT | MIT | OK | 多言語（23言語、日本語含む）。ゼロショット voice cloning |
| Zonos | Apache 2.0 | Apache 2.0 | OK | 英 / 日 / 中 / 仏 / 独。ゼロショット voice cloning。`espeak-ng` 必須 |
| ZONOS2 | MIT | Apache 2.0 | OK | 41言語（tier-1 英/中/日）。ゼロショット voice cloning。Mini-SGLang バックエンド。GPU sm_80+（L4/A100） |
| OuteTTS (0.6B) | Apache 2.0 | Apache 2.0 | OK | 日本語含む多言語、CPU 動作可、voice cloning |
| OuteTTS (1B)   | Apache 2.0 | CC-BY-NC-SA-4.0 + Llama 3.2 Community License | 不可 | Llama-3.2 ベース。重みは非商用 |
| Dia | Apache 2.0 | Apache 2.0 | OK | 英語のみ。`[S1]`/`[S2]` でマルチスピーカー対話 TTS |
| Kyutai-TTS | MIT (Python) / Apache 2.0 (Rust) | CC-BY-4.0 | OK（要 attribution） | 英語 / 仏。DSM ベースのストリーミング TTS。GPU 推奨 |
| Pocket-TTS (model) | MIT | CC-BY-4.0 | OK（要 attribution） | 100M パラメータ、CPU のみ。英 / 仏 / 独 / 伊 / 葡 / 西 |
| Pocket-TTS (voices) | — | voice ごとに異なる | 各 voice で要確認 | voice ライセンスは [kyutai/tts-voices](https://huggingface.co/kyutai/tts-voices) を参照。上流規約により非合意のなりすまし禁止 |
| Orpheus-TTS | Apache 2.0 | Apache 2.0 + Llama 3.2 Community License | 要注意 | ベースが Llama-3.2-3B-Instruct のため Llama Community License も実質適用。英語のみ。**現在動作不可: 重みが HF gated で Llama 3.2 ライセンス同意 + `HF_TOKEN` が必須** |
| CosyVoice2 | Apache 2.0 | Apache 2.0 | OK | 多言語（日本語含む）。ゼロショット voice cloning。Python 3.10 venv 必須 |
| Spark-TTS | Apache 2.0 | CC BY-NC-SA 4.0 | 不可 | 英 / 中のみ。重みは学習データ制約で Apache 2.0 から再ライセンス |
| OpenVoice-V2 | MIT | MIT | OK | 多言語（日本語含む）。voice cloning。現在動作不可: `faster-whisper==0.9.0` 経由の `av==10` が Python 3.13 でビルドできない |
| VibeVoice | MIT | MIT | 要注意（research-only） | 英 / 中のみ。現在は動作不可: upstream API 移行中（.wav speaker ファイル → .pt prompt cache へ移行） |
| Fish-Speech | Apache 2.0 | Apache 2.0 | OK | A100/L4 GPU 必須（VRAM 24GB+） |
| Bark | MIT | MIT | OK | 13言語（日本語含む）。生成的（笑い声 / SFX）。著者は重みを research-oriented と表記 |
| ChatTTS | AGPL-3.0+ | CC BY-NC 4.0 | **不可** | 英 / 中 の対話 TTS。重みには乱用防止用の高周波ノイズが意図的に入っている |
| CSM-1B | Apache 2.0 | Apache 2.0 | OK | 英のみ。対話型。Llama-3.2-1B も依存（Llama 3.2 Community License）。HF gated |
| MisoTTS | Modified MIT | Modified MIT | OK | 8B の Sesame-CSM フォーク、**A100 必須**（~32GB F32 ckpt → bf16 ~16GB）。英語中心。トークナイザは ungated な `unsloth/Llama-3.2-1B` ミラーから取得（Llama 3.2 Community License は適用。`HF_TOKEN` 不要）。ゼロショット音声クローン。SilentCipher ウォーターマーク。MAU 5,000万超 or 月商 $1,000万超は UI に "Miso Labs" 表示義務 |
| StyleTTS2 (code) | MIT | — | — | sidharthrajaram/StyleTTS2 を使用（MIT、gruut ベース — upstream の GPL phonemizer を回避） |
| StyleTTS2 (LibriTTS 重み) | — | Custom (yl4579/StyleTTS2-LibriTTS) | 要注意 | 合成であることの開示が必要。voice cloning には話者の明示的同意が必要 |
| MaskGCT | MIT | CC BY-NC 4.0 | **不可** | 英 / 中 ゼロショット音声クローン。重みは非商用 |
| GPT-SoVITS | MIT | MIT | OK | 中 / 英 / 日 / 韓 / 粤 few-shot voice cloning。参照音声 + 書き起こし必須 |
| Higgs-Audio-v2 (code) | Apache 2.0 | — | — | LLM ベース音声基盤モデル。英語中心 |
| Higgs-Audio-v2 (重み) | — | Boson Higgs Audio 2 Community License | 要注意 | Llama 派生 community license。MAU 10万超は追加ライセンス必須、出力で他 LLM 学習禁止 |
| Higgs-Audio-v3 (code) | Apache 2.0 | — | — | SGLang-Omni が配信。4B chat-native TTS、100+ 言語（日本語含む） |
| Higgs-Audio-v3 (重み) | — | Boson Higgs Audio v3 Research and Non-Commercial License | 要注意 | **非商用のみ。** 個人利用 / 短期評価は許可、hosted API / production / 収益化は別途商用ライセンス必須。ダウンロードは gated ではない（HF_TOKEN 不要） |
| dots.tts | Apache 2.0 | Apache 2.0 | OK | 2B 連続 AR TTS、24 言語（日本語含む）、ゼロショット cloning。コードと全チェックポイント（base / soar / mf）が Apache-2.0。backbone は Qwen2.5-1.5B-Base 由来。ダウンロードは gated ではない（HF_TOKEN 不要） |
| LFM2.5-Audio-JP | LFM Open License v1.0 | LFM Open License v1.0 | 要注意 | 日本語特化の音声・テキストモデル、TTS パス利用。**年商 $10M 未満は商用 OK**、超過は別途商用ライセンス必須。ungated（HF_TOKEN 不要） |
| LFM2.5-Audio-JP (audio encoder) | Apache 2.0 | — | OK | FastConformer encoder は NVIDIA NeMo 由来 |
| LFM2.5-Audio-JP (audio codec / Mimi) | — | CC-BY 4.0 | OK | Kyutai Mimi codec（24 kHz、8 codebooks）。帰属表示が必要 |
| Ming-omni-TTS | MIT | Apache 2.0 | OK | 16.8B-A3B MoE 音声 LM（active ~3B）。**A100 40GB 必須**（bf16 で ~34GB の重み）。中国語/英語が中心、方言制御あり。zero-shot cloning。コード MIT（GitHub）、重み Apache-2.0（HF カード）。ungated（HF_TOKEN 不要） |
| DramaBox | LTX-2 Community License (Lightricks) | LTX-2 Community License | **年商 $10M+ の組織は商用ライセンス必須** | 英語。非競合条項あり、再配布時は同ライセンス継承必須、Perth ウォーターマーク常時付与 |
| Scenema (code) | MIT | — | — | リポジトリ: `ScenemaAI/scenema-audio` |
| Scenema (音声重み) | — | LTX-2 Community License (Lightricks) | **年商 $10M+ の組織は商用ライセンス必須** | 音声 transformer は LTX-2.3 派生のため LTX-2 Community License が継承される。DramaBox と同じ注意点（非競合条項、acceptable use 制限、再配布時のライセンス継承） |
| Scenema (Gemma 3 12B IT) | — | Gemma Terms of Use (Google) | 要注意 | HF gated。モデルカードで同意のうえ `HF_TOKEN` を設定。商用利用は Gemma 規約の prohibited use policy 遵守が条件 |

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
- Irodori-TTS-Lite
  https://github.com/kizuna-intelligence/Irodori-TTS-Lite
- Kokoro
  https://github.com/hexgrad/kokoro
- Kokoro-ONNX
  https://huggingface.co/nvidia/kokoro-82M-onnx-opt
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
- Supertonic
  https://github.com/supertone-inc/supertonic
- MOSS-TTS-Nano
  https://github.com/OpenMOSS/MOSS-TTS-Nano
- MOSS-TTS-v1.5
  https://github.com/OpenMOSS/MOSS-TTS
- MOSS-TTS-Local-v1.5
  https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5
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
- ZONOS2
  https://github.com/Zyphra/Zonos2
- OuteTTS
  https://github.com/edwko/OuteTTS
- Dia
  https://github.com/nari-labs/dia
- Kyutai-TTS (delayed-streams-modeling)
  https://github.com/kyutai-labs/delayed-streams-modeling
- Pocket-TTS
  https://github.com/kyutai-labs/pocket-tts
- Orpheus-TTS
  https://github.com/canopyai/Orpheus-TTS
- Spark-TTS
  https://github.com/SparkAudio/Spark-TTS
- OpenVoice
  https://github.com/myshell-ai/OpenVoice
- VibeVoice
  https://github.com/microsoft/VibeVoice
- Fish Speech
  https://github.com/fishaudio/fish-speech
- CosyVoice
  https://github.com/FunAudioLLM/CosyVoice
- Bark
  https://github.com/suno-ai/bark
- ChatTTS
  https://github.com/2noise/ChatTTS
- CSM (Conversational Speech Model)
  https://github.com/SesameAILabs/csm
- MisoTTS
  https://github.com/MisoLabsAI/MisoTTS
- StyleTTS 2 (upstream)
  https://github.com/yl4579/StyleTTS2
- StyleTTS 2 (pip 化フォーク)
  https://github.com/sidharthrajaram/StyleTTS2
- MaskGCT (Amphion)
  https://github.com/open-mmlab/Amphion/tree/main/models/tts/maskgct
- GPT-SoVITS
  https://github.com/RVC-Boss/GPT-SoVITS
- Higgs Audio v2
  https://github.com/boson-ai/higgs-audio
- Higgs Audio v3 (Hugging Face)
  https://huggingface.co/bosonai/higgs-audio-v3-tts-4b
- SGLang-Omni (Higgs Audio v3 backend)
  https://github.com/sgl-project/sglang-omni
- dots.tts
  https://github.com/rednote-hilab/dots.tts
- dots.tts (Hugging Face)
  https://huggingface.co/rednote-hilab/dots.tts-base
- LFM2.5-Audio-JP (Hugging Face)
  https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B-JP
- liquid-audio (LFM2-Audio library)
  https://github.com/Liquid4All/liquid-audio
- Ming-omni-tts
  https://github.com/inclusionAI/Ming-omni-tts
- Ming-omni-tts-16.8B-A3B (Hugging Face)
  https://huggingface.co/inclusionAI/Ming-omni-tts-16.8B-A3B
- DramaBox
  https://github.com/resemble-ai/DramaBox
- DramaBox (Hugging Face)
  https://huggingface.co/ResembleAI/Dramabox
- Scenema Audio
  https://github.com/ScenemaAI/scenema-audio
- Scenema Audio (Hugging Face)
  https://huggingface.co/ScenemaAI/scenema-audio
