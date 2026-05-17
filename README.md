# local-tts-on-google-colab

![Logo](./images/logo.png)

**English** | [日本語](README.ja.md)

A sample project that temporarily launches a selected local TTS engine on Google Colab as an OpenAI-compatible `/v1/audio/speech` endpoint for quick evaluation.

Supported engines:

| Engine | Colab Status | Languages |
|---|---|---|
| Kokoro | Works | Japanese / English / Chinese and more |
| Irodori-TTS | Works | Japanese |
| Piper | Works | English (default) / multilingual |
| Piper-Plus | Works | Japanese / English / Chinese and 6 languages |
| Qwen3-TTS | Works (GPU required) | Japanese / English / Chinese and 10 languages |
| VoxCPM2 | Works (GPU required) | Japanese / English / Chinese and 30 languages |
| MOSS-TTS-Nano | Works (output truncated to ~2s) | Japanese / English / Chinese and 20 languages |
| NeuTTS | Works (CPU OK, voice cloning) | English / Spanish / German / French |
| TinyTTS | Works | English |
| Supertonic | Works (CPU OK, ONNX, ~99M params) | English / Japanese / Korean and 31 languages |
| Voxtral-TTS | Works (GPU required, VRAM 16GB+) | English / French / Spanish and 9 languages |
| Orpheus-TTS | Not working (HF-gated weights, requires Llama 3.2 license acceptance + `HF_TOKEN`) | English (Llama-3.2-3B base, vLLM) |
| CosyVoice2 | Works (GPU recommended, Python 3.10 venv) | Japanese / English / Chinese / Korean / German and 9 languages |
| Spark-TTS | Works (GPU recommended) | English / Chinese (non-commercial weights) |
| Sarashina-TTS | Works (GPU required, ~6GB VRAM) | Japanese / English |
| F5-TTS | Works (GPU required) | English / Chinese (Japanese via separate model) |
| Chatterbox | Works (GPU recommended) | Japanese / English / Chinese and 23 languages |
| Zonos | Works (GPU required, ~6GB VRAM) | Japanese / English / Chinese / French / German |
| OuteTTS | Works (CPU OK) | Japanese / English / Chinese and many languages |
| Dia | Works (GPU recommended) | English (multi-speaker dialogue) |
| Kyutai-TTS | Works (GPU recommended) | English / French |
| Pocket-TTS | Works (CPU OK, ~6x real-time) | English / French / German / Italian / Portuguese / Spanish |
| OpenVoice-V2 | Not working (Python 3.13 / `av==10` build failure) | Japanese / English / Spanish / French / Chinese / Korean |
| VibeVoice | Not working (upstream API churn) | English / Chinese (long-form, up to 4 speakers) |
| Fish-Speech | Not working | Japanese / English / Chinese and 80+ languages |
| MeloTTS | Not working | - |
| Style-Bert-VITS2 | Not working | - |
| Bark | Working on Colab (GPU recommended, ~12GB / 8GB small) | English / Japanese / Chinese and 13 languages |
| ChatTTS | Working on Colab (GPU recommended, **non-commercial**) | English / Chinese |
| CSM-1B | Not working by default (HF-gated weights for `sesame/csm-1b` + `meta-llama/Llama-3.2-1B`, requires accepting both licenses + `HF_TOKEN`) | English (Llama-3.2-1B base + Mimi codec) |
| StyleTTS2 | Working on Colab (GPU recommended, Python 3.11 venv) | English |
| MaskGCT | Working on Colab (GPU required, ~10-12GB, **non-commercial weights**) | English / Chinese |
| GPT-SoVITS | Engine starts on Colab — reference audio required for synthesis (no default speaker mode; pass `--gpt-sovits-prompt-wav` + `--gpt-sovits-prompt-text`) | Chinese / English / Japanese / Korean / Cantonese |
| Higgs-Audio-v2 | Not working by default (upstream HF checkpoint requires unreleased `boson_multimodal` / transformers 5.x; engine starts but inference fails inside the audio tokenizer loader) | English |
| DramaBox | Works on Colab A100 (GPU required, VRAM ~24GB peak, **LTX-2 Community License — non-compete clause**) | English |
| Scenema | **Not verified on Colab** — text encoder is Gemma 3 12B IT (HF-gated), so running this engine requires accepting the Gemma Terms of Use on Hugging Face and providing `HF_TOKEN` via Colab Secrets. Code paths are in place but end-to-end Colab verification was deferred because `HF_TOKEN` setup is out of scope for this repo's default workflow. **Requires Colab A100 (40GB VRAM)**. First-run downloads ~38GB. Audio model derived from LTX-2.3 → **LTX-2 Community License** (same as DramaBox) | English-centric multilingual |

`MeloTTS` and `Style-Bert-VITS2` currently have dependency resolution issues under Colab's uv + venv environment and do not work.

`Fish-Speech` requires 24GB+ VRAM and targets A100/L4 GPUs. On Colab, the runtime crashes with OOM (out of memory) during model loading, so it currently does not work.

`VOICEVOX` is not included.

## Usage

### Quickest path

On Colab, it is recommended to paste the following code into a single cell and run it.

The cell automatically does the following:

- Clones/checks out the specified `REPO_URL` / `REPO_REF`
- Calls `colab/bootstrap.py` to start the selected TTS
- Optionally creates a `trycloudflare` public URL

`REPO_REF` accepts `main`, a tag, or a commit SHA. For reproducibility, a tag or commit SHA is recommended for normal use.

Key points:

- Start by only touching `ENGINE` and `REPO_REF`
- Only adjust engine-specific parameters when you actually need to
- The same cell contents are available in [multi_tts_openai_colab.py](multi_tts_openai_colab.py)

```python
#@title Local TTS on Google Colab -> OpenAI Compatible `/v1/audio/speech`
REPO_URL = "https://github.com/shinshin86/local-tts-on-google-colab.git"  #@param {type:"string"}
REPO_REF = "main"  #@param {type:"string"}
WORKDIR = "/content/local-tts-on-google-colab"  #@param {type:"string"}

ENGINE = "Kokoro"  #@param ["Bark", "ChatTTS", "Chatterbox", "CosyVoice2", "CSM-1B", "Dia", "DramaBox", "F5-TTS", "Fish-Speech", "GPT-SoVITS", "Higgs-Audio-v2", "Irodori-TTS", "Kokoro", "Kyutai-TTS", "MaskGCT", "MeloTTS", "MOSS-TTS-Nano", "NeuTTS", "OpenVoice-V2", "Orpheus-TTS", "OuteTTS", "Piper", "Piper-Plus", "Pocket-TTS", "Qwen3-TTS", "Sarashina-TTS", "Scenema", "Spark-TTS", "Style-Bert-VITS2", "StyleTTS2", "Supertonic", "TinyTTS", "VibeVoice", "VoxCPM2", "Voxtral-TTS", "Zonos"]
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
IRODORI_HF_CHECKPOINT = "Aratako/Irodori-TTS-500M-v3"  #@param ["Aratako/Irodori-TTS-500M-v3", "Aratako/Irodori-TTS-500M-v2", "Aratako/Irodori-TTS-500M"]
IRODORI_CODEC_REPO = "Aratako/Semantic-DACVAE-Japanese-32dim"  #@param {type:"string"}
IRODORI_MODEL_PRECISION = "fp32"  #@param ["fp32", "bf16", "fp16"]
IRODORI_CODEC_PRECISION = "fp32"  #@param ["fp32", "bf16", "fp16"]

#@markdown ---
#@markdown Kokoro
KOKORO_DEFAULT_VOICE = "jf_alpha"  #@param ["jf_alpha", "jf_gongitsune", "jm_kumo", "af_heart", "af_bella", "am_adam", "bf_emma", "bm_george", "zf_xiaobei"]
KOKORO_DEFAULT_LANG_CODE = "j"  #@param ["j", "a", "b", "e", "f", "h", "i", "p", "z"]

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
#@markdown NeuTTS (CPU OK, EN/ES/DE/FR, voice cloning)
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
CSM_HF_MODEL = "sesame/csm-1b"  #@param {type:"string"}
CSM_LLAMA_MODEL = "meta-llama/Llama-3.2-1B"  #@param {type:"string"}
CSM_DEFAULT_VOICE = "default"  #@param {type:"string"}
CSM_DEFAULT_SPEAKER = 0  #@param {type:"integer"}
CSM_MAX_AUDIO_LENGTH_MS = 10000  #@param {type:"integer"}
CSM_TEMPERATURE = 0.9  #@param {type:"number"}

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
        "--kokoro-default-voice",
        KOKORO_DEFAULT_VOICE,
        "--kokoro-default-lang-code",
        KOKORO_DEFAULT_LANG_CODE,
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
    cmd.append("--expose-public-url" if EXPOSE_PUBLIC_URL else "--no-expose-public-url")
    return cmd


def main():
    workdir = Path(WORKDIR)
    ensure_repo(REPO_URL, REPO_REF, workdir)
    run(build_bootstrap_command(workdir), cwd=str(workdir))


main()
```

### After running

On success, you will see the following in order:

- The local URL
- `/v1/models`
- `/v1/voices`
- The output path of the test WAV
- Optionally, the `trycloudflare` public URL

For your first try, `Kokoro` is recommended.

This setup assumes "one engine per runtime". To try a different engine, restart the runtime before re-running.

### For advanced users

If you want to launch directly from an already-cloned repository, you can call `colab/bootstrap.py`.

```python
!python colab/bootstrap.py --engine Kokoro --expose-public-url
```

If you just want to check the configuration without installing dependencies or starting the server, use `--dry-run`.

```python
!python colab/bootstrap.py --engine Kokoro --dry-run
```

## OpenAI Compatibility Scope

Supported endpoints:

- `GET /`
- `GET /v1/models`
- `GET /v1/voices`
- `POST /v1/audio/speech`

Main compatible inputs:

- `model`
- `input`
- `voice`
- `speed`
- `response_format`

This sample is fixed to `wav`. Conversion to formats like `mp3` is not performed.

## Engine-specific notes

### Kokoro

A lightweight TTS using [hexgrad/kokoro](https://github.com/hexgrad/kokoro), supporting Japanese, English, and Chinese. The default voice is the Japanese `jf_alpha`, and 9 voices can be selected from the form.

### Irodori-TTS

A Japanese TTS using [Aratako/Irodori-TTS](https://github.com/Aratako/Irodori-TTS). By default it uses the Hugging Face model `Aratako/Irodori-TTS-500M-v3` (a Rectified Flow DiT trained from scratch). To fall back to V2 set `IRODORI_HF_CHECKPOINT=Aratako/Irodori-TTS-500M-v2`; for V1 use `Aratako/Irodori-TTS-500M`. Output is high-quality 48 kHz, but there is no voice switching.

V3 adds two upstream changes that this wrapper handles automatically:

- **Duration Predictor**: with V3 the wrapper passes `seconds=None`, letting the model estimate output length from the input text (V2 / V1 stay on the previous 30-second fixed slot).
- **Integrated SilentCipher watermark**: V3 weights ship with [SilentCipher](https://github.com/sony/silentcipher) and upstream initializes it unconditionally inside `InferenceRuntime` — there is no public kill-switch and `RuntimeKey` no longer accepts an `enable_watermark` flag. Generated audio is watermarked whenever the SilentCipher weights are importable. **Do not strip the watermark**; it is part of the model release.

### Piper

Launches the [piper-tts](https://github.com/OHF-Voice/piper1-gpl) built-in HTTP server as a backend and puts an OpenAI-compatible wrapper in front of it. The default is the English `en_US-lessac-medium`. Dependencies are light and setup is stable.

### Piper-Plus

A lightweight, Japanese-capable TTS based on [ayutaz/piper-plus](https://github.com/ayutaz/piper-plus). It enhances the original Piper with better Japanese quality (OpenJTalk + prosody) and a GPL-free (MIT) license. No GPU required, runs quickly on CPU. The default model is `tsukuyomi` (Japanese female).

### Qwen3-TTS

A high-quality multilingual TTS using [QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS). It includes 9 speakers and supports 10 languages including Japanese. A GPU runtime (T4 or higher) is required. The default is the 0.6B model (lightweight), and the 1.7B model can also be selected from the form. Apache 2.0 licensed.

### VoxCPM2

A high-quality TTS using [OpenBMB/VoxCPM](https://github.com/OpenBMB/VoxCPM). A 2B-parameter model that supports 30 languages including Japanese, with automatic language detection. Features include zero-shot TTS, voice design (generate voice from text description), and voice cloning. A GPU runtime (T4 or higher, ~8GB VRAM) is required. License: Apache 2.0.

### MOSS-TTS-Nano

A lightweight multilingual TTS using [OpenMOSS/MOSS-TTS-Nano](https://github.com/OpenMOSS/MOSS-TTS-Nano). Only 0.1B (100M) parameters, supports 20 languages including Japanese / English / Chinese, and runs on CPU without a GPU. Default Hugging Face model: `OpenMOSS-Team/MOSS-TTS-Nano-100M`. Launched in `continuation` mode (plain TTS without a prompt audio). Output is 48 kHz stereo. License: Apache-2.0. Note: audio is generated successfully, but output is currently truncated to roughly the first ~2 seconds regardless of input length. The wrapper delegates generation to MOSS-TTS-Nano's `model.inference()`; exposing a length parameter on the upstream `inference()` API is likely needed to fix this.

### NeuTTS

An on-device TTS using [neuphonic/neutts](https://github.com/neuphonic/neutts). Uses **instant voice cloning** — every request is rendered in the voice of a reference audio file, so there is no preset speaker concept. Five reference voices bundled in the upstream repo are exposed via the OpenAI `voice` parameter:

| voice | language | sex |
|---|---|---|
| `dave`     | English | male   |
| `jo`       | English | female |
| `mateo`    | Spanish | male   |
| `greta`    | German  | female |
| `juliette` | French  | female |

Default backbone: `neuphonic/neutts-air` (~360M params, English only, Apache 2.0). Other languages have separate Nano backbones (`neuphonic/neutts-nano-french` / `-german` / `-spanish`, NeuTTS Open License 1.0). **Use a reference voice whose language matches the backbone** — mixing languages produces accented or garbled output. The wrapper lazy-encodes each reference on first use and caches it in memory. Japanese is **not** supported. License: code Apache-2.0; weights vary per backbone (see Licenses below). Adding your own reference voice is technically possible but should only be done with audio you have rights to (consent of the speaker).

### TinyTTS

An ultra-lightweight English TTS using [ecyht2/tiny-tts](https://github.com/ecyht2/tiny-tts). The model has only 1.6M parameters (~3.4 MB), no GPU required, and can synthesize speech at 53× real-time on CPU alone. Audio is output at 44.1 kHz. There is no voice switching. License: Apache 2.0.

### Supertonic

An ultra-lightweight on-device TTS using [supertone-inc/supertonic](https://github.com/supertone-inc/supertonic) by Supertone Inc. The `supertonic-3` model (~99M params, ~305MB ONNX assets) supports 31 languages including English, Japanese, and Korean, plus a `na` fallback for unknown text. Runs entirely on CPU via ONNX Runtime (no GPU required). The wrapper loads the model on first request and caches voice styles between requests.

The `voice` parameter exposes the 10 built-in presets:

| voice | description |
|---|---|
| `M1` – `M5` | Male presets (M1: upbeat / M2: deep / M3: authoritative / M4: gentle / M5: warm storyteller) |
| `F1` – `F5` | Female presets (F1: calm / F2: cheerful / F3: announcer / F4: confident / F5: gentle) |
| `default` | Alias for `--supertonic-default-voice` (defaults to `M1`) |

Voice cloning is **not** supported by the public Python SDK — cloned voices are produced through Supertone's separate Voice Builder service. Requesting `voice=clone` returns a 4xx with a clear message.

Supertonic-3 needs a language hint at synthesis time. OpenAI's `/v1/audio/speech` schema has no `language` field, so the wrapper accepts an extra `language` field in the JSON body. When `language` is omitted, the wrapper auto-detects the script (Hiragana/Katakana/CJK → `ja`, Hangul → `ko`) and otherwise falls back to `--supertonic-default-lang` (`en`). Use `"na"` for text whose language is unknown.

License: **code MIT** (https://github.com/supertone-inc/supertonic), **weights OpenRAIL-M** (https://huggingface.co/Supertone/supertonic-3). Commercial use is permitted; use-based ethical restrictions apply (no impersonation, deepfakes, defamation, etc. — see the OpenRAIL-M license text for the full list).

### Voxtral-TTS

A multilingual TTS using [mistralai/Voxtral-4B-TTS-2603](https://huggingface.co/mistralai/Voxtral-4B-TTS-2603). A 4B-parameter model supporting 9 languages: English, French, Spanish, German, Italian, Portuguese, Dutch, Arabic, and Hindi. It includes 20 preset voices and supports multiple formats such as wav / mp3 / flac / aac / opus. The backend uses vLLM + vllm-omni. A GPU runtime (VRAM 16GB or more) is required. Verified working on Colab A100 (40GB VRAM); may not work on the free-tier T4 (15GB) due to VRAM requirements. License: CC BY-NC 4.0 (non-commercial only).

### Sarashina-TTS

A Japanese-centric TTS using [sbintuitions/sarashina2.2-tts](https://huggingface.co/sbintuitions/sarashina2.2-tts) by SB Intuitions. An 0.8B-parameter LLM-based TTS supporting Japanese (primary) and English, with zero-shot voice cloning support. Default Hugging Face model: `sbintuitions/sarashina2.2-tts`. The HuggingFace transformers backend needs ~6GB VRAM (a Colab T4 fits); the optional vLLM backend (`--sarashina-use-vllm`) needs more VRAM but is faster. Generated audio is 24 kHz and contains an inaudible SilentCipher watermark by default — per the upstream model terms, do not remove it. **License: Sarashina Model NonCommercial License Agreement (commercial use prohibited).**

The `voice` parameter exposes:

| voice | description |
|---|---|
| `default` | Plain TTS without any reference audio (no zero-shot cloning). |
| `clone` | Zero-shot voice cloning. Only available when both `--sarashina-prompt-wav` and `--sarashina-prompt-text` are configured. The transcript must match the reference audio. |

For voice cloning, only use reference audio you have rights to (consent of the speaker).

### F5-TTS

A zero-shot voice cloning TTS using [SWivid/F5-TTS](https://github.com/SWivid/F5-TTS). It mimics the voice quality of a reference audio to generate speech. Uses the default reference audio bundled with the package (English female). To use a Japanese model, specify a community-provided Japanese checkpoint with `--f5tts-ckpt-file` / `--f5tts-vocab-file`. A GPU runtime (T4 or higher) is required. License: code MIT / model CC-BY-NC.

### Chatterbox

A multilingual TTS using [resemble-ai/chatterbox](https://github.com/resemble-ai/chatterbox) by Resemble AI. The Chatterbox Multilingual model supports 23 languages including Japanese, English, Chinese, French, German, Spanish, Korean, etc., and supports zero-shot voice cloning. Default language is `ja` (Japanese). When `--chatterbox-prompt-wav` is provided, the `clone` voice becomes available and uses the reference audio. A GPU runtime is recommended (VRAM ~2-4GB). License: MIT (both code and weights).

The `voice` parameter exposes:

| voice | description |
|---|---|
| `default` | Plain TTS without any reference audio. |
| `clone` | Zero-shot voice cloning. Only available when `--chatterbox-prompt-wav` is configured. |

For voice cloning, only use reference audio you have rights to (consent of the speaker).

### Zonos

A multilingual TTS using [Zyphra/Zonos](https://github.com/Zyphra/Zonos). Supports English, Japanese, Chinese, French, and German with zero-shot voice cloning. Default model: `Zyphra/Zonos-v0.1-transformer` (Apache 2.0). Phonemization is done by `espeak-ng`, which is installed automatically. The wrapper uses the bundled `assets/exampleaudio.mp3` as the default speaker reference; supplying `--zonos-prompt-wav` enables a `clone` voice with your own reference audio. A GPU runtime is required (VRAM 6GB+, T4 OK). The optional Hybrid backbone needs an Ampere or newer GPU and additional `mamba-ssm` deps; the Transformer backbone is used by default for portability. License: Apache 2.0 (code and weights).

The `voice` parameter exposes:

| voice | description |
|---|---|
| `default` | Uses the bundled reference audio shipped in the upstream repo. |
| `clone` | Zero-shot voice cloning. Only available when `--zonos-prompt-wav` is configured. |

For voice cloning, only use reference audio you have rights to (consent of the speaker).

### OuteTTS

A lightweight multilingual TTS using [edwko/OuteTTS](https://github.com/edwko/OuteTTS). Supports many languages including Japanese, with two model sizes (`0.6B` and `1B`) and multiple backends (`HF` for transformers, `LLAMACPP` for GGUF). Voice cloning is exposed via `--outetts-prompt-wav` (and an optional `--outetts-prompt-text` transcript). The default voice uses one of the bundled built-in speaker profiles, configurable via `--outetts-default-speaker` (e.g., `EN-FEMALE-1-NEUTRAL`). For best Japanese results, create a Japanese speaker profile from a reference clip with `clone`. Runs on CPU or GPU.

**License (depends on model size):**

| Model | Code | Weights | Commercial use |
|---|---|---|---|
| `OuteAI/OuteTTS-1.0-0.6B` | Apache 2.0 | Apache 2.0 | OK |
| `OuteAI/Llama-OuteTTS-1.0-1B` | Apache 2.0 | CC-BY-NC-SA-4.0 + Llama 3.2 Community License | **Not allowed** |

The default model size in this wrapper is `0.6B` (Apache 2.0). If you switch to `1B`, the weights become non-commercial.

The `voice` parameter exposes:

| voice | description |
|---|---|
| `default` | Plain TTS using the built-in speaker profile selected by `--outetts-default-speaker`. |
| `clone` | Voice cloning. Only available when `--outetts-prompt-wav` is configured. |

For voice cloning, only use reference audio you have rights to (consent of the speaker).

### Dia

A dialogue-oriented TTS using [nari-labs/dia](https://github.com/nari-labs/dia). The 1.6B-parameter model generates multi-speaker conversations in a single pass via `[S1]` / `[S2]` speaker tags directly in the prompt. English-only at the moment. The wrapper automatically prepends `[S1]` if your input has no speaker tag, so plain text still works for single-speaker TTS. Default model: `nari-labs/Dia-1.6B-0626`. With `--dia-prompt-wav` and `--dia-prompt-text`, the `clone` voice becomes available and conditions on a reference clip. A GPU runtime is recommended (VRAM ~4.4GB at float16/bfloat16, ~7.9GB at float32). License: Apache 2.0 (code and weights).

The `voice` parameter exposes:

| voice | description |
|---|---|
| `default` | Plain TTS without any reference. Use `[S1]` / `[S2]` tags in `input` for multi-speaker dialogue. |
| `clone` | Voice cloning. Only available when both `--dia-prompt-wav` and `--dia-prompt-text` are configured. |

For voice cloning, only use reference audio you have rights to (consent of the speaker).

### Kyutai-TTS

A streaming TTS using [kyutai-labs/delayed-streams-modeling](https://github.com/kyutai-labs/delayed-streams-modeling) — Kyutai Labs' English / French TTS built on the Delayed Streams Modeling (DSM) framework. The default model is `kyutai/tts-1.6b-en_fr` (1.6B parameters, English + French). Voices are loaded from a separate Hugging Face voice repository (default `kyutai/tts-voices`); the wrapper looks up `KYUTAI_VOICE` (default `expresso/ex03-ex01_happy_001_channel1_334s.wav`) inside that repo for the `default` voice. When `--kyutai-prompt-wav` is provided (a local `.wav` or pre-extracted `.safetensors` voice cache), the `clone` voice becomes available; you can also pass any voice path inside the voice repo directly as the `voice` parameter. A GPU runtime is recommended (CUDA, VRAM ~6GB). Japanese is **not** supported. License: code is MIT (Python) / Apache 2.0 (Rust); model weights are CC-BY-4.0.

### Pocket-TTS

An ultra-lightweight CPU TTS using [kyutai-labs/pocket-tts](https://github.com/kyutai-labs/pocket-tts) — Kyutai Labs' 100M-parameter on-device TTS that runs at ~6× real-time on a MacBook Air M4 using only 2 CPU cores. GPU is **not** required. Default Hugging Face model: `kyutai/pocket-tts`; voices are sourced from `kyutai/tts-voices`. Six language models are available (`english` / `english_2026-01` / `english_2026-04` / `french_24l` / `german_24l` / `italian` / `portuguese` / `spanish_24l`); pick one via `--pocket-language`. The default voice uses the `POCKET_DEFAULT_SPEAKER` preset (default: `alba`); supplying `--pocket-prompt-wav` enables a `clone` voice from your own audio file or a `.safetensors` voice cache. The 21 built-in preset names (`alba`, `anna`, `charles`, …) can also be passed directly as the `voice` parameter. License: code is MIT, model weights are CC-BY-4.0; **individual voice licenses vary** (see [kyutai/tts-voices](https://huggingface.co/kyutai/tts-voices)). **Prohibited use:** voice impersonation or cloning without explicit and lawful consent, and disinformation, are explicitly forbidden by the upstream terms.

### Spark-TTS

A bilingual zero-shot voice cloning TTS using [SparkAudio/Spark-TTS](https://github.com/SparkAudio/Spark-TTS). The 0.5B-parameter Qwen2.5-based LLM-TTS supports **English and Chinese** (Japanese is **not** supported), with two generation modes: voice cloning from a reference clip, or controllable generation by gender / pitch / speed without any reference. Output is 16 kHz mono WAV. A GPU is recommended (VRAM ~4GB).

The `voice` parameter exposes:

| voice | description |
|---|---|
| `default` | Plain TTS without any reference. Uses the configured `--spark-default-gender` (`male` / `female`), `--spark-default-pitch` (`very_low` / `low` / `moderate` / `high` / `very_high`), and `--spark-default-speed` (same five levels). |
| `clone` | Zero-shot voice cloning. Requires `--spark-prompt-wav`. `--spark-prompt-text` (optional transcript) improves quality when supplied. |

For voice cloning, only use reference audio you have rights to (consent of the speaker).

**License caveat:** code is Apache 2.0, but **the `Spark-TTS-0.5B` weights are CC BY-NC-SA 4.0 (non-commercial only)** because of training-data license constraints — the weights were previously Apache 2.0 and were re-licensed by upstream. Use the same way you would treat Sarashina-TTS / OuteTTS 1B / Voxtral-TTS in this repository: research and personal use are fine, commercial use is not allowed. The upstream model card also warns against unauthorized voice cloning, impersonation, fraud, and illegal use.

### Orpheus-TTS (currently not working — HF-gated weights)

Intended to use [canopyai/Orpheus-TTS](https://github.com/canopyai/Orpheus-TTS) by Canopy Labs — an English LLM-TTS built on `meta-llama/Llama-3.2-3B-Instruct` and served via vLLM through the `orpheus-speech` package. The default checkpoint `canopylabs/orpheus-tts-0.1-finetune-prod` ships with 8 English voices: `tara`, `leah`, `jess`, `leo`, `dan`, `mia`, `zac`, `zoe`. Output is 24 kHz mono WAV. Japanese is **not** supported.

**Why it does not work out-of-the-box on Colab:** the underlying weights repo `canopylabs/orpheus-3b-0.1-ft` is a **gated Hugging Face repository**, because the model is fine-tuned from Meta's `Llama-3.2-3B-Instruct` (Llama 3.2 Community License). With no token configured, vLLM fails at model load with:

```
OSError: You are trying to access a gated repo.
Access to model canopylabs/orpheus-3b-0.1-ft is restricted. You must have access to it and be authenticated to access it.
```

**To make it work, you must do all of the following before running the cell:**

1. Sign in to Hugging Face and request access to **both** repos. Acceptance is usually instant once you fill in the form:
   - <https://huggingface.co/canopylabs/orpheus-3b-0.1-ft>
   - <https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct>
2. Read and **agree to the Llama 3.2 Community License** on the Meta repo. The license restricts use cases (e.g. no use against Meta's Acceptable Use Policy) and applies regardless of the Apache-2.0 tag on the Orpheus repo itself.
3. Create a Hugging Face access token at <https://huggingface.co/settings/tokens> and expose it as `HF_TOKEN` in the Colab runtime — the simplest path is *Tools → Secrets → New secret*, key `HF_TOKEN`, value your token, then enable “Notebook access”. The wrapper picks it up via `os.environ` and forwards it to the engine subprocess.

The wrapper pins `vllm==0.7.3` (newer 0.7.x has a regression that breaks Orpheus' streaming generator) and creates a Python 3.12 venv (`xgrammar==0.1.11` has no cp313 wheel). A GPU is required — L4 / A100 recommended (VRAM ~10–12GB for the 3B weights plus vLLM KV cache).

**License (when access is granted):** code is Apache 2.0. The weights repo is tagged Apache 2.0, but in practice the **Llama 3.2 Community License** also applies because the model is fine-tuned from `Llama-3.2-3B-Instruct` (same situation as OuteTTS 1B). Always read both licenses before any commercial use.

### OpenVoice-V2 (currently not working)

Intended to use [myshell-ai/OpenVoice](https://github.com/myshell-ai/OpenVoice) V2 — a two-stage voice cloning TTS that first synthesises base speech with MeloTTS, then runs a ToneColorConverter (V2 checkpoints) to match the timbre of a reference clip. Both the code and the weights are MIT, so commercial use is allowed.

**Why it fails on Colab today**: OpenVoice's `pyproject.toml` hard-pins `faster-whisper==0.9.0`, which transitively pins `av>=10.dev0,<11.dev0`. The 10.x line of `av` does not have wheels for Python 3.13 (Colab's current default) and its Cython source no longer compiles against Cython 3.x:

```
av/logging.pyx:216:22: Cannot assign type 'const char *(void *) except?
NULL nogil' to 'const char *(*)(void *) noexcept nogil'.
```

Pre-installing `faster-whisper>=1.0` (which has `av==17.x` with py3.13 wheels) does not help — uv respects OpenVoice's pin and downgrades back to 0.9.0. Working around it would require `--no-deps` plus reconstructing the entire OpenVoice + MeloTTS dependency tree by hand, which sweeps in the standalone `MeloTTS` engine's own Rust-toolchain breakage as well.

The wrapper code is kept in tree so OpenVoice V2 can be reactivated once upstream relaxes its pins. **License (when working):** MIT for both code and weights (since April 2024).

### VibeVoice (currently not working)

Intended to use [microsoft/VibeVoice](https://github.com/microsoft/VibeVoice) — a 1.5B-parameter long-form multi-speaker TTS that can generate up to ~90 minutes of audio with up to 4 speakers in a single pass. The wrapper has been verified end-to-end up to model load on a Colab L4 GPU, but the upstream Microsoft repository is in the middle of a breaking API migration and synthesis cannot complete cleanly today:

- The reference inference class was renamed: `VibeVoiceForConditionalGenerationInference` → `VibeVoiceForConditionalGeneration` (this part the wrapper now handles).
- `model.set_ddpm_inference_steps(...)` has been removed; DDPM steps must now be set via `model.model.noise_scheduler.set_timesteps(...)` (handled).
- The bigger break: upstream **stopped shipping reference `.wav` speaker files** in `demo/voices/`. They now ship pre-extracted `.pt` prompt caches (e.g. `en-Carter_man.pt`, `jp-Spk1_woman.pt`) and the recommended path is `processor.process_input_with_cached_prompt(cached_prompt=torch.load(...))` rather than `processor(text=..., voice_samples=[wav_path])`. The non-streaming `voice_samples`-based path the wrapper currently uses no longer has working defaults.

The wrapper code is kept in tree so it can be rebuilt against the upstream API once it stabilises. **License caveat:** even when working, the model card tags VibeVoice as **"research purpose only"**: non-EN/ZH languages, voice impersonation, disinformation, and real-time voice conversion are prohibited. Don't ship into commercial / real-world products regardless of how the API ends up.

### Fish-Speech (currently not working)

A high-quality TTS using [fishaudio/fish-speech](https://github.com/fishaudio/fish-speech). Japanese is Tier 1 supported (highest quality) and it supports 80+ languages. It requires 24GB+ VRAM and targets A100/L4 GPUs, but on Colab the runtime crashes with OOM during model loading, so it currently does not work. License: Apache 2.0.

### CosyVoice2

A multilingual zero-shot voice cloning TTS using [FunAudioLLM/CosyVoice](https://github.com/FunAudioLLM/CosyVoice). The 0.5B-parameter v2 checkpoint (`FunAudioLLM/CosyVoice2-0.5B`) supports 9 common languages — **Japanese**, English, Chinese, Korean, German, Spanish, French, Italian, Russian — plus 18+ Chinese dialects, with cross-lingual zero-shot cloning. The wrapper forces a **Python 3.10 venv** (`uv venv --python 3.10`) because upstream pins (`torch==2.3.1`, `openai-whisper==20231117`, `onnxruntime-gpu==1.18.0`, etc.) do not resolve under Colab's default Python 3.12. The repo is cloned with `--recursive` to pick up the `Matcha-TTS` submodule. A GPU is recommended (VRAM ~4GB).

The `voice` parameter exposes:

| voice | description |
|---|---|
| `default` | Uses the bundled `asset/zero_shot_prompt.wav` reference (Chinese female) via `inference_cross_lingual`, which works regardless of input language. |
| `clone` | Zero-shot voice cloning with `--cosyvoice-prompt-wav`. If `--cosyvoice-prompt-text` is also set, calls `inference_zero_shot` (better quality when transcript matches); otherwise falls back to `inference_cross_lingual`. |

For voice cloning, only use reference audio you have rights to (consent of the speaker).

License: Apache 2.0 for both code (CosyVoice repo) and the `CosyVoice2-0.5B` weights (per the Hugging Face model card).

### Bark

A text-prompted generative audio model from Suno using [suno-ai/bark](https://github.com/suno-ai/bark). Supports 13 languages (English, German, Spanish, French, Hindi, Italian, **Japanese**, Korean, Polish, Portuguese, Russian, Turkish, Simplified Chinese) and can produce non-verbal sounds (laughter, sighs, simple SFX). Voice presets follow the upstream Speaker Library naming `v2/<lang>_speaker_<n>` (10 speakers per language).

The full model needs ~12 GB VRAM; set `BARK_USE_SMALL_MODELS=True` (or `--bark-use-small-models`) to drop to ~8 GB. Generation is stochastic — the same input can yield different audio.

License: code & weights both MIT (commercially usable). The author labels Bark as research-oriented and acknowledges potential dual-use; use responsibly.

### ChatTTS

A conversational TTS from 2noise using [2noise/ChatTTS](https://github.com/2noise/ChatTTS). Designed for daily dialogue with expressive features (laughter, hesitation, pauses). EN / ZH only.

The `voice` parameter exposes:

| voice | description |
|---|---|
| `default` | Reproducible speaker generated from `--chattts-seed` (default 2). Same seed yields the same speaker. |
| `random` | A fresh random speaker on every request via `chat.sample_random_speaker()`. |

**License caveat (important):** the code is **AGPL-3.0+** and the **weights are CC BY-NC 4.0**, so this engine is **research / educational use only — commercial use is not allowed**. The weights also contain intentional high-frequency noise added during training to discourage misuse, which slightly degrades audio quality.

### CSM-1B (Sesame Conversational Speech Model)

A conversational speech model from Sesame using [SesameAILabs/csm](https://github.com/SesameAILabs/csm). The architecture is a Llama-3.2-1B backbone plus an audio decoder that produces Mimi codec audio (the same codec lineage as Kyutai-TTS). EN only.

The wrapper forces a **Python 3.11 venv** because upstream pins `torch==2.4.0` / `torchtune==0.4.0` / `torchao==0.9.0`. It also sets `NO_TORCH_COMPILE=1` per upstream README. GPU recommended (VRAM ~6 GB).

**HF gated weights** — the model card requires accepting terms before downloads, and the Llama-3.2-1B base model is also gated under the Llama 3.2 Community License. To use it on Colab:

1. Visit `https://huggingface.co/sesame/csm-1b` and accept the conditions.
2. Visit `https://huggingface.co/meta-llama/Llama-3.2-1B`, accept the Llama 3.2 Community License.
3. Add `HF_TOKEN` to Colab Secrets (notebook access enabled).

The `voice` parameter exposes:

| voice | description |
|---|---|
| `default` | Uses speaker_id `--csm-default-speaker` (default 0). |
| `speaker_<int>` | Direct speaker_id (e.g. `speaker_1`). |

License: code & CSM-1B weights are Apache 2.0. The Llama-3.2-1B base is under the Llama 3.2 Community License.

### StyleTTS2

A high-quality TTS using diffusion + adversarial training with large speech language models, by yl4579 ([yl4579/StyleTTS2](https://github.com/yl4579/StyleTTS2)). The wrapper uses the [sidharthrajaram/StyleTTS2](https://github.com/sidharthrajaram/StyleTTS2) pip package, which substitutes **gruut (MIT)** for the upstream **phonemizer (GPL-3.0)** to keep the dependency tree GPL-free.

EN only (gruut is English-focused). The wrapper uses a **Python 3.11 venv** to isolate the legacy pins from sidharthrajaram's package.

The `voice` parameter exposes:

| voice | description |
|---|---|
| `default` | Random style sampled from diffusion (no reference audio). |
| `clone` | Voice cloning. Only available when `--styletts2-prompt-wav` is configured. |

`--styletts2-alpha` controls timbre similarity to the reference (0=full reference, 1=full sampling). `--styletts2-beta` controls prosody similarity. `--styletts2-diffusion-steps` and `--styletts2-embedding-scale` are upstream knobs for variation and emotion strength.

**License caveat:** code (both upstream and the pip fork) is MIT, but the LibriTTS pretrained checkpoint (`yl4579/StyleTTS2-LibriTTS`) is under a **custom license that requires disclosing the audio is synthesized**. Voice cloning requires explicit consent of the speaker.

### MaskGCT

A Masked Generative Codec Transformer TTS from Amphion ([open-mmlab/Amphion](https://github.com/open-mmlab/Amphion/tree/main/models/tts/maskgct)). Fully non-autoregressive, eliminating the explicit alignment step. Zero-shot voice cloning — every request needs a reference audio. EN / ZH (with optional `prompt_lang` / `target_lang` for other languages).

The wrapper sparse-checks-out only the `models/tts/maskgct`, `models/codec`, and `utils` paths from Amphion (the full repo is huge), and forces a **Python 3.10 venv** because upstream pins `torch==2.0.1` / `transformers==4.41.2` / `numpy==1.26.0`. Requires `espeak-ng` system package and the same `setuptools<70` pre-install trick as CosyVoice2 (transitive `openai-whisper==20231117` legacy setup.py needs `pkg_resources`).

The `voice` parameter exposes:

| voice | description |
|---|---|
| `default` | Uses the bundled `models/tts/maskgct/wav/prompt.wav` (English female) shipped in the upstream repo. |
| `clone` | Voice cloning. Only available when both `--maskgct-prompt-wav` and `--maskgct-prompt-text` are configured. |

GPU required (VRAM ~10-12 GB).

**License caveat:** code is MIT, but the weights (`amphion/MaskGCT`) are under **CC BY-NC 4.0**, so this engine is **non-commercial only**.

### GPT-SoVITS

A few-shot voice cloning TTS from RVC-Boss using [RVC-Boss/GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS). 5 seconds of reference audio is enough for zero-shot inference; 1 minute can be used for fine-tuning. Supports **Chinese, English, Japanese, Korean, Cantonese**. Multiple model versions (v1, v2, v2Pro, v2ProPlus, v3, v4) are exposed via `--gpt-sovits-version` (default `v2`).

GPT-SoVITS is fundamentally a few-shot cloning model — there is no built-in "default speaker" mode. The wrapper requires `--gpt-sovits-prompt-wav` and `--gpt-sovits-prompt-text` at startup; without them every request returns 400.

The wrapper uses a **Python 3.11 venv** (upstream documents Python 3.10 / 3.11). On first run it selectively snapshot-downloads only the v2 weights (`gsv-v2final-pretrained/` ~1.2 GB) plus `chinese-hubert-base/` and `chinese-roberta-wwm-ext-large/` from `lj1995/GPT-SoVITS` instead of the full 5.3 GB repo. GPU recommended (VRAM ~4-6 GB).

The `voice` parameter exposes:

| voice | description |
|---|---|
| `default` / `clone` | Both use the configured prompt audio + transcript. (GPT-SoVITS has no plain-TTS fallback.) |

License: code is MIT, weights (`lj1995/GPT-SoVITS`) are MIT — commercially usable.

### Higgs-Audio-v2

An LLM-based audio foundation model from Boson AI ([boson-ai/higgs-audio](https://github.com/boson-ai/higgs-audio)). The 3B-base generation model is paired with a separate audio tokenizer (`bosonai/higgs-audio-v2-tokenizer`). Voice cloning works either by referencing a bundled preset (e.g. `belinda`, `broom_salesman` under `examples/voice_prompts/`) or by passing a custom audio path.

**Hardware**: requires ~24 GB VRAM for the 3B model, so **A100 / L4 (Colab Pro) is required**. T4 (free tier) cannot host this model.

The wrapper uses a **Python 3.10 venv** (matching upstream NVIDIA container target) and `pip install -e .` to expose the `boson_multimodal` package. The default reference voice is `belinda`; pass `--higgs-default-ref-voice <name>` to change it (any preset under `examples/voice_prompts/`).

The `voice` parameter exposes:

| voice | description |
|---|---|
| `default` | Uses preset `--higgs-default-ref-voice` (default `belinda`). |
| `clone` | Voice cloning. Only available when both `--higgs-prompt-wav` and `--higgs-prompt-text` are configured. |
| `<preset_name>` | Any preset name under `examples/voice_prompts/` (e.g. `broom_salesman`). |

**License caveat (important):** code is Apache-2.0, but the weights (`bosonai/higgs-audio-v2-generation-3B-base`) are under the **Boson Higgs Audio 2 Community License** — a derivative of Meta's Llama 3 Community License. Restrictions:

- Commercial use is permitted, but if your product has more than 100,000 monthly active users, you must request an expanded license from Boson AI.
- Outputs cannot be used to train other large language models.
- Attribution required for redistribution.
- Compliance with Meta's Acceptable Use Policy is required.

**Status (currently not working by default):** the published HF checkpoint expects an unreleased branch of `boson-ai/higgs-audio` plus transformers 5.x, while the released `boson_multimodal` PyPI package targets transformers 4.46.x. The engine wiring (install / venv / app / voice list / cloudflared) is correct and `/`, `/v1/models`, `/v1/voices` all return 200 on Colab L4, but `/v1/audio/speech` returns 500 because the audio tokenizer loader (`load_higgs_audio_tokenizer`) passes the flat HF config keys (`acoustic_model_config`, `semantic_model_config`) to a `HiggsAudioTokenizer.__init__` that does not accept them. Two earlier mismatches are already worked around in this wrapper (text-config defaults that broke `nn.Embedding(padding_idx=128001, num_embeddings=32000)`, and a `tokenizer_class=TokenizersBackend` reference that only exists in unreleased transformers 5.x) but the audio-tokenizer schema drift requires an upstream fix in `boson-ai/higgs-audio` itself. Once upstream realigns code with the published config, this engine should work without further changes here.

### DramaBox

A directable / expressive TTS from Resemble AI ([resemble-ai/DramaBox](https://github.com/resemble-ai/DramaBox)). It is an IC-LoRA fine-tune of Lightricks' LTX-2.3 audio-only branch, with a Gemma 3 12B text encoder; users can drive emotion, pacing, laughs, sighs, and other paralinguistic cues directly from the English text prompt. Voice cloning uses any 10+ second audio reference.

**Hardware**: requires ~24 GB VRAM peak, so **Google Colab A100 (40 GB) is required**. T4 / V100 cannot host this model.

The wrapper clones the upstream GitHub repo, installs `requirements.txt` (which pins `torch==2.8.0`, `pydantic==2.10.6`, `bitsandbytes`, `resemble-perth`, etc.), and points the FastAPI app at `<repo>/src` + `<repo>/ltx2` for module resolution. First run downloads ~8.5 GB from `ResembleAI/Dramabox` plus the `unsloth/gemma-3-12b-it-bnb-4bit` snapshot.

Prompts work best in director-instruction English, for example:

```
A woman speaks warmly, "Hello, how are you today?" She laughs, "Hahaha, it is so good to see you!"
```

The `voice` parameter exposes:

| voice | description |
|---|---|
| `default` | Uses preset `--dramabox-default-ref-voice` (default `female_american`). |
| `clone` | Voice cloning. Only available when `--dramabox-prompt-wav` is configured (10+ seconds recommended). |
| `<preset_name>` | Any bundled preset under `DramaBox/assets/voices/`: `female_american`, `female_shadowheart`, `male_arnie`, `male_conan`, `male_harvey_keitel`, `male_old_movie`, `male_petergriffin`, `male_samuel_j`. |

**Watermarking**: Every generated audio file is invisibly watermarked with Resemble's **Perth** implicit watermarker (`perth.PerthImplicitWatermarker`). This is left enabled as the upstream design intends — do not patch it out. The watermark is imperceptible to humans but allows Resemble's tools to identify AI-generated audio.

**License caveat (important):** code and weights are released under the **LTX-2 Community License Agreement** (Lightricks), which is materially more restrictive than Apache 2.0 / MIT / Llama Community License:

- Free for non-commercial use, and free for organizations with annual revenue below USD 10 M; organizations above that threshold must obtain a paid commercial license from Lightricks.
- A **non-compete clause** forbids using the model (or derivatives) to train other competing models or to build products that directly compete with Lightricks' offerings.
- When redistributing derivatives, the same license must be propagated (use-restrictions and acceptable-use policy intact).
- Acceptable-use restrictions are extensive: no deepfakes / impersonation without consent, no undisclosed AI-generated content, no misinformation, no medical advice, no automated legal decisions, no military / weapons / malware uses, etc.

This repository wraps DramaBox only for personal / research evaluation on Colab — operators are responsible for confirming their own use case complies with the LTX-2 Community License.

### Scenema

A zero-shot expressive voice cloning / speech-generation engine from Scenema AI ([ScenemaAI/scenema-audio](https://github.com/ScenemaAI/scenema-audio)). Its audio diffusion transformer is extracted from Lightricks' LTX-2 (22B audiovisual model) and conditioned with Gemma 3 12B IT, paired with SeedVC for voice identity transfer and a MelBandRoFormer separator for clean speech. The marquee feature is "performance": describe a voice in free-form English and the model delivers it with intent, pacing, breath control, and emotional arcs — including emotions the reference speaker never recorded.

**Status: not verified on Colab.** The full install / wiring / API layer is implemented (installer, app, voice presets, XML pass-through), but Scenema's text encoder is **Gemma 3 12B IT, which is HF-gated**. End-to-end verification on Colab A100 was deferred because configuring `HF_TOKEN` and accepting Google's Gemma Terms of Use sits outside this repo's default verification workflow. If you want to actually run this engine on Colab, you need to do the setup yourself (see below). Confirmed-good Colab logs from a user would be welcome — please open an issue / PR with the trycloudflare evidence.

**Setup required before launch:**

1. Sign in to Hugging Face and accept the Gemma Terms of Use on the [google/gemma-3-12b-it](https://huggingface.co/google/gemma-3-12b-it) model card ("Acknowledge license").
2. Create a Read-scoped Hugging Face token at https://huggingface.co/settings/tokens.
3. In the Colab notebook, open the left sidebar **🔑 Secrets** panel, add a new secret named **`HF_TOKEN`** with the token value, and **enable notebook access**.

Without this, the lifespan startup `snapshot_download("google/gemma-3-12b-it")` returns `401 Unauthorized` and `AudioProcessor` never becomes ready — `/v1/audio/speech` will return 503.

**Hardware**: **Colab A100 (40 GB) is required.** The wrapper uses the INT8 audio transformer + NF4 Gemma profile (~13 GB resident VRAM, fits in 24 GB and comfortably on 40 GB). T4 / V100 cannot host this model. First run downloads ~38 GB total: audio transformer (~5 GB), pipeline checkpoint (~7 GB), Gemma 3 12B IT (~24 GB), SeedVC (~1.6 GB), BigVGAN v2 + Whisper Small.

**OpenAI API mapping.** This repo exposes Scenema through the standard `/v1/audio/speech` endpoint. Because Scenema's native input is a structured `<speak>` XML with `<action>` performance directions, two input paths are supported:

1. **Plain text** (default): `input` is wrapped automatically as `<speak voice="<preset description>" gender="<gender>">…</speak>`. The `voice` parameter selects a preset or — for non-preset values — is used as the voice description verbatim. This is what OpenAI SDK clients will use.
2. **XML pass-through** (advanced): if `input` begins with `<speak`, the string is forwarded to Scenema unchanged. Use this to add `<action>angry shout</action>` / `<sound>thunder</sound>` / multi-segment performance direction.

The `voice` parameter exposes:

| voice | description |
|---|---|
| `default` | Warm British-accented male narrator (baseline neutral voice). |
| `warm_male` | Warm middle-aged male narrator. Deep but gentle tone. |
| `smoky_female` | Smoky low-register female voice, intimate confessional tone. |
| `child_girl` | Bright six-year-old girl, breathless and excited. |
| `elderly_male` | Weathered elderly male storyteller, deep baritone. |
| `elderly_female` | Soft alto, woman in her seventies. |
| `clone` | Zero-shot voice cloning. Requires `--scenema-prompt-wav` (10–20 s of reference audio with some emotional variability). |
| `<any other string>` | Used directly as a Scenema voice description (e.g. `"Male, mid 60s. Deep baritone with gravel. Slight Southern American inflection."`). |

**License caveat (important):** code is MIT, but the **audio transformer weights inherit the LTX-2 Community License Agreement** (Lightricks) — the same restrictive license as DramaBox. Gemma 3 12B IT is additionally bound by the Gemma Terms of Use (Google). Operators are responsible for confirming their use case complies with both. Acceptable-use limits are extensive (no impersonation, no deepfakes, no disinformation, no military / weapons / medical-advice uses, etc.).

### MeloTTS (currently not working)

Intended to use [myshell-ai/MeloTTS](https://github.com/myshell-ai/MeloTTS), but the dependency `tokenizers` requires a Rust compiler to build, so installation fails in the current Colab environment.

### Style-Bert-VITS2 (currently not working)

Intended to use [litagin02/Style-Bert-VITS2](https://github.com/litagin02/Style-Bert-VITS2), but `setuptools` / `torch` / `scipy` dependency conflicts cannot be resolved, and speech synthesis is not reachable in the current Colab environment.

## License

The license for each engine is as follows. When using them, always check each project's latest license terms.

| Engine | Code | Model Weights | Commercial Use | Notes |
|---|---|---|---|---|
| Kokoro | Apache 2.0 | Apache 2.0 | OK | |
| Irodori-TTS | MIT | MIT (v1 / v2 / v3) | OK | Ethical policy prohibits impersonation / deepfake generation. V3 ships with SilentCipher watermarking — do not strip |
| Piper | GPL-3.0 | MIT | Caution | The default voice `en_US-lessac-medium` is trained on the Blizzard 2013 dataset (Lessac Technologies), which is research-only and prohibits commercial use |
| Piper-Plus | MIT | MIT | OK | |
| Qwen3-TTS | Apache 2.0 | Apache 2.0 | OK | |
| VoxCPM2 | Apache 2.0 | Apache 2.0 | OK | |
| MOSS-TTS-Nano | Apache 2.0 | Apache 2.0 | OK | 100M params, CPU OK |
| NeuTTS | Apache 2.0 | Apache 2.0 (Air) / NeuTTS Open License 1.0 (Nano) | OK (Air) / Check terms (Nano) | Voice cloning. EN / ES / DE / FR |
| TinyTTS | Apache 2.0 | Apache 2.0 | OK | |
| Supertonic | MIT | OpenRAIL-M | OK | 31 languages incl JP/KO/EN. CPU OK (ONNX). Use-based ethical restrictions (no impersonation/deepfakes) |
| Voxtral-TTS | — | CC BY-NC 4.0 | Not allowed | Via vLLM + vllm-omni. Non-commercial due to voice dataset license constraints |
| Orpheus-TTS | Apache 2.0 | Apache 2.0 + Llama 3.2 Community License | Caution | Llama-3.2-3B-Instruct base; Llama Community License applies in practice. EN only. **Currently not working: weights are HF-gated and require Llama 3.2 license acceptance + `HF_TOKEN`** |
| CosyVoice2 | Apache 2.0 | Apache 2.0 | OK | Multilingual incl JP. Zero-shot voice cloning. Requires Python 3.10 venv |
| Spark-TTS | Apache 2.0 | CC BY-NC-SA 4.0 | Not allowed | EN / ZH only. Weights re-licensed from Apache 2.0 due to training-data constraints |
| Sarashina-TTS | — | Sarashina Model NonCommercial License | Not allowed | Japanese / English. Zero-shot voice cloning. Output contains a SilentCipher watermark (do not remove) |
| F5-TTS | MIT | CC-BY-NC | Not allowed (model) | Model weights are non-commercial due to Emilia dataset constraints |
| Chatterbox | MIT | MIT | OK | Multilingual (23 languages incl JP). Zero-shot voice cloning |
| Zonos | Apache 2.0 | Apache 2.0 | OK | EN/JA/ZH/FR/DE. Zero-shot voice cloning. Requires `espeak-ng` |
| OuteTTS (0.6B) | Apache 2.0 | Apache 2.0 | OK | Multilingual incl JP. CPU OK. Voice cloning |
| OuteTTS (1B)   | Apache 2.0 | CC-BY-NC-SA-4.0 + Llama 3.2 Community License | Not allowed | Llama-3.2-based; non-commercial weights |
| Dia | Apache 2.0 | Apache 2.0 | OK | EN only. Multi-speaker `[S1]`/`[S2]` dialogue TTS |
| Kyutai-TTS | MIT (Python) / Apache 2.0 (Rust) | CC-BY-4.0 | OK (with attribution) | EN / FR. DSM-based streaming TTS. GPU recommended |
| Pocket-TTS (model) | MIT | CC-BY-4.0 | OK (with attribution) | 100M params, CPU-only. EN / FR / DE / IT / PT / ES |
| Pocket-TTS (voices) | — | Per-voice (mixed) | Check per voice | Voice licenses listed at [kyutai/tts-voices](https://huggingface.co/kyutai/tts-voices); upstream prohibits non-consensual impersonation |
| OpenVoice-V2 | MIT | MIT | OK | Multilingual (incl JP). Voice cloning. Currently not working: `av==10` (via `faster-whisper==0.9.0` pin) won't build on Python 3.13 |
| VibeVoice | MIT | MIT | Caution (research-only) | EN/ZH only per model card. Currently not working: upstream is mid-API migration (.wav speakers replaced with .pt caches) |
| Fish-Speech | Apache 2.0 | Apache 2.0 | OK | Requires A100/L4 GPU (VRAM 24GB+) |
| Bark | MIT | MIT | OK | 13 languages incl JP. Generative (laughter / SFX). Author labels weights as research-oriented |
| ChatTTS | AGPL-3.0+ | CC BY-NC 4.0 | **Not allowed** | EN / ZH conversational TTS. Weights contain intentional high-frequency noise to deter misuse |
| CSM-1B | Apache 2.0 | Apache 2.0 | OK | EN only. Conversational. Llama-3.2-1B is also pulled in (Llama 3.2 Community License). HF gated |
| StyleTTS2 (code) | MIT | — | — | Uses sidharthrajaram/StyleTTS2 (MIT, gruut-based — avoids upstream's GPL phonemizer) |
| StyleTTS2 (LibriTTS weights) | — | Custom (yl4579/StyleTTS2-LibriTTS) | Caution | Requires disclosing that audio is synthesized; voice cloning needs explicit consent |
| MaskGCT | MIT | CC BY-NC 4.0 | **Not allowed** | EN / ZH zero-shot voice cloning. Weights are non-commercial |
| GPT-SoVITS | MIT | MIT | OK | ZH / EN / JA / KO / YUE few-shot voice cloning. Reference audio + transcript required |
| Higgs-Audio-v2 (code) | Apache 2.0 | — | — | LLM-based audio foundation model. EN focus |
| Higgs-Audio-v2 (weights) | — | Boson Higgs Audio 2 Community License | Caution | Llama-derived community license. Restricted: >100k MAU needs extra license; outputs cannot be used to train other LLMs |
| DramaBox | LTX-2 Community License (Lightricks) | LTX-2 Community License | **Not allowed without commercial license** for orgs with annual revenue $10M+ | English. Non-compete clause; redistributions must propagate the same license. Perth watermark always applied |
| Scenema (code) | MIT | — | — | Repo: `ScenemaAI/scenema-audio` |
| Scenema (audio weights) | — | LTX-2 Community License (Lightricks) | **Not allowed without commercial license** for orgs with annual revenue $10M+ | Audio transformer is derived from LTX-2.3; the LTX-2 Community License flows through. Same caveats as DramaBox (non-compete, acceptable-use restrictions, propagation requirement) |
| Scenema (Gemma 3 12B IT) | — | Gemma Terms of Use (Google) | Caution | HF-gated; accept on the model card and set `HF_TOKEN`. Commercial use is permitted under Gemma terms but the prohibited-use policy applies |

**About Piper**: The `piper-tts` package is GPL-3.0. Also, the default `en_US-lessac-medium` voice is trained on the Blizzard 2013 dataset provided by Lessac Technologies, and its license prohibits commercial use. If you need commercial use, choose another voice model trained with a permissive license.

This repository itself is intended for short-term operational verification and technical evaluation.

## Notes

- On Colab's managed runtime, external exposure and proxy usage are not suitable for continuous operation. This repository is for short-term verification only.
- Because each engine has heavy dependencies, switching between engines assumes a runtime restart.
- For the license of each engine and voice model, please check the "License" section above and each project's official information.

## References

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
- Supertonic
  https://github.com/supertone-inc/supertonic
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
- StyleTTS 2 (upstream)
  https://github.com/yl4579/StyleTTS2
- StyleTTS 2 (pip-installable fork)
  https://github.com/sidharthrajaram/StyleTTS2
- MaskGCT (Amphion)
  https://github.com/open-mmlab/Amphion/tree/main/models/tts/maskgct
- GPT-SoVITS
  https://github.com/RVC-Boss/GPT-SoVITS
- Higgs Audio v2
  https://github.com/boson-ai/higgs-audio
- DramaBox
  https://github.com/resemble-ai/DramaBox
- DramaBox (Hugging Face)
  https://huggingface.co/ResembleAI/Dramabox
- Scenema Audio
  https://github.com/ScenemaAI/scenema-audio
- Scenema Audio (Hugging Face)
  https://huggingface.co/ScenemaAI/scenema-audio
