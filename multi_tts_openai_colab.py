"""
Google Colab 向けの最小 bootstrap セル。

このコードを Colab の 1 セルに貼り付けて実行すると、
指定した ref のリポジトリを clone / checkout し、
`colab/bootstrap.py` を呼び出して対象エンジンを起動する。
"""

#@title Local TTS on Google Colab -> OpenAI Compatible `/v1/audio/speech`
REPO_URL = "https://github.com/shinshin86/local-tts-on-google-colab.git"  #@param {type:"string"}
REPO_REF = "main"  #@param {type:"string"}
WORKDIR = "/content/local-tts-on-google-colab"  #@param {type:"string"}

ENGINE = "Kokoro"  #@param ["Bark", "ChatTTS", "Chatterbox", "CosyVoice2", "CSM-1B", "Dia", "DramaBox", "F5-TTS", "Fish-Speech", "GPT-SoVITS", "Higgs-Audio-v2", "Irodori-TTS", "Irodori-TTS-Lite", "Kokoro", "Kokoro-ONNX", "Kyutai-TTS", "MaskGCT", "MeloTTS", "MisoTTS", "MOSS-TTS-Nano", "MOSS-TTS-v1.5", "NeuTTS", "OpenVoice-V2", "Orpheus-TTS", "OuteTTS", "Piper", "Piper-Plus", "Pocket-TTS", "Qwen3-TTS", "Sarashina-TTS", "Scenema", "Spark-TTS", "Style-Bert-VITS2", "StyleTTS2", "Supertonic", "TinyTTS", "VibeVoice", "VoxCPM2", "Voxtral-TTS", "Zonos"]
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
#@markdown - **HF gated**: the tokenizer loads `meta-llama/Llama-3.2-1B`. Accept the Llama 3.2 Community License at https://huggingface.co/meta-llama/Llama-3.2-1B and set `HF_TOKEN` in Colab Secrets, or the first request fails with `OSError: gated repo ... 401`.
#@markdown - License: MisoTTS code & weights are **Modified MIT** ([MisoLabsAI/MisoTTS](https://github.com/MisoLabsAI/MisoTTS), [MisoLabs/MisoTTS](https://huggingface.co/MisoLabs/MisoTTS)) — commercial use OK, but products with >50M MAU or >$10M/month revenue must display "Miso Labs" in the UI. The gated Llama-3.2-1B tokenizer adds the **Llama 3.2 Community License**. Output carries an inaudible SilentCipher watermark applied inside `generate()` (do not remove).
#@markdown - The ~32GB F32 checkpoint loads as bf16 (~16GB on GPU). `voice="clone"` needs `MISOTTS_PROMPT_WAV` (optionally `MISOTTS_PROMPT_TEXT`); otherwise use `voice="default"` / `speaker_<int>`.
MISOTTS_HF_MODEL = "MisoLabs/MisoTTS"  #@param {type:"string"}
MISOTTS_DEFAULT_VOICE = "default"  #@param ["default", "clone"]
MISOTTS_DEFAULT_SPEAKER = 0  #@param {type:"integer"}
MISOTTS_PROMPT_WAV = ""  #@param {type:"string"}
MISOTTS_PROMPT_TEXT = ""  #@param {type:"string"}
MISOTTS_MAX_AUDIO_LENGTH_MS = 30000  #@param {type:"integer"}
MISOTTS_TEMPERATURE = 0.9  #@param {type:"number"}
MISOTTS_TOPK = 50  #@param {type:"integer"}

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
    cmd.append("--expose-public-url" if EXPOSE_PUBLIC_URL else "--no-expose-public-url")
    return cmd


def main():
    workdir = Path(WORKDIR)
    ensure_repo(REPO_URL, REPO_REF, workdir)
    run(build_bootstrap_command(workdir), cwd=str(workdir))


main()
