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

ENGINE = "Kokoro"  #@param ["Chatterbox", "CosyVoice2", "Dia", "F5-TTS", "Fish-Speech", "Irodori-TTS", "Kokoro", "Kyutai-TTS", "MeloTTS", "MOSS-TTS-Nano", "NeuTTS", "OpenVoice-V2", "Orpheus-TTS", "OuteTTS", "Piper", "Piper-Plus", "Pocket-TTS", "Qwen3-TTS", "Sarashina-TTS", "Spark-TTS", "Style-Bert-VITS2", "TinyTTS", "VibeVoice", "VoxCPM2", "Voxtral-TTS", "Zonos"]
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
    ]
    if SARASHINA_USE_VLLM:
        cmd.append("--sarashina-use-vllm")
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

A Japanese TTS using [Aratako/Irodori-TTS](https://github.com/Aratako/Irodori-TTS). By default it uses the Hugging Face model `Aratako/Irodori-TTS-500M-v2` (to use V1, change it to `Aratako/Irodori-TTS-500M`). Output is high-quality 48 kHz, but there is no voice switching.

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

### MeloTTS (currently not working)

Intended to use [myshell-ai/MeloTTS](https://github.com/myshell-ai/MeloTTS), but the dependency `tokenizers` requires a Rust compiler to build, so installation fails in the current Colab environment.

### Style-Bert-VITS2 (currently not working)

Intended to use [litagin02/Style-Bert-VITS2](https://github.com/litagin02/Style-Bert-VITS2), but `setuptools` / `torch` / `scipy` dependency conflicts cannot be resolved, and speech synthesis is not reachable in the current Colab environment.

## License

The license for each engine is as follows. When using them, always check each project's latest license terms.

| Engine | Code | Model Weights | Commercial Use | Notes |
|---|---|---|---|---|
| Kokoro | Apache 2.0 | Apache 2.0 | OK | |
| Irodori-TTS | MIT | MIT | OK | Ethical policy prohibits impersonation / deepfake generation |
| Piper | GPL-3.0 | MIT | Caution | The default voice `en_US-lessac-medium` is trained on the Blizzard 2013 dataset (Lessac Technologies), which is research-only and prohibits commercial use |
| Piper-Plus | MIT | MIT | OK | |
| Qwen3-TTS | Apache 2.0 | Apache 2.0 | OK | |
| VoxCPM2 | Apache 2.0 | Apache 2.0 | OK | |
| MOSS-TTS-Nano | Apache 2.0 | Apache 2.0 | OK | 100M params, CPU OK |
| NeuTTS | Apache 2.0 | Apache 2.0 (Air) / NeuTTS Open License 1.0 (Nano) | OK (Air) / Check terms (Nano) | Voice cloning. EN / ES / DE / FR |
| TinyTTS | Apache 2.0 | Apache 2.0 | OK | |
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
