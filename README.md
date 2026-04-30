# local-tts-on-google-colab

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
| Sarashina-TTS | Works (GPU required, ~6GB VRAM) | Japanese / English |
| F5-TTS | Works (GPU required) | English / Chinese (Japanese via separate model) |
| Chatterbox | Works (GPU recommended) | Japanese / English / Chinese and 23 languages |
| Zonos | Works (GPU required, ~6GB VRAM) | Japanese / English / Chinese / French / German |
| OuteTTS | Works (CPU OK) | Japanese / English / Chinese and many languages |
| Dia | Works (GPU recommended) | English (multi-speaker dialogue) |
| OpenVoice-V2 | At your own risk (MeloTTS deps) | Japanese / English / Spanish / French / Chinese / Korean |
| VibeVoice | Works (GPU required) - research-only license | English / Chinese (long-form, up to 4 speakers) |
| Fish-Speech | Not working | Japanese / English / Chinese and 80+ languages |
| MeloTTS | Not working | - |
| Style-Bert-VITS2 | Not working | - |
| CosyVoice2 | Not working | - |

`MeloTTS`, `Style-Bert-VITS2`, and `CosyVoice2` currently have dependency resolution issues under Colab's uv + venv environment and do not work.

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

ENGINE = "Kokoro"  #@param ["Chatterbox", "Dia", "F5-TTS", "Fish-Speech", "Irodori-TTS", "Kokoro", "MeloTTS", "MOSS-TTS-Nano", "NeuTTS", "OpenVoice-V2", "OuteTTS", "Piper", "Piper-Plus", "Qwen3-TTS", "Sarashina-TTS", "Style-Bert-VITS2", "TinyTTS", "VibeVoice", "VoxCPM2", "Voxtral-TTS", "Zonos"]
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

### OpenVoice-V2

A two-stage voice cloning TTS using [myshell-ai/OpenVoice](https://github.com/myshell-ai/OpenVoice) V2. The pipeline first synthesizes base speech with MeloTTS in the chosen language (EN / ES / FR / ZH / JP / KR), then applies a ToneColorConverter (V2 checkpoints) to match the timbre of a reference clip. Default language: `JP` (Japanese). The wrapper uses `resources/example_reference.mp3` from the upstream repo as the default reference; supplying `--openvoice-prompt-wav` enables a `clone` voice with your own reference audio. License: MIT (both code and weights, since April 2024) — commercial use is allowed.

**Caveat:** OpenVoice V2 depends on MeloTTS as the base TTS, which is what currently breaks the standalone `MeloTTS` engine in this repo (the `tokenizers` build needs a Rust toolchain that Colab's `uv + venv` setup does not always provide). OpenVoice V2 may fail at install time for the same reason. If MeloTTS install succeeds in your runtime, OpenVoice V2 should work.

The `voice` parameter exposes:

| voice | description |
|---|---|
| `default` | Uses `resources/example_reference.mp3` shipped in the OpenVoice repo as the timbre reference. |
| `clone` | Uses `--openvoice-prompt-wav` as the timbre reference. Only available when configured. |

For voice cloning, only use reference audio you have rights to (consent of the speaker).

### VibeVoice

A long-form multi-speaker TTS using [microsoft/VibeVoice](https://github.com/microsoft/VibeVoice). The 1.5B-parameter model can generate up to ~90 minutes of audio with up to 4 distinct speakers in a single pass — useful for podcasts and long dialogues. Each speaker conditions on a short reference clip. The wrapper resolves the default reference from `demo/voices/<VIBEVOICE_DEFAULT_SPEAKER>.wav` shipped in the upstream repo (e.g., `en-Alice_woman`); supplying `--vibevoice-prompt-wav` enables a `clone` voice. Use `Speaker 1: ... \n Speaker 2: ...` formatting in `input` for multi-speaker dialogues; the wrapper auto-prepends `Speaker 1:` for plain text. A GPU runtime is required (bf16). Tunable: `--vibevoice-ddpm-steps` (default 10) and `--vibevoice-cfg-scale` (default 1.3).

**License caveat:** the code and weights are nominally MIT, **but** Microsoft tags VibeVoice as **"research purpose only"** on the model card and explicitly prohibits non-EN/ZH languages, voice impersonation, disinformation, and real-time voice conversion. Treat this engine as research / evaluation only — do not ship into commercial or real-world products.

The `voice` parameter exposes:

| voice | description |
|---|---|
| `default` | Uses the bundled `demo/voices/<--vibevoice-default-speaker>.wav` reference. |
| `clone` | Uses `--vibevoice-prompt-wav` as the reference. Only available when configured. |

For voice cloning, only use reference audio you have rights to (consent of the speaker).

### Fish-Speech (currently not working)

A high-quality TTS using [fishaudio/fish-speech](https://github.com/fishaudio/fish-speech). Japanese is Tier 1 supported (highest quality) and it supports 80+ languages. It requires 24GB+ VRAM and targets A100/L4 GPUs, but on Colab the runtime crashes with OOM during model loading, so it currently does not work. License: Apache 2.0.

### CosyVoice2 (currently not working)

Intended to use [FunAudioLLM/CosyVoice](https://github.com/FunAudioLLM/CosyVoice), but dependent packages (openai-whisper, onnxruntime-gpu, grpcio, deepspeed, lightning, etc.) do not support Python 3.12+, so setup does not complete in Colab's uv + venv environment.

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
| Sarashina-TTS | — | Sarashina Model NonCommercial License | Not allowed | Japanese / English. Zero-shot voice cloning. Output contains a SilentCipher watermark (do not remove) |
| F5-TTS | MIT | CC-BY-NC | Not allowed (model) | Model weights are non-commercial due to Emilia dataset constraints |
| Chatterbox | MIT | MIT | OK | Multilingual (23 languages incl JP). Zero-shot voice cloning |
| Zonos | Apache 2.0 | Apache 2.0 | OK | EN/JA/ZH/FR/DE. Zero-shot voice cloning. Requires `espeak-ng` |
| OuteTTS (0.6B) | Apache 2.0 | Apache 2.0 | OK | Multilingual incl JP. CPU OK. Voice cloning |
| OuteTTS (1B)   | Apache 2.0 | CC-BY-NC-SA-4.0 + Llama 3.2 Community License | Not allowed | Llama-3.2-based; non-commercial weights |
| Dia | Apache 2.0 | Apache 2.0 | OK | EN only. Multi-speaker `[S1]`/`[S2]` dialogue TTS |
| OpenVoice-V2 | MIT | MIT | OK | Multilingual (incl JP). Voice cloning. Depends on MeloTTS (may not install) |
| VibeVoice | MIT | MIT | Caution (research-only) | EN/ZH only per model card. Microsoft tags this "research purpose only" |
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
- OpenVoice
  https://github.com/myshell-ai/OpenVoice
- VibeVoice
  https://github.com/microsoft/VibeVoice
- Fish Speech
  https://github.com/fishaudio/fish-speech
- CosyVoice
  https://github.com/FunAudioLLM/CosyVoice
