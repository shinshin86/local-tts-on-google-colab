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
| TinyTTS | Works | English |
| Voxtral-TTS | Unverified (GPU required, VRAM 16GB+) | English / French / Spanish and 9 languages |
| F5-TTS | Works (GPU required) | English / Chinese (Japanese via separate model) |
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

ENGINE = "Kokoro"  #@param ["Irodori-TTS", "Kokoro", "MeloTTS", "Piper", "Piper-Plus", "Qwen3-TTS", "Style-Bert-VITS2", "TinyTTS", "Voxtral-TTS"]
EXPOSE_PUBLIC_URL = True  #@param {type:"boolean"}
TEST_TEXT = "こんにちは。これは OpenAI 互換 TTS の動作確認です。"  #@param {type:"string"}
TEST_SPEED = 1.0  #@param {type:"number"}
TEST_VOICE = ""  #@param {type:"string"}
OPENAI_MODEL_ID = ""  #@param {type:"string"}

#@markdown ---
#@markdown Irodori-TTS
# To use V1: checkpoint="Aratako/Irodori-TTS-500M", codec_repo="facebook/dacvae-watermarked"
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

### TinyTTS

An ultra-lightweight English TTS using [ecyht2/tiny-tts](https://github.com/ecyht2/tiny-tts). The model has only 1.6M parameters (~3.4 MB), no GPU required, and can synthesize speech at 53× real-time on CPU alone. Audio is output at 44.1 kHz. There is no voice switching. License: Apache 2.0.

### Voxtral-TTS

A multilingual TTS using [mistralai/Voxtral-4B-TTS-2603](https://huggingface.co/mistralai/Voxtral-4B-TTS-2603). A 4B-parameter model supporting 9 languages: English, French, Spanish, German, Italian, Portuguese, Dutch, Arabic, and Hindi. It includes 20 preset voices and supports multiple formats such as wav / mp3 / flac / aac / opus. The backend uses vLLM + vllm-omni. A GPU runtime (VRAM 16GB or more) is required, and it may not work on Colab's free-tier T4 (15GB). License: CC BY-NC 4.0 (non-commercial only).

### F5-TTS

A zero-shot voice cloning TTS using [SWivid/F5-TTS](https://github.com/SWivid/F5-TTS). It mimics the voice quality of a reference audio to generate speech. Uses the default reference audio bundled with the package (English female). To use a Japanese model, specify a community-provided Japanese checkpoint with `--f5tts-ckpt-file` / `--f5tts-vocab-file`. A GPU runtime (T4 or higher) is required. License: code MIT / model CC-BY-NC.

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
| TinyTTS | Apache 2.0 | Apache 2.0 | OK | |
| Voxtral-TTS | — | CC BY-NC 4.0 | Not allowed | Via vLLM + vllm-omni. Non-commercial due to voice dataset license constraints |
| F5-TTS | MIT | CC-BY-NC | Not allowed (model) | Model weights are non-commercial due to Emilia dataset constraints |
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
- Voxtral-TTS
  https://huggingface.co/mistralai/Voxtral-4B-TTS-2603
- F5-TTS
  https://github.com/SWivid/F5-TTS
- Fish Speech
  https://github.com/fishaudio/fish-speech
- CosyVoice
  https://github.com/FunAudioLLM/CosyVoice
