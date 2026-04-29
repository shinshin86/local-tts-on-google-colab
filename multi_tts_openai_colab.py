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

ENGINE = "Kokoro"  #@param ["Chatterbox", "F5-TTS", "Fish-Speech", "Irodori-TTS", "Kokoro", "MeloTTS", "MOSS-TTS-Nano", "NeuTTS", "OuteTTS", "Piper", "Piper-Plus", "Qwen3-TTS", "Sarashina-TTS", "Style-Bert-VITS2", "TinyTTS", "VoxCPM2", "Voxtral-TTS", "Zonos"]
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
#@markdown OuteTTS (CPU OK, multilingual incl JP, voice cloning, Apache 2.0)
OUTETTS_MODEL_SIZE = "0.6B"  #@param ["0.6B", "1B"]
OUTETTS_BACKEND = "HF"  #@param ["HF", "LLAMACPP"]
OUTETTS_DEFAULT_SPEAKER = "EN-FEMALE-1-NEUTRAL"  #@param {type:"string"}
OUTETTS_PROMPT_WAV = ""  #@param {type:"string"}
OUTETTS_PROMPT_TEXT = ""  #@param {type:"string"}
OUTETTS_DEFAULT_VOICE = "default"  #@param ["default", "clone"]

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
