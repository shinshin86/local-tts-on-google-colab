from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


KOKORO_VOICE_PRESETS = [
    "jf_alpha",
    "jf_gongitsune",
    "jm_kumo",
    "af_heart",
    "af_bella",
    "am_adam",
    "bf_emma",
    "bm_george",
    "zf_xiaobei",
]

KOKORO_ONNX_VOICE_PRESETS = [
    "af_heart",
    "af_bella",
    "am_adam",
    "am_michael",
    "bf_emma",
    "bm_george",
    "jf_alpha",
    "jf_gongitsune",
    "jm_kumo",
    "zf_xiaobei",
    "zm_yunjian",
    "ef_dora",
    "ff_siwis",
    "hf_alpha",
    "if_sara",
    "pf_dora",
]

NEUTTS_VOICE_PRESETS = ["dave", "jo", "greta", "juliette", "mateo"]

MELO_VOICE_PRESETS = [
    "JP",
    "EN-Default",
    "EN-US",
    "EN-BR",
    "EN_INDIA",
    "EN-AU",
    "ZH",
    "ES",
    "FR",
    "KR",
]


def default_repo_dir() -> Path:
    return Path(__file__).resolve().parent.parent


@dataclass
class Settings:
    engine: str
    expose_public_url: bool = True
    dry_run: bool = False
    test_text: str = "こんにちは。これは OpenAI 互換 TTS の動作確認です。"
    test_speed: float = 1.0
    test_voice: str = ""
    openai_model_id: str = ""
    f5tts_model: str = "F5TTS_v1_Base"
    fish_speech_model: str = "fishaudio/s2-pro"
    f5tts_ckpt_file: str = ""
    f5tts_vocab_file: str = ""
    # V1: "Aratako/Irodori-TTS-500M", V2: "Aratako/Irodori-TTS-500M-v2"
    irodori_hf_checkpoint: str = "Aratako/Irodori-TTS-500M-v3"
    # V1を利用する場合: "facebook/dacvae-watermarked"
    irodori_codec_repo: str = "Aratako/Semantic-DACVAE-Japanese-32dim"
    irodori_model_precision: str = "fp32"
    irodori_codec_precision: str = "fp32"
    # Irodori-TTS-Lite: int4-quantized runtime that monkey-patches the upstream Irodori-TTS.
    # Default: voice-design int4 (no Duration Predictor; seconds derived from text).
    # Switch to "kizuna-intelligence/Irodori-TTS-500M-v3-int4" with checkpoint_file="model.safetensors"
    # to use the v3 int4 (with Duration Predictor).
    irodori_lite_hf_checkpoint: str = "kizuna-intelligence/Irodori-TTS-Lite-int4"
    irodori_lite_checkpoint_file: str = "dit_int4.safetensors"
    irodori_lite_codec_repo: str = "Aratako/Semantic-DACVAE-Japanese-32dim"
    irodori_lite_codec_int4: bool = False
    kokoro_default_voice: str = "jf_alpha"
    kokoro_default_lang_code: str = "j"
    kokoro_onnx_hf_model: str = "nvidia/kokoro-82M-onnx-opt"
    kokoro_onnx_default_voice: str = "jf_alpha"
    kokoro_onnx_default_lang_code: str = "j"
    kokoro_onnx_provider: str = "auto"
    kyutai_hf_repo: str = "kyutai/tts-1.6b-en_fr"
    kyutai_voice_repo: str = "kyutai/tts-voices"
    kyutai_voice: str = "expresso/ex03-ex01_happy_001_channel1_334s.wav"
    kyutai_prompt_wav: str = ""
    kyutai_default_voice: str = "default"
    pocket_language: str = "english"
    pocket_default_speaker: str = "alba"
    pocket_prompt_wav: str = ""
    pocket_default_voice: str = "default"
    melo_language: str = "JP"
    melo_default_voice: str = "JP"
    style_bert_model_repo: str = "litagin/style_bert_vits2_jvnv"
    style_bert_model_subdir: str = "jvnv-F2-jp"
    style_bert_model_name: str = "jvnv-F2-jp"
    style_bert_speaker_id: int = 0
    style_bert_style: str = "Neutral"
    piper_voice: str = "en_US-lessac-medium"
    piper_speaker_id: int = -1
    piper_plus_model: str = "tsukuyomi"
    qwen3_hf_model: str = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
    qwen3_language: str = "Japanese"
    qwen3_default_speaker: str = "ono_anna"
    moss_tts_nano_hf_model: str = "OpenMOSS-Team/MOSS-TTS-Nano-100M"
    moss_tts_nano_mode: str = "continuation"
    moss_tts_v1_5_hf_model: str = "OpenMOSS-Team/MOSS-TTS-v1.5"
    moss_tts_v1_5_language: str = "Japanese"
    moss_tts_v1_5_prompt_wav: str = ""
    moss_tts_v1_5_default_voice: str = "default"
    moss_tts_v1_5_attn_impl: str = "sdpa"
    moss_tts_v1_5_max_new_tokens: int = 4096
    moss_local_v1_5_hf_model: str = "OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5"
    moss_local_v1_5_language: str = "Japanese"
    moss_local_v1_5_prompt_wav: str = ""
    moss_local_v1_5_default_voice: str = "default"
    moss_local_v1_5_attn_impl: str = "sdpa"
    moss_local_v1_5_max_new_tokens: int = 4096
    neutts_backbone_repo: str = "neuphonic/neutts-air"
    neutts_codec_repo: str = "neuphonic/neucodec"
    neutts_default_voice: str = "jo"
    voxcpm_hf_model: str = "openbmb/VoxCPM2"
    voxcpm_cfg_value: float = 2.0
    voxcpm_inference_timesteps: int = 10
    voxtral_hf_model: str = "mistralai/Voxtral-4B-TTS-2603"
    voxtral_default_voice: str = "neutral_female"
    voxtral_backend_port: int = 5001
    sarashina_hf_model: str = "sbintuitions/sarashina2.2-tts"
    sarashina_use_vllm: bool = False
    sarashina_prompt_wav: str = ""
    sarashina_prompt_text: str = ""
    sarashina_default_voice: str = "default"
    chatterbox_language: str = "ja"
    chatterbox_prompt_wav: str = ""
    chatterbox_default_voice: str = "default"
    zonos_hf_model: str = "Zyphra/Zonos-v0.1-transformer"
    zonos_language: str = "ja"
    zonos_prompt_wav: str = ""
    zonos_default_voice: str = "default"
    # ZONOS2 (Zyphra): MoE backbone + DAC tokens + ECAPA-TDNN embedding served
    # by the bundled Mini-SGLang server (GPU sm_80+ required: flashinfer /
    # sgl_kernel / cutlass kernels). Multilingual (tier-1: en/zh/ja). We launch
    # `uv run python -m zonos2` as a backend and proxy its /tts/generate
    # (44.1 kHz float32 PCM). default = a shipped reference voice; clone =
    # --zonos2-prompt-wav. Code MIT (pyproject), weights Apache-2.0 (HF card).
    zonos2_hf_model: str = "Zyphra/ZONOS2"
    zonos2_language: str = "ja"
    zonos2_prompt_wav: str = ""
    zonos2_default_voice: str = "default"
    # Bare filename → looked up inside the cloned repo's default_voices/.
    zonos2_default_ref: str = "AmericanFemale.mp3"
    zonos2_accurate_mode: bool = True
    # >= 0 forces a fixed seed for reproducible output; -1 leaves it random.
    zonos2_seed: int = -1
    zonos2_backend_port: int = 5003
    outetts_model_size: str = "0.6B"
    outetts_backend: str = "HF"
    outetts_default_speaker: str = "EN-FEMALE-1-NEUTRAL"
    outetts_prompt_wav: str = ""
    outetts_prompt_text: str = ""
    outetts_default_voice: str = "default"
    dia_hf_model: str = "nari-labs/Dia-1.6B-0626"
    dia_compute_dtype: str = "float16"
    dia_prompt_wav: str = ""
    dia_prompt_text: str = ""
    dia_default_voice: str = "default"
    openvoice_language: str = "JP"
    openvoice_prompt_wav: str = ""
    openvoice_default_voice: str = "default"
    orpheus_hf_model: str = "canopylabs/orpheus-tts-0.1-finetune-prod"
    orpheus_default_voice: str = "tara"
    orpheus_max_model_len: int = 2048
    cosyvoice_hf_model: str = "FunAudioLLM/CosyVoice2-0.5B"
    cosyvoice_prompt_wav: str = ""
    cosyvoice_prompt_text: str = ""
    cosyvoice_default_voice: str = "default"
    spark_hf_model: str = "SparkAudio/Spark-TTS-0.5B"
    spark_default_voice: str = "default"
    spark_default_gender: str = "female"
    spark_default_pitch: str = "moderate"
    spark_default_speed: str = "moderate"
    spark_prompt_wav: str = ""
    spark_prompt_text: str = ""
    vibevoice_hf_model: str = "microsoft/VibeVoice-1.5B"
    vibevoice_default_speaker: str = "en-Alice_woman"
    vibevoice_prompt_wav: str = ""
    vibevoice_default_voice: str = "default"
    vibevoice_ddpm_steps: int = 10
    vibevoice_cfg_scale: float = 1.3
    bark_default_voice: str = "v2/en_speaker_6"
    bark_use_small_models: bool = False
    chattts_default_voice: str = "default"
    chattts_seed: int = 2
    chattts_temperature: float = 0.3
    csm_hf_model: str = "sesame/csm-1b"
    csm_llama_model: str = "meta-llama/Llama-3.2-1B"
    csm_default_voice: str = "default"
    csm_default_speaker: int = 0
    csm_max_audio_length_ms: int = 10000
    csm_temperature: float = 0.9
    misotts_hf_model: str = "MisoLabs/MisoTTS"
    misotts_default_voice: str = "default"
    misotts_default_speaker: int = 0
    misotts_prompt_wav: str = ""
    misotts_prompt_text: str = ""
    misotts_max_audio_length_ms: int = 30000
    misotts_temperature: float = 0.9
    misotts_topk: int = 50
    # Ungated mirror of the Llama 3.2 tokenizer (byte-identical to the gated
    # meta-llama/Llama-3.2-1B). Set to "meta-llama/Llama-3.2-1B" (+ HF_TOKEN) for the official source.
    misotts_tokenizer_repo: str = "unsloth/Llama-3.2-1B"
    # MioTTS (Aratako): LLM-based TTS. A Qwen3-1.7B-Base backbone autoregressively
    # generates MioCodec tokens (25 Hz) that decode to 44.1 kHz audio. Triple
    # process: a llama-cpp-python OpenAI server (prebuilt CUDA wheel) hosts the
    # GGUF, run_server.py is the synthesis backend, and our app.py proxies
    # /v1/audio/speech -> /v1/tts. default = a shipped preset; one of the preset
    # names selects it directly; clone = --miotts-prompt-wav reference audio.
    # Code MIT, MioTTS-1.7B weights Apache-2.0, MioCodec Apache-2.0. NOTE: the
    # bundled default presets are derived from T5Gemma-TTS / Gemini 2.5 Pro TTS
    # and CANNOT be used commercially — clone your own voice for commercial use.
    miotts_gguf_repo: str = "Aratako/MioTTS-GGUF"
    miotts_gguf_file: str = "MioTTS-1.7B-Q8_0.gguf"
    miotts_codec_model: str = "Aratako/MioCodec-25Hz-44.1kHz-v2"
    # llama-cpp-python CUDA wheel index tag (cu121..cu125). Match Colab's CUDA.
    miotts_llama_cuda: str = "cu124"
    miotts_n_ctx: int = 8192
    miotts_default_preset: str = "jp_female"
    miotts_prompt_wav: str = ""
    miotts_default_voice: str = "default"
    # Upstream DefaultLLMParams (miotts_server/config.py).
    miotts_temperature: float = 0.8
    miotts_top_p: float = 1.0
    miotts_repetition_penalty: float = 1.0
    miotts_max_tokens: int = 700
    miotts_llm_port: int = 5004
    miotts_backend_port: int = 5005
    styletts2_default_voice: str = "default"
    styletts2_prompt_wav: str = ""
    styletts2_alpha: float = 0.3
    styletts2_beta: float = 0.7
    styletts2_diffusion_steps: int = 5
    styletts2_embedding_scale: float = 1.0
    maskgct_default_voice: str = "default"
    maskgct_prompt_wav: str = ""
    maskgct_prompt_text: str = ""
    maskgct_prompt_lang: str = "en"
    maskgct_target_lang: str = "en"
    gpt_sovits_version: str = "v2"
    gpt_sovits_default_voice: str = "default"
    gpt_sovits_prompt_wav: str = ""
    gpt_sovits_prompt_text: str = ""
    gpt_sovits_prompt_lang: str = "en"
    gpt_sovits_target_lang: str = "en"
    higgs_hf_model: str = "bosonai/higgs-audio-v2-generation-3B-base"
    higgs_hf_tokenizer: str = "bosonai/higgs-audio-v2-tokenizer"
    higgs_default_voice: str = "default"
    higgs_default_ref_voice: str = "belinda"
    higgs_prompt_wav: str = ""
    higgs_prompt_text: str = ""
    higgs_max_new_tokens: int = 1024
    higgs_temperature: float = 0.7
    # Higgs Audio v3: a separate 4B chat-native TTS served by SGLang-Omni
    # (Qwen3-4B backbone). Weights are ungated (no HF_TOKEN). Research /
    # non-commercial license only — see the README license table.
    higgs_v3_hf_model: str = "bosonai/higgs-audio-v3-tts-4b"
    higgs_v3_default_voice: str = "default"
    higgs_v3_prompt_wav: str = ""
    higgs_v3_prompt_text: str = ""
    higgs_v3_temperature: float = 0.7
    higgs_v3_top_k: int = 50
    higgs_v3_max_new_tokens: int = 2048
    higgs_v3_backend_port: int = 5002
    supertonic_model: str = "supertonic-3"
    supertonic_default_voice: str = "M1"
    supertonic_default_lang: str = "en"
    supertonic_total_steps: int = 5
    dramabox_hf_model: str = "ResembleAI/Dramabox"
    dramabox_gemma_repo: str = "unsloth/gemma-3-12b-it-bnb-4bit"
    dramabox_default_voice: str = "default"
    dramabox_default_ref_voice: str = "female_american"
    dramabox_prompt_wav: str = ""
    dramabox_dtype: str = "bf16"
    dramabox_cfg_scale: float = 2.5
    dramabox_stg_scale: float = 1.5
    dramabox_duration_multiplier: float = 1.1
    dramabox_seed: int = 42
    dramabox_compile: bool = False
    dramabox_bnb_4bit: bool = True
    scenema_default_voice: str = "default"
    scenema_default_gender: str = "male"
    scenema_prompt_wav: str = ""
    scenema_gemma_quantize: str = "nf4"
    scenema_seed: int = -1
    scenema_pace: float = 1.5
    scenema_validate: bool = True
    scenema_min_match_ratio: float = 0.90
    scenema_skip_vc: bool = False
    scenema_vc_steps: int = 25
    scenema_vc_cfg_rate: float = 0.5
    scenema_background_sfx: bool = False
    # dots.tts (rednote-hilab): 2B fully continuous AR TTS, 24 languages incl.
    # Japanese, Apache-2.0 for both code and weights. Fundamentally a zero-shot
    # cloning model: `default` = no reference (random-voice sampling), `clone` =
    # reference audio (+ optional transcript for continuation cloning).
    dots_tts_hf_model: str = "rednote-hilab/dots.tts-base"
    dots_tts_default_voice: str = "default"
    dots_tts_prompt_wav: str = ""
    dots_tts_prompt_text: str = ""
    dots_tts_language: str = "auto_detect"
    dots_tts_precision: str = "bfloat16"
    dots_tts_num_steps: int = 10
    dots_tts_guidance_scale: float = 1.2
    dots_tts_speaker_scale: float = 1.5
    dots_tts_max_generate_length: int = 500
    # LFM2.5-Audio-JP (LiquidAI): 1.5B end-to-end speech-text LM (speech-to-speech
    # / ASR / TTS), Japanese-only. Single built-in Japanese voice via system
    # prompt, no reference/cloning. Weights ungated (no HF_TOKEN). LFM Open
    # License v1.0 (commercial OK under $10M annual revenue).
    lfm2_audio_jp_hf_model: str = "LiquidAI/LFM2.5-Audio-1.5B-JP"
    lfm2_audio_jp_system_prompt: str = "Perform TTS in japanese."
    lfm2_audio_jp_max_new_tokens: int = 1024
    lfm2_audio_jp_audio_temperature: float = 0.8
    lfm2_audio_jp_audio_top_k: int = 64
    # Ming-omni-TTS (inclusionAI): 16.8B-A3B MoE audio LM (~3B active) with a
    # 12.5 Hz continuous tokenizer + DiT head. Runs in-process. `default` = the
    # built-in voice (zero speaker-embedding, no reference); `clone` = zero-shot
    # cloning from a reference wav (+ optional transcript). Code MIT (GitHub),
    # weights Apache-2.0 (HF card). ~34GB of weights → A100 40GB required.
    ming_omni_tts_hf_model: str = "inclusionAI/Ming-omni-tts-16.8B-A3B"
    ming_omni_tts_default_voice: str = "default"
    ming_omni_tts_prompt_wav: str = ""
    ming_omni_tts_prompt_text: str = ""
    ming_omni_tts_gen_prompt: str = "Please generate speech based on the following description.\n"
    ming_omni_tts_max_decode_steps: int = 200
    ming_omni_tts_cfg: float = 2.0
    ming_omni_tts_sigma: float = 0.25
    ming_omni_tts_temperature: float = 0.0
    # Prompt-driven control (all optional; empty = current behavior). task picks
    # what to generate (speech / music / tta); style/emotion/dialect are
    # natural-language voice design mapped to the instruction keys 风格/情感/方言.
    ming_omni_tts_task: str = "speech"
    ming_omni_tts_style: str = ""
    ming_omni_tts_emotion: str = ""
    ming_omni_tts_dialect: str = ""
    root_dir: Path = Path("/content/openai-compatible-local-tts")
    repo_dir: Path = field(default_factory=default_repo_dir)
    app_port: int = 8000
    piper_backend_port: int = 5000

    @property
    def engines_dir(self) -> Path:
        return self.root_dir / "engines"

    @property
    def log_dir(self) -> Path:
        return self.root_dir / "logs"

    @property
    def output_dir(self) -> Path:
        return self.root_dir / "outputs"

    @property
    def cloudflared_path(self) -> Path:
        return self.root_dir / "cloudflared"

    def ensure_directories(self) -> None:
        for path in (self.root_dir, self.engines_dir, self.log_dir, self.output_dir):
            path.mkdir(parents=True, exist_ok=True)

    def read_repo_text(self, relative_path: str) -> str:
        source_path = self.repo_dir / relative_path
        if not source_path.exists():
            raise FileNotFoundError(
                f"Required source file not found: {source_path}. "
                "Run this script from a cloned local-tts-on-google-colab repository."
            )
        return source_path.read_text(encoding="utf-8")
