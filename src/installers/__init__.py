from .bark import install as install_bark
from .chatterbox import install as install_chatterbox
from .chattts import install as install_chattts
from .cosyvoice2 import install as install_cosyvoice2
from .csm import install as install_csm
from .dia import install as install_dia
from .dramabox import install as install_dramabox
from .f5tts import install as install_f5tts
from .fish_speech import install as install_fish_speech
from .gpt_sovits import install as install_gpt_sovits
from .higgs_audio import install as install_higgs_audio
from .irodori import install as install_irodori
from .irodori_lite import install as install_irodori_lite
from .qwen3_tts import install as install_qwen3_tts
from .kokoro import install as install_kokoro
from .kokoro_onnx import install as install_kokoro_onnx
from .kyutai_tts import install as install_kyutai_tts
from .maskgct import install as install_maskgct
from .melo import install as install_melo
from .moss_tts_nano import install as install_moss_tts_nano
from .moss_tts_v1_5 import install as install_moss_tts_v1_5
from .neutts import install as install_neutts
from .openvoice_v2 import install as install_openvoice_v2
from .orpheus import install as install_orpheus
from .outetts import install as install_outetts
from .piper import install as install_piper
from .piper_plus import install as install_piper_plus
from .pocket_tts import install as install_pocket_tts
from .sarashina_tts import install as install_sarashina_tts
from .scenema import install as install_scenema
from .spark_tts import install as install_spark_tts
from .style_bert import install as install_style_bert
from .styletts2 import install as install_styletts2
from .supertonic import install as install_supertonic
from .vibevoice import install as install_vibevoice
from .voxcpm import install as install_voxcpm
from .tiny_tts import install as install_tiny_tts
from .voxtral import install as install_voxtral
from .zonos import install as install_zonos


INSTALLERS = {
    "Bark": install_bark,
    "ChatTTS": install_chattts,
    "Chatterbox": install_chatterbox,
    "CosyVoice2": install_cosyvoice2,
    "CSM-1B": install_csm,
    "Dia": install_dia,
    "DramaBox": install_dramabox,
    "F5-TTS": install_f5tts,
    "Fish-Speech": install_fish_speech,
    "GPT-SoVITS": install_gpt_sovits,
    "Higgs-Audio-v2": install_higgs_audio,
    "Irodori-TTS": install_irodori,
    "Irodori-TTS-Lite": install_irodori_lite,
    "Kokoro": install_kokoro,
    "Kokoro-ONNX": install_kokoro_onnx,
    "Kyutai-TTS": install_kyutai_tts,
    "MaskGCT": install_maskgct,
    "MeloTTS": install_melo,
    "MOSS-TTS-Nano": install_moss_tts_nano,
    "MOSS-TTS-v1.5": install_moss_tts_v1_5,
    "NeuTTS": install_neutts,
    "OpenVoice-V2": install_openvoice_v2,
    "Orpheus-TTS": install_orpheus,
    "OuteTTS": install_outetts,
    "Style-Bert-VITS2": install_style_bert,
    "StyleTTS2": install_styletts2,
    "Supertonic": install_supertonic,
    "Piper": install_piper,
    "Piper-Plus": install_piper_plus,
    "Pocket-TTS": install_pocket_tts,
    "Qwen3-TTS": install_qwen3_tts,
    "Sarashina-TTS": install_sarashina_tts,
    "Scenema": install_scenema,
    "Spark-TTS": install_spark_tts,
    "TinyTTS": install_tiny_tts,
    "VibeVoice": install_vibevoice,
    "VoxCPM2": install_voxcpm,
    "Voxtral-TTS": install_voxtral,
    "Zonos": install_zonos,
}
