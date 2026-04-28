from .f5tts import install as install_f5tts
from .fish_speech import install as install_fish_speech
from .irodori import install as install_irodori
from .qwen3_tts import install as install_qwen3_tts
from .kokoro import install as install_kokoro
from .melo import install as install_melo
from .moss_tts_nano import install as install_moss_tts_nano
from .neutts import install as install_neutts
from .piper import install as install_piper
from .piper_plus import install as install_piper_plus
from .sarashina_tts import install as install_sarashina_tts
from .style_bert import install as install_style_bert
from .voxcpm import install as install_voxcpm
from .tiny_tts import install as install_tiny_tts
from .voxtral import install as install_voxtral


INSTALLERS = {
    "F5-TTS": install_f5tts,
    "Fish-Speech": install_fish_speech,
    "Irodori-TTS": install_irodori,
    "Kokoro": install_kokoro,
    "MeloTTS": install_melo,
    "MOSS-TTS-Nano": install_moss_tts_nano,
    "NeuTTS": install_neutts,
    "Style-Bert-VITS2": install_style_bert,
    "Piper": install_piper,
    "Piper-Plus": install_piper_plus,
    "Qwen3-TTS": install_qwen3_tts,
    "Sarashina-TTS": install_sarashina_tts,
    "TinyTTS": install_tiny_tts,
    "VoxCPM2": install_voxcpm,
    "Voxtral-TTS": install_voxtral,
}
