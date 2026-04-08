from .cosyvoice import install as install_cosyvoice
from .f5tts import install as install_f5tts
from .fish_speech import install as install_fish_speech
from .irodori import install as install_irodori
from .qwen3_tts import install as install_qwen3_tts
from .kokoro import install as install_kokoro
from .melo import install as install_melo
from .piper import install as install_piper
from .piper_plus import install as install_piper_plus
from .style_bert import install as install_style_bert
from .voxcpm import install as install_voxcpm
from .voxtral import install as install_voxtral


INSTALLERS = {
    "CosyVoice2": install_cosyvoice,
    "F5-TTS": install_f5tts,
    "Fish-Speech": install_fish_speech,
    "Irodori-TTS": install_irodori,
    "Kokoro": install_kokoro,
    "MeloTTS": install_melo,
    "Style-Bert-VITS2": install_style_bert,
    "Piper": install_piper,
    "Piper-Plus": install_piper_plus,
    "Qwen3-TTS": install_qwen3_tts,
    "VoxCPM2": install_voxcpm,
    "Voxtral-TTS": install_voxtral,
}
