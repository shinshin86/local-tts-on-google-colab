from .chatterbox import install as install_chatterbox
from .cosyvoice2 import install as install_cosyvoice2
from .dia import install as install_dia
from .f5tts import install as install_f5tts
from .fish_speech import install as install_fish_speech
from .irodori import install as install_irodori
from .qwen3_tts import install as install_qwen3_tts
from .kokoro import install as install_kokoro
from .kyutai_tts import install as install_kyutai_tts
from .melo import install as install_melo
from .moss_tts_nano import install as install_moss_tts_nano
from .neutts import install as install_neutts
from .openvoice_v2 import install as install_openvoice_v2
from .orpheus import install as install_orpheus
from .outetts import install as install_outetts
from .piper import install as install_piper
from .piper_plus import install as install_piper_plus
from .pocket_tts import install as install_pocket_tts
from .sarashina_tts import install as install_sarashina_tts
from .style_bert import install as install_style_bert
from .vibevoice import install as install_vibevoice
from .voxcpm import install as install_voxcpm
from .tiny_tts import install as install_tiny_tts
from .voxtral import install as install_voxtral
from .zonos import install as install_zonos


INSTALLERS = {
    "Chatterbox": install_chatterbox,
    "CosyVoice2": install_cosyvoice2,
    "Dia": install_dia,
    "F5-TTS": install_f5tts,
    "Fish-Speech": install_fish_speech,
    "Irodori-TTS": install_irodori,
    "Kokoro": install_kokoro,
    "Kyutai-TTS": install_kyutai_tts,
    "MeloTTS": install_melo,
    "MOSS-TTS-Nano": install_moss_tts_nano,
    "NeuTTS": install_neutts,
    "OpenVoice-V2": install_openvoice_v2,
    "Orpheus-TTS": install_orpheus,
    "OuteTTS": install_outetts,
    "Style-Bert-VITS2": install_style_bert,
    "Piper": install_piper,
    "Piper-Plus": install_piper_plus,
    "Pocket-TTS": install_pocket_tts,
    "Qwen3-TTS": install_qwen3_tts,
    "Sarashina-TTS": install_sarashina_tts,
    "TinyTTS": install_tiny_tts,
    "VibeVoice": install_vibevoice,
    "VoxCPM2": install_voxcpm,
    "Voxtral-TTS": install_voxtral,
    "Zonos": install_zonos,
}
