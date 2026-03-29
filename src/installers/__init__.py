from .irodori import install as install_irodori
from .qwen3_tts import install as install_qwen3_tts
from .kokoro import install as install_kokoro
from .melo import install as install_melo
from .piper import install as install_piper
from .style_bert import install as install_style_bert


INSTALLERS = {
    "Irodori-TTS": install_irodori,
    "Kokoro": install_kokoro,
    "MeloTTS": install_melo,
    "Style-Bert-VITS2": install_style_bert,
    "Piper": install_piper,
    "Qwen3-TTS": install_qwen3_tts,
}
