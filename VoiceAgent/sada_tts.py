import os
import re
import sys

# Add the TTS project to sys.path so its internal packages (models, text, etc.) are importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tts-arabic-pytorch-master'))

from utils.app_utils import TTSManager

# Only keep Arabic script characters and spaces — the TTS model
# crashes on periods, commas, digits, Latin letters, etc.
_ARABIC_ONLY = re.compile(
    r'[^\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF'
    r'\uFB50-\uFDFF\uFE70-\uFEFF\s]+'
)


class SadaTTS:
    def __init__(self, out_dir='./app/static', use_cuda=True):
        """
        Initializes the TTS Engine. 
        Kareem will initialize this ONCE in his agent script so the 
        model weights stay loaded in the GPU memory.
        """
        print("[SADA-TTS] Initializing engine and loading acoustic models...")
        
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        self.engine = TTSManager(out_dir=out_dir, use_cuda_if_available=use_cuda)
        self.out_dir = out_dir
        
        print("[SADA-TTS] Ready for inference!")

    def speak(self, text, speed=1.0, denoise=0.01):
        """
        Generates speech and returns the file path of the saved audio.
        Strips non-Arabic characters that the acoustic model can't handle.
        """
        # Sanitize: keep only Arabic + whitespace
        clean = _ARABIC_ONLY.sub(' ', text).strip()
        clean = re.sub(r'\s+', ' ', clean)  # collapse whitespace

        if not clean:
            print(f"[SADA-TTS] Skipping empty text after sanitization")
            return None

        print(f"[SADA-TTS] Synthesizing speech...")
        
        response_data = self.engine.tts(clean, speed=speed, denoise=denoise)

        model_id = response_data[0]['id']
        audio_filepath = os.path.join(self.out_dir, f"wave{model_id}.wav")
        
        print(f"[SADA-TTS] Audio generated: {audio_filepath}")
        
        return audio_filepath