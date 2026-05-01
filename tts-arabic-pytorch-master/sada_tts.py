import os
from utils.app_utils import TTSManager

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

    def speak(self, text_buckw, speed=1.0, denoise=0.01):
        """
        Generates speech and returns the file path of the saved audio.
        """
        print(f"[SADA-TTS] Synthesizing speech...")
        
        response_data = self.engine.tts(text_buckw, speed=speed, denoise=denoise)
        

        model_id = response_data[0]['id']
        audio_filepath = os.path.join(self.out_dir, f"wave{model_id}.wav")
        
        print(f"[SADA-TTS] Audio successfully generated at: {audio_filepath}")
        
        return audio_filepath