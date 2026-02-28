import sounddevice as sd
import numpy as np
import torch
import queue
import sys
import time
import threading
from collections import deque # <-- NEW: Import deque
#import torchaudio

class VADAudioIngestion:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.chunk_size = 512 
        self.audio_queue = queue.Queue()
        
        print("Loading Silero VAD model...")
        torch.set_num_threads(1) 
        self.model, _ = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False
        )
        self.model.eval()
        
        self.speech_start_threshold = 0.5  
        self.speech_end_threshold = 0.3    
        self.max_silence_chunks = int((0.8 * sample_rate) / self.chunk_size) 
        
        self.is_speaking = False
        self.silence_counter = 0
        self.utterance_buffer = [] 
        
        # --- NEW: Pre-roll Buffer Setup ---
        # We want to save 200ms of audio before speech is detected.
        # (0.2 seconds * 16000 samples/sec) / 512 samples/chunk = ~6 chunks
        self.pre_roll_chunks = int((0.2 * sample_rate) / self.chunk_size)
        
        # deque automatically discards the oldest chunk when maxlen is reached
        self.pre_roll_buffer = deque(maxlen=self.pre_roll_chunks) 

    def audio_callback(self, indata, frames, time, status):
        if status:
            print(f"[WARNING] {status}", file=sys.stderr)
        self.audio_queue.put(indata.copy())

    def process_vad(self):
        print("Listening for speech... (Try speaking naturally)")
        while True:
            chunk = self.audio_queue.get()
            if chunk is None: break 
            
            audio_tensor = torch.from_numpy(chunk.flatten()).float()
            
            with torch.no_grad():
                speech_prob = self.model(audio_tensor, self.sample_rate).item()
            
            # --- UPDATED: Endpointing Logic with Pre-roll ---
            if speech_prob >= self.speech_start_threshold:
                if not self.is_speaking:
                    print("\n[🎙️ Speech Started - Injecting Pre-roll]")
                    self.is_speaking = True
                    
                    # Dump the saved 200ms of background audio into the utterance first!
                    self.utterance_buffer.extend(self.pre_roll_buffer)
                    self.pre_roll_buffer.clear() # Reset for the next time
                
                self.silence_counter = 0
                self.utterance_buffer.append(chunk)
                print("█", end="", flush=True) 
                
            elif speech_prob < self.speech_end_threshold and self.is_speaking:
                self.silence_counter += 1
                self.utterance_buffer.append(chunk)
                print("▒", end="", flush=True) 
                
                if self.silence_counter >= self.max_silence_chunks:
                    print(f"\n[🛑 Speech Ended - Captured {len(self.utterance_buffer)} chunks]")
                    
                    self.is_speaking = False
                    self.silence_counter = 0
                    self.utterance_buffer = []
            
            else:
                # --- NEW: Background Noise Handling ---
                # If we are NOT speaking, keep saving chunks to the sliding window
                self.pre_roll_buffer.append(chunk)

    def start(self):
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            callback=self.audio_callback,
            blocksize=self.chunk_size,
            dtype=np.float32
        )
        self.stream.start()

    def stop(self):
        self.stream.stop()
        self.stream.close()
        self.audio_queue.put(None)

if __name__ == "__main__":
    vad_system = VADAudioIngestion()
    processing_thread = threading.Thread(target=vad_system.process_vad)
    processing_thread.start()
    
    vad_system.start()
    
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nShutting down...")
        vad_system.stop()
        processing_thread.join()