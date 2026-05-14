import queue
import numpy as np
import sounddevice as sd
import webrtcvad
import sys
import pygame

# Import our custom modules
from stt_engine import transcribe_audio
from chatbot import generate_response
from hamsa_tts import HamsaTTS
# ==========================================
# AUDIO CONFIGURATION
# ==========================================
SAMPLE_RATE = 16000  # Whisper expects 16kHz audio
CHANNELS = 1         # Mono audio
FRAME_DURATION_MS = 30  # WebRTCVAD requires 10, 20, or 30ms frames
FRAME_SIZE = int(SAMPLE_RATE * (FRAME_DURATION_MS / 1000.0))  # 480 samples per frame

# Silence threshold to determine when the user has stopped speaking
SILENCE_DURATION_MS = 600
SILENCE_FRAMES_THRESHOLD = int(SILENCE_DURATION_MS / FRAME_DURATION_MS)

# Initialize Voice Activity Detection (3 is the most aggressive noise filtering)
vad = webrtcvad.Vad(3)

# Queue to safely transfer audio chunks from the audio thread to the main thread
audio_queue = queue.Queue()

# ==========================================
# AUDIO CALLBACK FUNCTION
# ==========================================
def audio_callback(indata, frames, time_info, status):
    """
    This function is called automatically by sounddevice for every audio frame.
    It puts the raw bytes into a queue so the main thread can process them.
    """
    if status:
        print(f"[Mic Status] {status}", file=sys.stderr)
    
    # We use RawInputStream, so indata is a buffer of raw bytes
    audio_queue.put(bytes(indata))

# ==========================================
# PROCESSING PIPELINE
# ==========================================
def process_audio_buffer(byte_data: bytes, tts: HamsaTTS):
    """
    Takes the recorded raw bytes, converts them for Whisper, 
    gets the text, and feeds it to the LLM.
    """
    # 1. Convert raw 16-bit PCM bytes to a float32 NumPy array ([-1.0 to 1.0]) for Whisper
    audio_np = np.frombuffer(byte_data, dtype=np.int16).astype(np.float32) / 32768.0
    
    # 2. Run Speech-to-Text
    transcribed_text = transcribe_audio(audio_np)
    
    # 3. If we caught actual words, pass to the LLM
    if transcribed_text:
        
        for raw, cleaned in generate_response(transcribed_text):
            try:
                audio_file_path = tts.speak(cleaned)
                if audio_file_path is None:
                    continue
                pygame.mixer.music.load(audio_file_path)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    pygame.time.Clock().tick(10)
            except Exception as e:
                print(f"[TTS Error] {e}")

# ==========================================
# MAIN LOOP
# ==========================================
def main():
    print("=" * 50)
    print(" Starting Real-Time Voice Agent Pipeline...")
    print("=" * 50)
    
    # Initialize pygame mixer for audio playback
    pygame.mixer.init()
    
    # Initialize TTS engine (loads model weights into GPU)
    tts = HamsaTTS()
    
    # Open the microphone stream
    try:
        stream = sd.RawInputStream(
            samplerate=SAMPLE_RATE, 
            blocksize=FRAME_SIZE,
            device=None, # Uses default system microphone
            channels=CHANNELS, 
            dtype='int16',
            callback=audio_callback
        )
        
        with stream:
            print("\n System Ready! Start speaking...\n")
            
            buffer = bytearray()
            is_recording = False
            silence_counter = 0
            
            while True:
                # Get the next 30ms frame of audio from the queue
                frame = audio_queue.get()
                
                # Check if this frame contains human speech
                is_speech = vad.is_speech(frame, SAMPLE_RATE)
                
                if is_speech:
                    silence_counter = 0 # Reset silence counter
                    if not is_recording:
                        is_recording = True
                        print("\n[ Listening...]")
                    
                    # Append the audio frame to our memory buffer
                    buffer.extend(frame)
                
                else:
                    if is_recording:
                        silence_counter += 1
                        # Keep adding silence to the buffer so it sounds natural to Whisper
                        buffer.extend(frame)
                        
                        # If we've hit our silence threshold, the user finished their sentence
                        if silence_counter >= SILENCE_FRAMES_THRESHOLD:
                            is_recording = False
                            print("[ Processing...]")
                            
                            # Send the buffer to STT -> LLM
                            process_audio_buffer(buffer, tts)
                            
                            # Clear the buffer for the next time the user speaks
                            buffer.clear()
                            print("\n[ Ready for next input]")
                            
    except KeyboardInterrupt:
        print("\n\n Voice Agent stopped by user. Shutting down gracefully.")
    except Exception as e:
        print(f"\n Fatal Error: {e}")

if __name__ == "__main__":
    main()
