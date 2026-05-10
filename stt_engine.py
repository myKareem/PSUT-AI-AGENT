from faster_whisper import WhisperModel
import time
import numpy as np

print("Loading STT model...")
# Model loads ONCE into VRAM when this file is imported
model = WhisperModel(
    "HebArabNlpProject/WhisperLevantine",
    device="cuda",
    compute_type="float16",
    num_workers=4,
    download_root="./models"
)
print("STT Model ready!\n")

def transcribe_audio(audio_data: np.ndarray) -> str:
    """
    Takes an in-memory NumPy array of audio data (16kHz, mono, float32)
    and returns the transcribed text.
    """
    start_time = time.perf_counter()
    
    # We pass the in-memory array directly instead of "audio2.wav"
    segments, info = model.transcribe(
        audio_data,
        beam_size=1,
        language="ar",
        vad_filter=True,
        vad_parameters=dict(
            min_silence_duration_ms=500,
            threshold=0.5,
            min_speech_duration_ms=250
        ),
        temperature=0.0,
        condition_on_previous_text=False,
        compression_ratio_threshold=2.4,
        log_prob_threshold=-1.0,
        no_speech_threshold=0.6
    )
    
    segments_list = list(segments)
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    
    # Combine all transcribed segments into one continuous string
    full_text = " ".join([segment.text.strip() for segment in segments_list]).strip()
    
    # Print the latency and result for debugging, similar to your original script
    if full_text:
        print(f"[STT] Time: {elapsed_ms:.0f}ms | Conf: {info.language_probability:.2f}")
        print(f"[STT] User said: {full_text}")
        print("-" * 50)
        
    return full_text