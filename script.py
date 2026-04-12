from faster_whisper import WhisperModel
import time

print("Loading model...")
model = WhisperModel(
    "HebArabNlpProject/WhisperLevantine",
    device="cuda",
    compute_type="float16",
    num_workers=4,           
    download_root="./models"
)
print("Model ready!\n")

start_time = time.perf_counter()

segments, info = model.transcribe(
    "audio2.wav",
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

print(f"Detected language: {info.language} (confidence: {info.language_probability:.2f})")
print("-" * 50)
for segment in segments_list:
    print(f"[{segment.start:>7.2f}s -> {segment.end:<7.2f}s] {segment.text.strip()}")

with open("jordanian_transcription.txt", "w", encoding="utf-8") as f:
    for segment in segments_list:
        f.write(f"[{segment.start:.2f}-{segment.end:.2f}] {segment.text.strip()}\n")

print(f"\n Transcription saved ({elapsed_ms:.0f}ms)")