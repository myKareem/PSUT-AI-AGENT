import sounddevice as sd
import numpy as np
import threading
import time
import sys

class AudioRingBuffer:
    def __init__(self, sample_rate=16000, channels=1, buffer_duration_sec=5):
        self.sample_rate = sample_rate
        self.channels = channels
        self.buffer_size = sample_rate * buffer_duration_sec
        
        # Initialize a zero-filled numpy array for the buffer
        self.buffer = np.zeros((self.buffer_size, channels), dtype=np.float32)
        self.write_index = 0
        self.lock = threading.Lock()

    def add_data(self, data):
        """Writes new audio frames into the ring buffer."""
        frames = len(data)
        with self.lock:
            # If the new data is larger than the entire buffer, only keep the latest part
            if frames >= self.buffer_size:
                self.buffer[:] = data[-self.buffer_size:]
                self.write_index = 0
                return

            end_idx = self.write_index + frames

            # Handle wrap-around
            if end_idx <= self.buffer_size:
                self.buffer[self.write_index:end_idx] = data
            else:
                overflow = end_idx - self.buffer_size
                first_part = frames - overflow
                self.buffer[self.write_index:] = data[:first_part]
                self.buffer[:overflow] = data[first_part:]

            self.write_index = (self.write_index + frames) % self.buffer_size

    def get_latest_data(self, duration_sec):
        """Retrieves the most recent audio frames."""
        frames_to_get = int(duration_sec * self.sample_rate)
        if frames_to_get > self.buffer_size:
            frames_to_get = self.buffer_size

        with self.lock:
            start_idx = self.write_index - frames_to_get
            if start_idx >= 0:
                return self.buffer[start_idx:self.write_index].copy()
            else:
                # Handle wrap-around read
                part1 = self.buffer[start_idx:]
                part2 = self.buffer[:self.write_index]
                return np.concatenate((part1, part2))

class AudioIngestion:
    def __init__(self, sample_rate=16000, channels=1):
        self.sample_rate = sample_rate
        self.channels = channels
        self.ring_buffer = AudioRingBuffer(sample_rate, channels, buffer_duration_sec=5)
        self.stream = None

    def audio_callback(self, indata, frames, time, status):
        """This function is called by sounddevice for every new audio block."""
        if status:
            # This logs dropouts or glitches (e.g., input overflow)
            print(f"\n[WARNING] Stream status: {status}", file=sys.stderr)
            
        self.ring_buffer.add_data(indata)

    def start_stream(self):
        """Starts the non-blocking audio stream."""
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            callback=self.audio_callback,
            blocksize=1024 # Process audio in chunks of 1024 frames
        )
        self.stream.start()
        print(f"Started audio stream: {self.sample_rate}Hz, {self.channels} Channel(s)")

    def stop_stream(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()
            print("Audio stream stopped.")

# --- Testing the Implementation ---
if __name__ == "__main__":
    # Initialize and start
    audio_system = AudioIngestion(sample_rate=16000, channels=1)
    audio_system.start_stream()

    try:
        print("Capturing audio... Press Ctrl+C to stop.")
        while True:
            time.sleep(1) # Simulate main thread doing other work
            
            # Fetch the last 1 second of audio to verify it's working
            recent_audio = audio_system.ring_buffer.get_latest_data(duration_sec=1.0)
            
            # Calculate volume (Root Mean Square) to show live activity
            if len(recent_audio) > 0:
                volume_rms = np.sqrt(np.mean(recent_audio**2))
                bar_length = int(volume_rms * 500) # Scale for terminal visualization
                print(f"\rLive Audio Level: [{'#' * bar_length}{'-' * (50 - bar_length)}]", end="")
                
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        audio_system.stop_stream()