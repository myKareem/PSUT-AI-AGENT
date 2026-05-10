"""
HamsaTTS — Cloud-based Arabic TTS via Hamsa WebSocket API.
Drop-in replacement for SadaTTS: speak(text) → file path to WAV.
"""
import asyncio
import json
import os
import numpy as np
import soundfile as sf
import websockets

HAMSA_API_KEY = "6c287387-ced2-4eba-a2bc-0b0a01dae1fb"
HAMSA_URI = f"wss://api.tryhamsa.com/v1/realtime/ws?api_key={HAMSA_API_KEY}"
HAMSA_SAMPLE_RATE = 16000


class HamsaTTS:
    def __init__(self, out_dir='./app/static', speaker='ASSY', dialect='jor'):
        """
        Initializes the Hamsa TTS client.
        No local model loading — all inference is cloud-based.
        """
        self.out_dir = out_dir
        self.speaker = speaker
        self.dialect = dialect

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        print("[HAMSA-TTS] Ready (cloud-based, no local models)")

    async def _synthesize(self, text: str) -> bytes:
        """Calls the Hamsa WebSocket API and returns raw PCM bytes."""
        async with websockets.connect(HAMSA_URI) as ws:
            await ws.send(json.dumps({
                "type": "tts",
                "payload": {
                    "text": text,
                    "speaker": self.speaker,
                    "dialect": self.dialect,
                    "languageId": "ar",
                    "mulaw": False
                }
            }))

            audio_chunks = []
            async for message in ws:
                if isinstance(message, bytes):
                    audio_chunks.append(message)
                else:
                    data = json.loads(message)
                    if data.get("type") == "end":
                        break

        return b"".join(audio_chunks)

    def speak(self, text: str, speed=1.0, denoise=0.01) -> str | None:
        """
        Generates speech and returns the file path of the saved WAV.
        Same interface as SadaTTS.speak() for drop-in compatibility.
        speed/denoise params are kept for interface compatibility but unused.
        """
        if not text or not text.strip():
            return None

        print("[HAMSA-TTS] Synthesizing speech...", flush=True)

        # Run the async WebSocket call synchronously.
        # This works from both sync code (main.py) and from
        # asyncio.to_thread() in server.py (thread has no event loop).
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # We're inside an async context — create a new loop in this thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                pcm_bytes = pool.submit(
                    asyncio.run, self._synthesize(text)
                ).result()
        else:
            pcm_bytes = asyncio.run(self._synthesize(text))

        if not pcm_bytes:
            print("[HAMSA-TTS] No audio received from API", flush=True)
            return None

        # Convert PCM bytes to numpy and save as WAV
        audio_np = np.frombuffer(pcm_bytes, dtype=np.int16)
        save_path = os.path.join(self.out_dir, "wave0.wav")
        sf.write(save_path, audio_np, HAMSA_SAMPLE_RATE)

        print(f"[HAMSA-TTS] Audio generated: {save_path}", flush=True)
        return save_path