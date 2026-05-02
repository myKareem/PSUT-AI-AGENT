"""
PSUT Voice Agent — WebSocket Server
"""
import os, sys
# Fix Windows cp1252 — force UTF-8 on stdout/stderr for Arabic text
os.environ["PYTHONUTF8"] = "1"
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import asyncio
import json
import os
import time
import traceback
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

# ── TTS path setup ──────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tts-arabic-pytorch-master'))

from stt_engine import transcribe_audio
from chatbot import generate_response
from sada_tts import SadaTTS

# ── Initialize at module level (before uvicorn) ────────────
GREETING_TEXT = "مرحبا معك المساعد الذكي لجامعة الاميرة سمية، كيف ممكن أساعدك؟"

print("[SERVER] Initializing TTS engine...", flush=True)
tts = SadaTTS()

print("[SERVER] Pre-generating greeting audio...", flush=True)
_greet_path = tts.speak(GREETING_TEXT)
with open(_greet_path, "rb") as _f:
    GREETING_WAV = _f.read()
print(f"[SERVER] Greeting ready ({len(GREETING_WAV)} bytes)", flush=True)

# ── FastAPI ─────────────────────────────────────────────────
app = FastAPI()
app.mount("/static", StaticFiles(directory="web"), name="static")


@app.get("/")
async def index():
    return FileResponse("web/index.html")


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    print("[WS] Client connected", flush=True)

    try:
        # 1. Send greeting
        await ws.send_json({"type": "greeting", "text": GREETING_TEXT})
        await ws.send_bytes(GREETING_WAV)
        print("[WS] Greeting sent", flush=True)

        # Wait for greeting to finish playing on client before enabling mic
        await ws.send_json({"type": "ready"})

        while True:
            # 2. Wait for a message from client
            raw = await ws.receive()

            if raw.get("type") == "websocket.disconnect":
                break

            # Handle text control messages
            if "text" in raw:
                try:
                    ctrl = json.loads(raw["text"])
                    if ctrl.get("type") == "hangup":
                        print("[WS] Client hung up", flush=True)
                        break
                    if ctrl.get("type") == "interrupt":
                        print("[WS] Interrupt received (ignored outside generation)", flush=True)
                        continue
                except json.JSONDecodeError:
                    continue
                continue

            # Handle binary audio
            if "bytes" not in raw:
                continue

            audio_bytes = raw["bytes"]
            audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

            if len(audio_np) < 1600:  # <100ms, skip noise
                await ws.send_json({"type": "ready"})
                continue

            await ws.send_json({"type": "processing"})

            # 3. STT
            t0 = time.perf_counter()
            text = await asyncio.to_thread(transcribe_audio, audio_np)
            stt_ms = (time.perf_counter() - t0) * 1000
            print(f"[WS] STT {stt_ms:.0f}ms", flush=True)

            if not text:
                await ws.send_json({"type": "ready"})
                continue

            await ws.send_json({"type": "transcription", "text": text})

            # 4. LLM + TTS sentence-by-sentence
            def _pipeline():
                results = []
                for sentence in generate_response(text):
                    audio_path = tts.speak(sentence)
                    if audio_path is None:
                        continue  # text was empty after sanitization
                    with open(audio_path, "rb") as f:
                        wav = f.read()
                    results.append((sentence, wav))
                return results

            sentences = await asyncio.to_thread(_pipeline)

            for stxt, wav_data in sentences:
                await ws.send_json({"type": "sentence", "text": stxt})
                await ws.send_bytes(wav_data)

            await ws.send_json({"type": "done"})
            await ws.send_json({"type": "ready"})

    except WebSocketDisconnect:
        print("[WS] Client disconnected", flush=True)
    except Exception as e:
        print(f"[WS] ERROR: {e}", flush=True)
        traceback.print_exc()
    finally:
        try:
            await ws.close()
        except Exception:
            pass


if __name__ == "__main__":
    print("[SERVER] Ready — open http://localhost:8000", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
