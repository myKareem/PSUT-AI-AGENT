"""
Arabic STT Service  —  VAD → faster-whisper pipeline
=====================================================
Jordanian Arabic university customer service bot.

Full in-memory pipeline:
  Upload → decode → resample → Silero VAD → collect speech
         → NumPy float32 → faster-whisper (Levantine) → JSON transcript

No temporary files. No re-encoding between stages.
Models are pre-loaded at startup; inference begins instantly per request.
"""

import io
import logging
import time
from contextlib import asynccontextmanager
from typing import Annotated

import numpy as np
import torch
import torchaudio
from faster_whisper import WhisperModel
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

# ──────────────────────────────────────────────────────────────────────────────
# Logging  (single line format — minimal overhead)
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("stt_service")

# ──────────────────────────────────────────────────────────────────────────────
# ① VAD Configuration
# ──────────────────────────────────────────────────────────────────────────────
VAD_SAMPLE_RATE: int   = 16_000   # Silero supports 8k / 16k; 16k for Arabic
VAD_THRESHOLD:   float = 0.50     # raise to 0.65 in loud university lobbies
MIN_SPEECH_MS:   int   = 250      # reject bursts shorter than 250 ms
SPEECH_PAD_MS:   int   = 100      # 100 ms context pad per segment edge

# ──────────────────────────────────────────────────────────────────────────────
# ② Whisper Configuration
# ──────────────────────────────────────────────────────────────────────────────
# Replaced with the model from script.py
WHISPER_MODEL_SIZE: str = "HebArabNlpProject/WhisperLevantine"

# float16 as defined in script.py
WHISPER_COMPUTE_TYPE: str = "float16"

# Beam size 1 = greedy decoding.  Fastest possible path; WER penalty for
# Arabic is negligible on short (<30 s) customer service utterances.
WHISPER_BEAM_SIZE: int = 1

# Forced language — never let the model auto-detect; saves ~100 ms per call.
WHISPER_LANGUAGE: str = "ar"

# ──────────────────────────────────────────────────────────────────────────────
# ③ Global model handles  (populated once at startup, read-only during serving)
# ──────────────────────────────────────────────────────────────────────────────
_vad_model:            torch.nn.Module | None = None
_get_speech_timestamps: object                = None
_collect_chunks:        object                = None
_whisper_model:        WhisperModel | None    = None


# ──────────────────────────────────────────────────────────────────────────────
# ④ Model loaders
# ──────────────────────────────────────────────────────────────────────────────
def _load_vad() -> None:
    """
    Pull Silero VAD from torch.hub and pin it to CUDA / CPU.

    torch.hub caches the weights in %USERPROFILE%\\.cache\\torch\\hub on
    Windows after the first download.  Subsequent starts are <100 ms.

    onnx=False  →  PyTorch TorchScript backend.
    Faster than ONNX for single short clips because it avoids the ONNX
    runtime DLL load and the extra memory copy into the OrtValue buffer.
    """
    global _vad_model, _get_speech_timestamps, _collect_chunks

    log.info("[VAD ] Loading Silero VAD ...")
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        onnx=False,
        trust_repo=True,
    )
    (
    _get_speech_timestamps,
    _save_audio,
    _read_audio,
    _VADIterator,
    _collect_chunks
) = utils

    model.eval()
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(_device)
    _vad_model = model
    log.info("[VAD ] Ready — device=%s", _device)


def _load_whisper() -> None:
    """
    Load faster-whisper (CTranslate2 backend) with the Levantine model.
    """
    global _whisper_model

    _use_cuda = torch.cuda.is_available()
    _device   = "cuda" if _use_cuda else "cpu"
    _compute  = WHISPER_COMPUTE_TYPE   

    log.info(
        "[STT ] Loading faster-whisper '%s' -> device=%s compute=%s ...",
        WHISPER_MODEL_SIZE, _device, _compute,
    )
    
    # Updated to match script.py initialization
    _whisper_model = WhisperModel(
        WHISPER_MODEL_SIZE,
        device=_device,
        compute_type=_compute,
        cpu_threads=0,    
        num_workers=4,             # Imported from script.py 
        download_root="./models"   # Imported from script.py
    )
    log.info("[STT ] faster-whisper ready — device=%s", _device)


# ──────────────────────────────────────────────────────────────────────────────
# ⑤ Lifespan — both models load before the first request is accepted
# ──────────────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Sequential model loading at startup; graceful log on shutdown."""
    _load_vad()
    _load_whisper()
    log.info("All models loaded — service ready.")
    yield
    log.info("Service shutting down.")


# ──────────────────────────────────────────────────────────────────────────────
# ⑥ FastAPI app
# ──────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Arabic STT Service",
    description=(
        "Voice note -> Silero VAD silence removal -> "
        "faster-whisper transcription -> JSON. "
        "Optimised for Jordanian Arabic university customer service."
    ),
    version="2.0.0",
    lifespan=lifespan,
)


# ──────────────────────────────────────────────────────────────────────────────
# ⑦ Internal pipeline helpers
# ──────────────────────────────────────────────────────────────────────────────
def _decode_audio(raw_bytes: bytes) -> tuple[torch.Tensor, int]:
    """
    Decode any supported codec (WAV / MP3 / OGG / FLAC / M4A) held in
    *raw_bytes* into a mono float32 tensor.  Zero disk I/O.
    """
    buf = io.BytesIO(raw_bytes)
    waveform, sr = torchaudio.load(buf)     # shape: (channels, samples)

    if waveform.shape[0] > 1:              # stereo / multi-channel -> mono
        waveform = waveform.mean(dim=0, keepdim=True)

    return waveform.squeeze(0), sr         # shape: (samples,)


def _resample(waveform: torch.Tensor, src_sr: int) -> torch.Tensor:
    """Resample to 16 kHz if the source differs; no-op otherwise."""
    if src_sr == VAD_SAMPLE_RATE:
        return waveform
    return torchaudio.functional.resample(waveform, src_sr, VAD_SAMPLE_RATE)


def _apply_vad(waveform: torch.Tensor) -> torch.Tensor:
    """
    Run Silero VAD and return a concatenated tensor of speech-only samples.
    """
    vad_device   = next(_vad_model.parameters()).device
    waveform_dev = waveform.to(vad_device)

    timestamps = _get_speech_timestamps(
        waveform_dev,
        _vad_model,
        sampling_rate=VAD_SAMPLE_RATE,
        threshold=VAD_THRESHOLD,
        min_speech_duration_ms=MIN_SPEECH_MS,
        speech_pad_ms=SPEECH_PAD_MS,
        return_seconds=False,
    )

    if not timestamps:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "no_speech_detected",
                "message": (
                    "Silero VAD found no speech above the configured threshold. "
                    "Check audio quality or lower VAD_THRESHOLD."
                ),
            },
        )

    # collect_chunks always returns a CPU tensor
    speech = _collect_chunks(timestamps, waveform_dev.cpu())
    return speech    # shape: (speech_samples,) on CPU


def _tensor_to_numpy(speech: torch.Tensor) -> np.ndarray:
    """
    Convert the speech tensor to a 1-D float32 NumPy array.
    """
    audio_np = speech.numpy()

    # Guarantee float32 — faster-whisper rejects float64
    if audio_np.dtype != np.float32:
        audio_np = audio_np.astype(np.float32)

    return audio_np


def _transcribe(audio_np: np.ndarray) -> dict:
    """
    Run faster-whisper inference and return a structured result dict.
    """
    segments_gen, info = _whisper_model.transcribe(
        audio_np,
        language=WHISPER_LANGUAGE,
        beam_size=WHISPER_BEAM_SIZE,
        vad_filter=False, # We already ran dedicated Silero VAD above
        without_timestamps=False,
        condition_on_previous_text=False,
        temperature=0.0,
        compression_ratio_threshold=2.4, # Added from script.py
        log_prob_threshold=-1.0,         # Added from script.py
        no_speech_threshold=0.6          # Added from script.py
    )

    # Materialise the lazy generator
    segments        = []
    full_text_parts = []
    for seg in segments_gen:
        segments.append(
            {
                "start":      round(seg.start, 3),
                "end":        round(seg.end,   3),
                "text":       seg.text.strip(),
                "confidence": round(float(np.exp(seg.avg_logprob)), 4),
            }
        )
        full_text_parts.append(seg.text.strip())

    return {
        "text":       " ".join(full_text_parts).strip(),
        "segments":   segments,
        "language":   info.language,
        "duration_s": round(info.duration, 3),
    }


# ──────────────────────────────────────────────────────────────────────────────
# ⑧ POST /transcribe  — the single public endpoint
# ──────────────────────────────────────────────────────────────────────────────
@app.post(
    "/transcribe",
    summary="Transcribe an Arabic voice note",
    response_description=(
        "JSON with full transcript, per-segment detail, and timing metadata."
    ),
    responses={
        200: {"description": "Successful transcription."},
        400: {"description": "Empty file or unsupported/corrupt audio format."},
        422: {"description": "Audio contains no detectable speech."},
        500: {"description": "Unexpected inference error."},
    },
)
async def transcribe(
    file: Annotated[
        UploadFile,
        File(description="Voice note — WAV, MP3, OGG, FLAC, or M4A."),
    ],
) -> JSONResponse:
    t0 = time.perf_counter()

    # ── Step 1: Ingest ────────────────────────────────────────────────────────
    raw: bytes = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    log.info("Received file='%s'  size=%d B", file.filename, len(raw))

    # ── Step 2: Decode ────────────────────────────────────────────────────────
    try:
        waveform, src_sr = _decode_audio(raw)
    except Exception as exc:
        log.warning("Decode error: %s", exc)
        raise HTTPException(
            status_code=400,
            detail={"error": "decode_failed", "message": str(exc)},
        ) from exc

    # ── Step 3: Resample ──────────────────────────────────────────────────────
    waveform = _resample(waveform, src_sr)

    # ── Step 4: VAD ───────────────────────────────────────────────────────────
    # Raises HTTP 422 automatically if no speech detected
    speech_tensor = _apply_vad(waveform)

    t_vad = time.perf_counter()
    log.info(
        "VAD done in %.0f ms -> %.2f s speech retained",
        (t_vad - t0) * 1000,
        speech_tensor.shape[0] / VAD_SAMPLE_RATE,
    )

    # ── Step 5: Tensor -> NumPy (zero re-encoding overhead) ───────────────────
    audio_np: np.ndarray = _tensor_to_numpy(speech_tensor)

    # ── Step 6: Transcribe ────────────────────────────────────────────────────
    try:
        result = _transcribe(audio_np)
    except Exception as exc:
        log.error("Whisper inference error: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={"error": "inference_failed", "message": str(exc)},
        ) from exc

    total_ms = round((time.perf_counter() - t0) * 1000)
    log.info(
        "Done — transcript='%s'  latency=%d ms",
        result["text"][:80],
        total_ms,
    )

    return JSONResponse(
        content={
            "transcript":        result["text"],
            "segments":          result["segments"],
            "language":          result["language"],
            "speech_duration_s": result["duration_s"],
            "total_latency_ms":  total_ms,
        }
    )


# ──────────────────────────────────────────────────────────────────────────────
# ⑨ Health probe — Docker / Kubernetes liveness check
# ──────────────────────────────────────────────────────────────────────────────
@app.get("/health", include_in_schema=False)
async def health() -> JSONResponse:
    cuda_ok = torch.cuda.is_available()
    return JSONResponse(
        content={
            "status":         "ok",
            "vad_loaded":     _vad_model is not None,
            "whisper_loaded": _whisper_model is not None,
            "cuda_available": cuda_ok,
            "cuda_device":    torch.cuda.get_device_name(0) if cuda_ok else None,
        }
    )


# ──────────────────────────────────────────────────────────────────────────────
# ⑩ Windows entry-point guard
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        workers=1,        # Keep to 1 on Windows; use a Linux host for multi-worker
        log_level="info",
        reload=False,     # Never use --reload with global GPU models
    )