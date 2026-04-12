# Arabic STT Benchmark — `benchmark.py`

Standalone evaluation script for the **Jordanian Arabic VAD → faster-whisper
pipeline** defined in `main.py`.  
Run it once against a labelled test set and get a structured terminal report
covering accuracy, speed, and GPU memory usage — no FastAPI server required.

---

## Metrics

| Metric | Symbol | Definition |
|---|---|---|
| Word Error Rate | **WER** | `(S + D + I) / N` — substitutions + deletions + insertions over reference tokens.  0 % = perfect match; values > 100 % are possible when the hypothesis has many extra insertions. |
| Real-Time Factor | **RTF** | `inference_time_s / audio_duration_s`.  RTF < 1.0 means the model is faster than real-time. |
| Speed to First Token | **SFTT** | Wall-clock time from the `transcribe()` call until the first segment is yielded from the lazy generator.  Relevant for streaming / low-latency applications. |
| Peak VRAM | **VRAM** | `torch.cuda.max_memory_allocated()` sampled separately for (a) model loading and (b) each inference call.  Reported in MB / GB; "N/A (CPU-only)" on machines without CUDA. |

---

## Files

```
project/
├── main.py            Original FastAPI STT service
├── benchmark.py       This benchmarking script
├── requirements.txt   Pinned production dependencies
└── data/
    ├── test_manifest.csv        Example manifest (see below)
    └── audio/
        ├── utt_001.wav
        └── utt_002.mp3
```

---

## Installation

### 1 — Satisfy `requirements.txt`

Follow the instructions in `requirements.txt` to install the correct
`torch` + `torchaudio` build for your hardware (CPU-only or CUDA).

### 2 — Install the optional WER dependency

`jiwer` is **not** listed in `requirements.txt` because it is only needed for
benchmarking (not production serving).  Install it separately:

```bash
pip install jiwer
```

If `jiwer` is absent the script still runs — WER columns are shown as `—`.

### 3 — Verify your environment

```bash
python -c "import torch; print(torch.cuda.is_available())"
python -c "from faster_whisper import WhisperModel; print('ok')"
```

---

## Manifest format

The manifest is a UTF-8 CSV file (Excel BOM `\xef\xbb\xbf` is handled
automatically).

```csv
file_path,reference_text
audio/utt_001.wav,مرحبا كيف حالك
audio/utt_002.mp3,أريد معلومات عن التسجيل
audio/utt_003.wav,
```

| Column | Required | Notes |
|---|---|---|
| `file_path` | ✅ | Absolute path **or** path relative to the manifest's directory. |
| `reference_text` | ✗ | Leave blank to skip WER for that file.  The file is still benchmarked for speed and VRAM. |

---

## Usage

### Basic run

```bash
python benchmark.py --manifest data/test_manifest.csv
```

### All options

```bash
python benchmark.py \
  --manifest       data/test_manifest.csv \
  --model          HebArabNlpProject/WhisperLevantine \
  --compute-type   float16 \
  --device         auto \
  --beam-size      1 \
  --output-json    results/run_001.json
```

| Flag | Default | Description |
|---|---|---|
| `--manifest` | *required* | Path to CSV manifest. |
| `--model` | `HebArabNlpProject/WhisperLevantine` | faster-whisper model ID (HuggingFace Hub) or path to a local CTranslate2 model directory. |
| `--compute-type` | `float16` | CTranslate2 weight type. Use `int8` for maximum CPU speed or on older GPUs. `float16` requires a CUDA-capable GPU. |
| `--device` | `auto` | `auto` picks CUDA when available, otherwise CPU. Pass `cpu` to force CPU even when a GPU is present. |
| `--beam-size` | `1` | Whisper beam width. `1` = greedy decoding (fastest; matches production). Increase to `3`–`5` to trade speed for accuracy. |
| `--output-json` | *(none)* | Write the full per-file results + config + VRAM tracker to a JSON file for offline analysis or CI diffing. |

---

## Sample terminal output

```
══════════════════════════════════════════════════════════════════════════
  JORDANIAN ARABIC STT — BENCHMARK REPORT
══════════════════════════════════════════════════════════════════════════

  Model                        HebArabNlpProject/WhisperLevantine
  Compute type                 float16
  Device                       cuda
  Beam size                    1
  Language                     ar
  VAD threshold                0.5

──────────────────────────────────────────────────────────────────────────
  FILES
──────────────────────────────────────────────────────────────────────────
  Total submitted          20
  Processed successfully   19
  Skipped / failed         1

  utt_broken.wav                         Decode error: ...

──────────────────────────────────────────────────────────────────────────
  ACCURACY
──────────────────────────────────────────────────────────────────────────
  Average WER              12.34 %
  Best  WER                utt_001.wav  →  2.1 %
  Worst WER                utt_017.wav  →  38.5 %

──────────────────────────────────────────────────────────────────────────
  SPEED
──────────────────────────────────────────────────────────────────────────
  Average audio duration   4.82 s  (post-VAD)
  Average RTF              0.1823  (< 1.0 = faster than real-time)
  RTF range                0.0941 – 0.4102
  Average SFTT             287.4 ms  (wall-clock to first segment)
  SFTT p50 / p95           241.0 ms / 510.3 ms

──────────────────────────────────────────────────────────────────────────
  GPU MEMORY  (VRAM)
──────────────────────────────────────────────────────────────────────────
  Peak during model load   1.73 GB
  Peak during inference    1.89 GB
  Absolute peak (run)      1.89 GB

──────────────────────────────────────────────────────────────────────────
  PER-FILE RESULTS
──────────────────────────────────────────────────────────────────────────
  File                              Dur(s)    RTF  SFTT ms   WER %  VRAM MB
  ──────────────────────────────── ────── ────── ─────── ────── ───────
  utt_001.wav                         3.21  0.1204    241.0     2.1     1934
  utt_002.mp3                         6.88  0.2341    310.5    15.3     1941
  ...
══════════════════════════════════════════════════════════════════════════
```

---

## How VRAM tracking works

PyTorch exposes a built-in high-watermark counter per CUDA device.

```
torch.cuda.reset_peak_memory_stats()   ← zero the counter
... code under measurement ...
torch.cuda.max_memory_allocated()      ← read the high-watermark in bytes
```

The script calls this pair at **two distinct phases**:

1. **Model load phase** — wraps both `_load_vad()` + `_load_whisper()` via
   the `vram_phase("model_load", tracker)` context manager.  This captures
   the memory footprint of loading weights onto the GPU, which is typically
   the highest sustained allocation in the whole process.

2. **Per-file inference phase** — the counter is reset again before every
   `transcribe()` call so you can see the *incremental* VRAM cost of
   running inference on top of the already-loaded model weights.  The
   global maximum across all files is stored as `inference_peak`.

The **Absolute peak** row in the report is `max(model_load_peak, inference_peak)`.

> **Note**: `max_memory_allocated()` tracks *allocated* bytes — tensors that
> are live or recently freed but not yet returned to the OS.  It does **not**
> include the reserved-but-unused PyTorch CUDA cache
> (`max_memory_reserved()`).  Allocated is the more meaningful number for
> budgeting real model footprint.

---

## JSON output schema

When `--output-json` is supplied the file has the following shape:

```json
{
  "config": {
    "whisper_model_size":   "HebArabNlpProject/WhisperLevantine",
    "whisper_compute_type": "float16",
    "whisper_beam_size":    1,
    "whisper_language":     "ar",
    "device":               "auto",
    "download_root":        "./models",
    "num_workers":          4
  },
  "vram_tracker": {
    "model_load":       1773.4,
    "inference_peak":   1934.1,
    "resolved_device":  "cuda"
  },
  "results": [
    {
      "file_path":      "/abs/path/to/utt_001.wav",
      "audio_duration": 3.21,
      "total_latency":  0.386,
      "sftt":           0.241,
      "rtf":            0.1204,
      "wer":            0.021,
      "hypothesis":     "مرحبا كيف حالك",
      "reference":      "مرحبا كيف حالك",
      "vram_peak_mb":   1934.1,
      "error":          ""
    }
  ]
}
```

All float fields use Python-native `float` (IEEE 754 double).  `wer: -1.0`
means WER was not computed for that file.

---

## Architecture notes

### Why a standalone script (not a test against the live server)?

Benchmarking via HTTP would include network stack overhead, JSON
serialisation, and uvicorn's event-loop scheduling — none of which is part
of the model's actual latency.  `benchmark.py` imports the pipeline logic
directly, giving clean, reproducible hardware numbers.

### Pipeline fidelity

Every helper (`_decode_audio`, `_resample`, `_apply_vad`, `_tensor_to_numpy`)
and every `transcribe()` parameter (`compression_ratio_threshold`,
`log_prob_threshold`, `no_speech_threshold`, `temperature`, etc.) is copied
verbatim from `main.py`.  If you change `main.py` you should update the
corresponding sections in `benchmark.py` to keep the results meaningful.

### VAD skips vs inference skips

The script distinguishes two failure modes:

- **VAD skip** — Silero found no speech.  The file is counted as failed in
  the report, but it is not a model error — it may simply be a silent clip.
- **Inference error** — faster-whisper raised an exception.  Investigate
  these; they are usually dtype or shape mismatches.

---

## Interpreting results

| Metric | Good | Investigate |
|---|---|---|
| Average WER | < 20 % | > 40 % |
| Average RTF | < 0.5 | > 1.0 (slower than real-time) |
| Average SFTT | < 400 ms | > 1 000 ms |
| Absolute VRAM peak | Fits in your GPU's VRAM | OOM errors during load |

A high WER on specific files often points to:
- Heavy background noise surviving VAD
- Code-switching (English phrases inside Arabic)
- Very short utterances (< 1 s) where Whisper has little context

A high RTF on a CUDA machine usually means:
- The audio is being processed in small chunks, causing repeated kernel launches
- Compute type `float32` instead of `float16`
- The GPU is being shared with another process

---

## Python 3.11 compatibility notes

- All type hints use `X | Y` union syntax (PEP 604, available from 3.10+).
- `list[tuple[Path, str]]` built-in generic syntax (PEP 585, Python 3.9+).
- No `match` statements or other 3.10+ syntax that would break 3.9.
- Tested against the dependency versions pinned in `requirements.txt`.
