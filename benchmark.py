#!/usr/bin/env python3
"""
benchmark.py — Jordanian Arabic STT Pipeline Benchmarker
=========================================================
Evaluates the VAD → faster-whisper pipeline defined in main.py across a
labelled test set.  All pipeline code is reproduced here verbatim from
main.py so the benchmark measures exactly what production runs.

Metrics
-------
  WER   Word Error Rate            (requires: pip install jiwer)
  RTF   Real-Time Factor           inference_time / audio_duration_s
  SFTT  Speed to First Token       wall-clock seconds until first segment yields
  VRAM  Peak GPU memory (MB / GB)  tracked separately for load vs inference phases

Usage
-----
  python benchmark.py --manifest data/test_manifest.csv

Manifest CSV (UTF-8, with BOM accepted)
----------------------------------------
  file_path,reference_text
  audio/utt_001.wav,"مرحبا كيف حالك"
  audio/utt_002.mp3,"أريد معلومات عن التسجيل"

  • file_path    — absolute path OR path relative to the manifest's directory
  • reference_text — ground-truth Arabic transcript (leave blank to skip WER)

Full options
------------
  --manifest       Path to CSV manifest (required)
  --model          Whisper model ID or local path
                   (default: HebArabNlpProject/WhisperLevantine)
  --compute-type   float16 | int8 | float32  (default: float16)
  --device         auto | cuda | cpu          (default: auto)
  --beam-size      integer ≥ 1                (default: 1)
  --output-json    Optional path to save full results as JSON

Python ≥ 3.11  |  Compatible with requirements.txt
"""

# ── Standard library ──────────────────────────────────────────────────────────
import argparse
import csv
import io
import json
import logging
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Generator, Optional

# ── Third-party (all pinned in requirements.txt) ──────────────────────────────
import numpy as np
import torch
import torchaudio
from faster_whisper import WhisperModel

# jiwer is an *optional* dependency — WER is simply skipped when absent.
# Install with:  pip install jiwer
try:
    import jiwer
    _JIWER_AVAILABLE = True
except ImportError:
    _JIWER_AVAILABLE = False

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("benchmark")


# ──────────────────────────────────────────────────────────────────────────────
# ① Pipeline constants — identical to main.py
# ──────────────────────────────────────────────────────────────────────────────
VAD_SAMPLE_RATE: int   = 16_000
VAD_THRESHOLD:   float = 0.50
MIN_SPEECH_MS:   int   = 250
SPEECH_PAD_MS:   int   = 100


# ──────────────────────────────────────────────────────────────────────────────
# ② Benchmark configuration dataclass
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class BenchmarkConfig:
    """
    All tuneable parameters for one benchmark run.
    Defaults mirror the constants in main.py exactly.
    """
    whisper_model_size:   str = "HebArabNlpProject/WhisperLevantine"
    whisper_compute_type: str = "float16"
    whisper_beam_size:    int = 1
    whisper_language:     str = "ar"
    device:               str = "auto"        # resolved to "cuda" or "cpu" at runtime
    download_root:        str = "./models"
    num_workers:          int = 4


# ──────────────────────────────────────────────────────────────────────────────
# ③ VRAM tracking utilities
# ──────────────────────────────────────────────────────────────────────────────
def _reset_vram_peak() -> None:
    """
    Zero out PyTorch's internal peak-memory counter for the active CUDA device.
    No-op on CPU-only machines.
    """
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _read_vram_peak_mb() -> float:
    """
    Return peak *allocated* VRAM in MB since the last reset (or process start).
    Uses torch.cuda.max_memory_allocated() which reflects the high-watermark of
    tensors actually allocated by PyTorch — not the reserved (cached) pool.
    Returns 0.0 when no CUDA device is present.
    """
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024 ** 2)


@contextmanager
def vram_phase(label: str, tracker: dict) -> Generator:
    """
    Context manager that brackets a code block with VRAM peak measurement.

    Steps
    -----
    1. Call _reset_vram_peak()   — clears the high-watermark counter
    2. Yield into the body       — arbitrary code runs here
    3. Call _read_vram_peak_mb() — captures the new high-watermark
    4. Store result in tracker[label]

    Example
    -------
        with vram_phase("model_load", tracker):
            models = load_models(cfg)
        print(tracker["model_load"])   # → e.g. 1843.2  (MB)
    """
    _reset_vram_peak()
    log.debug("[VRAM] Starting phase: '%s'", label)
    try:
        yield
    finally:
        peak_mb = _read_vram_peak_mb()
        tracker[label] = peak_mb
        log.debug("[VRAM] Phase '%s' peak: %.1f MB", label, peak_mb)


# ──────────────────────────────────────────────────────────────────────────────
# ④ Model loader — mirrors _load_vad() + _load_whisper() from main.py
# ──────────────────────────────────────────────────────────────────────────────
def load_models(cfg: BenchmarkConfig) -> dict:
    """
    Instantiate Silero VAD and faster-whisper and return a dict of handles.

    The implementation is a direct copy of main.py's two loader functions,
    with one addition: float16 → int8 fallback when running on CPU (CTranslate2
    silently degrades on CPU float16, so we make the switch explicit).

    Returns
    -------
    dict with keys:
        vad_model              torch.nn.Module
        get_speech_timestamps  callable
        collect_chunks         callable
        whisper_model          WhisperModel
        device                 "cuda" or "cpu"
    """
    # ── Resolve device ────────────────────────────────────────────────────────
    if cfg.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = cfg.device

    # float16 is meaningless / broken on CPU with CTranslate2; fall back silently
    compute_type = cfg.whisper_compute_type
    if device == "cpu" and compute_type == "float16":
        compute_type = "int8"
        log.warning("[STT ] float16 is unsupported on CPU — using int8 instead")

    # ── Silero VAD ────────────────────────────────────────────────────────────
    log.info("[VAD ] Loading Silero VAD from torch.hub ...")
    vad_model, vad_utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        onnx=False,           # TorchScript backend — faster for short clips
        trust_repo=True,
    )
    (
        get_speech_timestamps,
        _save_audio,          # unused
        _read_audio,          # unused
        _VADIterator,         # unused
        collect_chunks,
    ) = vad_utils

    vad_model.eval()
    vad_model.to(torch.device(device))
    log.info("[VAD ] Ready — device=%s", device)

    # ── faster-whisper ────────────────────────────────────────────────────────
    log.info(
        "[STT ] Loading faster-whisper '%s' → device=%s compute=%s ...",
        cfg.whisper_model_size, device, compute_type,
    )
    whisper_model = WhisperModel(
        cfg.whisper_model_size,
        device=device,
        compute_type=compute_type,
        cpu_threads=0,             # let CTranslate2 auto-select thread count
        num_workers=cfg.num_workers,
        download_root=cfg.download_root,
    )
    log.info("[STT ] faster-whisper ready — device=%s", device)

    return {
        "vad_model":             vad_model,
        "get_speech_timestamps": get_speech_timestamps,
        "collect_chunks":        collect_chunks,
        "whisper_model":         whisper_model,
        "device":                device,
    }


# ──────────────────────────────────────────────────────────────────────────────
# ⑤ Pipeline helpers — copied verbatim from main.py (no FastAPI dependencies)
# ──────────────────────────────────────────────────────────────────────────────
def _decode_audio(raw_bytes: bytes) -> tuple[torch.Tensor, int]:
    """
    Decode any torchaudio-supported codec held in *raw_bytes* to a mono
    float32 tensor.  Zero disk I/O — mirrors main.py._decode_audio().
    """
    buf = io.BytesIO(raw_bytes)
    waveform, sr = torchaudio.load(buf)         # shape: (channels, samples)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform.squeeze(0), sr              # shape: (samples,)


def _resample(waveform: torch.Tensor, src_sr: int) -> torch.Tensor:
    """Resample to 16 kHz; no-op when source already matches."""
    if src_sr == VAD_SAMPLE_RATE:
        return waveform
    return torchaudio.functional.resample(waveform, src_sr, VAD_SAMPLE_RATE)


def _apply_vad(
    waveform: torch.Tensor,
    vad_model: torch.nn.Module,
    get_speech_timestamps,
    collect_chunks,
) -> Optional[torch.Tensor]:
    """
    Run Silero VAD and return concatenated speech samples.
    Returns None (instead of raising HTTP 422) when no speech is found,
    allowing the benchmark to mark the file as skipped without crashing.
    """
    vad_device   = next(vad_model.parameters()).device
    waveform_dev = waveform.to(vad_device)

    timestamps = get_speech_timestamps(
        waveform_dev,
        vad_model,
        sampling_rate=VAD_SAMPLE_RATE,
        threshold=VAD_THRESHOLD,
        min_speech_duration_ms=MIN_SPEECH_MS,
        speech_pad_ms=SPEECH_PAD_MS,
        return_seconds=False,
    )

    if not timestamps:
        return None  # caller handles this gracefully

    # collect_chunks always returns a CPU tensor
    return collect_chunks(timestamps, waveform_dev.cpu())


def _tensor_to_numpy(speech: torch.Tensor) -> np.ndarray:
    """
    Convert speech tensor to 1-D float32 NumPy array.
    faster-whisper rejects float64, so we enforce dtype.
    """
    audio_np = speech.numpy()
    if audio_np.dtype != np.float32:
        audio_np = audio_np.astype(np.float32)
    return audio_np


# ──────────────────────────────────────────────────────────────────────────────
# ⑥ Per-file result dataclass
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class FileResult:
    """
    All metrics recorded for a single audio file.

    Fields
    ------
    file_path       : str   — absolute path used during the run
    audio_duration  : float — seconds of speech passed to Whisper (post-VAD)
    total_latency   : float — wall-clock seconds for the inference stage
    sftt            : float — seconds from inference start until first segment
    rtf             : float — real-time factor  (total_latency / audio_duration)
    wer             : float — word error rate in [0, ∞);  -1.0 = unavailable
    hypothesis      : str   — model transcript
    reference       : str   — ground-truth transcript (may be empty)
    vram_peak_mb    : float — peak VRAM (MB) during THIS file's inference
    error           : str   — non-empty string if the file was skipped
    """
    file_path:      str   = ""
    audio_duration: float = 0.0
    total_latency:  float = 0.0
    sftt:           float = 0.0
    rtf:            float = 0.0
    wer:            float = -1.0   # sentinel — "not computed"
    hypothesis:     str   = ""
    reference:      str   = ""
    vram_peak_mb:   float = 0.0
    error:          str   = ""


# ──────────────────────────────────────────────────────────────────────────────
# ⑦ Core benchmark runner — one file at a time
# ──────────────────────────────────────────────────────────────────────────────
def run_file(
    file_path:    Path,
    reference:    str,
    models:       dict,
    cfg:          BenchmarkConfig,
    vram_tracker: dict,
) -> FileResult:
    """
    Execute the full pipeline on one audio file and return a FileResult.

    Pipeline stages (matching main.py exactly)
    -------------------------------------------
    1. Read bytes from disk
    2. Decode + down-mix to mono float32
    3. Resample to 16 kHz
    4. Silero VAD  →  collect speech chunks
    5. Tensor → NumPy float32
    6. faster-whisper transcribe  (lazy generator → materialise)

    Metrics captured
    ----------------
    • total_latency  — step 6 wall-clock time
    • sftt           — time from transcribe() call to first generator yield
    • rtf            — total_latency / audio_duration
    • vram_peak_mb   — torch peak allocated during step 6 only
    • wer            — jiwer.wer(reference, hypothesis) when jiwer is installed
    """
    result = FileResult(file_path=str(file_path), reference=reference)

    # ── 1. Read ───────────────────────────────────────────────────────────────
    try:
        raw_bytes = file_path.read_bytes()
    except OSError as exc:
        result.error = f"Cannot read file: {exc}"
        return result

    # ── 2. Decode ─────────────────────────────────────────────────────────────
    try:
        waveform, src_sr = _decode_audio(raw_bytes)
    except Exception as exc:
        result.error = f"Decode error: {exc}"
        return result

    # ── 3. Resample ───────────────────────────────────────────────────────────
    try:
        waveform = _resample(waveform, src_sr)
    except Exception as exc:
        result.error = f"Resample error: {exc}"
        return result

    # ── 4. VAD ────────────────────────────────────────────────────────────────
    try:
        speech_tensor = _apply_vad(
            waveform,
            models["vad_model"],
            models["get_speech_timestamps"],
            models["collect_chunks"],
        )
    except Exception as exc:
        result.error = f"VAD error: {exc}"
        return result

    if speech_tensor is None:
        result.error = "No speech detected (VAD returned no timestamps)"
        return result

    # ── 5. Tensor → NumPy ────────────────────────────────────────────────────
    audio_np = _tensor_to_numpy(speech_tensor)
    audio_duration = len(audio_np) / VAD_SAMPLE_RATE
    result.audio_duration = audio_duration

    # ── 6. Transcribe (timed + VRAM tracked) ─────────────────────────────────
    # Reset VRAM counter just before inference so we isolate this file's peak.
    _reset_vram_peak()
    t_infer_start = time.perf_counter()
    first_token_time: Optional[float] = None

    try:
        # transcribe() returns a lazy generator — nothing runs until we iterate.
        segments_gen, _info = models["whisper_model"].transcribe(
            audio_np,
            language=cfg.whisper_language,
            beam_size=cfg.whisper_beam_size,
            vad_filter=False,               # dedicated Silero VAD already ran
            without_timestamps=False,
            condition_on_previous_text=False,
            temperature=0.0,
            compression_ratio_threshold=2.4,
            log_prob_threshold=-1.0,
            no_speech_threshold=0.6,
        )

        # Materialise the generator.
        # SFTT = time from transcribe() call to the first segment yield.
        text_parts: list[str] = []
        for seg in segments_gen:
            if first_token_time is None:
                # The clock started at t_infer_start (before the transcribe()
                # call) so SFTT includes model scheduling overhead — exactly
                # what matters for latency budgeting.
                first_token_time = time.perf_counter() - t_infer_start
            text_parts.append(seg.text.strip())

    except Exception as exc:
        result.error = f"Inference error: {exc}"
        return result

    t_infer_end = time.perf_counter()
    inference_time = t_infer_end - t_infer_start

    # ── Collect VRAM peak for this specific file ──────────────────────────────
    file_vram_peak = _read_vram_peak_mb()
    result.vram_peak_mb = file_vram_peak

    # Update the running global maximum for the "inference_peak" phase.
    # This value ends up in the report's "Absolute peak during inference" row.
    old_peak = vram_tracker.get("inference_peak", 0.0)
    vram_tracker["inference_peak"] = max(old_peak, file_vram_peak)

    # ── Populate metrics ──────────────────────────────────────────────────────
    result.hypothesis    = " ".join(text_parts).strip()
    result.total_latency = inference_time
    # If the generator never yielded a segment (silent-after-VAD edge-case),
    # fall back to the total inference time for SFTT.
    result.sftt = first_token_time if first_token_time is not None else inference_time
    result.rtf  = (
        inference_time / audio_duration if audio_duration > 0.0 else float("inf")
    )

    # ── WER ───────────────────────────────────────────────────────────────────
    if _JIWER_AVAILABLE and reference.strip():
        try:
            # jiwer.wer() returns a float in [0, ∞).
            # Values > 1.0 are possible when the hypothesis is much longer
            # than the reference due to insertions.
            result.wer = jiwer.wer(reference.strip(), result.hypothesis)
        except Exception as exc:
            log.warning("[WER ] Computation failed for %s: %s", file_path.name, exc)

    return result


# ──────────────────────────────────────────────────────────────────────────────
# ⑧ Manifest loader
# ──────────────────────────────────────────────────────────────────────────────
def load_manifest(manifest_path: Path) -> list[tuple[Path, str]]:
    """
    Parse a UTF-8 CSV manifest file.

    Expected columns (order-independent, header required)
    ------------------------------------------------------
      file_path      — path to audio file (absolute or relative to manifest dir)
      reference_text — ground-truth transcript (may be empty)

    Returns
    -------
    List of (audio_path, reference_text) tuples.
    """
    entries: list[tuple[Path, str]] = []
    manifest_dir = manifest_path.parent

    # utf-8-sig handles optional BOM written by Excel on Windows
    with manifest_path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            raw_path = row.get("file_path", "").strip()
            if not raw_path:
                continue
            audio_path = Path(raw_path)
            if not audio_path.is_absolute():
                audio_path = (manifest_dir / audio_path).resolve()
            entries.append((audio_path, row.get("reference_text", "").strip()))

    log.info("Manifest loaded — %d entries", len(entries))
    return entries


# ──────────────────────────────────────────────────────────────────────────────
# ⑨ Terminal report
# ──────────────────────────────────────────────────────────────────────────────
_SEP_THIN  = "─" * 74
_SEP_THICK = "═" * 74

# ANSI colour codes — suppressed when stdout is not a TTY (e.g. log files)
_CYAN   = "\033[96m"
_GREEN  = "\033[92m"
_YELLOW = "\033[93m"
_RED    = "\033[91m"
_BOLD   = "\033[1m"
_RESET  = "\033[0m"


def _c(text: str, code: str) -> str:
    """Wrap *text* in ANSI *code* when running in an interactive terminal."""
    if sys.stdout.isatty():
        return f"{code}{text}{_RESET}"
    return text


def _mb_str(mb: float) -> str:
    """Human-readable VRAM string: N/A on CPU, MB or GB on GPU."""
    if mb == 0.0:
        return "N/A (CPU-only)"
    return f"{mb / 1024:.2f} GB" if mb >= 1024 else f"{mb:.1f} MB"


def print_report(
    results:      list[FileResult],
    vram_tracker: dict,
    cfg:          BenchmarkConfig,
) -> None:
    """
    Render the full benchmark report to stdout.

    Sections
    --------
      Configuration summary
      File overview  (total / ok / failed, with skip reasons)
      Accuracy       (WER: average, best, worst)
      Speed          (RTF, SFTT, duration distribution)
      GPU Memory     (load-phase peak, inference-phase peak, absolute peak)
      Per-file table (one row per successful file)
    """
    ok     = [r for r in results if not r.error]
    failed = [r for r in results if r.error]
    wer_ok = [r for r in ok if r.wer >= 0.0]

    # ── Aggregate statistics ──────────────────────────────────────────────────
    avg_wer  = sum(r.wer  for r in wer_ok) / len(wer_ok) if wer_ok else None
    avg_rtf  = sum(r.rtf  for r in ok)     / len(ok)     if ok else None
    avg_sftt = sum(r.sftt for r in ok)     / len(ok)     if ok else None
    avg_dur  = sum(r.audio_duration for r in ok) / len(ok) if ok else None

    peak_load_mb  = vram_tracker.get("model_load", 0.0)
    peak_infer_mb = vram_tracker.get("inference_peak", 0.0)
    abs_peak_mb   = max(peak_load_mb, peak_infer_mb)
    resolved_dev  = vram_tracker.get("resolved_device", "unknown")

    # ── Header ────────────────────────────────────────────────────────────────
    print()
    print(_c(_SEP_THICK, _BOLD))
    print(_c("  JORDANIAN ARABIC STT — BENCHMARK REPORT", _BOLD))
    print(_c(_SEP_THICK, _BOLD))

    # ── Config ────────────────────────────────────────────────────────────────
    print(f"\n  {'Model':<28} {cfg.whisper_model_size}")
    print(f"  {'Compute type':<28} {cfg.whisper_compute_type}")
    print(f"  {'Device':<28} {resolved_dev}")
    print(f"  {'Beam size':<28} {cfg.whisper_beam_size}")
    print(f"  {'Language':<28} {cfg.whisper_language}")
    print(f"  {'VAD threshold':<28} {VAD_THRESHOLD}")

    # ── Files overview ────────────────────────────────────────────────────────
    print(f"\n{_c(_SEP_THIN, _CYAN)}")
    print(_c("  FILES", _BOLD))
    print(f"{_c(_SEP_THIN, _CYAN)}")
    print(f"  Total submitted          {len(results)}")
    print(f"  Processed successfully   {_c(str(len(ok)), _GREEN)}")
    skip_col = _RED if failed else _GREEN
    print(f"  Skipped / failed         {_c(str(len(failed)), skip_col)}")

    if failed:
        print(f"\n  {'File':<38}  {'Reason'}")
        print(f"  {'─'*37}  {'─'*30}")
        for r in failed:
            name = Path(r.file_path).name[:37]
            print(f"  {name:<38}  {r.error}")

    # ── Accuracy ──────────────────────────────────────────────────────────────
    print(f"\n{_c(_SEP_THIN, _CYAN)}")
    print(_c("  ACCURACY", _BOLD))
    print(f"{_c(_SEP_THIN, _CYAN)}")

    if not _JIWER_AVAILABLE:
        print(
            "  WER                      N/A\n"
            "  → Install jiwer to enable WER:  pip install jiwer"
        )
    elif avg_wer is None:
        print(
            "  WER                      N/A\n"
            "  → No reference_text values found in manifest."
        )
    else:
        wer_pct = avg_wer * 100
        wer_col = _GREEN if wer_pct < 20 else (_YELLOW if wer_pct < 40 else _RED)
        print(f"  Average WER              {_c(f'{wer_pct:.2f} %', wer_col)}")
        if len(wer_ok) < len(ok):
            print(
                f"  (WER computed on {len(wer_ok)} of {len(ok)} "
                "files — remaining had no reference)"
            )

        best  = min(wer_ok, key=lambda r: r.wer)
        worst = max(wer_ok, key=lambda r: r.wer)
        print(f"  Best  WER                {Path(best.file_path).name}"
              f"  →  {best.wer * 100:.1f} %")
        print(f"  Worst WER                {Path(worst.file_path).name}"
              f"  →  {worst.wer * 100:.1f} %")

    # ── Speed ─────────────────────────────────────────────────────────────────
    print(f"\n{_c(_SEP_THIN, _CYAN)}")
    print(_c("  SPEED", _BOLD))
    print(f"{_c(_SEP_THIN, _CYAN)}")

    if not ok:
        print("  No successful inferences — cannot compute speed metrics.")
    else:
        rtf_col = (
            _GREEN  if avg_rtf < 0.5  else
            _YELLOW if avg_rtf < 1.0  else
            _RED
        )
        min_rtf = min(r.rtf for r in ok)
        max_rtf = max(r.rtf for r in ok)

        print(f"  Average audio duration   {avg_dur:.2f} s  (post-VAD)")
        print(
            f"  Average RTF              {_c(f'{avg_rtf:.4f}', rtf_col)}"
            "  (< 1.0 = faster than real-time)"
        )
        print(f"  RTF range                {min_rtf:.4f} – {max_rtf:.4f}")
        print(
            f"  Average SFTT             {avg_sftt * 1000:.1f} ms"
            "  (wall-clock to first segment)"
        )

        # Show percentile spread on SFTT if we have enough samples
        if len(ok) >= 5:
            sftt_sorted = sorted(r.sftt for r in ok)
            p50 = sftt_sorted[len(sftt_sorted) // 2]
            p95_idx = min(int(len(sftt_sorted) * 0.95), len(sftt_sorted) - 1)
            p95 = sftt_sorted[p95_idx]
            print(f"  SFTT p50 / p95           {p50*1000:.1f} ms / {p95*1000:.1f} ms")

    # ── GPU Memory ────────────────────────────────────────────────────────────
    print(f"\n{_c(_SEP_THIN, _CYAN)}")
    print(_c("  GPU MEMORY  (VRAM)", _BOLD))
    print(f"{_c(_SEP_THIN, _CYAN)}")
    print(f"  Peak during model load   {_mb_str(peak_load_mb)}")
    print(f"  Peak during inference    {_mb_str(peak_infer_mb)}")
    print(
        f"  Absolute peak (run)      {_c(_mb_str(abs_peak_mb), _YELLOW)}"
    )
    if resolved_dev == "cpu":
        print("  (VRAM metrics unavailable — running on CPU)")

    # ── Per-file table ────────────────────────────────────────────────────────
    if ok:
        print(f"\n{_c(_SEP_THIN, _CYAN)}")
        print(_c("  PER-FILE RESULTS", _BOLD))
        print(f"{_c(_SEP_THIN, _CYAN)}")
        header = (
            f"  {'File':<32}"
            f" {'Dur(s)':>7}"
            f" {'RTF':>7}"
            f" {'SFTT ms':>8}"
            f" {'WER %':>7}"
            f" {'VRAM MB':>8}"
        )
        print(header)
        print(f"  {'─'*31} {'─'*6} {'─'*6} {'─'*7} {'─'*6} {'─'*7}")
        for r in ok:
            name    = Path(r.file_path).name[:31]
            wer_str = f"{r.wer * 100:.1f}" if r.wer >= 0 else "  —"
            vram_s  = f"{r.vram_peak_mb:.0f}" if r.vram_peak_mb > 0 else "CPU"
            print(
                f"  {name:<32}"
                f" {r.audio_duration:>7.2f}"
                f" {r.rtf:>7.4f}"
                f" {r.sftt * 1000:>8.1f}"
                f" {wer_str:>7}"
                f" {vram_s:>8}"
            )

    print()
    print(_c(_SEP_THICK, _BOLD))
    print()


# ──────────────────────────────────────────────────────────────────────────────
# ⑩ CLI argument parser
# ──────────────────────────────────────────────────────────────────────────────
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Benchmark the Jordanian Arabic STT pipeline "
            "(VAD → faster-whisper) on a labelled test set."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--manifest", required=True, type=Path,
        metavar="PATH",
        help="CSV manifest with columns: file_path, reference_text.",
    )
    p.add_argument(
        "--model",
        default="HebArabNlpProject/WhisperLevantine",
        metavar="MODEL_ID",
        help="faster-whisper model ID or local directory path.",
    )
    p.add_argument(
        "--compute-type",
        default="float16",
        choices=["float16", "int8", "float32"],
        dest="compute_type",
        help="CTranslate2 weight precision.",
    )
    p.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Inference device. 'auto' picks CUDA when available.",
    )
    p.add_argument(
        "--beam-size",
        default=1, type=int,
        dest="beam_size",
        metavar="N",
        help="Whisper beam search width (1 = greedy).",
    )
    p.add_argument(
        "--output-json",
        default=None, type=Path,
        dest="output_json",
        metavar="PATH",
        help="Optional path to write full results as a JSON file.",
    )
    return p


# ──────────────────────────────────────────────────────────────────────────────
# ⑪ Entry point
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    args = _build_parser().parse_args()

    # ── Build config from CLI ─────────────────────────────────────────────────
    cfg = BenchmarkConfig(
        whisper_model_size   = args.model,
        whisper_compute_type = args.compute_type,
        whisper_beam_size    = args.beam_size,
        device               = args.device,
    )

    # This dict accumulates VRAM peaks keyed by phase name.
    # It is shared (mutated in-place) between load_models() and run_file().
    vram_tracker: dict = {}

    # ── Load manifest ─────────────────────────────────────────────────────────
    if not args.manifest.exists():
        log.error("Manifest not found: %s", args.manifest)
        sys.exit(1)

    entries = load_manifest(args.manifest)
    if not entries:
        log.error("Manifest is empty — nothing to benchmark.")
        sys.exit(1)

    # ── Load models (VRAM-tracked) ────────────────────────────────────────────
    log.info("=" * 60)
    log.info("Phase 1 / 2 — Model loading (VRAM tracked)")
    log.info("=" * 60)
    with vram_phase("model_load", vram_tracker):
        models = load_models(cfg)

    # Store the resolved device string for the report
    vram_tracker["resolved_device"] = models["device"]
    log.info(
        "Model load complete.  VRAM peak: %s",
        _mb_str(vram_tracker.get("model_load", 0.0)),
    )

    # ── Run inference on every file ───────────────────────────────────────────
    log.info("=" * 60)
    log.info("Phase 2 / 2 — Inference  (%d files)", len(entries))
    log.info("=" * 60)

    results: list[FileResult] = []
    for idx, (audio_path, reference) in enumerate(entries, start=1):
        log.info("[%d/%d] Processing: %s", idx, len(entries), audio_path.name)
        r = run_file(audio_path, reference, models, cfg, vram_tracker)
        results.append(r)

        if r.error:
            log.warning("       SKIPPED — %s", r.error)
        else:
            wer_tag = f"  WER={r.wer * 100:.1f}%" if r.wer >= 0 else ""
            log.info(
                "       RTF=%.4f  SFTT=%d ms%s  VRAM=%s",
                r.rtf,
                int(r.sftt * 1000),
                wer_tag,
                _mb_str(r.vram_peak_mb),
            )

    # ── Print terminal report ─────────────────────────────────────────────────
    print_report(results, vram_tracker, cfg)

    # ── Optional JSON export ──────────────────────────────────────────────────
    if args.output_json:
        payload = {
            "config": asdict(cfg),
            "vram_tracker": {
                k: v for k, v in vram_tracker.items()
                if isinstance(v, (int, float, str))
            },
            "results": [asdict(r) for r in results],
        }
        args.output_json.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        log.info("Full results written to: %s", args.output_json)


if __name__ == "__main__":
    main()
