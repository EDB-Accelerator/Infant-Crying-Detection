#!/usr/bin/env python3
"""
random_extract.py
─────────────────
End-to-end helper script for **automated dataset curation** from long LENA™
recordings.  It repeatedly picks a random 10-minute clip, chops it into
5-second chunks, applies a lightweight cry-energy heuristic, and keeps the
first clip whose chunk-level detections reach the desired threshold.

Outputs
-------
* Extracted clip WAV files: ``<out_dir>/<subject>_<idx>.wav``
* A CSV summary  : ``<out_dir>/extraction_summary.csv``

Typical usage
-------------
```bash
python random_extract.py \
    --data-root /path/to/LENA \
    --out-dir    /path/to/out  \
    --subjects   3999 4319     \
    --min-detections 10        \
    --max-tries 10
```

Requirements
------------
```bash
pip install pydub pandas numpy scipy soundfile
```

Licence
-------
MIT © 2025 Kyunghun Lee
"""
from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import soundfile as sf  # robust WAV reader
from pydub import AudioSegment
from scipy import signal as sig

# ────────────────────────────── constants ────────────────────────────────
TEN_MIN_MS = 10 * 60 * 1000      # 10 minutes in milliseconds
CHUNK_MS   = 5 * 1000            # 5-second inference window
LOW_FREQ   = 250                 # band-pass range (Hz)
HIGH_FREQ  = 3_000               # …
CRY_RATIO  = 1.5                 # heuristic threshold

# ────────────────────────────── helpers ──────────────────────────────────

def wav_to_mono_array(wav_path: Path) -> Tuple[np.ndarray, int]:
    """Read *wav_path* and return (signal, sample_rate).  Stereo → mono."""
    audio, sr = sf.read(wav_path, always_2d=True)
    mono      = audio.mean(axis=1).astype(np.float32)
    return mono, sr


def bandpass(y: np.ndarray, sr: int, low: int = LOW_FREQ, high: int = HIGH_FREQ) -> np.ndarray:
    nyq = sr / 2
    b, a = sig.butter(4, (low / nyq, high / nyq), btype="band")
    return sig.filtfilt(b, a, y)


def cry_detect(wav_path: Path) -> bool:
    """Return **True** if *wav_path* likely contains infant cry."""
    y, sr = wav_to_mono_array(wav_path)
    y_f   = bandpass(y, sr)

    # sliding-window energy
    win   = int(sr * 0.1)   # 100 ms
    step  = int(sr * 0.05)  # 50 ms
    energy = np.array([np.sum(y_f[i:i + win] ** 2)
                       for i in range(0, len(y_f) - win, step)])
    if len(energy) < 10:  # avoid div-by-zero on very short clips
        return False
    short, long = energy.mean(), energy[-10:].mean()
    return (short / long) > CRY_RATIO


def chunk_and_detect(segment: AudioSegment, tmp_dir: Path) -> int:
    """Slice *segment* into 5-second WAVs in *tmp_dir* and count detections."""
    tmp_dir.mkdir(exist_ok=True)
    for f in tmp_dir.glob("segment_*.wav"):
        f.unlink()  # clean previous run

    detections = 0
    for idx, start in enumerate(range(0, len(segment), CHUNK_MS)):
        end = start + CHUNK_MS
        chunk_path = tmp_dir / f"segment_{idx+1}.wav"
        segment[start:end].export(chunk_path, format="wav")
        if cry_detect(chunk_path):
            detections += 1
    return detections


def random_clip(src_wav: Path, start_ms: int, end_ms: int) -> AudioSegment:
    """Return a *10-min* random clip between *start_ms* and *end_ms* (ms)."""
    audio = AudioSegment.from_wav(src_wav)
    max_start = (end_ms or len(audio)) - TEN_MIN_MS
    if max_start - start_ms < TEN_MIN_MS:
        raise ValueError("Not enough room for 10-minute clip in range.")
    clip_start = random.randint(start_ms, max_start)
    return audio[clip_start:clip_start + TEN_MIN_MS]


# ────────────────────────────── main routine ─────────────────────────────

def process_file(src_wav: Path, out_dir: Path, tmp_dir: Path, *,
                 min_detections: int, max_tries: int,
                 start_ms: int = 0, end_ms: int | None = None) -> Tuple[int, int, int]:
    """Extract a qualifying clip from *src_wav* and save it.
    Returns (num_detections, clip_start_ms, clip_end_ms).
    """
    subject = src_wav.parent.name
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(exist_ok=True)

    for attempt in range(1, max_tries + 1):
        clip = random_clip(src_wav, start_ms, end_ms or -1)
        dets = chunk_and_detect(clip, tmp_dir)
        logging.info("%s – try %d/%d: %d detections", src_wav.name, attempt, max_tries, dets)
        if dets >= min_detections:
            out_path = out_dir / f"{subject}_{src_wav.stem}_rand10m.wav"
            clip.export(out_path, format="wav")
            return dets, clip.start_second * 1000, clip.end_second * 1000  # type: ignore[attr-defined]
    # no success – still export last attempt for inspection
    out_path = out_dir / f"{subject}_{src_wav.stem}_rand10m_inspect.wav"
    clip.export(out_path, format="wav")
    return dets, clip.start_second * 1000, clip.end_second * 1000


def gather_wavs(data_root: Path, subjects: List[str] | None) -> List[Path]:
    pattern = "*/*.wav" if not subjects else f"*({'|'.join(subjects)})/*.wav"  # noqa: E501
    return sorted(data_root.glob(pattern))


def main():
    parser = argparse.ArgumentParser(description="Random 10-min extraction with minimal cry detections.")
    parser.add_argument("--data-root", type=Path, required=True,
                        help="Root folder containing subject sub-folders with WAV files.")
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="Destination for extracted clips and summary CSV.")
    parser.add_argument("--subjects", nargs="*", help="Process only these subject IDs (optional).")
    parser.add_argument("--min-detections", type=int, default=10,
                        help="Minimum # of 5-s chunks flagged as cry to accept a clip (default: 10).")
    parser.add_argument("--max-tries", type=int, default=10,
                        help="Maximum attempts per source file (default: 10).")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level, format="%(levelname)s | %(message)s")
    tmp_dir = args.out_dir / "_tmp"

    summary = []
    for wav_path in gather_wavs(args.data_root, args.subjects):
        try:
            dets, s_ms, e_ms = process_file(wav_path, args.out_dir, tmp_dir,
                                             min_detections=args.min_detections,
                                             max_tries=args.max_tries)
            summary.append({
                "subject"      : wav_path.parent.name,
                "source_wav"   : str(wav_path),
                "detections"   : dets,
                "clip_start_ms": s_ms,
                "clip_end_ms"  : e_ms,
            })
        except Exception as ex:
            logging.error("%s – skipped (%s)", wav_path.name, ex)

    if summary:
        df = pd.DataFrame(summary)
        df.to_csv(args.out_dir / "extraction_summary.csv", index=False)
        logging.info("Saved summary with %d rows to %%s", len(df))


if __name__ == "__main__":
    main()
