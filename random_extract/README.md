# Random 10‑Minute Clip Extractor for LENA Recordings

`random_extract.py` is a stand‑alone utility for generating fixed‑length excerpts from day‑long audio files or other long‑form audio recordings. It selects a random 10‑minute window, segments it into 5‑second chunks, applies a lightweight cry‑energy heuristic, and keeps the first window that meets a user‑defined minimum number of detected cries. The script is intended to streamline dataset curation by producing manageable WAV files with a guaranteed amount of infant‑cry content.

---

## Features

* **Randomised sampling**   – draws windows uniformly within a user‑specified time range.
* **Cry‑energy heuristic** – band‑pass filter (250–3000 Hz) and sliding‑window energy ratio.
* **Configurable thresholds** – minimum detections and maximum attempts per source file.
* **Batch processing** – handles entire directories (optionally filtered by subject IDs).
* **Reproducible summary** – writes an `extraction_summary.csv` listing source files, detection counts, and clip time‑stamps.

---

## Installation

Python 3.9+ is recommended.

```bash
pip install pydub pandas numpy scipy soundfile
```

`pydub` requires `ffmpeg` in your system path. On macOS:

```bash
brew install ffmpeg
```

---

## Quick‑start

```bash
python random_extract.py \
    --data-root   /path/to/LENA \
    --out-dir     /path/to/output \
    --min-detections 10 \
    --max-tries      10
```

### Important Arguments

| Flag               | Default    | Description                                                           |
| ------------------ | ---------- | --------------------------------------------------------------------- |
| `--data-root`      | *required* | Root folder containing subject sub‑folders with WAV files.            |
| `--out-dir`        | *required* | Destination for extracted clips and summary CSV.                      |
| `--subjects`       | *all*      | Space‑separated list of subject IDs to process.                       |
| `--min-detections` | `10`       | Minimum number of 5‑s chunks flagged as cry in a window to accept it. |
| `--max-tries`      | `10`       | Maximum number of random windows tested per source file.              |
| `--log-level`      | `INFO`     | Logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`).              |

---

## Output

* **Extracted clips** — `out_dir/<subject>_<source>_rand10m.wav`
* **Inspection clips** — If no window meets the detection threshold, the last attempt is saved with `_inspect` in the filename.
* **`extraction_summary.csv`** — One row per processed source WAV with columns:

  * `subject`
  * `source_wav`
  * `detections` – number of detected cry chunks
  * `clip_start_ms`, `clip_end_ms` – time‑bounds of the kept window (milliseconds)

---

## Algorithm Details

1. **Band‑pass filtering**: 4th‑order Butterworth, 250–3000 Hz.
2. **Energy computation**: 100 ms windows with 50 ms hop.
3. **Detection rule**: ratio of short‑term to long‑term energy must exceed `1.5` (`CRY_RATIO`).
4. **Chunk summarisation**: count the number of 5‑s chunks within the 10‑minute window that satisfy the rule.

These design choices favour speed over accuracy; replace `cry_detect` with a model‑based detector for higher precision.

---

## Performance Tips

* Run with `--log-level DEBUG` to audit intermediate detection counts.
* Parallelisation is possible but not implemented; process a subset of subjects per machine to scale horizontally.
* Store `out_dir/_tmp` on a fast local drive to minimise I/O overhead when writing chunk WAVs.

---

## License

Released under the MIT License (see top of `random_extract.py`).
