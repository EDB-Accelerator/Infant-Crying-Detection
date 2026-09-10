# Infant Crying Detection

**Deep-learning–based detection of infant crying in long-duration, naturalistic audio.**

[![Python](https://img.shields.io/badge/Python-3.9%2B-111827?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-implementation-111827?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Hugging%20Face-Transformers-111827?style=flat-square&logo=huggingface&logoColor=white)](https://huggingface.co/docs/transformers/)
[![License](https://img.shields.io/badge/License-MIT-111827?style=flat-square)](LICENSE)

> An end-to-end research pipeline for identifying crying segments in daylong, real-world audio recordings.

<p align="center">
  <img src="assets/infant-crying-detection-hero.jpg" alt="AI-generated illustration of infant audio monitoring with waveform and spectrogram overlays" width="100%">
</p>

<p align="center"><sub><strong>Illustration:</strong> AI-generated image for project presentation only; it is not a photograph from the study or dataset.</sub></p>

---

## Overview

Infant crying is a salient acoustic signal, but detecting it reliably in naturalistic recordings is substantially harder than classifying clean, isolated clips. Background speech, household noise, movement, reverberation, recording-device differences, and long recording sessions all introduce variability.

This repository provides a **PyTorch-based, configurable pipeline** for infant-cry detection in daylong audio. It re-implements and extends the approach introduced by Yao et al. (ICASSP 2022), with additional feature representations, model backends, and subject-level evaluation utilities.

The original development focused on recordings collected as part of the **When2Worry** study and is intended as research software that can be adapted to other WAV corpora.

## Why this repository exists

The goal is not simply to produce a classifier that works on a curated benchmark. The code is organized around a more practical research question:

> **Can we identify meaningful crying events in long, imperfect, real-world recordings well enough to support downstream behavioral and clinical analyses?**

That framing drives the repository toward:

- long-form audio processing rather than isolated clips;
- configurable feature and model pipelines;
- subject-level train/test separation;
- reproducible batch inference and evaluation; and
- outputs that can be consumed by downstream analyses.

## Method at a glance

```text
Raw WAV recordings
        │
        ▼
Signal preprocessing
        │
        ├── filtering / conditioning
        ├── windowing
        └── spectrogram generation
        │
        ▼
Feature representation
        │
        ├── MFCC
        ├── chroma
        ├── spectral contrast
        └── Wav2Vec 2.0 embeddings
        │
        ▼
Model / classifier
        │
        ├── CNN / spectrogram model
        ├── SVM-based pipeline
        └── gradient-boosting pipeline
        │
        ▼
Segment-level prediction
        │
        └── 1 = crying · 0 = non-cry
        │
        ▼
CSV summaries for downstream analysis
```

The repository currently includes preprocessing, CNN training, hybrid CNN+SVM training, end-to-end prediction, and utilities for extracting shorter candidate clips from long recordings.

## Repository structure

| Path | Purpose |
| --- | --- |
| `cry_detection_pipeline.py` | Main training / inference pipeline using Wav2Vec 2.0, traditional features, and gradient boosting. |
| `random_extract.py` | Extracts 10-minute clips containing at least *n* cry-like 5-second windows. |
| `src/preprocessing.py` | Signal conditioning, band-pass filtering, and mel-spectrogram generation. |
| `src/train_alex.py` | CNN / AlexNet-style spectrogram training with augmentation and early stopping. |
| `src/train_svm.py` | Hybrid CNN + SVM training pipeline. |
| `src/predict.py` | End-to-end WAV → preprocessing → CNN features → SVM inference. |
| `example/train.ipynb` | End-to-end training notebook. |
| `example/prediction.ipynb` | Batch inference notebook. |
| `lee2024/` | Project-specific research materials and analyses. |
| `assets/` | README and project presentation assets. |

## Installation

Python 3.9+ is recommended.

Install the project dependencies with:

```bash
pip install -r requirements.txt
```

Core dependencies include PyTorch, Transformers, librosa, pydub, NumPy, SciPy, pandas, scikit-learn, soundfile, and joblib.

GPU acceleration is optional, but may be useful for model training and embedding extraction.

## Data

The pipeline is designed for WAV audio and supports the project's original long-duration recordings. The repository documentation describes the target sample format as:

- 16-bit PCM
- 16 kHz
- mono

For additional data exploration, the project documentation also references the **HomeBank / TalkBank deBarbaro dataset** as a related corpus.

> **Data note:** This repository does not redistribute restricted participant recordings. Use only data for which you have the appropriate access and permissions.

## Training and inference

The project exposes training, prediction, and evaluation through its Python scripts and example notebooks. The exact flags and experiment configuration should be taken from the code in the current checkout rather than hard-coded into this README.

For a guided start, open:

- [`example/train.ipynb`](example/train.ipynb)
- [`example/prediction.ipynb`](example/prediction.ipynb)

The notebooks include Google Colab links for GPU-enabled execution.

## Outputs

The classifier produces binary segment-level predictions:

```text
1 → crying detected
0 → non-cry
```

CSV summaries can also retain per-segment probabilities and subject identifiers for downstream analysis.

## Reproducibility and evaluation

For research use, evaluation should be performed with **subject-level splits** so recordings from the same participant do not leak across training and test sets. The repository was extended with this evaluation structure specifically to better reflect generalization across participants.

When reporting new experiments, document at least:

- participant-level split strategy;
- feature representation;
- model backend;
- preprocessing parameters;
- decision threshold; and
- the evaluation metric used.

This README intentionally does **not** hard-code a single performance number: results depend on the dataset, split, feature set, and model configuration used for a particular experiment.

## Research context

This implementation builds on prior work in infant-cry detection and naturalistic audio analysis:

1. **Yao et al. (ICASSP 2022)** — *Infant Crying Detection in Real-World Environments.*
2. **Micheletti et al. (Behavior Research Methods, 2022)** — validation of infant-cry detection from naturalistic audio.
3. **Henry et al. (2025)** — *Detecting cry in daylong audio recordings using machine learning: The development and evaluation of binary classifiers.*

The current repository extends that line of work with a PyTorch implementation, additional feature sets, multiple model backends, and reproducible evaluation utilities.

## Citation

If you use this code or the associated methods in academic work, please cite:

> Henry, L. M., Lee, K., Hansen, E., Tandilashvili, E., Rozsypal, J., Erjo, T., Raven, J. G., Reynolds, H. M., Curtis, P., Haller, S. P., Pine, D. S., Norton, E. S., Wakschlag, L. S., Pereira, F., & Brotman, M. A. (2025). *Detecting cry in daylong audio recordings using machine learning: The development and evaluation of binary classifiers.* Assessment. https://doi.org/10.1177/10731911251395993

## License

This repository is released under the **MIT License**. See [`LICENSE`](LICENSE) for details.

## Contributors

- Kyunghun Lee — National Institutes of Health
- Lauren Henry — National Institutes of Health
- Laurie Wakschlag — Northwestern University
- Elizabeth Norton — Northwestern University
- Francisco Pereira — National Institutes of Health
- Melissa Brotman — National Institutes of Health

## Acknowledgments

This work builds on the open-source **Infant-Crying-Detection** implementation by Agnes May Yao and related prior research in naturalistic infant-vocalization analysis.

---

<p align="center"><sub>Research software for reproducible audio-based infant-cry detection.</sub></p>
