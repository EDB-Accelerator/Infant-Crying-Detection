# Infant Crying Detection Algorithm

<img src="assets/cryingnoncryingbabywaveform.jpeg" alt="Crying vs Non‑Crying Waveform" width="600"/>

This repository provides an end‑to‑end pipeline for infant‑cry detection in day‑long, naturalistic audio recordings. The implementation is based on PyTorch and re‑implements the TensorFlow approach of Yao *et al.* (ICASSP 2022) with additional feature sets, model options, and evaluation utilities. Although developed for 12‑month‑old infants enrolled in the **When2Worry** study (3R01MH107652‑05S1; 3R01DC016273‑05S1), the code can be adapted to any corpus of WAV files.

Accurate cry detection supports research into infant well‑being, developmental trajectories, and early identification of disorders such as colic by linking acoustic patterns to clinical or behavioural outcomes.

---

## Contributors

* Kyunghun Lee — National Institutes of Health
* Lauren Henry — National Institutes of Health
* Laurie Wakschlag — Northwestern University
* Elizabeth Norton — Northwestern University
* Francisco Pereira — National Institutes of Health
* Melissa Brotman — National Institutes of Health

---

## Citation

If you use this work in academic publications, please cite:

> L. M. Henry\*, K. Lee\*, E. Hansen, E. Tandilashvili, J. Rozsypal, T. Erjo, J. G. Raven, H. M. Reynolds, P. Curtis, S. P. Haller, D. S. Pine, E. S. Norton, L. S. Wakschlag, F. Pereira, & M. A. Brotman (2025). *Detecting cry in daylong audio recordings using machine learning: The development and evaluation of binary classifiers.* *Assessment*, Advance online publication. https://doi.org/10.1177/10731911251395993  
>
> \*Co-first authorship.

The approach builds on:

* X. Yao *et al.* “Infant Crying Detection in Real‑World Environments,” **ICASSP** 2022.
* M. Micheletti *et al.* “Validating a Model to Detect Infant Crying from Naturalistic Audio,” **Behavior Research Methods** 2022.

---

## Background

Our enhancements stem from the original [AgnesMayYao/Infant‑Crying‑Detection](https://github.com/AgnesMayYao/Infant-Crying-Detection) repository. Key changes include:

* Migration to PyTorch for wider hardware support.
* Integration of Wav2Vec 2.0 embeddings, MFCC, chroma, and spectral‑contrast features.
* Gradient‑boosting, SVM, and CNN back‑ends with configurable pipelines.
* Reproducible evaluation on subject‑level splits.

---

## Requirements

The project targets Python 3.9+. Core packages:

```
pytorch
transformers
librosa
pydub
pandas
numpy
scipy
soundfile
scikit‑learn
joblib
```

Install all dependencies with:

```bash
pip install -r requirements.txt
```

GPU support is optional but recommended.

---

## Sample Data

* **Format:** 16‑bit, 16 kHz, mono WAV.
* A small example clip is provided under `sample_data/`.
* Additional corpora: the [HomeBank deBarbaro dataset](https://homebank.talkbank.org/access/Password/deBarbaroCry.html).

---

## Outputs

Predictions are binary:

* `1` – crying detected
* `0` – non‑cry segment

CSV summaries include per‑segment probabilities and subject identifiers for downstream analysis.

---

## Source Code Overview

| File                        | Purpose                                                                                                                                                  |
| --------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `cry_detection_pipeline.py` | Full training / inference pipeline with Wav2Vec 2.0 + traditional features and gradient boosting. Toggle training, prediction, and evaluation via flags. |
| `random_extract.py`         | Automated extraction of 10‑minute clips that contain at least *n* cry‑like 5‑second windows. Useful for curating balanced datasets from long recordings. |
| `src/preprocessing.py`      | Signal conditioning, band‑pass filtering, and mel‑spectrogram generation.                                                                                |
| `src/train_alex.py`         | CNN (AlexNet‑style) training on spectrograms. Includes data augmentation, early stopping, and optional L2 regularisation.                                |
| `src/train_svm.py`          | Hybrid CNN + SVM training pipeline for robust classification.                                                                                            |
| `src/predict.py`            | End‑to‑end inference: preprocess WAV → CNN features → SVM classification.                                                                                |

---

## Jupyter / Colab Examples

* [`example/train.ipynb`](example/train.ipynb) – end‑to‑end model training.
* [`example/prediction.ipynb`](example/prediction.ipynb) – batch inference.
  Each notebook has a Google Colab link for GPU access without local setup.

---



## License

This project is released under the MIT License; see `LICENSE` for the full text.

```
