# Cry Detection Pipeline

End-to-end Python workflow for detecting infant cries in the **HomeBank deBarbaro** audio corpus.
The pipeline fuses **Wav2Vec 2.0** embeddings with classical acoustic features—MFCC-20, chroma, and spectral contrast—and classifies each segment using **Gradient Boosting** with automatic feature selection.

If this code proves useful in your work, please cite **at least one** of the publications listed below.

---

## Quick start

### 1 · Clone and create environment

Use **Miniconda** (recommended) or any virtual-env tool you prefer.

```bash
git clone https://github.com/<your-handle>/cry-detection.git
cd cry-detection

# create conda env
conda create -n crydetect python=3.9 -y
conda activate crydetect

# install requirements
pip install -r requirements.txt            # PyTorch, transformers, librosa, scikit-learn …
```

### 2 · Download data

1. Request access to [HomeBank](https://homebank.talkbank.org/access/Password/deBarbaroCry.html) (free for academic use).
2. Download the **deBarbaro** corpus and place WAV files under

```
data/deBarbaroCry/
    P123/xxx.wav
    P124/...
```

### 3 · Train, predict, evaluate

Open `cry_detection_pipeline.py` and set the toggles:

```python
go_train    = True    # train a new model from scratch
go_predict  = True    # run inference on the test split
go_evaluate = True    # generate reports and ROC curves
```

Then run

```bash
python cry_detection_pipeline.py
```

Outputs (classification reports and ROC plots) appear in `analysis/`.

*No pre-trained GBM model is shipped; the script will create one in `.train/` after training.*
(The Wav2Vec 2.0 acoustic model is fetched automatically from Hugging Face.)

---

## Requirements

* Python 3.9 +
* PyTorch ≥ 2.2 (CUDA optional but recommended)
* transformers · librosa · scikit-learn · joblib · pandas · numpy · scipy
  (exact versions pinned in `requirements.txt`)

---

## Pipeline at a glance

| Stage              | Details                                                       |
| ------------------ | ------------------------------------------------------------- |
| Pre-processing     | Silence filter + 300–3 000 Hz 5th-order Butterworth band-pass |
| Feature extraction | Wav2Vec 2.0 (base) + MFCC-20 + chroma + spectral contrast     |
| Feature selection  | Random-Forest top-500 importance filter                       |
| Classifier         | Gradient Boosting (100 estimators, depth = 3)                 |
| Metrics            | Per-subject & overall reports, ROC-AUC curves                 |

---

## References

1. Lee, K. *et al.* (2024). *Enhancing Infant Crying Detection with Gradient Boosting for Improved Emotional and Mental Health Diagnostics.* arXiv 2410.09236.
2. Henry, L.,Lee, K., Hansen, E., Tandilashvili, E., Rozsypal, J., Erjo, T., Raven, J.G., Reynolds, H.M., Curtis, P., Haller, S.P., Pine, D.S., Norton, E.S., Wakschlag, L.S., Pereira, F., Brotman, M.A.\* (2025). *Detecting Cry in Daylong Audio Recordings Using Machine Learning: The Development and Evaluation of Binary Classifiers.* OSF.


---

## BibTeX

```bibtex
@misc{lee2024cry,
  author       = {Lee, Kyunghun and others},
  title        = {Enhancing Infant Crying Detection with Gradient Boosting for Improved Emotional and Mental Health Diagnostics},
  year         = {2024},
  eprint       = {2410.09236},
  archivePrefix= {arXiv},
  primaryClass = {eess.AS}
}

@misc{henry2025cry,
  author       = {Henry, L. and Lee, K. and Hansen, E. and Tandilashvili, E. and Rozsypal, J. and Erjo, T. and Raven, J. G. and Reynolds, H. M. and Curtis, P. and Haller, S. P. and Pine, D. S. and Norton, E. S. and Wakschlag, L. S. and Pereira, F. and Brotman, M. A.},
  title        = {Detecting Cry in Daylong Audio Recordings Using Machine Learning: The Development and Evaluation of Binary Classifiers},
  year         = {2025},
  howpublished = {\\url{https://doi.org/10.31219/osf.io/3rd2y_v1}},
  note         = {OSF Preprint}
}
```

---

Questions or issues?
**Kyunghun Lee** — [kyunghun.lee@nih.gov](mailto:kyunghun.lee@nih.gov)
