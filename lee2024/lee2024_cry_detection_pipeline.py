#!/usr/bin/env python3
"""
cry_detection_pipeline.py
─────────────────────────
End-to-end pipeline for infant–cry detection:

1.  Load & label WAV files (`load_and_label_data`).
2.  Pre-process audio  (silence removal + band-pass filter).
3.  Extract features    (Wav2Vec 2.0 + MFCC + chroma + contrast).
4.  (Optional) feature selection with Random Forest.
5.  Train or load a Gradient-Boosting model.
6.  Predict & save CSV with scores.
7.  Evaluate per-subject and overall (calls `commonfunctions.evaluate`).

The script can be toggled via the `goTrain`, `goPredict`, `goEvaluate` flags.
"""

from __future__ import annotations
import os
from glob import glob
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from joblib import dump, load

from transformers import Wav2Vec2Processor, Wav2Vec2Model
import torch
import tensorflow as tf
import librosa   # after transformers to avoid thread-pinning warnings

# ─────────────────────────── GPU INFO ────────────────────────────
print(f"TensorFlow GPUs: {len(tf.config.list_physical_devices('GPU'))}")
print(f"PyTorch GPU   : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    dev_id = torch.cuda.current_device()
    print(f"  └─ device {dev_id} – {torch.cuda.get_device_name(dev_id)}")

# ─────────────────────────── CONSTANTS ───────────────────────────
FS              = 22_050               # original sample rate
TARGET_SR       = 16_000               # Wav2Vec expected SR
LOWCUT, HIGHCUT = 300, 3_000           # band-pass (Hz)

HOME            = Path.home()
DATA_ROOT       = HOME / "data/deBarbaroCry"
SPLIT_CSV       = Path("split/deBarBaro_split.csv")
MODEL_DIR       = Path(".train")
MODEL_PATH      = MODEL_DIR / "gbm_model-feature-selection.joblib"
RESULTS_DIR     = Path("analysis/analysis-06242024")
TMP_DIR         = HOME / "data/tmp"

# ─────────────────────── Wav2Vec 2.0 setup ───────────────────────
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
wav2vec   = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
wav2vec.to(DEVICE)

# ═════════════════════════ DATA I/O ══════════════════════════════
def load_and_label_data(data_dir: Path) -> pd.DataFrame:
    """Return DataFrame with columns [file_path, label, subject]."""
    wav_files = glob(str(data_dir / "P*/" / "*.wav"))
    df = pd.DataFrame(wav_files, columns=["file_path"])
    df["label"]   = df["file_path"].apply(lambda p: 0 if "notcry" in p else 1)
    df["subject"] = df["file_path"].apply(lambda p: Path(p).parent.name)
    return df


# ═════════════════════ AUDIO PRE-PROCESS ═════════════════════════
def is_silent(signal: np.ndarray, thresh: float = 0.01) -> bool:
    return np.max(np.abs(signal)) < thresh


def bandpass_filter(x: np.ndarray,
                    low: float = LOWCUT,
                    high: float = HIGHCUT,
                    fs: int = FS,
                    order: int = 5) -> np.ndarray:
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return lfilter(b, a, x)


def preprocess_audio(x: np.ndarray, *, mode: str) -> np.ndarray | None:
    """Return filtered signal or None (if silent in training mode)."""
    if mode == "train" and is_silent(x):
        return None
    return bandpass_filter(x)


# ═════════════════════ FEATURE EXTRACTION ════════════════════════
def extract_additional_features(x: np.ndarray, sr: int) -> np.ndarray:
    mfcc   = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=20).mean(axis=1)
    chroma = librosa.feature.chroma_stft(y=x, sr=sr).mean(axis=1)
    contr  = librosa.feature.spectral_contrast(y=x, sr=sr).mean(axis=1)
    return np.concatenate([mfcc, chroma, contr])


def extract_features(x: np.ndarray, sr: int) -> np.ndarray:
    if sr != TARGET_SR:
        x  = librosa.resample(x, orig_sr=sr, target_sr=TARGET_SR)
        sr = TARGET_SR

    inputs = processor(x, sampling_rate=sr, return_tensors="pt", padding=True)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        w2v_vec = wav2vec(**inputs).last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

    return np.concatenate([w2v_vec, extract_additional_features(x, sr)])


# ═════════════════════ FEATURE SELECTION ═════════════════════════
def feature_selection(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X, y)
    idx = np.argsort(rf.feature_importances_)[::-1][:500]
    return X[:, idx]


# ═════════════════════ DATASET PROCESSING ════════════════════════
def process_dataframe(df: pd.DataFrame,
                      *,
                      select_features: bool = False,
                      mode: str = "train") -> Tuple[np.ndarray, np.ndarray]:
    feats, lbls = [], []
    for _, row in df.iterrows():
        x, sr     = librosa.load(row["file_path"], sr=None)
        x_prep    = preprocess_audio(x, mode=mode)
        if x_prep is None:
            continue
        feats.append(extract_features(x_prep, sr))
        lbls.append(row["label"])

    X, y = np.asarray(feats), np.asarray(lbls)
    return (feature_selection(X, y) if select_features else X), y


# ═════════════════════ MODEL TRAIN / PREDICT ═════════════════════
def train_model(X: np.ndarray, y: np.ndarray) -> Pipeline:
    pipeline = Pipeline(steps=[
        ("scaler",           StandardScaler()),
        ("feature_selector", SelectFromModel(RandomForestClassifier(n_estimators=100,
                                                                    random_state=42),
                                             threshold=-np.inf,
                                             max_features=500)),
        ("classifier",       GradientBoostingClassifier(n_estimators=100,
                                                        learning_rate=0.1,
                                                        max_depth=3,
                                                        random_state=42)),
    ])
    pipeline.fit(X, y)
    dump(pipeline, MODEL_PATH)
    return pipeline


def predict(model_file: Path, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    model = load(model_file)
    return model.predict(X), model.predict_proba(X)[:, 1]


# ═════════════════════ MAIN EXECUTION ════════════════════════════
if __name__ == "__main__":
    # folders
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    # full dataset
    df_all = load_and_label_data(DATA_ROOT)

    # train / val / test split
    split_df          = pd.read_csv(SPLIT_CSV, names=["ID", "None", "Cry", "Split"])
    train_ids         = split_df.query("Split == 'train'")["ID"].tolist()
    validation_ids    = split_df.query("Split == 'validation'")["ID"].tolist()
    test_ids          = split_df.query("Split == 'test'")["ID"].tolist() + validation_ids

    df_train = df_all[df_all["file_path"].str.contains("|".join(train_ids))]
    df_test  = df_all[df_all["file_path"].str.contains("|".join(test_ids))]

    # ─────────────── toggles ────────────────
    go_train    = False
    go_predict  = True
    go_evaluate = True

    # ─────────────── TRAIN ──────────────────
    if go_train:
        X_train, y_train = process_dataframe(df_train)
        train_model(X_train, y_train)

    # ─────────────── PREDICT ────────────────
    if go_predict:
        X_test, y_test     = process_dataframe(df_test, mode="test")
        y_pred, y_prob     = predict(MODEL_PATH, X_test)

        df_test["ground_truth"]    = y_test
        df_test["predicted_label"] = y_pred
        df_test["decision_score"]  = y_prob
        df_test.to_csv(RESULTS_DIR / "09-gbm-feature_selection.csv", index=False)

    # ─────────────── EVALUATE ───────────────
    if go_evaluate:
        from commonfunctions import evaluate, plot_roc_curve   # local util

        df_eval = pd.read_csv(RESULTS_DIR / "09-gbm-feature_selection.csv")

        # per-subject
        for subj, df_sub in df_eval.groupby("subject"):
            y_true, y_pred, y_prob = df_sub["ground_truth"], df_sub["predicted_label"], df_sub["decision_score"]
            evaluate(y_true, y_pred, RESULTS_DIR, f"classification_report_{subj}.txt")
            plot_roc_curve(y_true, y_prob, RESULTS_DIR, TMP_DIR, f"roc_{subj}")

        # overall
        y_true_all, y_pred_all, y_prob_all = df_eval["ground_truth"], df_eval["predicted_label"], df_eval["decision_score"]
        evaluate(y_true_all, y_pred_all, RESULTS_DIR, "classification_report_overall.txt")
        plot_roc_curve(y_true_all, y_prob_all, RESULTS_DIR, TMP_DIR, "roc_overall")
