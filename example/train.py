from os.path import expanduser
import glob
import re
import pandas as pd
import os
import sys

# Append a directory to the system path for importing custom modules
sys.path.append('./src')

# Import training functions from custom modules
from train_alex import train_alex
from train_svm import train_svm


# Download sample dataset
import tarfile
import os
import subprocess

# URL of the file to download
url = "https://umd.box.com/shared/static/y5kbzxo827y4ohaq7rwzgzm25g7vkb1l.tar"

# Path to extract the files
extract_path = "data"
tar_file = "data.tar"

# Ensure the directory exists and is empty
os.makedirs(extract_path, exist_ok=True)

# Check if the folder is empty and proceed
if not os.listdir(extract_path):
    # Download the file using wget
    subprocess.run(["wget", "-O", tar_file, url], check=True)

    # Extract the tar file
    with tarfile.open(tar_file, "r") as tar:
        tar.extractall(path=".")  # Extract all maintains original structure

    print(f"File downloaded and extracted to {extract_path}")
    os.remove(tar_file)
else:
    print(f"The folder '{extract_path}' is not empty. Extraction skipped.")

# Gather all train/validation files
train_annotation_files = glob.glob(f"./data/train/*.csv")
train_audio_files = glob.glob(f"./data/train/*.wav")
validation_annotation_files = glob.glob(f"./data/validation//*.csv")
validation_audio_files = glob.glob(f"./data/validation/*.wav")

# Sort files for consistent processing
train_annotation_files.sort()
train_audio_files.sort()
validation_annotation_files.sort()
validation_audio_files.sort()

# Retraining process parameters and execution
alex_model_path = '.trained/alex_trained.h5'
svm_model_path = '.trained/svm_trained.joblib'
params = {
    "n_fft": 980,
    "hop_length": 490,
    "n_mels": 225,
    "img_rows": 225,
    "img_cols": 225,
    "batch_size": 128,
    "num_classes": 2
}

# Call training functions
train_alex(params, train_audio_files, train_annotation_files, validation_audio_files, validation_annotation_files, alex_model_path)
train_svm(params, train_audio_files, train_annotation_files, alex_model_path, svm_model_path, Cvalue=10000)
