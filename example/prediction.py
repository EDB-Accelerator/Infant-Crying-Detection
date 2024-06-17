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
extract_path = "data"
tar_file = "data.tar"
def downloadFileFromWeb(extract_path,tar_fil,url):
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
downloadFileFromWeb("data","data.tar","https://umd.box.com/shared/static/y5kbzxo827y4ohaq7rwzgzm25g7vkb1l.tar")

# Download pretrained dataset (Please change this if you trained the model.)
downloadFileFromWeb(".pretrained","pretrained.tar","https://umd.box.com/shared/static/qseveraeze15vbxztmq1aozbemil6yy9.tar")
alex_model_path = ".pretrained/alex_pretrained.h5"
svm_model_path = ".pretrained/svm_pretrained.joblib"

# Gather all test files
test_annotation_files = glob.glob(f"./data/test/*.csv")
test_audio_files = glob.glob(f"./data/test/*.wav")

# Sort files for consistent processing
test_annotation_files.sort()
test_audio_files.sort()

# Preprocessing
from preprocessing import preprocessing
os.makedirs("preprocessed", exist_ok=True) # Ensure the directory exists and is empty
test_preprocessed_files = []
for i in range(len(test_audio_files)):
    test_audio_file = test_audio_files[i]
    test_annotation_file = test_annotation_files[i]
    id = test_annotation_file.split('/')[-1].split('.csv')[0]
    file_path = './preprocessed/{id}.csv'
    test_preprocessed_files.append(file_path)
    preprocessing(test_audio_file,file_path)


# Prediction
from predict import predict_alex_svm
os.makedirs("prediction", exist_ok=True) # Ensure the directory exists and is empty
test_prediction_files = []
for i in range(len(test_audio_files)):
    test_audio_file = test_audio_files[i]
    test_annotation_file = test_annotation_files[i]
    id = test_annotation_file.split('/')[-1].split('.csv')[0]
    file_path = f'./prediction/{id}.csv'
    test_prediction_files.append(file_path)

params = {
    "n_fft": 980,
    "hop_length": 490,
    "n_mels": 225,
    "img_rows": 225,
    "img_cols": 225,
    "batch_size": 128,
    "num_classes": 2
}

predictions,decision_scores = predict_alex_svm(params,test_audio_files,test_preprocessed_files,test_prediction_files,alex_model_path,svm_model_path,'torch',decision_scores_only=False,best_threshold=None)
