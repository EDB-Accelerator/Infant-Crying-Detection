import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms

#for the CNN
import matplotlib
matplotlib.use('Agg')
import librosa
import os
import numpy as np
import librosa.display
import matplotlib.pyplot as plt
import csv
from scipy.signal import savgol_filter
from sklearn.cluster import KMeans
from sklearn.preprocessing import minmax_scale
from scipy.signal import find_peaks
from sklearn.metrics import classification_report, precision_recall_fscore_support, confusion_matrix, accuracy_score
from math import sqrt, pi, exp
from collections import Counter
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
#from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import ShortTermFeatures
import matplotlib as plt
import pandas as pd


n_fft = 980
hop_length = 490
n_mels = 225
img_rows, img_cols = 225, 225
batch_size = 128
num_classes = 2

import sys
sys.path.insert(1, 'src') # Path to the directory that contains the file, not the file itself
from train_alex import CustomPyTorchModel

def time_masking(mel_spectrogram, tau, time_masking_para=100, time_mask_num=2):
	mel_spectrogram = np.asarray(mel_spectrogram)
	for i in range(time_mask_num):
		t = np.random.randint(low = 0, high = time_masking_para)
		t0 = np.random.randint(low = 0, high = tau - t)
		# mel_spectrogram[:, t0:t0 + t] = 0
		mel_spectrogram[:, t0:(t0 + t)] = 0
	return list(mel_spectrogram)

def whatIsAnEvent(data, event_thre):
    previous = (-1, -1)
    start = (-1, -1)
    for i in range(len(data)):
        if data[i, 1] == 1 and previous[1] == -1:
            previous = (i, data[i, 0])
        elif data[i, 1] == 0 and previous[1] != -1 and data[i - 1, 1] == 1:
            start = (i, data[i, 0])
            if start[1] - previous[1] <= event_thre:
                data[previous[0] : start[0], 1] = 0
            previous = (-1, -1)
            start = (-1, -1)

    if previous[1] != -1 and data[-1, 0] - previous[1] + 1 <= event_thre:
        data[previous[0] :, 1] = 0
    return data

def combineIntoEvent(data, time_thre):
    previous = (-1, -1)
    for i in range(len(data)):
        if data[i, 1] == 1:
            start = (i, data[i, 0])
            if previous[1] > 0 and start[1] - previous[1] <= time_thre:
                data[previous[0] : start[0], 1] = 1
            previous = start

    if previous[1] > 0 and data[i - 1, 0] - previous[1] <= time_thre:
        data[previous[0] : i, 1] = 1

    return data


def label_to_num(input_label):
	if input_label == 'other':
		return 0
	elif input_label == 'fuss' or input_label =='cry':
		return 1
	# elif input_label == 'cry':
	# 	return 2
	# elif input_label == 'scream':
	# 	return 3
	else:
		return 2



# audio files: list of 10 min wav files (10 min mono)
# annotation files format:
# 0,10,notcry
# 10,15,cry
# 15,40,notcry
# predict_alex_svm(test_audio_files,test_annotation_filtered_files,test_prediction_files,alex_model_path,svm_model_path,'torch',decision_scores_only=False,best_threshold=None)

def predict_alex_svm(audio_files,annotation_filtered_files,prediction_files,alex_model_path,svm_model_path,alex_model_type='torch',decision_scores_only=False,best_threshold=None):

# test_audio_files,test_annotation_filtered_files,test_prediction_files,alex_model_path,svm_model_path,'torch',decision_scores_only=False,best_threshold=best_threshold)

	# predict_alex_svm(test_audio_files,test_annotation_filtered_files,alex_model_path,svm_model_path)
	# audio_files = test_audio_files
	# annotation_filtered_files = test_annotation_filtered_files
	# prediction_files = test_prediction_files

	all_data = []
	# all_predictions = []
	all_feature_data = []
	
	decision_scores = []
	predictions = []
	for i in range(len(audio_files)):
		audio_filename = audio_files[i]
		annotation_filename_filtered = annotation_filtered_files[i]

		def svm_input_generator(audio_filename,annotation_filename_filtered):

			y, sr = librosa.load(audio_filename)
			duration = librosa.get_duration(y = y, sr = sr)
			previous = 0

			# ra_annotations = []
			# with open(annotation_filename_ra, 'r') as csvfile:
			# 	csvreader = csv.reader(csvfile, delimiter=',')
			# 	for row in csvreader:
			# 		print(row)
			# 		if len(row) > 0:
			# 			row[2] = label_to_num(row[2])
			# 			if float(row[0]) - previous > 0 and int(row[2]) <= 2 and int(row[0]) <= duration // 10 :
			# 				ra_annotations.append([float(row[0]), min(duration // 10, float(row[1])), int(row[2])])
			#windows = [[0, 5], [1, 6],......]
			windows = []
			for j in range(0, int(duration) - 4):
				windows.append([j, j + 5])
			print(len(windows))

			previous = 0
			filtered_annotations = []
			with open(annotation_filename_filtered, 'r') as csvfile:
				csvreader = csv.reader(csvfile, delimiter=',')
				for row in csvreader:
					if float(row[0]) - previous > 0:
						filtered_annotations.extend([0] * int(float(row[0]) - previous))
					previous = float(row[1])
					filtered_annotations.extend([1] * int(float(row[1]) - float(row[0])))
			if duration - previous > 0:
				filtered_annotations.extend([0] * int(duration - previous))
			print(duration, len(filtered_annotations))

			S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, fmax=None, n_fft = n_fft, hop_length = hop_length)
			S = librosa.power_to_db(S, ref=np.max) + 80

			F, _ = ShortTermFeatures.feature_extraction(y, sr, 1 * sr, 0.5 * sr)
			F = F[:, 0::2]

			image_windows = []
			feature_windows = []
			for item in windows:
				spec = S[:, int(item[0] * sr / hop_length) : int(item[1] * sr / hop_length)]
				F_window = F[:, int(item[0]) : int(item[1])]
				F_feature = np.concatenate((np.mean(F_window, axis = 1), np.median(F_window, axis = 1), np.std(F_window, axis = 1)), axis = None)
				image_windows.append(spec)
				feature_windows.append(F_feature)

			image_windows = np.array(image_windows) 
			feature_windows = np.array(feature_windows)
			image_windows = image_windows.reshape(image_windows.shape[0], img_rows, img_cols, 1)
			image_windows = image_windows.astype('float32')
			image_windows /= 80.0
			print('image_windows shape:', image_windows.shape)
			
			if alex_model_type != 'tensorflow':
				model = CustomPyTorchModel(num_classes=2)
				model.load_state_dict(torch.load(alex_model_path))
				model.eval()  # Set the model to evaluation mode
				
				# Convert all_data to PyTorch tensor
				image_windows_tensor = torch.tensor(image_windows, dtype=torch.float32).permute(0, 3, 1, 2)	
				from torch.utils.data import TensorDataset, DataLoader
				# Assuming all_data_tensor is your input data and doesn't need labels for prediction
				dataset = TensorDataset(image_windows_tensor)
				data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)  # Set shuffle to False for prediction
				
				with torch.no_grad():
					prediction = []
					for inputs in data_loader:
						inputs = inputs[0]  # DataLoader wraps each batch in a tuple

						# If you have a GPU, move the data to the GPU
						# inputs = inputs.to('cuda')
						
						outputs = model(inputs)
						
						# Convert outputs to probabilities; for example, using softmax for classification
						probabilities = torch.softmax(outputs, dim=1)
						
						prediction.extend(probabilities.cpu().numpy())
						# Convert list of arrays to a single NumPy array
						image_vector = np.vstack(prediction)
						print("image_vector shape:",image_vector.shape)
			else:
				import h5py
				import copy
				from tensorflow import keras
				import tensorflow as tf
				from tensorflow.keras.models import Sequential, model_from_json, load_model
				from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation
				from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
				from tensorflow.keras import backend as K

				saved_model1 = load_model('./.trained_yao/deep_spectrum.h5')
				model1 = Sequential()
				for layer in saved_model1.layers[:-1]:
					model1.add(layer)

				for layer in model1.layers:
					layer.trainable = False
				image_vector = model1.predict(image_windows, batch_size=batch_size, verbose=1, steps=None)
			
			svm_input = np.concatenate((image_vector, feature_windows), axis = 1)
			return svm_input,filtered_annotations
		
		svm_input,filtered_annotations = svm_input_generator(audio_filename,annotation_filename_filtered)

		from sklearn.svm import SVC
		from joblib import dump, load
		clf = load(svm_model_path)
  
  		# Calculate decision function on the new training set
		# decision_scores.append(clf.decision_function(svm_input))
		# decision_score = clf.decision_function(svm_input)
		decision_score = clf.predict_proba(svm_input)[:,1]
		if best_threshold != None:
			# Convert decision scores to a numpy array for vectorized operations (if not already an array)
			decision_score_array = np.array(decision_score)
			# Apply the threshold to make binary predictions
			prediction = (decision_score_array > best_threshold).astype(int)
		elif decision_scores_only != True:
			prediction = clf.predict(svm_input)
  
  
		size_difference = len(filtered_annotations) - len(decision_score)
		if size_difference > 0:
			# Create an array of zeros to pad to decision_score
			padding = np.zeros(size_difference)
			# Append the zeros to the end of the decision_score array
			decision_score = np.concatenate([decision_score, padding])
		decision_scores.append(decision_score)
		if decision_scores_only:
			continue
		




		for ind, val in enumerate(filtered_annotations):
			if val >= 1:
				min_ind = max(ind - 4, 0)
				max_ind = min(len(prediction), ind + 1)
				#print(Counter(predictions[min_ind : max_ind]).most_common(1))
				#filtered_annotations[ind] = Counter(predictions[min_ind : max_ind]).most_common(1)[0][0]
				if sum(prediction[min_ind : max_ind]) >= 1:
					filtered_annotations[ind] = 1
				else:
					filtered_annotations[ind] = 0
		
		
		timed_filted = np.stack([np.arange(len(filtered_annotations)), filtered_annotations], axis = 1)
		timed_filted = combineIntoEvent(timed_filted, 5 )
		timed_filted = whatIsAnEvent(timed_filted, 5 )

		filtered_annotations = timed_filted[:, 1]
		predictions.append(filtered_annotations)
		# all_predictions.extend(filtered_annotations)
		df_record = pd.DataFrame(filtered_annotations)
		df_record.to_csv(prediction_files[i],header=False,index=False)
		print("generated:",prediction_files[i])
	
 
 
	if decision_scores_only:
		return decision_scores

	return predictions,decision_scores

######################

