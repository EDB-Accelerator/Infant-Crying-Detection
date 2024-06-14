import torch
import matplotlib
matplotlib.use('Agg')
import librosa
import numpy as np
import librosa.display
import csv
from collections import Counter
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from pyAudioAnalysis import ShortTermFeatures

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
	if input_label == 'other' or input_label == 'notcry':
		return 0
	elif input_label == 'fuss' or input_label =='cry':
		return 1
	else:
		return 2

def train_svm(audio_files,annotation_files,alex_model_path,model_output_path,Cvalue=1.0,imbalance='SMOTE',random_seed=None):
	import numpy as np

	all_data = []
	all_labels = []
	all_feature_data = []

	#get 5s windows with 1s overlap
	"""
	This for loop in your code is performing several key steps in processing and preparing audio data for a machine learning task. Here's a breakdown of its major components:
	Iterating Over Episodes: The loop iterates over a list of episodes. Each episode is likely a unique identifier for an audio recording.
	Loading Audio Data: For each episode, it constructs the path to the corresponding audio file (audio_filename) and annotation file (annotation_filename_ra). It then loads the audio file using librosa.load, which returns the audio time series y and its sampling rate sr. It also calculates the duration of the audio file.
	Processing Annotations: The annotations (presumably labels for segments of the audio) are read from a CSV file. The code expects each row in the annotation file to contain at least three elements. It appears to convert the third element of each row into a numerical label using label_to_num. It then filters and processes these annotations based on certain conditions (e.g., ensuring the start time of an annotation is within a certain range of the audio duration). The processed annotations are stored in ra_annotations.
	Generating Windows: The script creates 5-second windows from the audio, with each window starting 1 second apart (assuming a 1s overlap). For each annotation in ra_annotations, it checks if the duration of the annotation is at least 5 seconds and then generates these windows. It also assigns labels to these windows, depending on the condition in the annotation data.
	Extracting Audio Features: Two types of features are extracted from each window:
	Spectrogram Features: A Mel spectrogram is computed using librosa.feature.melspectrogram, which is then converted to a decibel scale. A slice of this spectrogram corresponding to each window is appended to image_windows.
	Statistical Features: Short-term audio features are extracted using a function (assumed to be ShortTermFeatures.feature_extraction). Statistical features (mean, median, standard deviation) of these are computed for each window and appended to feature_windows.
	Aggregating Data: Finally, 
	the labels, spectrogram slices, and statistical features for each window 
	are added to the lists all_labels, all_data, and all_feature_data, respectively. 
	These aggregated lists likely represent the dataset that will be used for 
	training or evaluating a machine learning model.
	"""
	for i in range(len(audio_files)):
		audio_filename = audio_files[i]
		annotation_filename_ra = annotation_files[i]

		y, sr = librosa.load(audio_filename)
		duration = librosa.get_duration(y = y, sr = sr)
		previous = 0

		#ra_annotations: [[0, 1.0, 'other'], [1.0, 243.0, 'fuss']...]
		ra_annotations = []
		rowTest = None
		with open(annotation_filename_ra, 'r') as csvfile:
			csvreader = csv.reader(csvfile, delimiter=',')
			for row in csvreader:
				print(row)
				if len(row) > 0:
					row[2] = label_to_num(row[2])
					print(row[2])
					if float(row[0]) - previous > 0 and int(row[2]) <= 2 and int(row[0]) <= duration // 10 :
						rowTest = row
						ra_annotations.append([float(row[0]), min(duration // 10, float(row[1])), int(row[2])])
		#windows = {'other': [[243.0, 248.0], [244.0, 249.0] ....}
		windows = []
		labels = []

		for item in ra_annotations:
			if item[1] - item[0] >= 5:
				for i in range(int(item[1]) - int(item[0]) - 4):
					windows.append([item[0] + i, item[0] + i + 5])
					if item[2] != 0:
						labels.append(1)
					else:
						labels.append(item[2])

		#print ra_annotations, duration
		#print windows
		#print len(labels)


		S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, fmax=None, n_fft = n_fft, hop_length = hop_length)
		S = librosa.power_to_db(S, ref=np.max) + 80

		#[Fs, x] = audioBasicIO.read_audio_file(audio_filename)
		#print(Fs, x)
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
		#image_windows = np.asarray(image_windows)

		#print image_windows.shape #(number of windows, n_mels, 5 * sr / hop_length)

		all_labels.extend(labels)
		all_data.extend(image_windows)
		all_feature_data.extend(feature_windows)
		# all_labels.append(labels)
		# all_data.append(image_windows)
		# all_feature_data.append(feature_windows)

	all_labels = np.array(all_labels)
	all_data = np.array(all_data) 
	all_feature_data = np.array(all_feature_data)

	all_data = all_data.reshape(all_data.shape[0], img_rows * img_cols)
	all_data = np.concatenate((all_data, all_feature_data), axis = 1)

	# rus = RandomUnderSampler(random_state=42)
	if imbalance == 'SMOTE':
		if random_seed == None:
			rus = SMOTE()
		else:
			rus = SMOTE(random_state=random_seed)
	elif imbalance == 'undersample':
		if random_seed == None:
			rus = RandomUnderSampler()
		else:
			rus = RandomUnderSampler(random_state=random_seed)

	if imbalance != 'weight':
		if len(np.unique(all_labels)) > 1:
			all_data, all_labels = rus.fit_resample(all_data, all_labels)
		else:
			print("Warning: all_labels contains only one class. Skipping resampling.")

	all_feature_data = all_data[:, img_rows * img_cols : ]
	all_data = all_data[:, : img_rows * img_cols]

	all_data = all_data.reshape(all_data.shape[0], img_rows, img_cols, 1)
	all_data = all_data.astype('float32')
	all_data /= 80.0

	print('all_data shape:', all_data.shape)
	print(all_data.shape[0], 'train samples')
	print(Counter(all_labels))

	model = CustomPyTorchModel(num_classes=2)
	model.load_state_dict(torch.load(alex_model_path))
	model.eval()  # Set the model to evaluation mode

	# Convert all_data to PyTorch tensor
	all_data_tensor = torch.tensor(all_data, dtype=torch.float32).permute(0, 3, 1, 2)	
	from torch.utils.data import TensorDataset, DataLoader

	# Assuming all_data_tensor is your input data and doesn't need labels for prediction
	dataset = TensorDataset(all_data_tensor)
	data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)  # Set shuffle to False for prediction

	# Disable gradient computation for efficiency and to reduce memory usage during inference
	with torch.no_grad():
		all_predictions = []
		for inputs in data_loader:
			inputs = inputs[0]  # DataLoader wraps each batch in a tuple

			# If you have a GPU, move the data to the GPU
			# inputs = inputs.to('cuda')
			
			outputs = model(inputs)
			
			# Convert outputs to probabilities; for example, using softmax for classification
			probabilities = torch.softmax(outputs, dim=1)
			
			all_predictions.extend(probabilities.cpu().numpy())

	# Convert list of arrays to a single NumPy array
	image_vector = np.vstack(all_predictions)
	print("image_vector shape:",image_vector.shape)
	svm_input = np.concatenate((image_vector, all_feature_data), axis = 1)
	
	# if modeltype == 'logisticregression':
	# 	# from sklearn.linear_model import LogisticRegression
	# 	# clf = LogisticRegression(random_state=0, class_weight='balanced')
	# 	from sklearn.preprocessing import StandardScaler
	# 	from sklearn.linear_model import LogisticRegression
	# 	from sklearn.pipeline import make_pipeline
	# 	clf = make_pipeline(
	# 	StandardScaler(),
	# 	LogisticRegression(random_state=0, class_weight='balanced', max_iter=1000, solver='saga')
	# 	)
	# else: 
	
	if imbalance == 'weight':
		from sklearn.utils.class_weight import compute_class_weight
		# Compute class weights
		class_labels = np.unique(all_labels)
		class_weights = compute_class_weight('balanced', classes=class_labels, y=all_labels)
		class_weight_dict = dict(zip(class_labels, class_weights))
	from sklearn.svm import SVC
	if imbalance == 'weight':
		clf = SVC(kernel='rbf', probability=True, C=Cvalue, class_weight=class_weight_dict)
	else:
		clf = SVC(kernel = 'rbf', probability = True,C=Cvalue)

	clf.fit(svm_input, all_labels)
	from joblib import dump, load
	

	# dump(clf, model_out_folder + '/svm_noflip.joblib')
	dump(clf, model_output_path)
######################

