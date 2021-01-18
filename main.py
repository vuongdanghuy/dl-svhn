import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
import config as cfg
from sklearn.model_selection import train_test_split
from train import train_detector
from keras.utils import to_categorical

# Get all samples from positive set
df_pos = pd.read_csv('../data/labels/posRP.csv')

posImg = []
posLabel = []

print('[INFO] Loading positive samples...')
for index, row in df_pos.iterrows():
	# Load image
	image = cv2.imread(os.path.join(cfg.EXTRA_PATH, row['name']))

	# Get patch
	window = image[row['y']:row['y']+row['h'],
					row['x']:row['x']+row['w'],:]

	# Convert to gray scale
	# image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# Resize image
	window = cv2.resize(window, cfg.IMG_SIZE, interpolation=cv2.INTER_AREA)

	# Get label from name
	# label = name.split('_')[-1]
	# label = label.split('.')[0]
	# posLabel.append(np.int(label))
	posLabel.append(row['label'])

	posImg.append(window)

posLabel = np.array(posLabel)

# Get all samples from negative set
df_neg = pd.read_csv('../data/labels/negRP.csv')
negImg = []
negLabel = []

print('[INFO] Loading negative samples...')
for index, row in df_neg.sample(np.int(len(posLabel)*1.5), random_state=13).iterrows():
	# Using only a portion of background samples
	if index > len(posLabel)*1.5:
		break
	# Load image
	image = cv2.imread(os.path.join(cfg.EXTRA_PATH, row['name']))

	# Get patch
	window = image[row['y']:row['y']+row['h'],
					row['x']:row['x']+row['w'],:]

	# # Convert to gray scale
	# image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# Resize image
	window = cv2.resize(window, cfg.IMG_SIZE, interpolation=cv2.INTER_AREA)

	negLabel.append(row['label'])
	negImg.append(window)

# Convert to numpy array and normalizatin
posImg = np.array(posImg, dtype=np.float32)/255.0
negImg = np.array(negImg, dtype=np.float32)/255.0

print('Shape of posImg: ', posImg.shape)
print('Shape of posImg: ', negImg.shape)

# Concat data and split into train and validation set
X = np.concatenate((posImg, negImg), axis=0)
label = np.concatenate((posLabel, negLabel))
y = to_categorical(label)

print('Shape of X: ', X.shape)
print('Shape of y: ', y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7)


print('Shape of X_train, X_test: ', X_train.shape, X_test.shape)
print('Shape of y_train, y_test: ', y_train.shape, y_test.shape)


# Train model
train_detector(X_train, X_test, y_train, y_test, nb_classes=y.shape[1], batch_size=cfg.BATCH_SIZE, epochs=cfg.EPOCHS, 
	do_augment=False, save_file=cfg.MODEL_PATH)