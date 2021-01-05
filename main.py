import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
import config as cfg
from sklearn.model_selection import train_test_split
from train import train_detector
from keras.utils import to_categorical

# Load all positive images and negative images
posImg = []
negImg = []

for name in os.listdir(cfg.DIGIT_PATH):
	# Load image
	image = cv2.imread(os.path.join(cfg.DIGIT_PATH, name))

	# Convert to gray scale
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	posImg.append(image)

for name in os.listdir(cfg.NO_DIGIT_PATH):
	# Load image
	image = cv2.imread(os.path.join(cfg.NO_DIGIT_PATH, name))

	# Convert to gray scale
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	negImg.append(image)

# Convert to numpy array and normalizatin
posImg = np.array(posImg, dtype=np.float)/255.0
negImg = np.array(negImg, dtype=np.float)/255.0

print('Shape of posImg: ', posImg.shape)
print('Shape of posImg: ', negImg.shape)

# Create label for each set
posLabel = np.ones(posImg.shape[0])
negLabel = np.zeros(negImg.shape[0])

# Concat data and split into train and validation set
X = np.concatenate((posImg, negImg), axis=0)
label = np.concatenate((posLabel, negLabel))
y = to_categorical(label)

print('Shape of X: ', X.shape)
print('Shape of y: ', y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

X_train = X_train[:, np.newaxis]
X_train = np.transpose(X_train, axes=(0,2,3,1))

X_test = X_test[:, np.newaxis]
X_test = np.transpose(X_test, axes=(0,2,3,1))


print('Shape of X_train, X_test: ', X_train.shape, X_test.shape)
print('Shape of y_train, y_test: ', y_train.shape, y_test.shape)


# Train model
train_detector(X_train, X_test, y_train, y_test, batch_size=cfg.BATCH_SIZE, epochs=cfg.EPOCHS, save_file=cfg.MODEL_PATH)