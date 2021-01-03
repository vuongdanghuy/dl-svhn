import numpy as np
import pandas as pd
import os
import argparse
import cv2
from selective_search import selective_search
from iou import iou

# Config
BASE_PATH = '../data/'
TRAIN_PATH = BASE_PATH + 'train/'
TEST_PATH = BASE_PATH + 'test/'
EXTRA_PATH = BASE_PATH + 'extra/'
DES_PATH = BASE_PATH + 'dataset/'
DIGIT_PATH = DES_PATH + 'digits/'
NO_DIGIT_PATH = DES_PATH + 'no_digits/'

LABEL_PATH = BASE_PATH + 'labels/'
TRAIN_LABEL_FILE = LABEL_PATH + 'train_label.csv'
TEST_LABEL_FILE = LABEL_PATH + 'test_label.csv'

# Check if destination folder is exist
if not(os.path.exists(DES_PATH)):
	print('[INFO] Creating destination folder and sub folders...')
	os.mkdir(DES_PATH)
	os.mkdir(DIGIT_PATH)
	os.mkdir(NO_DIGIT_PATH)
	print('[INFO] Folder created')
else:
	print('[INFO] Folder created.')

# Load label file
df = pd.read_csv(TRAIN_LABEL_FILE)

# Find all images need to process in label file
train_names = np.unique(df.name)

# Loop all images in training data
for name in train_names:
	# Load images
	print('[INFO] Proccessing image {}'.format(name))
	image = cv2.imread(os.path.join(TRAIN_PATH + name))
	print('[DBG] Image shape: ', image.shape)

	# Get ground truth bounding box in image
	gtbox = df[df.name == name]
	print('[DBG] gtbox shape: ', gtbox.shape)

	# Perform selective search on image
	rects = selective_search(image, method='quality', verbose=False, display=False)
	print('[INFO] Found {} region proposals'.format(len(rects)))

	# Compute IoU between region proposals and ground truth bounding box
	flag = False
	for (x, y, w, h) in rects:
		for i in range(len(gtbox)):
			bbox = [gtbox['left'].values[i], gtbox['left'].values[i] + gtbox['width'].values[i],
					gtbox['top'].values[i], gtbox['top'].values[i] + gtbox['height'].values[i]]
			rect = [x, x+w, y, y+h]

			if iou(bbox, rect) > 0.6:
				print('Found digit')
				output = image.copy()
				red = (0,0,255)
				green = (0,255,0)
				cv2.rectangle(output, (x,y), (x+w,y+h), color=red, thickness=2)
				cv2.rectangle(output, (bbox[0], bbox[2]), (bbox[1], bbox[3]), color=green, thickness=2)
				cv2.imshow('Digit', output)
				key = cv2.waitKey() & 0xFF
				if key == ord('q'):
					flag = True
					break

	if flag:
		break

