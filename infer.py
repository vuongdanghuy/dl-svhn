import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
import config as cfg
import pickle
from selective_search import selective_search
from keras.models import load_model
from keras.models import Model
from sklearn.metrics import average_precision_score
from nms import soft_nms, hard_nms
from iou import iou
import argparse

# Add argument parser to parse command line argument
ap = argparse.ArgumentParser()
ap.add_argument('-m', '--method', type=str, default='soft', 
	choices=['soft', 'hard', 'none'], help='Non-Maximal Suppression method')
ap.add_argument('-t', '--threshold', type=float, default=0.9, help='NMS threshold')
ap.add_argument('-i', '--iou', type=float, default=0.5, help='IoU overlap threshold')
args = vars(ap.parse_args())

# Load feature extractor model
model = load_model(cfg.MODEL_PATH)
output = model.get_layer(name='activation_5').output
extractor = Model(inputs=model.input, outputs=output)

# Load classifier model
f = open(cfg.CLASSIFY_PATH, 'rb')
classifier = pickle.load(f)
f.close()

# Load test label file
df = pd.read_csv(cfg.TEST_LABEL_FILE)

# Open file to write predicted bounding box coordinate
f = open(cfg.INFER_LABEL_FILE,'w')
f.write('name,left,top,width,height,score,label\n')

# Find all image name in test set
names = df.name.unique()

# Loop all images
for k,name in enumerate(names[1000:2000]):
	print('[INFO] Processing image {}/{}'.format(k+1,len(names)))
	# Load image
	image = cv2.imread(os.path.join(cfg.EXTRA_PATH, name))

	# Get image shape
	ih,iw = image.shape[:2]

	# Selective search
	rects = selective_search(image, method='quality', verbose=False, display=False)

	# Initialize bounding box
	rp = []
	for (x,y,w,h) in rects[:2000]:
		# Get predicted
		window = image[y:y+h, x:x+w]
		# Resize to target size
		im_rsz = cv2.resize(window, cfg.IMG_SIZE, interpolation=cv2.INTER_AREA)
		# Append to list
		rp.append(im_rsz)

	# Normalization
	rp = np.array(rp, dtype=np.float)/255.0

	# Extract features
	features = extractor.predict(rp)

	# Classify
	pred = classifier.predict_proba(features)
	
	# Find every bounding box with probability greater than threshold
	index = np.where(np.max(pred[:,1:], axis=1) >= 0.9)[0]
	pred = pred[index,:]
	bbox = rects[index,:]
	# print('[INFO] Found {} bounding box'.format(len(bbox)))

	# Non-Maximal Suppression for each class
	labels = np.argmax(pred, axis=1)
	uniqueLabel = np.unique(labels)
	result = np.array([])

	for label in uniqueLabel:
		labelIdx = np.where(labels == label)[0]
		boxes = bbox[labelIdx,:]
		scores = pred[labelIdx,label]

		if args['method'] == 'soft':
			nms_boxes, nms_scores = soft_nms(boxes, scores, threshold=args['threshold'])
			nms_scores = nms_scores[:,np.newaxis]
		elif args['method'] == 'hard':
			nms_boxes, nms_scores = hard_nms(boxes, scores, overlapThresh=args['threshold'])
			nms_scores = nms_scores[:,np.newaxis]
		else:
			nms_boxes, nms_scores = boxes, scores
			nms_scores = nms_scores[:,np.newaxis]

		label_box = np.hstack((nms_boxes, nms_scores, np.full((len(nms_scores),1), label)))
		if len(result):
			result = np.vstack((result, label_box))
		else:
			result = label_box
	# print('[INFO] After NMS there are {} bounding box'.format(len(result)))

	# Write to label file
	for x,y,w,h,prob,label in result:
		f.write('{},{},{},{},{},{},{}\n'.format(name,x,y,w,h,prob,label))

# Close file
f.close()