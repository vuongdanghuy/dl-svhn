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
from nms import soft_nms, hard_nms
import argparse

# Add argument parser to parse command line argument
ap = argparse.ArgumentParser()
ap.add_argument('-m', '--method', type=str, default='soft', 
	choices=['soft', 'hard', 'none'], help='Non-Maximal Suppression method')
ap.add_argument('-t', '--threshold', type=float, default=0.9, help='NMS threshold')
args = vars(ap.parse_args())

# Load test label
test_df = pd.read_csv(cfg.TEST_LABEL_FILE)

# Find all image names in test dataset
names = test_df.name.unique()

# Load detector and classifier model
model = load_model(cfg.MODEL_PATH)
output = model.get_layer(name='activation_5').output
extractor = Model(inputs=model.input, outputs=output)

f = open(cfg.CLASSIFY_PATH, 'rb')
classifier = pickle.load(f)
f.close()

f = open(cfg.REGRESSION_PATH, 'rb')
reg = pickle.load(f)
f.close()

threshold = 0.9

# Run test model in some test image
for name in names:
	print('Processing image {}'.format(name))
	image = cv2.imread(os.path.join(cfg.EXTRA_PATH, name))

	# Perform selective search
	rects = selective_search(image, method='quality', verbose=False, display=False)
	# print(rects.shape)

	# Initialize bounding box
	rp = []
	for (x,y,w,h) in rects[:2000]:
		# Get predicted
		window = image[y:y+h, x:x+w]
		# Resize to target size
		im_rsz = cv2.resize(window, cfg.IMG_SIZE)
		# Append to list
		rp.append(im_rsz)

	# Convert to grayscale and normalization
	rp = np.array(rp, dtype=np.float)/255.0

	# Predict
	features = extractor.predict(rp)
	pred = classifier.predict_proba(features)

	# Find every bounding box with probability greater than threshold
	index = np.where(np.max(pred[:,1:], axis=1) >= threshold)[0]
	pred = pred[index,:]
	bbox = rects[index,:]
	# print('[DBG] bbox:\n', bbox)

	if (len(bbox)==0):
		print('Pass')
		continue

	# Apply NMS for each class
	labels = np.argmax(pred, axis=1)
	uniqueLabel = np.unique(labels)
	# print('labels: ', labels)
	# print('uniqueLabel: ', uniqueLabel)
	result = np.array([])

	for label in uniqueLabel:
		labelIdx = np.where(labels == label)[0]
		boxes = bbox[labelIdx,:]
		scores = pred[labelIdx,label]
		# print('[DBG] boxes:\n', boxes)
		# print('[DBG] scores:\n', scores)

		if args['method'] == 'soft':
			nms_boxes, nms_scores = soft_nms(boxes, scores, threshold=args['threshold'])
			nms_scores = nms_scores[:,np.newaxis]
		elif args['method'] == 'hard':
			nms_boxes, nms_scores = hard_nms(boxes, scores, overlapThresh=args['threshold'])
			nms_scores = nms_scores[:,np.newaxis]
		else:
			nms_boxes, nms_scores = boxes, scores
			nms_scores = nms_scores[:,np.newaxis]

		# print('[DBG] nms_boxes:\n', nms_boxes)
		# print('[DBG] nms_scores:\n', nms_scores)

		tmp = np.hstack((nms_boxes, nms_scores, np.full((len(nms_scores),1), label)))
		if len(result):
			result = np.vstack((result, tmp))
		else:
			result = tmp

	flag = False
	print('[INFO] Number of predicted boxes: ', len(result))

	for x,y,w,h,score,label in result:
		output = image.copy()
		red = (0,0,255)
		green = (0,255,0)
		blue = (255,0,0)
		# Predict offset
		tmp = []
		tmp.append((x,y,w,h))
		tmp = np.array(tmp, dtype=np.float)
		dP = reg.predict(tmp)
		# Compute fine-tuned bounding box coordinate
		fx = w*dP[0,0]+x
		fy = h*dP[0,1]+y
		fw = w*np.exp(dP[0,2])
		fh = h*np.exp(dP[0,3])
		cv2.rectangle(output, (int(fx),int(fy)), (int(fx+fw),int(fy+fh)), color=blue, thickness=1)
		cv2.rectangle(output, (int(x),int(y)), (int(x+w),int(y+h)), color=red, thickness=1)
		cv2.putText(output, text='{}:{}'.format(int(label), np.around(score,4)), org=(int(x)+5,int(y)+5), 
			fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=green)
		cv2.imshow('Output', output)
		key = cv2.waitKey(0) & 0xFF

		if key == ord('q'):
			flag = True
			break

	if flag:
		break