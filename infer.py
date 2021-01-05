import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
import config as cfg
import pickle
from selective_search import selective_search
from keras.models import load_model
from nms import soft_nms, hard_nms

# Load test label
test_df = pd.read_csv(cfg.TEST_LABEL_FILE)

# Find all image names in test dataset
names = test_df.name.unique()

# Load detector model
# f = open(cfg.MODEL_PATH, 'r')
# model = pickle.load(f)
# f.close()
model = load_model(cfg.MODEL_PATH)

threshold = 0.8

# Run test model in some test image
for name in names:
	print('Processing image {}'.format(name))
	image = cv2.imread(os.path.join(cfg.TEST_PATH, name))

	# Perform selective search
	rects = selective_search(image, method='quality', verbose=False, display=False)
	print(rects.shape)

	# Initialize bounding box
	rp = []
	for (x,y,w,h) in rects:
		# Get predicted
		window = image[y:y+h, x:x+w]
		# Resize to target size
		im_rsz = cv2.resize(window, cfg.IMG_SIZE)
		# Convert to grayscale
		im_rsz = cv2.cvtColor(im_rsz, cv2.COLOR_BGR2GRAY)
		# Append to list
		rp.append(im_rsz)

	# Convert to grayscale and normalization
	rp = np.array(rp, dtype=np.float)/255.0

	# Add new dimension
	rp = rp[:, np.newaxis]
	rp = np.transpose(rp, axes=(0,2,3,1))

	# Predict 
	pred = model.predict(rp)
	
	# Find every bounding box with probability greater than threshold
	index = np.where(pred[:,1] >= threshold)[0]

	bbox = rects[index]
	print(bbox.shape)

	if (len(bbox)==0):
		print('Pass')
		continue
	
	scores = pred[index, 1]
	boxes = [bbox[:,0], bbox[:,0] + bbox[:,2], bbox[:,1], bbox[:,1] + bbox[:,3]]
	boxes = np.array(boxes)
	boxes = boxes.T
	# D, S = hard_nms(boxes, scores, overlapThresh=0.65)
	D, S = soft_nms(boxes, scores, threshold=0.8)

	flag = False
	# for i,(x,y,w,h) in zip(index,bbox):
	# 	output = image.copy()
	# 	red = (0,0,255)
	# 	cv2.rectangle(output, (x,y), (x+w,y+h), color=red, thickness=1)
	# 	cv2.putText(output, text=str(np.around(pred[i,1],4)), org=(x+5,y+5), 
	# 		fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=red)
	# 	cv2.imshow('Output', output)
	# 	key = cv2.waitKey(0) & 0xFF

	# 	if key == ord('q'):
	# 		flag = True
	# 		break

	for i,(xmin,xmax,ymin,ymax) in enumerate(D):
		output = image.copy()
		red = (0,0,255)
		cv2.rectangle(output, (xmin,ymin), (xmax,ymax), color=red, thickness=1)
		cv2.putText(output, text=str(np.around(S[i],4)), org=(xmin+5,ymin+5), 
			fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=red)
		cv2.imshow('Output', output)
		key = cv2.waitKey(0) & 0xFF

		if key == ord('q'):
			flag = True
			break

	if flag:
		break

