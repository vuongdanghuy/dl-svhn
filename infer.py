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

# Initialize True and Predict list
f = open('../data/labels/infer.csv','w')
f.write('name,x,y,w,h,score,label\n')

# Find all image name in test set
names = df.name.unique()

# Loop all images
for k,name in enumerate(names[1000:2000]):
	# name = '141256.png'
	print('[INFO] Processing image {}/{}'.format(k+1,len(names)))
	# Load image
	image = cv2.imread(os.path.join(cfg.EXTRA_PATH, name))

	# Get image shape
	ih,iw = image.shape[:2]

	# Get ground truth box in image
	# gtbox = df[df.name==name].copy()

	# # Mark all ground truth box as not found
	# for index, _ in gtbox.iterrows():
	# 	gtbox.loc[index, 'matched'] = False

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

	# Find TP, FP for each class
	for x,y,w,h,prob,label in result:
		f.write('{},{},{},{},{},{},{}\n'.format(name,x,y,w,h,prob,label))
		# If label is not in list then initialize
	# 	label = int(label)
	# 	if label not in P:
	# 		P[label] = []
	# 		T[label] = []
	# 	# Append label
	# 	P[label].append(prob)
	# 	found_match = 0

	# 	# Loop all ground truth bounding box
	# 	for index, row in gtbox.iterrows():
	# 		# Get bounding box coordinate
	# 		gtx = row['left']
	# 		gty = row['top']
	# 		gtw = row['width']
	# 		gth = row['height']

	# 		# If GT box label is not matched or GT box if found, then continue
	# 		if row['matched'] or row['label'] != label:
	# 			continue

	# 		# Find IoU
	# 		iou_score = iou((x,x+w,y,y+h), (gtx,gtx+gtw,gty,gty+gth))

	# 		# If IoU is greater than a threshold, mark this GT box is found
	# 		if iou_score >= args['iou']:
	# 			found_match = 1
	# 			gtbox.loc[index, 'matched'] = True
	# 			break
	# 	T[label].append(found_match)
	# # print('[DBG] gtbox:\n', gtbox)

	# # Loop all GT box to find what GT box is not found
	# for index, row in gtbox.iterrows():
	# 	if not row['matched']:
	# 		if row['label'] not in P:
	# 			P[row['label']] = []
	# 			T[row['label']] = []
	# 		P[row['label']].append(0)
	# 		T[row['label']].append(1)

f.close()
# Save T and P
# f = open('./output/T.pickle','wb')
# pickle.dump(T,f)
# f.close()

# f = open('./output/P.pickle','wb')
# pickle.dump(P,f)
# f.close()
# # Calculate AP for each class
# AP = []
# for key in T.keys():
# 	class_ap = average_precision_score(T[key], P[key])
# 	AP.append(AP)
# 	print('[INFO] Class {}: AP = {}'.format(key, class_ap))
# print('[INFO] mAP = {}'.format(np.mean(AP)))