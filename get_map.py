import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
import config as cfg
import pickle
from sklearn.metrics import average_precision_score
from iou import iou
import argparse

# Add argument parser to parse command line argument
ap = argparse.ArgumentParser()
ap.add_argument('-p', '--path', type=str, required=True, 
	help='Path to label files')
ap.add_argument('-t', '--true', type=str, required=True, help='Ground truth label file')
ap.add_argument('-g', '--guess', type=str, required=True, help='Predicted label file')
args = vars(ap.parse_args())

def get_map(path, true_df, pred_df):
	"""
	Calculate Average Precision
	@INPUT:
		- path: path to label file
		- true_df: Dataframe hold ground truth bounding box.
			Columns =['name','left','top','width','height','label']
		- pred_df: Dataframe hold predicted bounding box.
			Columns =['name','left','top','width','height','score','label']
	@OUTPUT
		- AP: Average Precision of each class
		- mAP
	"""
	# Load ground truth label file
	gt_df = pd.read_csv(os.path.join(path, true_df))

	# Load predicted label file
	df = pd.read_csv(os.path.join(path, pred_df))

	# Change all label 0 to label 10
	df.loc[df['label']==0, 'label'] = 10

	# Initialize True and Predict list
	T = {}
	P = {}

	# Find all image name in predicted dataframe
	names = df.name.unique()
	print('[INFO] There are {} image in predicted dataframe'.format(len(names)))
	print('[INFO] There are {} image in ground truth dataframe'.format(len(gt_df.name.unique())))

	# Loop all images
	for k,name in enumerate(names):
		# print('[INFO] Processing image {}/{}'.format(k,len(names)))

		# Get ground truth box in image
		gtbox = gt_df[gt_df.name==name].copy()

		# Mark all ground truth box as not found
		for index, _ in gtbox.iterrows():
			gtbox.loc[index, 'matched'] = False

		# Get all predicted box
		pbox = df[df.name==name]

		# Find TP, FP for each class
		for index, row in pbox.iterrows():
			# Get predicted box position, score and label
			x = row['left']
			y = row['top']
			w = row['width']
			h = row['height']
			prob = row['score']
			label = int(row['label'])
			# If label is not in list then initialize
			if label not in P:
				P[label] = []
				T[label] = []
			# Append label
			P[label].append(prob)
			found_match = 0

			# Loop all ground truth bounding box
			for index, row in gtbox.iterrows():
				# Get bounding box coordinate
				gtx = row['left']
				gty = row['top']
				gtw = row['width']
				gth = row['height']

				# If GT box label is not matched or GT box if found, then continue
				if row['matched'] or row['label'] != label:
					continue

				# Find IoU
				iou_score = iou((x,x+w,y,y+h), (gtx,gtx+gtw,gty,gty+gth))

				# If IoU is greater than a threshold, mark this GT box is found
				if iou_score >= 0.5:
					found_match = 1
					gtbox.loc[index, 'matched'] = True
					break
			T[label].append(found_match)
		# print('[DBG] gtbox:\n', gtbox)

		# Loop all GT box to find what GT box is not found
		for index, row in gtbox.iterrows():
			if not row['matched']:
				if row['label'] not in P:
					P[row['label']] = []
					T[row['label']] = []
				P[row['label']].append(0)
				T[row['label']].append(1)

	# Calculate AP for each class
	AP = []
	for key in T.keys():
		class_ap = average_precision_score(T[key], P[key])
		AP.append(class_ap)
		print('[INFO] Class {}: AP = {}'.format(key, class_ap))
	print('[INFO] mAP = {}'.format(np.mean(AP)))

if __name__ == '__main__':
	get_map(path=args['path'], true_df=args['true'], pred_df=args['guess'])