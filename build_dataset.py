import numpy as np
import pandas as pd
import os
import argparse
import cv2
from selective_search import selective_search
from iou import iou
import config as cfg
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('-n', '--nsamples', type=int, default=0, help='Number of images used to create dataset')
ap.add_argument('-m','--method',type=str, default='quality', 
				choices=['fast', 'quality'], help='Selective search method')
args = vars(ap.parse_args())

# Load label file
df = pd.read_csv(cfg.TRAIN_LABEL_FILE)

# Find all images need to process in label file
train_names = np.unique(df.name)

if not args['nsamples']:
	args['nsamples'] = len(train_names)

# Initialize counter to count number of positive and negative image
totalPositive = 0
totalNegative = 0

# Open label file to write region proposal coordinate
posFile = open(cfg.POSSITIVE_LABEL_FILE, 'w')
negFile = open(cfg.NEGATIVE_LABEL_FILE, 'w')
regFile = open(cfg.REGRESSION_LABEL_FILE, 'w')

posFile.write('name,x,y,w,h,label\n')
negFile.write('name,x,y,w,h,label\n')
regFile.write('name,x,y,w,h,gtx,gty,gtw,gth,label\n')

print('[INFO] Create dataset base on {} images'.format(args['nsamples']))
print('[INFO] Selective search method *{}*'.format(args['method']))

# Loop all images in training data
for name in train_names[:args['nsamples']]:
	# Load images
	print('[INFO] Proccessing image {}'.format(name))
	image = cv2.imread(os.path.join(cfg.EXTRA_PATH + name))

	# Get ground truth bounding box in image
	gtbox = df[df.name == name]

	# Perform selective search on image
	rects = selective_search(image, method=args['method'], verbose=False, display=False)

	# Initialize counter to count number of false positive, true positve proposed box
	fp_cnt = 0
	tp_cnt = 0

	# Compute IoU between region proposals and ground truth bounding box
	flag = False
	for (x, y, w, h) in rects:
		for i in range(len(gtbox)):
			label = gtbox['label'].values[i]
			bbox = [gtbox['left'].values[i], gtbox['left'].values[i] + gtbox['width'].values[i],
					gtbox['top'].values[i], gtbox['top'].values[i] + gtbox['height'].values[i]]
			rect = [x, x+w, y, y+h]

			# Determine if the proposed bounding box falls within the ground truth box
			fullOverlap = rect[0] >= bbox[0]
			fullOverlap = fullOverlap and (rect[1] <= bbox[1])
			fullOverlap = fullOverlap and (rect[2] >= bbox[2])
			fullOverlap = fullOverlap and (rect[3] <= bbox[3])

			# Calculate IoU
			overlapArea = iou(bbox, rect)

			# If IoU is greater than a threshold and number of positive samples do not hit the limit yet
			if overlapArea >= cfg.posThresh and tp_cnt < cfg.PRP_PER_IMAGE:
				posFile.write('{},{},{},{},{},{}\n'.format(name,x,y,w,h,label))
				if overlapArea >= cfg.regThresh:
					regFile.write('{},{},{},{},{},{},{},{},{},{}\n'.format(name,x,y,w,h,
						gtbox['left'].values[i],gtbox['top'].values[i],gtbox['width'].values[i],gtbox['height'].values[i],label))
				# # Increase counter
				totalPositive += 1
				tp_cnt += 1

			# If IoU is less than a threshold and number of negative samples do not hit the limit yet
			elif not fullOverlap and overlapArea <= cfg.negThresh and fp_cnt < cfg.NRP_PER_IMAGE:
				negFile.write('{},{},{},{},{},{}\n'.format(name,x,y,w,h,0))
				# Increase counter
				totalNegative += 1
				fp_cnt += 1

		flag = (fp_cnt == cfg.NRP_PER_IMAGE) and (tp_cnt == cfg.PRP_PER_IMAGE)

		if flag:
			break

# Close file
posFile.close()
negFile.close()
regFile.close()