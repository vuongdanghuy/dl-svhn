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

# Check if destination folder is exist
if not(os.path.exists(cfg.DES_PATH)):
	print('[INFO] Creating destination folder and sub folders...')
	os.mkdir(cfg.DES_PATH)
	os.mkdir(cfg.DIGIT_PATH)
	os.mkdir(cfg.NO_DIGIT_PATH)
	print('[INFO] Folder created')

# Load label file
df = pd.read_csv(cfg.TRAIN_LABEL_FILE)

# Find all images need to process in label file
train_names = np.unique(df.name)

if not args['nsamples']:
	args['nsamples'] = len(train_names)

# Initialize counter to count number of positive and negative image
totalPositive = 0
totalNegative = 0

# posFile = open('../data/labels/posRP.csv', 'w')
# negFile = open('../data/labels/negRP.csv', 'w')
regFile = open('../data/labels/regRP.csv', 'w')

# posFile.write('name,x,y,w,h,label\n')
# negFile.write('name,x,y,w,h,label\n')
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

	# for i in range(len(gtbox)):
	# 	label = gtbox['label'].values[i]
	# 	x = gtbox['left'].values[i]
	# 	y = gtbox['top'].values[i]
	# 	w = gtbox['width'].values[i]
	# 	h = gtbox['height'].values[i]

	# 	# Get ground truth box
	# 	window = image[y:y+h, x:x+w]
	# 	# Resize to target size
	# 	im_rsz = cv2.resize(window, cfg.IMG_SIZE)
	# 	# Save image
	# 	cv2.imwrite(os.path.join(cfg.DIGIT_PATH, str(totalPositive) + '_' + str(label) + '.png'), im_rsz)
	# 	# Increase counter
	# 	totalPositive += 1

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

			if overlapArea >= cfg.posThresh and tp_cnt < cfg.PRP_PER_IMAGE:
				# # Get predicted box
				# window = image[y:y+h, x:x+w]
				# # Resize to target size
				# im_rsz = cv2.resize(window, cfg.IMG_SIZE)
				# # Save image
				# cv2.imwrite(os.path.join(cfg.NO_DIGIT_PATH, str(totalPositive) + '_' + str(label) + '.png'), im_rsz)
				# posFile.write('{},{},{},{},{},{}\n'.format(name,x,y,w,h,label))

				if overlapArea >= cfg.regThresh:
					regFile.write('{},{},{},{},{},{},{},{},{},{}\n'.format(name,x,y,w,h,
						gtbox['left'].values[i],gtbox['top'].values[i],gtbox['width'].values[i],gtbox['height'].values[i],label))
				# # Increase counter
				totalPositive += 1
				tp_cnt += 1

			elif not fullOverlap and overlapArea <= cfg.negThresh and fp_cnt < cfg.NRP_PER_IMAGE:
				# # Get predicted box
				# window = image[y:y+h, x:x+w]
				# # Resize to target size
				# im_rsz = cv2.resize(window, cfg.IMG_SIZE)
				# # Save image
				# cv2.imwrite(os.path.join(cfg.NO_DIGIT_PATH, str(totalNegative) + '_' + str(label) + '.png'), im_rsz)
				# # Increase counter
				# negFile.write('{},{},{},{},{},{}\n'.format(name,x,y,w,h,0))
				totalNegative += 1
				fp_cnt += 1

		flag = (fp_cnt == cfg.NRP_PER_IMAGE) and (tp_cnt == cfg.PRP_PER_IMAGE)

		if flag:
			break

# posFile.close()
# negFile.close()
regFile.close()