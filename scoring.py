import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from iou import iou
from sklearn.metrics import auc

def mAP(gt, pred, threshold=0.5, plot=False):
	"""
	Calculate multiclass Average Precision
	@INPUT:
		- gt: ground truth label file. Columns=['x', 'y', 'width', 'height', 'label']
		- pred: predicted label file. Columns=['x', 'y', 'w', 'h', 'score', 'label']
		- threshold: overlapping threshold to decide if two boxes are match
	"""
	# Load ground truth label file
	gt_df = pd.read_csv(gt)
	test_names = gt_df.name.unique()
	test_names = test_names[:1000]
	gt_df = gt_df[gt_df['name'].isin(test_names)]

	# Load predicted label file
	pred_df = pd.read_csv(pred)

	# Get number of classes
	classes = np.unique(gt_df['label'].values)
	print('[INFO] Class: ', classes)

	# Loop all classes
	for cl in classes:
		# Get all ground truth of class
		gt_class = gt_df[gt_df.label == cl]
		# print('[INFO] class_df:\n', gt_class.head())

		# Get number of instances of that class
		total = len(gt_class)

		# Get all image name in ground truth of that class
		gt_names = gt_class.name.unique()

		# Get all predicted of that class
		pred_class = pred_df[pred_df.label == cl].copy()
		pred_class = pred_class.reset_index(drop=True)

		# Initialize all predicted boxes are false positive
		pred_class['tp'] = 0
		print('[INFO] pred_class:\n', pred_class.head())

		# Get all image name in ground truth of that class
		pred_names = pred_class.name.unique()

		# Find all image name that appear both in ground truth and predict, all other name
		# in predicted is marked as false positive
		names = list(set(gt_names) & set(pred_names))
		# print('[INFO] names:\n', names)
		print('[INFO] Number of images that predicted have class {}: {}'.format(cl, len(pred_names)))
		print('[INFO] Number of images that actual have that class {}: {}'.format(cl, len(names)))

		# Loop all name
		for name in names:
			# print('[INFO] Image {}'.format(name))
			# Find all ground truth in that image
			gt_boxes = gt_class[gt_class.name == name][['left', 'top', 'width', 'height']].values
			# print(gt_boxes)

			# Find all predicted box in that image
			pd_boxes = pred_class[pred_class.name == name][['x','y','w','h']].values
			index = pred_class[pred_class.name == name].index
			# print(pd_boxes)
			# print('[INFO] Index: ', index)

			# Flag to indicate which ground truth box is found
			flags = np.zeros(len(gt_boxes))

			# Loop all predicted boxes
			for i,(x,y,w,h) in zip(index,pd_boxes):
				a = [x,x+w,y,y+h]
				# Loop all ground truth box
				for j,(gtx,gty,gtw,gth) in enumerate(gt_boxes):
					# If ground truth box is not found and IoU is greater than threshold
					b = [gtx,gtx+gtw,gty,gty+gth]
					iou_score = iou(a,b)
					if not(flags[j]) and iou_score >= threshold:
						# Mark this predicted box is true positive
						pred_class.loc[i,'tp'] = 1
						# Mark this ground truth box is found
						flags[j] = 1
		print('Number of true positive: ', np.sum(pred_class.tp))
		score = pred_class.score.values
		tp = pred_class.tp.values
		
		class_ap, recall, precision = AP(score, tp, total)
		print('[INFO] Class {} average precision: ', class_ap)

		if plot:
			plt.plot(recall, precision, label='AP={}'.format(np.around(class_ap,3)))
			plt.xlabel('Recall')
			plt.ylabel('Precision')
			plt.title('Precision-Recall Curve of class {}'.format(cl))
			plt.legend()
			plt.show()

		# Calculate Average Precision for that class
		# class_AP = AP(class_df.values, pred_class.values)
		# exit()


def AP(score, tp, total):
	"""
	Calculate Average Precision
	@INPUT:
		- score:
		- tp:
		- total:
	@OUTPUT:
		- r: List of recall value
		- p: List of precision value
		- ap: average precision value
	"""
	# Sorting based on score
	index = np.argsort(score)
	index = index[::-1]
	score = score[index]
	tp = tp[index]

	# Calculate precision and recall
	r = np.zeros(len(score))
	p = r.copy()

	for i in range(len(score)):
		r[i] = np.sum(tp[:i+1])/total
		p[i] = np.sum(tp[:i+1])/(i+1)

	# Calculate average precision
	ap = auc(r, p)

	return ap, r, p