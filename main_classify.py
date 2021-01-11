import numpy as np
import pandas as pd
import config as cfg
import os
import cv2
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from keras.models import load_model
from keras.models import Model
import pickle

# Load detector model
model = load_model(cfg.MODEL_PATH)
print('[INFO] Detector summary:\n', model.summary())

# Create feature extraction model
output = model.get_layer(name='activation_5').output

extractor = Model(inputs=model.input, outputs=output)
print('[INFO] Extractor summary:\n', extractor.summary())

#### Get all ground truth box feature ####

# Load label file
gt_df = pd.read_csv(cfg.TRAIN_LABEL_FILE)

# Replace all row with left and top value < 0 by 0
gt_df.loc[gt_df.left < 0, 'left'] = 0
gt_df.loc[gt_df.top < 0, 'top'] = 0

gtBox = []
gtLabel = []

print('[INFO] Loading all ground truth image...')
for index, row in gt_df.iterrows():
	# Load image
	image = cv2.imread(os.path.join(cfg.EXTRA_PATH, row['name']))

	# Get ground truth box
	window = image[row['top']:row['top']+row['height'], row['left']:row['left']+row['width'],:]

	# Resize
	window = cv2.resize(window, cfg.IMG_SIZE, interpolation=cv2.INTER_AREA)

	# Append
	gtBox.append(window)
	gtLabel.append(row['label'])

# Convert to array and normalization
gtBox = np.array(gtBox, dtype=np.float)/255.0
gtLabel = np.array(gtLabel)

# Convert ground truth image to feature
gtFeatures = extractor.predict(gtBox)

print('[INFO] gtFeatures shape: ', gtFeatures.shape)
print('[INFO] gtLabel shape: ', gtLabel.shape)

#### Get all negative sample feature ####

# Load label file
neg_df = pd.read_csv('../data/labels/negRP.csv')

negBox = []
negLabel = []

print('[INFO] Loading all negative image...')
for index, row in neg_df.sample(np.int(len(gtFeatures)/10*1.5), random_state=25).iterrows():
	# Load image
	image = cv2.imread(os.path.join(cfg.EXTRA_PATH, row['name']))

	# Get ground truth box
	window = image[row['y']:row['y']+row['h'], row['x']:row['x']+row['w'],:]

	# Resize
	window = cv2.resize(window, cfg.IMG_SIZE, interpolation=cv2.INTER_AREA)

	# Append
	negBox.append(window)
	negLabel.append(row['label'])

# Convert to array and normalization
negBox = np.array(negBox, dtype=np.float)/255.0
negLabel = np.array(negLabel)

# Convert ground truth image to feature
negFeatures = extractor.predict(negBox)

print('[INFO] negFeatures shape: ', negFeatures.shape)
print('[INFO] negLabel shape: ', negLabel.shape)

#### Split into train and test data ####
x = np.concatenate((gtFeatures, negFeatures), axis=0)
y = np.concatenate((gtLabel, negLabel))

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=9)

print('[INFO] x_train, x_test shape: ', x_train.shape, x_test.shape)
print('[INFO] y_train, y_test shape: ', y_train.shape, y_test.shape)

#### Train Support Vector Machine Classifier ####
# params={
# 	'C':[1,5,10],
# 	'kernel':['linear','rbf']
# }
# clf = SVC()
# gsc = GridSearchCV(clf, param_grid=params,cv=5)
# grid_result = gsc.fit(x_train, y_train)
# best_params = grid_result.best_params_
# print(best_params)
# exit()

svm = SVC(C=10, kernel='rbf', probability=True, random_state=42)

svm.fit(x_train, y_train)

#### Display score #### 
print('[INFO] Score on training set: ', svm.score(x_train, y_train))
print('[INFO] Score on testing set: ', svm.score(x_test, y_test))

#### Confusion matrix on training and testing set ####
train_pred = svm.predict(x_train)
test_pred = svm.predict(x_test)

print('[INFO] Confusion matrix on training set:\n', confusion_matrix(y_train, train_pred))
print('[INFO] Confusion matrix on testing set:\n', confusion_matrix(y_test, test_pred))

#### Save model ####
f = open(cfg.CLASSIFY_PATH, 'wb')
pickle.dump(svm, f)
f.close()