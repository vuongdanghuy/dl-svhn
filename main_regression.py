import numpy as np
import pandas as pd
import config as cfg
import cv2
import os
import pickle
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from keras.models import Model, Sequential
from keras.models import load_model
from keras.layers import Dense, Dropout

#### Load pre-trained model ####
# # Load feature extraction model
# model = load_model(cfg.MODEL_PATH)
# output = model.get_layer(name='activation_5').output
# extractor = Model(inputs=model.input, outputs=output)

# # Load classifier model
# f = open(cfg.CLASSIFY_PATH, 'rb')
# classify = pickle.load(f)
# f.close()

#### Prepare data ####

# Load label file
df = pd.read_csv(cfg.REGRESSION_LABEL_FILE)

# Initialize array to hold input and target
inputs = []
targets = []
labels = []

# Loop all row
print('[INFO] Loading data...')
for index, row in df.iterrows():
	# if index >= 100000:
	# 	break
	# Load image
	image = cv2.imread(os.path.join(cfg.EXTRA_PATH, row['name']))
	# Get image size
	ih, iw = image.shape[:2]
	# Get predicted and ground truth bounding box
	x = row['x']
	y = row['y']
	w = row['w']
	h = row['h']
	gtx = row['gtx']
	gty = row['gty']
	gtw = row['gtw']
	gth = row['gth']
	# Get selective search patch
	window = image[y:y+h,x:x+w,:]
	# Resize
	window = cv2.resize(window, cfg.IMG_SIZE, interpolation=cv2.INTER_AREA)
	# Find center of predicted box
	px = x + w//2
	py = y + h//2
	pw = w
	ph = h
	# Find center of ground truth box
	gx = gtx + gtw//2
	gy = gty + gth//2
	gw = gtw
	gy = gth
	# Compute target for bounding box regression
	tx = np.float(gx - px)/pw
	ty = np.float(gy - py)/ph
	tw = np.log(gtw/pw)
	th = np.log(gth/ph)
	# Normalize input
	x = np.float(x)/iw
	y = np.float(y)/ih
	w = np.float(w)/iw
	h = np.float(h)/ih
	# Append
	inputs.append((x,y,w,h))
	targets.append((tx,ty,tw,th))
	labels.append(row['label'])

# Convert to array
inputs = np.array(inputs, dtype=np.float)
targets = np.array(targets, dtype=np.float)
labels = np.array(labels)

print('[INFO] Inputs shape: ', inputs.shape)
print('[INFO] Targets shape: ', targets.shape)

# # Extract features
# features = extractor.predict(inputs)

# # Predict based on those features
# pred = classify.predict(features)

# # Get only features that classify model predict is a number
# index = np.where(pred > 0)[0]

# pred = pred[index]
# features = features[index,:]
# labels = labels[index]
# targets = targets[index,:]

# Split into training and testing set
x_train, x_test, y_train, y_test = train_test_split(inputs, targets, test_size=0.3, random_state=29)
print('[INFO] x_train, x_test shape: ', x_train.shape, x_test.shape)
print('[INFO] y_train, y_test shape: ', y_train.shape, y_test.shape)

# Training bounding box regression model
# for i in range(-4,5):
# 	reg_x = Ridge(10**i)
# 	reg_x.fit(x_train,y_train)
# 	print('[INFO] For alpha = {}: score = {}, test_score = {}'.format(10**i,reg_x.score(x_train,y_train),reg_x.score(x_test,y_test)))
reg_x = Ridge(10000)
reg_x.fit(x_train,y_train)

f = open(cfg.REGRESSION_PATH, 'wb')
pickle.dump(reg_x, f)
f.close()