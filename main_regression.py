import numpy as np
import pandas as pd
import config as cfg
import cv2
import os
import pickle
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.models import load_model

#### Load pre-trained model ####
# Load feature extraction model
model = load_model(cfg.MODEL_PATH)
output = model.get_layer(name='activation_5').output
extractor = Model(inputs=model.input, outputs=output)

# Load classifier model
f = open(cfg.CLASSIFY_PATH, 'rb')
classify = pickle.load(f)
f.close()

#### Prepare data ####

# Load label file
df = pd.read_csv('../data/labels/regRP.csv')

df1 = df[df.x < 0]
print(df1.head())
exit()

# Replace all row with left and top value <= by 0
df.loc[df.gtx < 0, 'gtx'] = 0
df.loc[df.gty < 0, 'gty'] = 0

# Initialize array to hold input and target
inputs = []
targets = []

# Loop all row
print('[INFO] Loading data...')
for index, row in df.iterrows():
	if index >= 10000:
		break
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
	# Find center of predicted box
	px = (x + w/2.0)/iw
	py = (y + h/2.0)/ih
	pw = np.float(w)/iw
	ph = np.float(h)/ih
	# Find center of ground truth box
	gx = (gtx + gtw/2.0)/iw
	gy = (gty + gth/2.0)/ih
	gw = np.float(gtw)/iw
	gy = np.float(gth)/ih
	# Compute target for bounding box regression
	tx = (gx - px)/pw
	ty = (gy - py)/ph
	tw = np.log(gtw/pw)
	th = np.log(gth/ph)
	# Append
	inputs.append((px,py,pw,ph))
	targets.append((tx,ty,tw,th))

# Convert to array
inputs = np.array(inputs, dtype=np.float)
targets = np.array(targets, dtype=np.float)
print('[INFO] Inputs shape: ', inputs.shape)
print('[INFO] Targets shape: ', targets.shape)

# Split into training and testing set
x_train, x_test, y_train, y_test = train_test_split(inputs, targets, test_size=0.3, random_state=29)
print('[INFO] x_train, x_test shape: ', x_train.shape, x_test.shape)
print('[INFO] y_train, y_test shape: ', y_train.shape, y_test.shape)

print('[DBG] x_train:\n', x_train[:5,:])
print('[DBG] y_train:\n', y_train[:5,:])

# Training linear regression model
# clf = Ridge()
# params={
# 	'alpha':np.linspace(0,1000,100)
# }

# grid = GridSearchCV(clf, params, cv=5)
# grid.fit(x_train, y_train)
# print('[INFO] Best params: ', grid.best_params_)
# alpha_list = np.linspace(0, 1, 11)
# for alpha in alpha_list:
# 	clf = Ridge(alpha=alpha, fit_intercept=True)
# 	clf.fit(x_train, y_train)
# 	print('[INFO] Score on training set with alpha = {} is {}'.format(alpha, clf.score(x_train, y_train)))
# 	print('[INFO] Score on testing set with alpha = {} is {}'.format(alpha, clf.score(x_test, y_test)))
# 	f = open(cfg.REGRESSION_PATH + '_' + str(alpha), 'wb')
# 	pickle.dump(clf, f)
# 	f.close()

clf = LinearRegression()
clf.fit(x_train, y_train)
print('[INFO] Score on training set with is {}'.format(clf.score(x_train, y_train)))
print('[INFO] Score on testing set with is {}'.format(clf.score(x_test, y_test)))