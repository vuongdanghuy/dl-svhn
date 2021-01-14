import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import config as cfg

# Load extra label file
df = pd.read_csv(cfg.EXTRA_LABEL_FILE)

# Find number of image, number of instances for each class
names = df.name.unique()
print('[INFO] Number of images: ', len(names))

labels = df.label.unique()
class_cnt = []
for label in labels:
	class_cnt.append(len(df[df.label == label]))
	print('[INFO] Number of instances for class {}:{}'.format(label, len(df[df.label == label])))

# plt.bar(labels, class_cnt)
# plt.xlabel('Class')
# plt.ylabel('Number')
# plt.title('Number of instances in each class')
# plt.show()

# Find width and height distribution
df1 = df[(df.height >= 32) & (df.width >= 32)]

# Find number of image, number of instances for each class
names = df1.name.unique()
print('[INFO] Number of images: ', len(names))

labels = df1.label.unique()
class_cnt = []
for label in labels:
	class_cnt.append(len(df1[df1.label == label]))
	print('[INFO] Number of instances for class {}:{}'.format(label, len(df1[df.label == label])))

# plt.bar(labels, class_cnt)
# plt.xlabel('Class')
# plt.ylabel('Number')
# plt.title('Number of instances in each class')
# plt.show()

# Split into train and test set
split_ratio = 0.3
train_name = names[:np.int(len(names)*(1-split_ratio))]
test_name = names[np.int(len(names)*(1-split_ratio)):]

print('There are {} images in train set'.format(len(train_name)))
print('There are {} images in test set'.format(len(test_name)))

# Inspect train set
df_train = df1[df1['name'].isin(train_name)]
class_cnt = np.zeros(10)
for index, row in df_train.iterrows():
	class_cnt[row['label']-1] += 1

# plt.bar(labels, class_cnt)
# plt.xlabel('Class')
# plt.ylabel('Number')
# plt.title('Number of instances in each class')
# plt.show()

# Inspect test set
df_test = df1[df1['name'].isin(test_name)]
class_cnt = np.zeros(10)
for index, row in df_test.iterrows():
	class_cnt[row['label']-1] += 1

# plt.bar(labels, class_cnt)
# plt.xlabel('Class')
# plt.ylabel('Number')
# plt.title('Number of instances in each class')
# plt.show()

# Save label file
# df_train.to_csv('./train.csv',index=False)
# df_test.to_csv('./test.csv',index=False)
tmp = df[df['name'].isin(train_name)]
tmp.to_csv(cfg.TRAIN_LABEL_FILE,index=False)
tmp1 = df[df['name'].isin(test_name)]
tmp1.to_csv(cfg.TEST_LABEL_FILE,index=False)