#### Path to dataset ####
BASE_PATH = '../data/'

# Train dataset
TRAIN_PATH = BASE_PATH + 'train/'

# Test dataset
TEST_PATH = BASE_PATH + 'test/'

# Extra dataset
EXTRA_PATH = BASE_PATH + 'extra/'

#### Labels folder contains bounding box ####
LABEL_PATH = BASE_PATH + 'labels/'

# Train set label
TRAIN_LABEL_FILE = LABEL_PATH + 'train.csv'

# Test set label
TEST_LABEL_FILE = LABEL_PATH + 'test.csv'

# Extra set label
EXTRA_LABEL_FILE = LABEL_PATH + 'extra_label.csv'

# Predicted label
INFER_LABEL_FILE = LABEL_PATH + 'infer.csv'

# Positive samples label file
POSITIVE_LABEL_FILE = LABEL_PATH + 'posRP.csv'

# Negative samples label file
NEGATIVE_LABEL_FILE = LABEL_PATH + 'negRP.csv'

# Regression samples label file
REGRESSION_LABEL_FILE = LABEL_PATH + 'regRP.csv'

#### Build dataset parameters ####
# Number of region proposal per images
RP_PER_IMAGE = 2000

# Number of positive region per images
PRP_PER_IMAGE = 30

# Number of negative region per images
NRP_PER_IMAGE = 30

# IoU threshold to decide a box is near ground truth. Use for bounding box regression
regThresh = 0.7

# IoU threshold to decide a box is positive
posThresh = 0.5

# IoU threshold to decide a box is negative
negThresh = 0.2

# Image target size
IMG_SIZE = (32, 32)

#### MODEL PARAMETERS ####
BATCH_SIZE = 32
EPOCHS = 25

#### Output path ####
OUTPUT_PATH = './output/'

# Model path
MODEL_PATH = OUTPUT_PATH + 'model.hdf5'

# Classify model path
CLASSIFY_PATH = OUTPUT_PATH + 'classify.pkl'

# Regression model path
REGRESSION_PATH = OUTPUT_PATH + 'regression.pkl'