#### Path to dataset ####
BASE_PATH = '../data/'

# Train dataset
TRAIN_PATH = BASE_PATH + 'train/'

# Test dataset
TEST_PATH = BASE_PATH + 'test/'

# Extra dataset
EXTRA_PATH = BASE_PATH + 'extra/'

#### Path to processed dataset ####
DES_PATH = BASE_PATH + 'dataset/'

# Dataset that contains number
DIGIT_PATH = DES_PATH + 'digits/'

# Dataset that contains background
NO_DIGIT_PATH = DES_PATH + 'no_digits/'

#### Labels folder contains ground truth bounding box ####
LABEL_PATH = BASE_PATH + 'labels/'

# Train set label
TRAIN_LABEL_FILE = LABEL_PATH + 'train_label.csv'

# Test set label
TEST_LABEL_FILE = LABEL_PATH + 'test_label.csv'

#### Build dataset parameters ####
# Number of region proposal per images
RP_PER_IMAGE = 2000

# Number of positive region per images
PRP_PER_IMAGE = 30

# Number of negative region per images
NRP_PER_IMAGE = 30

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