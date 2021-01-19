# Street View Home Number Detection & Recognition using R-CNN

This project use Format 1 dataset from **SVHN dataset**, obtained from house numbers in Google Street View images.  

These are the original, variable-resolution, color house-number images with character level bounding boxes. The bounding box information are stored in digitStruct.mat instead of drawn directly on the images in the dataset.

Each tar.gz file contains the orignal images in png format, together with a **digitStruct.mat** file, which can be loaded using MATLAB. The **digitStruct.mat** file contains a struct called digitStruct with the same length as the number of original images. Each element in **digitStruct** has the following fields: **name** which is a string containing the filename of the corresponding image, **bbox** which is a struct array that contains the position, size and label of each digit bounding box in the image.  

Eg: digitStruct(300).bbox(2).height gives height of the 2nd digit bounding box in the 300th image.  

You can download this dataset [here](http://ufldl.stanford.edu/housenumbers/).

## Requirements

    numpy==1.19.3
    pandas==1.1.5
    keras==2.4.3
    tensorflow==2.4.0
    opencv-contrib-python==4.4.0.46
    matplotlib==3.3.3
    scikit-learn==0.24.0
    Flask==1.1.2

If you have pip installed, simply run:

    pip install -r requirements.txt

## How to use this project  

This project used R-CNN model to detect and classify house number in images. 

For more information about R-CNN please read the following article [[1]](https://arxiv.org/pdf/1311.2524.pdf) from authors of R-CNN. Or you can read a wonderful tutorial that [Adrian Rosebrock, PhD](https://www.pyimagesearch.com/) has [here](https://www.pyimagesearch.com/2020/07/13/r-cnn-object-detection-with-keras-tensorflow-and-deep-learning/).

SVHN format 1 comes with 3 datasets: **train**, **test** and **extra**. Images in **train** and **test** set are very noisy mean while images in **extra** set are less difficult. So we'll create our own training and testing set from **extra** set and use them to build and evaluate model. Provided **train** and **test** set will be used as additional testing set.

1. Download dataset from this link [here](http://ufldl.stanford.edu/housenumbers/).
2. Extract training, testing and extra dataset to folder <code>*/data/*</code>. Create a folder <code>/labels/</code> inside <code>*/data/*</code> to hold all the label files.
3. Clone this Git repository, you'll have a folder <code>*/dl-svhn/*</code>
4. Check out branch *dev* contains all the source code. Your working folder now would look like this:
```
.
|-- data
|   |-- extra
|   |-- labels
|   |-- test
|   `-- train
`-- dl-svhn
    |-- README.md
    |-- __pycache__
    |-- build_dataset.py
    |-- config.py
    |-- convert_mat.m
    |-- get_map.py
    |-- infer.py
    |-- inspect_data.py
    |-- iou.py
    |-- main.py
    |-- main_classify.py
    |-- main_regression.py
    |-- nms.py
    |-- output
    |-- requirements.txt
    |-- selective_search.py
    |-- server.py
    |-- templates
    |-- train.py
    |-- uploads
    `-- visualize.py
```
5. If you have MATLAB installed, you can run <code>convert_mat.m</code> file to create .csv label files. It loads **digitStruct.mat** files, creates 3 csv files named *train_label.csv*, *test_label.csv* and *extra_label.csv* and saves them in <code>*/data/labels/*</code>.

6. Run <code>inspect_data.py</code> file to create our own training and tesing set. It removes all images with ground truth box size < 32x32 and creates 2 new label file: *train.csv* and *test.csv* file

7. **Training**

    + Run <code>build_dataset.py</code> file: This file loads all images in *train.csv*, performs Selective Search to get region proposal coordinates and classifies it as: Positive, Negative or Neutral. This file also creates 3 new file: *posRP.csv*, *negRP.csv* and *regRP.csv* used for training.
    + Run <code>main.py</code> file: This file loads data from *posRP.csv*, *negRP.csv* file, splits it into train and validation set, trains a feature extraction model and saves that model in <code>*/dl-svhn/output/*</code> folder.
    + Run <code>main_classify.py</code> file: This file loads data from *negRP.csv*, *train.csv*, gets data features by running feature extraction model from above step and uses those features to train a SVM classify model. Trained SVM model are saved in <code>*/dl-svhn/output/*</code>.
    + Run <code>main_regression.py</code> file: This file trains a bounding box regression model to fine-tune region proposal coordinates. Trained regression model is saved in <code>*/dl-svhn/output/*</code>.

8. **Testing**

    + <code>visualize.py</code>: This file is used to visualize model results.
    + <code>infer.py</code>: This file loads all images in *test.csv* file and passes them through R-CNN model. Results are written into a label file named *infer.csv*
    + Run <code>get_map.py</code> function: Passing path to *test.csv* and *infer.csv* file as argument to compute mAP value.
    
