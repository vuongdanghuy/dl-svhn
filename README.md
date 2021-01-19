# Street View Home Number Detection & Recognition

This project use Format 1 dataset from **SVHN dataset**, obtained from house numbers in Google Street View images.  

These are the original, variable-resolution, color house-number images with character level bounding boxes. The bounding box information are stored in digitStruct.mat. 

Each tar.gz file contains the orignal images in png format, together with a **digitStruct.mat** file, which can be loaded using Matlab. The **digitStruct.mat** file contains a struct called digitStruct with the same length as the number of original images. Each element in **digitStruct** has the following fields: **name** which is a string containing the filename of the corresponding image, **bbox** which is a struct array that contains the position, size and label of each digit bounding box in the image.  

Eg: digitStruct(300).bbox(2).height gives height of the 2nd digit bounding box in the 300th image.  

You can download this dataset [here](http://ufldl.stanford.edu/housenumbers/).

## How to use this project  

1. Download dataset from this link [here](http://ufldl.stanford.edu/housenumbers/).
2. Extract training, testing and extra dataset to folder name **data**
3. Clone this Git repository, you'll have a folder name **dl-svhn** contain all the source code.
4. Checkout the branch you want to use  
    Currently there are 3 branches in this repo with different solution:
    + <code>*dev*</code>: Use R-CNN model
    + <code>*frcnn*</code>: Use Faster R-CNN model
    + <code>*yolo*</code>: Use Tiny YOLO model
    
    R-CNN model is the baseline model. It's simple and easy to understand. It also doesn't require much computation power to train but on the other hand take a long time to test and its accuracy is just so so.
    
    YOLO model is the best model. It's fast and has high accuracy. But it need high computation power to train.
    
    Faster R-CNN model is somewhat between. It is not fast and accurate as YOLO but it's still much faster and have higher accuracy than R-CNN model. It's also need high computation power.
    
    Select your prefer solution and simply run:
    
    > git checkout branch_name
