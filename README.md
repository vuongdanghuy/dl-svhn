# Street View Home Number Detection & Recognition

This project use Format 1 dataset from **SVHN dataset**, obtained from house numbers in Google Street View images.  

These are the original, variable-resolution, color house-number images with character level bounding boxes, as shown in the examples images above. (The blue bounding boxes in the example image are just for illustration purposes. The bounding box information are stored in digitStruct.mat instead of drawn directly on the images in the dataset.)  

Each tar.gz file contains the orignal images in png format, together with a **digitStruct.mat** file, which can be loaded using Matlab. The **digitStruct.mat** file contains a struct called digitStruct with the same length as the number of original images. Each element in **digitStruct** has the following fields: **name** which is a string containing the filename of the corresponding image, **bbox** which is a struct array that contains the position, size and label of each digit bounding box in the image.  

Eg: digitStruct(300).bbox(2).height gives height of the 2nd digit bounding box in the 300th image.  

You can download this dataset [here](http://ufldl.stanford.edu/housenumbers/).

## How to use this branch of the project  
### Training

1. Download dataset from this link [here](http://ufldl.stanford.edu/housenumbers/).
2. Extract training and testing dataset to folder name **data**. Delete the two label files in extra. You should archive the extra folder in the data
3. After cloning this repo, all you really need is the two notebook file located in SVHN_Yolo. Upload those to your Google Colab
4. Run the Yolo_setup.ipynb notebook. 
5. First mount and create the folder SVHN_Yolo under `/content/gdrive/MyDrive/`. Follow the instructions and clone the darknet repo inside the folder we just created.
6. Cd to `./darknet/`. Open the Makefile for darknet and change GPU, OpenCV, and Cuda to 1, then use `make` to build `darknet`.
7. Next thing to do is to extract the `extra.rar` file into `/content/extra`
8. Get the pretrained weights for tiny-yolo4 `yolov4-tiny.conv.29` and the label file `train.csv` and `test.csv` and put them under `darknet/data/`
9. cd to `darknet/data/`, run the cells that contain libraries, functions definions, prepare training process, create necessary file. Sucessfully do so would:  
    - split the extra folder into train/test/valid set with labels under `/content/` 
    - createtrain/test/valid.txt files that contain the path to all images in the sets under `darknet/data/`
    - create obj.names that contains the name of the class, here it's only 0-9
    - create obj.data that contains the path to the files above. This will be the file you pass to darknet to create the model
10. Copy the `svhn.cfg` in this repo and put it under `darknet/cfg`
11. Everything is now ready. cd to `darknet/` and use this command to start the training process:
```
!./darknet detector train ./data/obj.data cfg/svhn.cfg ./data/yolov4-tiny.conv.29 -dont_show -mjpeg_port 8090 -map
```
12. To test use:
```
!./darknet detector test data/obj.data cfg/svhn.cfg backup/svhn_best.weights -ext_output -dont_show -out result.json < ./data/test.txt
```

- for a single image you can replace `< ./data/test.txt` to your custom image path 
13. To plot the result from result.json use:
```
OneFuncToDoEverything(steps='a', disp =1, return_df =0)
```    

This is the layout this repo with the necessary files
```
|-- dl-svhn/
    |-- SVHN_Yolo
        |-- yolo_setup.ipynb
        |-- yolo_flask.ipynb
        |-- svhn_best.weights
        |-- svhn.cfg
        |-- train.csv
        |-- test.csv
        |-- index.html

```

## To create a live, flask server for demo
1. Upload the Yolo_flask.ipynb
2. Create /darknet/templates and put index.html inside
3. Create /darknet/data/uploads 
2. Run the entire notebook
3. Click on the `*.io` link to get to the demo website
4. Upload file and submit. `.png` files are preferred over `.jpg`.



