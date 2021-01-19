# House number detection - Faster R-CNN on Google colab
Clone from [source](https://github.com/kbardool/keras-frcnn)  

***Dependencies:***  
* tensorflow-gpu: 1.14.0  
* keras: 2.2.4
* h5py: 2.10.0  
  
**1. Train model**  
  * **Bước 1:** Chuẩn bị file txt với mỗi dòng có format như sau: `filepath,x1,y1,x2,y2,class_name`  
  ***Ví dụ:***  
/content/1.png,837,346,981,456,1  
/content/2.png,215,312,279,391,9   
  * **Bước 2:** Sử dụng `train_frcnn.py` để train model: `python train_frcnn.py -o simple -p my_data.txt`  
  * **Bước 3:** Chạy `train_frcnn.py` sẽ lưu các tham số weights của model vào file **hdf5**, mọi cài đặt trong quá trình trainning được lưu vào một `pickle` file. Những cài đặt này được load bởi `test_frcnn.py` cho quá trình testing.  
    
***Note:*** Cấu trúc thư mục:  
  ```  
    |-- keras_frcnn/  
    |  |-- data/  
    |  |-- keras_frcnn/  
   ```
  * `keras_frcnn` chứa folder keras_frcnn, file `pickle`, file `lib.txt` lưu tên các thư viện cần cài đặt, file `model_frcnn.dhf5` chứa pre-trained weights, `train_frcnn.py`, `test_frcnn.py`, file `training.ipynb` quá trình training và testing trên google colab.  
  * data: chứa file `train.txt`, `test.txt` và `val.txt` phục vụ cho quá trình training và testing; file `inference_result_frcnn.csv` lưu kết quả inference của tập test.  
  * Dữ liệu từ các tập train, val và test được nén và đưa lên drive, sau đó được extract vào `content` trong mỗi phiên làm việc với google colab.  
    
**2. Infer model**  
  Từ pre-trained weights và file config (`pickle`), thực hiện infer tập ảnh test như sau: `python test_frcnn.py -p <path to test data>`.  
    
**3. Results**  
  * Inference time: 1s/image  
  * mAP = 96.8 %  
  [img]
