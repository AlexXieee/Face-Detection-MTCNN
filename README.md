## Face detection with Multi-task CNN (MTCNN)

Multi-task CNN (MTCNN) algorithm for face detection and alignment.  
Users can choose CPU or GPU to process the data with tensorflow.  
Users can process images and videos. Especially, you can **import existed video** and export result video, or **process real-time video** from camera then export.

## Script Description
* *fd\_adaboost\_import.py* Import existed video and export result.
* *fd\_adaboost\_realtime.py* Process real-time video from camera without export.
* *fd\_adaboost\_realtime\_save.py* Process real-time video from camera and export.
* Suffix: *\_cpu* or *\_gpu* Choose CPU/GPU

* **You must manually add data path with '(root)/data/images' and '(root)/data/videos'**

## Environment
* Python 3.6
* OpenCV 4.2
* TensorFlow 1.14
* Python package - numpy tensorflow cv2 tqdm 

## Reference 
> [1] Zhang K, Zhang Z, Li Z, et al. Joint Face Detection and Alignment Using Multitask Cascaded Convolutional Networks[J]. IEEE Signal Processing Letters, 2016, 23(10): 1499-1503.  
> MATLAB Code: https://github.com/kpzhang93/MTCNN_face_detection_alignment 

## License
This code is distributed under MIT LICENSE

## Contact
xieboxuann@gmail.com