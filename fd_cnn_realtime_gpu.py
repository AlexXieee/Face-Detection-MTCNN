import warnings
warnings.filterwarnings('ignore')

import os
import sys
import random
from tqdm import tqdm
from scipy import misc

import cv2
import matplotlib.pyplot as plt

import tensorflow as tf

import detect_face

ROOT_DIR = os.getcwd()
DATA_PATH = os.path.join(ROOT_DIR, "data")
TEST_IMGS_PATH = os.path.join(DATA_PATH, "images")
TEST_VIDEOS_PATH = os.path.join(DATA_PATH, "videos")

minsize = 20  # minimum face area
threshold = [0.6, 0.7, 0.7]  # threshold of P-net, R-net, O-net
factor = 0.709  # scale factor

gpu_memory_fraction = 0.4

print('Creating networks and loading parameters')

# tensorflow config
with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False)) #use GPU
    # sess = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}, log_device_placement=True))  # use CPU
    with sess.as_default():
        pnet, rnet, onet = detect_face.create_mtcnn(sess, None)  # build P-net, R-net, O-net

video_capture = cv2.VideoCapture(0)

(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

total_faces_detected = 0

while (video_capture.isOpened()):
    ret, bgr_image = video_capture.read()  # read real-time frame

    rgb_image = bgr_image[:, :, ::-1]

    bounding_boxes, _ = detect_face.detect_face(rgb_image, minsize, pnet, rnet, onet, threshold, factor)

    total_faces_detected += len(bounding_boxes)

    # iterate every face position (x, y, w, h), left top = (x, y), width and height of rectangle = (w, h)
    for face_position in bounding_boxes:
        face_position = face_position.astype(int)
        # print(face_position[0:4])
        x1 = face_position[0] if face_position[0] > 0 else 0
        y1 = face_position[1] if face_position[1] > 0 else 0
        x2 = face_position[2] if face_position[2] > 0 else 0
        y2 = face_position[3] if face_position[3] > 0 else 0

        # draw the boundary by openCV
        cv2.rectangle(bgr_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # output handled video by openCV
    cv2.imshow('Video', bgr_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()

print("Total faces detected: ", total_faces_detected)


