# https://github.com/Practical-CV/EYE-BLINK-DETECTION-WITH-OPENCV-AND-DLIB

from scipy.spatial import distance as dist
from imutils import face_utils

import numpy as np

import time
import dlib
import os
import cv2

# load facial landmark detector
model_root = os.getcwd() + '/models'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(model_root + "/shape_predictor_68_face_landmarks_GTX.dat")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']

# cnn detector + predictor
import caffe
os.environ['GLOG_minloglevel'] = '2' 

model_weight = model_root + "/mmod_human_face_detector.dat"
detector_cnn = dlib.cnn_face_detection_model_v1(model_weight)

net_work_path = model_root + '/landmark_deploy.prototxt'
weight_path = model_root + '/VanFace.caffemodel'

net = caffe.Net(net_work_path, weight_path, caffe.TEST)
net.name = 'FaceThink_face_landmark'


def current_time():
    return int(round(time.time() * 1000))


def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])

    return (A + B) / (2.0 * C)


def face_rect(frame, rect):

    x1 = max(0, rect.left())
    y1 = max(0, rect.top())
    x2 = min(rect.right(), frame.shape[1])
    y2 = min(rect.bottom(), frame.shape[0])

    gray = frame[y1:y2+1, x1:x2+1,]
    # gray = cv2.cvtColor(roi, cv2.COLOR_GRAY2GRAY)

    w, h = 60, 60
    res = cv2.resize(gray, (w, h), 0.0, 0.0, interpolation=cv2.INTER_CUBIC)
    resize_mat = np.float32(res)

    m = np.zeros((w, h))
    sd = np.zeros((w, h))
    mean, std_dev = cv2.meanStdDev(resize_mat, m, sd)
    new_m = mean[0][0]
    new_sd = std_dev[0][0]
    img = (resize_mat - new_m) / (0.000001 + new_sd)

    return img

# get 68 points
def landmarks(points, rect, frame):
    x1 = max(0, rect.left())
    y1 = max(0, rect.top())
    x2 = min(rect.right(), frame.shape[1])
    y2 = min(rect.bottom(), frame.shape[0])

    res = []
    for i in range(len(points) // 2):
        x = points[2*i] * (x2 - x1) + x1
        y = points[2*i+1] * (y2 - y1) + y1
        res.append([x, y])

    return np.array(res)


def extract_ears(img, grayed=False, cnn=False, debug=False):
    gray = img
    if not grayed:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    start_ms = current_time()
    ear = -1.0
    center = np.array([.0, .0])

    if cnn:
        rects = [r.rect for r in detector_cnn(gray, 0)]
    else:
        rects = detector(gray, 0)

    # take only the first
    for rect in rects[:1]:
        # if cnn:
        #     net.blobs['data'].data[...] = face_rect(gray, rect)
        #     net.forward()
        #     shape = net.blobs['Dense3'].data[0].flatten()
        #     shape = landmarks(shape, rect, gray)        

        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        l_center = sum(leftEye) / len(leftEye)
        r_center = sum(rightEye) / len(rightEye)

        ear = (leftEAR + rightEAR) / 2.0
        center = (l_center + r_center) / 2.0

        if debug:
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)

            cv2.drawContours(img, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(img, [rightEyeHull], -1, (0, 255, 0), 1)

        duration_ms = current_time() - start_ms
        print("Eye aspect ratio took:     ", str(duration_ms / 1000) + "s")

    return ear, center
