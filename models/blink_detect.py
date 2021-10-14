# https://github.com/Practical-CV/EYE-BLINK-DETECTION-WITH-OPENCV-AND-DLIB

from scipy.spatial import distance as dist
from imutils import face_utils

import numpy as np

import time
import dlib
import os
import cv2

# config for blink detection
COUNTER = 0
TOTAL = 0

# load facial landmark detector
model_root = os.getcwd() + '/models'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(model_root +
                                 "/shape_predictor_68_face_landmarks_GTX.dat")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']


def current_time():
    return int(round(time.time() * 1000))


def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])

    return (A + B) / (2.0 * C)


def extract_ears(img, grayed=False, debug=False):
    gray = img
    if not grayed:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    start_ms = current_time()
    rects = detector(gray, 0)
    ear = -1.0
    center = np.array([.0, .0])

    # take only the first
    for rect in rects[:1]:
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
