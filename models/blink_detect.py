# https://github.com/Practical-CV/EYE-BLINK-DETECTION-WITH-OPENCV-AND-DLIB

from scipy.spatial import distance as dist
from imutils import face_utils

import time
import dlib
import os

# config for blink detection
EYE_AR_THRESH = 0.30
EYE_AR_CONSEC_FRAMES = 3

# count blinks
COUNTER = 0
TOTAL = 0

# load facial landmark detector
model_root = os.getcwd() + '/models'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(model_root + "/shape_predictor_68_face_landmarks.dat")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']


def current_time():
    return int(round(time.time() * 1000))


def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])

    return (A + B) / (2.0 * C)


def extract_blinks(gray):
    global COUNTER, TOTAL

    start_ms = current_time()
    rects = detector(gray, 0)

    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        ear = (leftEAR + rightEAR) / 2.0

        if ear < EYE_AR_THRESH:
            COUNTER += 1
        else:
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                TOTAL += 1

            COUNTER = 0

    duration_ms = current_time() - start_ms
    print("Eye blink detection took:     ", str(duration_ms / 1000) + "s")

    return COUNTER
