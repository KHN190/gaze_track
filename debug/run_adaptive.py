from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import time
import numpy as np
import imutils
import cv2
import dlib


def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    C = dist.euclidean(eye[0], eye[3])

    ear = (A + B) / (2.0 * C)
    return ear

def eye_center(eye):
    return sum(eye) * 1.0 / len(eye)


TOTAL = 0

print('[INFO] Loading facial landmark predictor...')
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(
    '../models/shape_predictor_68_face_landmarks_GTX.dat')

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']

print('[INFO] Starting video stream thread...')
vs = VideoStream(src=0).start()
fileStream = False

time.sleep(1.0)

pred = 0
last_pred = 0
recent_ears = [-1.] * 10
recent_ears_long = [-1.] * 20
centers = [-1.] * 10
eye_move_thres = 30
thres = 0.89


def update_recent_ear(recent_ears, ear):
    if ear > 0:
        recent_ears.append(ear)
        recent_ears = recent_ears[1:]
    return recent_ears


def calc_mean_ear(recent_ears):
    recent = [x for x in recent_ears if x > 0]
    if recent != []:
        return sum(recent) * 1.0 / len(recent)
    return -1


while True:
    if fileStream and not vs.more():
        break

    frame = vs.read()
    frame = imutils.resize(frame, width=800)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 0)
    center = 0

    if len(rects) > 0:
        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)

            l_center = eye_center(leftEye)
            r_center = eye_center(rightEye)

            ear = (leftEAR + rightEAR) / 2.0
            center = (l_center + r_center) / 2.0

            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
    else:
        # no eyes detected
        ear = -1.

    # predict with adaptive thres
    recent_ears = update_recent_ear(recent_ears, ear)
    recent_ears_long = update_recent_ear(recent_ears_long, ear)

    centers.append(center)
    centers = centers[1:]

    mean_ear = calc_mean_ear(recent_ears)
    mean_ear_long = calc_mean_ear(recent_ears_long)

    mean_center = sum(centers) * 1.0 / len(centers)
    diff_center = dist.euclidean(mean_center, center)

    cv2.putText(frame, "Eye diff: {:.2f}".format(diff_center), (10, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 225, 0), 2)

    # only detect when light changes not too fast,
    # and that ear is stable
    if abs(mean_ear_long - mean_ear) < 0.02 and diff_center < eye_move_thres:

        pred = 1 if (ear > 0 and ear <= mean_ear * thres) else 0

        # ignore consecutive blink detection
        if pred > 0 and last_pred != pred:
            TOTAL += 1

        last_pred = pred

    cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, "EAR mean 1: {:.2f}".format(mean_ear), (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 225, 0), 2)
    cv2.putText(frame, "EAR mean 2: {:.2f}".format(mean_ear_long), (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 225, 0), 2)
    
    if diff_center >= eye_move_thres:

        cv2.putText(frame, "Eye center moved.", (10, 500),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 225, 0), 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    time.sleep(0.05)

    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()
