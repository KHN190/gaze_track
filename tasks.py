import os

os.environ['GLOG_minloglevel'] = '2'

import cv2
import json
import time
import caffe
import yaml
import pickle

with open('config.yml') as f:
    config = yaml.load(f.read(), Loader=yaml.CLoader)

if config.get('gpu'):
    caffe.set_mode_gpu()
    caffe.set_device(0)

from redis import Redis

from models.face_detect import *
from models.load_model import *
from models.blink_detect import *

import numpy as np


def face_track(cap=None, conn=None, states={}):
    # capture frame from camera
    if cap is None:
        cap = cv2.VideoCapture(0)

    # get frame from camera
    ret, frame = cap.read()
    if ret:
        # redis conn
        if conn is None:
            conn = Redis()
        # face + eye detection
        img, faces, face_feats = extract_frame_features(frame, grayed=False)
        # blink detection
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ear = extract_ears(img)

        # get last frames data
        blinks = states['blinks']
        states['recent_ears'] = update_recent_ear(states['recent_ears'], ear)
        mean_ear = calc_mean_ear(states['recent_ears'])

        # normalize with train set mean value
        offset = mean_ear - 0.2744 if mean_ear > 0 else 0
        states['cons_ear'] = curr_ear_window(states['cons_ear'], ear, states['n'], offset)

        # predict using svm
        X = np.array(states['cons_ear']).reshape(1, -1)
        pred = states['svm'].predict(X)[0]
        if pred > 0 and pred != states['last_pred']:
            blinks += 1
            print(' * blink detected\n')

        states['blinks'] = blinks
        states['last_pred'] = pred

        conn.set("faces", pickle.dumps(faces), ex=2)
        conn.set("face_feats", pickle.dumps(face_feats), ex=2)
        conn.set("blinks", str(blinks), ex=2)


def gaze_track(conn=None):

    # redis conn
    if conn is None:
        conn = Redis()

    faces = conn.get("faces")
    face_feats = conn.get("face_feats")

    if faces != '' and face_feats != '' and faces != None and face_feats != None:

        faces = pickle.loads(faces)
        face_feats = pickle.loads(face_feats)

        # detect gaze
        outputs = extract_gazes(faces, face_feats)

        res = ';'.join(
            [','.join([str(_) for _ in x.tolist()]) for x in outputs])

        if res is not None and len(res) > 0:

            ts = time.strftime("%b %d %Y %H:%M:%S", time.gmtime(time.time()))

            conn.set("gaze_coord", json.dumps(res), ex=2)
            conn.set("updated", json.dumps(ts), ex=2)


# Utils


# use default mean & var consistent with training data
# to normalize ear numbers
def curr_ear_window(cons_ear, ear, n, offset=0):
    if cons_ear == None or cons_ear == []:
        cons_ear = [-1.] * n

    if ear > 0:
        ear -= offset

    cons_ear.append(ear)
    cons_ear = cons_ear[1:]

    ears = [round(x, 3) for x in cons_ear]
    print(f" * ears: {ears}")

    return cons_ear


def update_recent_ear(recent_ears, ear):
    if ear > 0:
        recent_ears.append(ear)
        recent_ears = recent_ears[1:]
    return recent_ears


# calculate average mean
def calc_mean_ear(recent_ears):
    recent = [x for x in recent_ears if x > 0]
    if recent != []:
        return sum(recent) * 1.0 / len(recent)
    return -1
