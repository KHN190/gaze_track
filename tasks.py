import os
os.environ['GLOG_minloglevel'] = '2' 

import caffe

# caffe.set_mode_gpu()
# caffe.set_device(0)

import cv2
import json
import time

from redis import Redis

from models.face_detect import *
from models.load_model import *
from models.blink_detect import *

import pickle


def face_track(cap=None, conn=None):
    # capture frame from camera
    if cap is None:
        cap = cv2.VideoCapture(0)

    # get frame from camera
    ret, frame = cap.read()
    if ret:
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img, faces, face_feats = extract_frame_features(img, grayed=True)

        blinks = extract_blinks(img)

        # redis conn
        if conn is None:
            conn = Redis()
        conn.set("faces", pickle.dumps(faces), ex=2)
        conn.set("face_feats", pickle.dumps(face_feats), ex=2)
        conn.set("blinks", pickle.dumps(blinks), ex=2)


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

        res = ';'.join([','.join([str(_) for _ in x.tolist()]) for x in outputs])

        if res is not None and len(res) > 0:

            ts = time.strftime("%b %d %Y %H:%M:%S", time.gmtime(time.time()))

            conn.set("gaze_coord", json.dumps(res), ex=2)
            conn.set("updated", json.dumps(ts), ex=2)
