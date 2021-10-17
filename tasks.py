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

# capture video

from threading import Thread

class VideoGet:
    """
    Class that continuously gets frames from a VideoCapture object
    with a dedicated thread.
    """

    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        self.stopped = False

    def start(self):    
        Thread(target=self.read, args=()).start()
        return self

    def read(self):
        if not self.stopped:
            return self.stream.read()

    def stop(self):
        self.stopped = True
        self.stream.release()


def face_track(q, conn, states={}):
    frame = q.get(False)

    # face + eye detection
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img, faces, face_feats = extract_frame_features(frame, grayed=False)

    conn.set('faces', pickle.dumps(faces), ex=2)
    conn.set('face_feats', pickle.dumps(face_feats), ex=2)


def blink_track(frame, conn, states={}, method='adaptive'):
    debug = config['debug']
    use_cnn = config['cnn_face_detect']

    # face + eye detection
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if debug:
        ear, center = extract_ears(frame, grayed=False, cnn=use_cnn, debug=True)
        states['frame'] = frame
    else:
        ear, center = extract_ears(img, grayed=True, cnn=use_cnn)

    globals()[f"blink_{method}"](ear, center, states, debug=debug)

    if debug:
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): return False

    conn.set('blinks', str(states['blinks']), ex=2)

    return True


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



# blink detection methods

def blink_svm(ear, center, states, debug=False):
    # get last frames data
    blinks = states['blinks']
    states['recent_ears'] = update_recent_ear(states['recent_ears'], ear)
    mean_ear = calc_mean_ear(states['recent_ears'])

    # normalize with train set mean value
    offset = mean_ear - 0.2744 if mean_ear > 0 else 0
    states['cons_ear'] = curr_ear_window(states['cons_ear'], ear, states['n'],
                                         offset)

    if debug:
        frame = states['frame']

        cv2.putText(frame, "Blinks: {}".format(blinks), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # predict using svm
    X = np.array(states['cons_ear']).reshape(1, -1)
    pred = states['svm'].predict(X)[0]
    if pred > 0 and pred != states['last_pred']:
        blinks += 1
        print(' * blink detected\n')

    states['blinks'] = blinks
    states['last_pred'] = pred


def blink_adaptive(ear, center, states, debug=False):
    # get last frames data
    blinks = states['blinks']

    # predict with adaptive thres
    states['recent_ears'] = update_recent_ear(states['recent_ears'], ear)
    states['recent_ears_long'] = update_recent_ear(states['recent_ears_long'],
                                                   ear)
    states['eye_centers'] = update_history(states['eye_centers'], center)

    mean_ear = calc_mean_ear(states['recent_ears'])
    mean_ear_long = calc_mean_ear(states['recent_ears_long'])

    mean_center = mean(states['eye_centers'])
    diff_center = eye_center_dist(center, mean_center)

    if (states.get('diffs', None) is None):
        states['diffs'] = [diff_center] * 10

    states['diffs'] = update_history(states['diffs'], diff_center)

    thres = states['eye_move_thres']

    # penalty on eye move stability, and ear stability
    penalty = mean(states['diffs']) / thres * 0.08 * (mean_ear / 0.45)

    ear_thres = mean_ear * states['sensi'] - penalty

    # debug in frame
    if debug:
        frame = states['frame']

        cv2.putText(frame, "Eye diff: {:.2f}".format(diff_center), (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 225, 0), 2)
        cv2.putText(frame, "Blinks: {}".format(blinks), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "EAR mean 1: {:.2f}".format(mean_ear), (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 225, 0), 2)
        cv2.putText(frame, "EAR mean 2: {:.2f}".format(mean_ear_long),
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 225, 0), 2)
        cv2.putText(frame, "EAR thres: {:.3f}".format(ear_thres), (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 225, 0), 2)
        cv2.putText(frame, "penalty: {:.3f}".format(penalty), (10, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 225, 0), 2)
        if diff_center >= thres:
            cv2.putText(frame, "Eye center moved.", (10, 500),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 225, 0), 2)

    # only detect when light changes not too fast,
    # and that eye position is stable
    if abs(mean_ear_long - mean_ear) >= 0.02:
        states['msg'] = 'Ignore unstable light change.'
        return

    if diff_center >= thres:
        states['msg'] = 'Ignore moving head.'
        return

    pred = 1 if ear > 0 and ear <= ear_thres else 0

    # ignore consecutive blink detection
    if pred > 0 and pred != states['last_pred']:
        blinks += 1
        print(' * blink detected\n')
    else:
        states['msg'] = ''

    states['blinks'] = blinks
    states['last_pred'] = pred


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
        return mean(recent)
    return -1


def update_history(history, c):
    history.append(c)
    history = history[1:]

    return history


def eye_center_dist(c1, c2):
    from scipy.spatial import distance as dist
    return dist.euclidean(c1, c2)


def mean(cons):
    return sum(cons) * 1.0 / len(cons)


def var(cons):
    return np.var(cons)
