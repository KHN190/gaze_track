from tasks import face_track, blink_track, VideoGet

from redis import Redis
from threading import Thread
from queue import Queue

import cv2
import time
import yaml

import numpy as np

with open('config.yml') as f:
    config = yaml.load(f.read(), Loader=yaml.CLoader)

from svm.utils import load_model

if __name__ == '__main__':
    cap = VideoGet(0)
    cap.start()

    conn = Redis()

    method = config['method']
    if method not in ['svm', 'adaptive']:
        raise Exception('Eye blink method must be svm or adaptive.')

    if method == 'svm':
        n = int(config['svm']['frames'])
        svm = load_model(config['svm']['model'])

        states = {
            'last_pred': -1.,
            'recent_ears': [-1.] * n * 2,
            'cons_ear': [-1.] * n,
            'blinks': 0,
            'n': n,
            'svm': svm,
        }

    elif method == 'adaptive':
        n = int(config['adaptive']['frames'])
        m = int(config['adaptive']['frames_long'])
        i = int(config['adaptive']['eye_move_frames'])
        t = int(config['adaptive']['eye_move_thres'])
        s = float(config['adaptive']['sensi'])

        states = {
            'last_pred': -1.,
            'recent_ears': [-1.] * n,
            'recent_ears_long': [-1.] * m,
            'eye_move_thres': t,
            'eye_centers': [np.array([.0, .0])] * i,
            'blinks': 0,
            'sensi': s,
        }

    else:
        raise Exception('Eye blink method not found.')

    print(f"### running method: {method}")
    print(f"\nstates: {states}\n")

    q = Queue()

    while True:
        ret, frame = cap.read()
        if ret:
            q.put(frame, timeout=1)

            Thread(target=face_track, args=(q, conn, states)).start()
            if not blink_track(frame, conn, states, method): 
                break

    cap.stop()
    cv2.destroyAllWindows()
