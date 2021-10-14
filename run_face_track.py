from tasks import face_track

from redis import Redis

import cv2
import time
import yaml

import numpy as np

with open('config.yml') as f:
    config = yaml.load(f.read(), Loader=yaml.CLoader)

from svm.utils import load_model

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
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

    while True:
        face_track(cap=cap, conn=conn, states=states, method=method)

    cap.release()
    cv2.destroyAllWindows()
