from tasks import face_track

from redis import Redis

import cv2
import time
import yaml

with open('config.yml') as f:
    config = yaml.load(f.read(), Loader=yaml.CLoader)

from svm.utils import load_model

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    conn = Redis()
    svm = load_model(config['svm']['model'])

    n = int(config['svm']['frames'])

    states = {
        'last_pred': -1.,
        'recent_ears': [-1.] * n * 2,
        'cons_ear': [-1.] * n,
        'blinks': 0,
        'svm': svm,
        'n': n,
    }

    while True:
        face_track(cap=cap, conn=conn, states=states)

    cap.release()
    cv2.destroyAllWindows()
