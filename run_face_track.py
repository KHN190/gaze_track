from tasks import face_track

from redis import Redis

import cv2
import time

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    conn = Redis()

    while True:
        face_track(cap=cap, conn=conn)

    cap.release()
    cv2.destroyAllWindows()
