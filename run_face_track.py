from tasks import face_track

from redis import Redis

import cv2
import time

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    conn = Redis()
    last = time.time()

    blinks = 0
    while True:
        now = time.time()
        if now - last < 500: # 200ms
            time.sleep(0.1)
        last = now

        blinks = face_track(cap=cap, conn=conn, blinks=blinks)

    cap.release()
    cv2.destroyAllWindows()
