from tasks import gaze_track

from redis import Redis

if __name__ == '__main__':
    conn = Redis()

    while True:
        try:
            gaze_track(conn=conn)
        
        except IndexError:
            pass

        except Exception as e:
            print(e)
            raise e

