# Gaze Capture Server

Forward camera to OpenCV + Gaze Capture neural network, return eye track data in a local server. The initial purpose is to use it along with Unity.

Code referenced from

* [GazeCapture](https://github.com/CSAILVision/GazeCapture)
* [Presence](https://github.com/oveddan/presence)

## Dependency

* PyCaffe

Others install with:

```bash
pip3 install -r requirements.txt

```

## Start Tasks

```bash
# start redis
sudo service redis start

# start camera & face detection
python3 run_face_track.py &> face.log

# start gaze detection
python3 run_gaze_track.py &> gaze.log

# start server, or directly pull from redis
flask run
```

## API

* '/' for timestamp
* '/gaze' for eye coords
