from flask import Flask

from redis import Redis


app = Flask('Gaze Capture Server')
conn = Redis()

@app.route("/")
def ping():

    rep = conn.get("updated")
    
    if rep is not None:
        return rep
    return ''

# get last frame gaze coords
# 
# fetch data from queue
# 
@app.route("/gaze")
def gaze_track():

    rep = conn.get("gaze_coord")

    if rep is not None:
        return rep
    return ''
