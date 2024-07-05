import cv2
import numpy as np
from flask import Flask, make_response, jsonify, request

from tracking import Tracker, TrackedObject

app = Flask(__name__)
tracker = Tracker("yolov8n.pt")
tracker.SHOW_PREDS = False
tracker.SAVE = False


@app.route("/model/api/v1.0/track", methods=["POST"])
async def predict():
    save = request.args.get('save_frame', default=False, type=lambda x: x.lower() == "true")
    files = request.files

    img = np.frombuffer(files["files"].stream.read(), dtype=np.uint8)
    img = cv2.imdecode(img, flags=1)

    results: list[TrackedObject] = tracker.track_next_frame(img)

    results = list(map(lambda x: x.json(send_frame=save), results))

    return make_response(jsonify(results), 200)


if __name__ == "__main__":
    app.run(debug=False)
