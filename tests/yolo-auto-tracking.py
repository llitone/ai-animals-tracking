from collections import defaultdict

import cv2
import numpy as np

from ultralytics import YOLO

model = YOLO("yolov8n.pt")

video_path = "12345.mp4"
cap = cv2.VideoCapture(video_path)

track_history = defaultdict(lambda: [])

while cap.isOpened():
    success, frame = cap.read()

    if success:
        results = model.track(frame, persist=True, conf=0.3, iou=0.5, verbose=False)
        if results[0].boxes.id is None:
            cv2.imshow("YOLOv8 Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        annotated_frame = results[0].plot()

        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            track = track_history[track_id]
            track.append((float(x), float(y)))
            if len(track) > 90:
                track.pop(0)

            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated_frame, [points], isClosed=False, color=(0, 230, 0), thickness=4)

        cv2.imshow("YOLOv8 Tracking", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()