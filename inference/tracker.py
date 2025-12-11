# CSRT/ BYTETRACK / DEEPSORT Tracker Inference Script

import cv2
import numpy as np
from ultralytics import YOLO

class Tracker:
    def __init__(self, model_path, tracker_config="bytetrack.yaml"):
        """
        Ultralytics tabanlı ByteTrack Wrapper.
        """
        self.model = YOLO(model_path)
        self.tracker_config = tracker_config
        print(f"[Tracker] ByteTrack yüklendi. Config: {tracker_config}")

    def update(self, frame):
        """
        Frame üzerinde hem tespit hem takip yapar.
        Döndürür: tracks [[x1, y1, x2, y2, track_id, score, class_id], ...]
        """
        # persist=True -> Takip belleğini korur (ID'ler değişmez)
        results = self.model.track(
            source=frame,
            persist=True,
            tracker=self.tracker_config,
            verbose=False,
            device=1 # GPU
        )
        
        tracks = []
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            clss = results[0].boxes.cls.cpu().numpy()

            for box, track_id, conf, cls in zip(boxes, track_ids, confs, clss):
                x1, y1, x2, y2 = box.astype(int)
                # Format: [x1, y1, x2, y2, track_id, score, class_id]
                tracks.append([x1, y1, x2, y2, int(track_id), conf, int(cls)])
        
        return tracks