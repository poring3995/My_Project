import sys
import os
import cv2
import numpy as np
from collections import defaultdict
from ultralytics import YOLO
from datetime import datetime
import time

class EggInspection:
    def __init__(self):
        self.model_detection = YOLO(r"D:\backup before crash\egg inspection\all model weight\Weight_egg_detection_l_10ep__model\weights\best.pt", "v8")
        self.model_classification = YOLO(r"d:\backup before crash\egg inspection\all model weight\RY_egg_detectclassify_n_20ep__model\weights\best_n_20_ry.pt", "v8")
        
        self.classify_history = defaultdict(list)
        self.last_positions = {}
        self.missing_frames = defaultdict(int)

        self.output_text_path = r"D:\backup before crash\egg inspection\testing realtime save\result.txt"
        self.save_defect = r"D:\backup before crash\egg inspection\testing realtime save\crack"
        self.save_normal = r"D:\backup before crash\egg inspection\testing realtime save\normal"
        
        self.max_missing = 5  # If missing for 5 frames = out of frame
        self.defect_threshold = 1  # Minimum defect frames to be considered defect

        os.makedirs(self.save_defect, exist_ok=True)
        os.makedirs(self.save_normal, exist_ok=True)

    def crop_box(self, frame, xyxy):
        x1, y1, x2, y2 = map(int, xyxy)
        return frame[y1:y2, x1:x2]

    def summarize_and_save(self, egg_id):
        history = self.classify_history[egg_id]
        last_pos = self.last_positions.get(egg_id, (0, 0))
        decision = "defect" if history.count("defect") >= 1 else "normal"

        with open(self.output_text_path, 'a') as f:
            f.write(f"egg id : {egg_id} , answer : {decision} , position (x,y) : ({last_pos[0]},{last_pos[1]})\n")

        # Cleanup
        del self.classify_history[egg_id]
        del self.last_positions[egg_id]
        del self.missing_frames[egg_id]

    def run(self):
        cap = cv2.VideoCapture(r"C:\Users\Nattanon\MVS\Data\MV-CS050-10GC (DA2967405)\Video_20250520173047050.avi")

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            results_t = self.model_detection.track(frame, persist=True, tracker="bytetrack.yaml")
            if not results_t or results_t[0].boxes is None:
                continue

            frame_plot = results_t[0].plot()
            boxes = results_t[0].boxes
            ids = boxes.id.int().tolist() if boxes.id is not None else []

            # Track which eggs are seen this frame
            current_eggs = set()

            for i, box in enumerate(boxes.xyxy.cpu().numpy()):
                egg_id = ids[i]
                current_eggs.add(egg_id)
                self.last_positions[egg_id] = ((int(box[0]) + int(box[2])) // 2, (int(box[1]) + int(box[3])) // 2)

                crop = self.crop_box(frame, box)

                # Run classification model
                result_cls = self.model_classification.track(crop, persist=True, tracker="bytetrack.yaml")
                found_defect = result_cls[0].boxes is not None and len(result_cls[0].boxes) > 0

                # Save image
                img_name = f"{egg_id}_{datetime.now().strftime('%H%M%S%f')}.jpg"
                if found_defect:
                    out_img = result_cls[0].plot()
                    cv2.imwrite(os.path.join(self.save_defect, img_name), out_img)
                    self.classify_history[egg_id].append("defect")
                else:
                    cv2.imwrite(os.path.join(self.save_normal, img_name), crop)
                    self.classify_history[egg_id].append("normal")

            # Update missing frame count
            for egg_id in list(self.last_positions.keys()):
                if egg_id not in current_eggs:
                    self.missing_frames[egg_id] += 1
                    if self.missing_frames[egg_id] >= self.max_missing:
                        self.summarize_and_save(egg_id)
                else:
                    self.missing_frames[egg_id] = 0

       
if __name__ == "__main__":
    inspection = EggInspection()
    inspection.run()
