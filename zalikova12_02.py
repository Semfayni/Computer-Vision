import cv2
import os
import csv
import time
import subprocess
from ultralytics import YOLO

YOUTUBE_URL = "https://www.youtube.com/watch?v=Lxqcg1qt0XU"

def get_stream_url(url):
    command = f'yt-dlp -f best -g "{url}"'
    return subprocess.check_output(command, shell=True).decode().strip()

stream_url = get_stream_url(YOUTUBE_URL)

PROJECT_DIR = os.path.dirname(__file__)
OUTPUT_DIR = os.path.join(PROJECT_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

CSV_PATH = os.path.join(OUTPUT_DIR, "speed_results.csv")

MODEL_PATH = "yolov8n.pt"
CONF_THRESH = 0.5
TRACKER = "bytetrack.yaml"

VEHICLE_CLASSES = ["car", "truck", "bus", "motorcycle"]

LINE1_P1 = (950,450)
LINE1_P2 = (1620,510)

LINE2_P1 = (560,560)
LINE2_P2 = (1400,680)

REAL_DISTANCE_METERS = 10

model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(stream_url)

object_prev_centers = {}
cross_data = {}
calculated_ids = set()
object_speeds = {}
total_vehicle_ids = set()

with open(CSV_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Vehicle_ID", "Speed_km_h"])

def ccw(A, B, C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

def intersect(A, B, C, D):
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(frame, conf=CONF_THRESH,
                          tracker=TRACKER, persist=True)
    r = results[0]

    cv2.line(frame, LINE1_P1, LINE1_P2, (0,255,255), 3)
    cv2.line(frame, LINE2_P1, LINE2_P2, (0,0,255), 3)

    if r.boxes is not None and len(r.boxes) > 0:

        boxes = r.boxes
        xyxy = boxes.xyxy.cpu().numpy()
        cls = boxes.cls.cpu().numpy()
        track_id = boxes.id.cpu().numpy() if boxes.id is not None else None

        for i in range(len(xyxy)):

            class_id = int(cls[i])
            class_name = model.names[class_id]

            if class_name not in VEHICLE_CLASSES:
                continue

            x1, y1, x2, y2 = xyxy[i].astype(int)
            tid = int(track_id[i]) if track_id is not None else -1

            center = ((x1+x2)//2, (y1+y2)//2)

            if tid in object_prev_centers:
                prev_center = object_prev_centers[tid]

                if tid not in cross_data:

                    if intersect(prev_center, center, LINE1_P1, LINE1_P2):
                        cross_data[tid] = ("L1", time.time())
                        total_vehicle_ids.add(tid)

                    elif intersect(prev_center, center, LINE2_P1, LINE2_P2):
                        cross_data[tid] = ("L2", time.time())
                        total_vehicle_ids.add(tid)

                if tid in cross_data and tid not in calculated_ids:

                    first_line, t1 = cross_data[tid]

                    crossed = False

                    if first_line == "L1" and intersect(prev_center, center, LINE2_P1, LINE2_P2):
                        crossed = True

                    elif first_line == "L2" and intersect(prev_center, center, LINE1_P1, LINE1_P2):
                        crossed = True

                    if crossed:
                        t2 = time.time()
                        delta_time = t2 - t1

                        if delta_time > 0:

                            speed_mps = REAL_DISTANCE_METERS / delta_time
                            speed_kmh = speed_mps * 3.6

                            calculated_ids.add(tid)
                            object_speeds[tid] = speed_kmh

                            with open(CSV_PATH, "a", newline="") as f:
                                writer = csv.writer(f)
                                writer.writerow([f"Vehicle {tid}", round(speed_kmh,2)])

            object_prev_centers[tid] = center

            speed_text = ""
            if tid in object_speeds:
                speed_text = f"{object_speeds[tid]:.1f} km/h"

            label = f"{class_name} ID {tid} {speed_text}"

            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(frame, label, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

    total_vehicles = len(total_vehicle_ids)

    cv2.putText(frame,
                f"TOTAL VEHICLES: {total_vehicles}",
                (40, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 255),
                3)

    cv2.imshow("Speed Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()