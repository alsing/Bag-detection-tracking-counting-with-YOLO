import cv2
import torch
from ultralytics import YOLO
import numpy as np
from sort import Sort
import os

CAP_FILE = os.getenv('CAP_FILE')
DATA_FOLDER = '/data'
OUT_FILE = 'output.mp4'

CONF = 0.7

cap = cv2.VideoCapture(os.path.join(DATA_FOLDER, CAP_FILE))

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

out = cv2.VideoWriter(os.path.join(DATA_FOLDER, OUT_FILE), cv2.VideoWriter_fourcc(*'DIVX'), cap.get(cv2.CAP_PROP_FPS), (frame_width,frame_height))

model = YOLO(os.path.join(DATA_FOLDER, 'best.pt')).to(device)

global_count = set()

tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

check_line=[190, 180, 270, 210]

def draw_boxes(frame, bbox, ids=None, label='bag'):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = map(int, box)
        box_id = int(ids[i]) if ids is not None else 0

        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(frame, (cx, cy), 3, (0, 0, 255), cv2.FILLED)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, f'{label}', (x1, y1 - 5), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)

        if check_line[0] < cx < check_line[2] and check_line[1] < cy < check_line[3]:
            if box_id not in global_count:
                global_count.add(box_id)
                cv2.line(frame, (check_line[0], check_line[1]), (check_line[2], check_line[3]), (0, 255, 0), 3)
        
        cv2.putText(frame, f'bags count: {len(global_count)}', (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 3)
    return frame

while True:
    ret, frame = cap.read()
    if ret:
        detections = np.zeros((0, 6))
        result = list(model.predict(frame, conf=CONF))[0]
        bbox_xyxys = result.boxes.xyxy.tolist()
        confidences = result.boxes.conf.tolist()
        labels = map(int, result.boxes.cls)
        for bbox_xyxy, confidence, label in zip(bbox_xyxys, confidences, labels):
            bbox = np.array(bbox_xyxy)
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            cur_array = np.array([x1, y1, x2, y2, confidence, label])
            detections = np.vstack((detections, cur_array))
        
        cv2.line(frame, (check_line[0], check_line[1]), (check_line[2], check_line[3]), (255, 0, 0), 3)
        tracker_dets = tracker.update(detections)
        if len(tracker_dets):
            bbox_xyxy = tracker_dets[:, :4]
            ids = tracker_dets[:, 4]
            draw_boxes(frame, bbox_xyxy, ids)
        
        out.write(frame)
        cv2.waitKey(1)
    else:
        break

out.release()
cap.release()
