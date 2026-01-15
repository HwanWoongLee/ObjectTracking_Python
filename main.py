from ultralytics import YOLO
import cv2
from SORT import Sort


def main():
    model = YOLO("yolov8n.pt")
    tracker = Sort(max_age=3, min_hits=3, iou_threshold=0.3)

    cap = cv2.VideoCapture("./tracking_video.mp4")

    if not cap.isOpened():
        print("video not found")
        return

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        results = model(frame)

        for r in results:
            boxes = r.boxes
            xyxy = boxes.xyxy.cpu().numpy()

            tracks = tracker.update(xyxy)

            for bbox in xyxy:
                x1 = int(bbox[0])
                y1 = int(bbox[1])
                x2 = int(bbox[2])
                y2 = int(bbox[3])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

            for t in tracks:
                x1 = int(t[0])
                y1 = int(t[1])
                x2 = int(t[2])
                y2 = int(t[3])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
                cv2.putText(frame, "id:{}".format(t[4]), (x1, y1), 0, 1, (0, 0, 255), 1)

        cv2.imshow("tracking result", frame)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
