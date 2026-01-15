import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import List


def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


class KalmanTracker:
    count = 0

    def __init__(self, bbox: np.ndarray):
        # 고유 ID 할당
        self.id = KalmanTracker.count
        KalmanTracker.count += 1

        # cv2.KalmanFilter 초기화 (상태 7차원, 관측 4차원)
        self.kf = cv2.KalmanFilter(7, 4)

        # 상태 전이 행렬 F (등속 운동 모델)
        self.kf.transitionMatrix = np.array(
            [
                [1, 0, 0, 0, 1, 0, 0],  # cx' = cx + vx
                [0, 1, 0, 0, 0, 1, 0],  # cy' = cy + vy
                [0, 0, 1, 0, 0, 0, 1],  # area' = area + va
                [0, 0, 0, 1, 0, 0, 0],  # ratio' = ratio
                [0, 0, 0, 0, 1, 0, 0],  # vx' = vx
                [0, 0, 0, 0, 0, 1, 0],  # vy' = vy
                [0, 0, 0, 0, 0, 0, 1],  # va' = va
            ],
            dtype=np.float32,
        )

        # 관측 행렬 H
        self.kf.measurementMatrix = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0],
            ],
            dtype=np.float32,
        )

        # 프로세스 노이즈 공분산 Q
        self.kf.processNoiseCov = np.eye(7, dtype=np.float32) * 0.01
        self.kf.processNoiseCov[4, 4] = 0.01
        self.kf.processNoiseCov[5, 5] = 0.01
        self.kf.processNoiseCov[6, 6] = 0.0001

        # 관측 노이즈 공분산 R
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 1.0

        # 초기 오차 공분산 P
        self.kf.errorCovPost = np.eye(7, dtype=np.float32) * 10.0

        # 초기 상태 설정
        init_state = self._bbox_to_state(bbox)
        self.kf.statePost = np.array(
            [
                [init_state[0]],  # cx
                [init_state[1]],  # cy
                [init_state[2]],  # area
                [init_state[3]],  # ratio
                [0],  # vx
                [0],  # vy
                [0],  # va
            ],
            dtype=np.float32,
        )

        self.time_since_update = 0
        self.hits = 1
        self.hit_streak = 1
        self.age = 0

    def _bbox_to_state(self, bbox: np.ndarray) -> np.ndarray:
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        return np.array(
            [
                bbox[0] + w / 2,  # cx
                bbox[1] + h / 2,  # cy
                w * h,  # area
                w / (h + 1e-6),  # ratio
            ],
            dtype=np.float32,
        )

    def _state_to_bbox(self, state: np.ndarray) -> np.ndarray:
        cx, cy, area, ratio = state.flatten()[:4]
        w = np.sqrt(max(area * ratio, 1))
        h = area / (w + 1e-6)
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])

    def predict(self) -> np.ndarray:
        if self.kf.statePost[2] + self.kf.statePost[6] <= 0:
            self.kf.statePost[6] = 0

        predicted = self.kf.predict()

        if predicted[2] <= 0:
            predicted[2] = 1.0

        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1

        return self._state_to_bbox(predicted)

    def update(self, bbox: np.ndarray):
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1

        measurement = self._bbox_to_state(bbox).reshape(4, 1)
        self.kf.correct(measurement)

    def get_state(self) -> np.ndarray:
        return self._state_to_bbox(self.kf.statePost)


class Sort:
    def __init__(self, max_age: int = 3, min_hits: int = 3, iou_threshold: float = 0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers: List[KalmanTracker] = []
        self.frame_count = 0

    def update(self, detections: np.ndarray) -> np.ndarray:
        # 1. 프레임 카운트 증가
        self.frame_count += 1

        predicted_boxes = []
        valid_trackers = []

        # 2. Kalman filter 예측
        for tracker in self.trackers:
            pred = tracker.predict()
            if not np.any(np.isnan(pred)):
                predicted_boxes.append(pred)
                valid_trackers.append(tracker)

        self.trackers = valid_trackers

        if len(detections) == 0:
            return self._get_output()

        det_boxes = detections[:, :4]

        if len(self.trackers) == 0:
            for det in det_boxes:
                self.trackers.append(KalmanTracker(det))
            return self._get_output()

        # 3. IoU 계산 + Hungraian Algorithm (track <-> detection 매칭)
        predicted_boxes = np.array(predicted_boxes)
        matched, unmatched_dets, _ = self._match(det_boxes, predicted_boxes)

        # 4. 매칭된 track update
        for det_idx, trk_idx in matched:
            self.trackers[trk_idx].update(det_boxes[det_idx])
        
        # 5. 매칭되지 않은 detection -> 새 track 생성
        for det_idx in unmatched_dets:
            self.trackers.append(KalmanTracker(det_boxes[det_idx]))

        # 6. 오래 update되지 않은 track 제거
        self.trackers = [
            t for t in self.trackers if t.time_since_update <= self.max_age
        ]

        return self._get_output()

    def _match(self, dets: np.ndarray, trks: np.ndarray):
        cost_matrix = np.zeros((len(dets), len(trks)))
        for d in range(len(dets)):
            for t in range(len(trks)):
                cost_matrix[d, t] = 1 - iou(dets[d], trks[t])

        det_idx, trk_idx = linear_sum_assignment(cost_matrix)

        matched = []
        unmatched_dets = list(range(len(dets)))
        unmatched_trks = list(range(len(trks)))

        for d, t in zip(det_idx, trk_idx):
            if 1 - cost_matrix[d, t] >= self.iou_threshold:
                matched.append((d, t))
                unmatched_dets.remove(d)
                unmatched_trks.remove(t)

        return matched, unmatched_dets, unmatched_trks

    def _get_output(self) -> np.ndarray:
        output = []
        for t in self.trackers:
            if (
                t.hit_streak >= self.min_hits or self.frame_count <= self.min_hits
            ) and t.time_since_update == 0:
                output.append([*t.get_state(), t.id])

        return np.array(output) if output else np.empty((0, 5))
