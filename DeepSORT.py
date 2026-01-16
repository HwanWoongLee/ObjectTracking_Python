import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import List, Tuple
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18


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


class FeatureExtractor:
    """Deep Learning 기반 appearance feature 추출기"""

    def __init__(self, model_path=None, use_cuda=True):
        self.device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")

        # ResNet18 백본 사용 (경량화)
        self.model = resnet18(pretrained=True)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 128)  # 128차원 feature

        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))

        self.model.to(self.device)
        self.model.eval()

        # 이미지 전처리
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 64)),  # DeepSORT 표준 크기
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def extract_features(self, image: np.ndarray, bboxes: np.ndarray) -> np.ndarray:
        """
        이미지에서 bounding box들의 appearance feature 추출

        Args:
            image: BGR 이미지 (H, W, 3)
            bboxes: bounding boxes (N, 4) [x1, y1, x2, y2]

        Returns:
            features: (N, 128) 정규화된 feature vectors
        """
        if len(bboxes) == 0:
            return np.empty((0, 128))

        features = []
        for bbox in bboxes:
            x1, y1, x2, y2 = map(int, bbox)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)

            if x2 <= x1 or y2 <= y1:
                features.append(np.zeros(128))
                continue

            # crop & transform
            patch = image[y1:y2, x1:x2]
            patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
            patch_tensor = self.transform(patch).unsqueeze(0).to(self.device)

            # feature extraction
            with torch.no_grad():
                feature = self.model(patch_tensor).cpu().numpy().flatten()
                # L2 normalization
                feature = feature / (np.linalg.norm(feature) + 1e-6)
                features.append(feature)

        return np.array(features)


class KalmanTrackerDeep:
    """DeepSORT용 Kalman Filter 기반 Tracker (appearance feature 포함)"""
    count = 0

    def __init__(self, bbox: np.ndarray, feature: np.ndarray):
        # 고유 ID 할당
        self.id = KalmanTrackerDeep.count
        KalmanTrackerDeep.count += 1

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

        # appearance feature gallery (최근 100개 유지)
        self.features = [feature]
        self.max_features = 100

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

    def update(self, bbox: np.ndarray, feature: np.ndarray):
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1

        measurement = self._bbox_to_state(bbox).reshape(4, 1)
        self.kf.correct(measurement)

        # feature gallery 업데이트
        self.features.append(feature)
        if len(self.features) > self.max_features:
            self.features.pop(0)

    def get_state(self) -> np.ndarray:
        return self._state_to_bbox(self.kf.statePost)

    def get_feature(self) -> np.ndarray:
        """최근 feature들의 평균 반환 (smooth feature)"""
        return np.mean(self.features, axis=0)


class DeepSORT:
    """DeepSORT: SORT + Deep Appearance Features"""

    def __init__(
        self,
        max_age: int = 70,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        max_cosine_distance: float = 0.2,
        nn_budget: int = 100,
        use_cuda: bool = True,
        model_path: str = None
    ):
        """
        Args:
            max_age: tracker가 매칭 안될 때 최대 유지 프레임 수
            min_hits: 출력하기 위한 최소 히트 수
            iou_threshold: IoU 매칭 임계값
            max_cosine_distance: appearance feature 코사인 거리 임계값
            nn_budget: feature gallery 최대 크기
            use_cuda: GPU 사용 여부
            model_path: 사전 학습된 feature extractor 모델 경로
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.max_cosine_distance = max_cosine_distance
        self.nn_budget = nn_budget

        self.trackers: List[KalmanTrackerDeep] = []
        self.frame_count = 0

        # Feature Extractor 초기화
        self.feature_extractor = FeatureExtractor(model_path, use_cuda)

    def update(self, detections: np.ndarray, image: np.ndarray) -> np.ndarray:
        """
        Args:
            detections: (N, 4+) [x1, y1, x2, y2, ...] detection 결과
            image: (H, W, 3) BGR 이미지 (feature 추출용)

        Returns:
            tracks: (M, 5) [x1, y1, x2, y2, track_id]
        """
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

        # 3. Appearance feature 추출
        det_features = self.feature_extractor.extract_features(image, det_boxes)

        if len(self.trackers) == 0:
            for det, feat in zip(det_boxes, det_features):
                self.trackers.append(KalmanTrackerDeep(det, feat))
            return self._get_output()

        # 4. Cascaded Matching (DeepSORT의 핵심)
        predicted_boxes = np.array(predicted_boxes)
        matched, unmatched_dets, unmatched_trks = self._cascaded_match(
            det_boxes, det_features, predicted_boxes
        )

        # 5. 매칭된 track update
        for det_idx, trk_idx in matched:
            self.trackers[trk_idx].update(det_boxes[det_idx], det_features[det_idx])

        # 6. 매칭되지 않은 detection -> 새 track 생성
        for det_idx in unmatched_dets:
            self.trackers.append(KalmanTrackerDeep(det_boxes[det_idx], det_features[det_idx]))

        # 7. 오래 update되지 않은 track 제거
        self.trackers = [
            t for t in self.trackers if t.time_since_update <= self.max_age
        ]

        return self._get_output()

    def _cascaded_match(self, dets: np.ndarray, det_features: np.ndarray, trks: np.ndarray):
        matched = []
        unmatched_dets = list(range(len(dets)))
        unmatched_trks = list(range(len(trks)))

        # Cascade by age (최근 매칭된 tracker 우선)
        max_cascade_age = 30
        for age in range(max_cascade_age):
            if len(unmatched_dets) == 0:
                break

            # 현재 age에 해당하는 tracker 선택
            age_trackers = [
                i for i in unmatched_trks
                if self.trackers[i].time_since_update == age + 1
            ]

            if len(age_trackers) == 0:
                continue

            # Cost matrix 계산 (appearance + motion)
            cost_matrix = self._compute_cost_matrix(
                dets, det_features, trks,
                unmatched_dets, age_trackers
            )

            # Hungarian Algorithm
            if cost_matrix.size > 0:
                det_indices, trk_indices = linear_sum_assignment(cost_matrix)

                # 먼저 매칭 수집
                matches_in_cascade = []
                for d_idx, t_idx in zip(det_indices, trk_indices):
                    det_id = unmatched_dets[d_idx]
                    trk_id = age_trackers[t_idx]

                    # cost가 임계값 이하일 때만 매칭
                    if cost_matrix[d_idx, t_idx] < 1.0 - self.iou_threshold:
                        matched.append((det_id, trk_id))
                        matches_in_cascade.append((det_id, trk_id))

                # 매칭된 것들 제거
                for det_id, trk_id in matches_in_cascade:
                    unmatched_dets.remove(det_id)
                    unmatched_trks.remove(trk_id)

        # IOU matching for unmatched tracks (time_since_update가 큰 경우)
        if len(unmatched_dets) > 0 and len(unmatched_trks) > 0:
            iou_cost = np.zeros((len(unmatched_dets), len(unmatched_trks)))
            for i, d_idx in enumerate(unmatched_dets):
                for j, t_idx in enumerate(unmatched_trks):
                    iou_cost[i, j] = 1 - iou(dets[d_idx], trks[t_idx])

            det_indices, trk_indices = linear_sum_assignment(iou_cost)

            for d_idx, t_idx in zip(det_indices, trk_indices):
                det_id = unmatched_dets[d_idx]
                trk_id = unmatched_trks[t_idx]

                if 1 - iou_cost[d_idx, t_idx] >= self.iou_threshold:
                    matched.append((det_id, trk_id))

        # 최종 unmatched 갱신
        matched_dets = [m[0] for m in matched]
        matched_trks = [m[1] for m in matched]
        unmatched_dets = [d for d in range(len(dets)) if d not in matched_dets]
        unmatched_trks = [t for t in range(len(trks)) if t not in matched_trks]

        return matched, unmatched_dets, unmatched_trks

    def _compute_cost_matrix(
        self,
        dets: np.ndarray,
        det_features: np.ndarray,
        trks: np.ndarray,
        det_indices: List[int],
        trk_indices: List[int]
    ) -> np.ndarray:
        """
        Appearance (cosine distance) + Motion (IoU) 기반 cost matrix
        """
        cost_matrix = np.zeros((len(det_indices), len(trk_indices)))

        for i, d_idx in enumerate(det_indices):
            for j, t_idx in enumerate(trk_indices):
                # Appearance cost (cosine distance)
                trk_feature = self.trackers[t_idx].get_feature()
                det_feature = det_features[d_idx]
                cosine_dist = 1 - np.dot(trk_feature, det_feature)

                # Motion cost (1 - IoU)
                iou_dist = 1 - iou(dets[d_idx], trks[t_idx])

                # Gating: appearance similarity로 필터링
                if cosine_dist > self.max_cosine_distance:
                    cost_matrix[i, j] = 1e5  # 무한대 cost (매칭 불가)
                else:
                    # Combined cost (appearance 우선, motion 보조)
                    cost_matrix[i, j] = 0.7 * cosine_dist + 0.3 * iou_dist

        return cost_matrix

    def _get_output(self) -> np.ndarray:
        output = []
        for t in self.trackers:
            if (
                t.hit_streak >= self.min_hits or self.frame_count <= self.min_hits
            ) and t.time_since_update == 0:
                output.append([*t.get_state(), t.id])

        return np.array(output) if output else np.empty((0, 5))
