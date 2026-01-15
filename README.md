# Object Tracking (YOLO + SORT)

YOLOv8과 SORT 알고리즘을 사용한 실시간 객체 추적 프로젝트.

## 설치

```bash
pip install opencv-python numpy scipy ultralytics
```

## 실행

```bash
python main.py
```

## 주요 구성

- **SORT.py**: SORT 추적 알고리즘 구현
- **main.py**: YOLOv8 검출 + SORT 추적 실행

## 동작 원리

1. YOLOv8이 객체 검출
2. Kalman Filter로 위치 예측
3. Hungarian Algorithm으로 매칭
4. 트랙 업데이트 및 ID 부여

## 참고

- [SORT 논문](https://arxiv.org/abs/1602.00763)
- [YOLOv8](https://docs.ultralytics.com/)
