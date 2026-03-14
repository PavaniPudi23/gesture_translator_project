import os
import cv2
import numpy as np
import urllib.request

import mediapipe as mp

# ---------- Detect which MediaPipe API is available ----------
_USE_LEGACY_API = hasattr(mp, "solutions")

if _USE_LEGACY_API:
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
else:
    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision as mp_vision

# Hand connections for drawing landmarks (used with new API)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),       # Index finger
    (5, 9), (9, 10), (10, 11), (11, 12),   # Middle finger
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring finger
    (0, 17), (13, 17), (17, 18), (18, 19), (19, 20),  # Pinky + palm
]

# Task model for new API
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)
_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models",
    "hand_landmarker.task",
)


def _ensure_task_model():
    """Download the hand landmarker .task model if not present."""
    if not os.path.exists(_MODEL_PATH):
        print("[preprocessing] Downloading hand landmarker model...")
        os.makedirs(os.path.dirname(_MODEL_PATH), exist_ok=True)
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
        print("[preprocessing] Download complete.")


def create_hand_detector():
    """
    Create a hand detection model.
    Automatically picks the correct API for the installed MediaPipe version.

    - mediapipe <= 0.10.14: uses mp.solutions.hands (legacy)
    - mediapipe >= 0.10.30: uses mp.tasks.vision.HandLandmarker (new)
    """
    if _USE_LEGACY_API:
        return mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.4,
            min_tracking_confidence=0.4,
        )
    else:
        _ensure_task_model()
        base_options = mp_tasks.BaseOptions(model_asset_path=_MODEL_PATH)
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.4,
            min_hand_presence_confidence=0.4,
        )
        return mp_vision.HandLandmarker.create_from_options(options)


def _draw_landmarks_manual(image, hand_landmarks):
    """Draw hand landmarks on the image using OpenCV (for new API)."""
    h, w, _ = image.shape
    points = []
    for lm in hand_landmarks:
        px, py = int(lm.x * w), int(lm.y * h)
        points.append((px, py))
        cv2.circle(image, (px, py), 4, (0, 255, 0), -1)

    for start, end in HAND_CONNECTIONS:
        if start < len(points) and end < len(points):
            cv2.line(image, points[start], points[end], (0, 200, 0), 2)

    return image


def extract_hand_landmarks(frame, detector):
    """
    Run hand landmark detection on a BGR frame.

    Works with both legacy (mp.solutions) and new (mp.tasks) MediaPipe APIs.

    Returns:
        annotated_image: the frame with hand landmarks drawn on it
        landmarks: list of 63 floats (21 landmarks x 3 coords),
                   or zeros if no hand is detected
    """
    landmarks = [0.0] * 63  # default: no hand detected

    if _USE_LEGACY_API:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = detector.process(rgb)
        rgb.flags.writeable = True
        annotated = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(
                annotated, hand, mp_hands.HAND_CONNECTIONS
            )
            row = []
            for lm in hand.landmark:
                row.extend([lm.x, lm.y, lm.z])
            if len(row) == 63:
                landmarks = row
    else:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect(mp_image)

        annotated = frame.copy()

        if result.hand_landmarks:
            hand = result.hand_landmarks[0]
            annotated = _draw_landmarks_manual(annotated, hand)
            row = []
            for lm in hand:
                row.extend([lm.x, lm.y, lm.z])
            if len(row) == 63:
                landmarks = row

    return annotated, landmarks


def process_frame(frame, detector):
    """
    Process a single video frame.

    Args:
        frame: BGR image (numpy array)
        detector: a hand detector (legacy Hands or new HandLandmarker)

    Returns:
        annotated_image: frame with landmarks drawn
        landmarks: list of 63 floats (hand keypoints)
    """
    return extract_hand_landmarks(frame, detector)