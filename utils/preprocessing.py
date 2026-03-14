import cv2
import numpy as np
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def extract_hand_landmarks(frame, hand_model):
    """
    Run MediaPipe Hands on a BGR frame.

    Returns:
        annotated_image: the frame with hand landmarks drawn on it
        landmarks: list of 63 floats (21 landmarks × 3 coords),
                   or zeros if no hand is detected
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    result = hand_model.process(rgb)
    rgb.flags.writeable = True
    annotated = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    landmarks = [0.0] * 63  # default: no hand detected

    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]

        # Draw landmarks on the frame
        mp_drawing.draw_landmarks(
            annotated, hand, mp_hands.HAND_CONNECTIONS
        )

        # Extract landmark coordinates
        row = []
        for lm in hand.landmark:
            row.extend([lm.x, lm.y, lm.z])

        if len(row) == 63:
            landmarks = row

    return annotated, landmarks


def process_frame(frame, hand_model):
    """
    Process a single video frame.

    Args:
        frame: BGR image (numpy array)
        hand_model: a MediaPipe Hands instance

    Returns:
        annotated_image: frame with landmarks drawn
        landmarks: list of 63 floats (hand keypoints)
    """
    return extract_hand_landmarks(frame, hand_model)