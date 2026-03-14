import os
import numpy as np
import joblib

from utils.constants import SEQUENCE_LENGTH, CONFIDENCE_THRESHOLD, FEATURES_PER_FRAME


class GestureDetector:
    """Loads a scikit-learn .pkl model and its LabelEncoder for prediction."""

    def __init__(self, mode="word"):
        self.model = None
        self.label_encoder = None
        self.sequence_length = SEQUENCE_LENGTH
        self.confidence_threshold = CONFIDENCE_THRESHOLD

        # Pick the correct model/encoder pair based on mode
        if mode == "phrase":
            model_file = "phrase_model.pkl"
            encoder_file = "phrase_label_encoder.pkl"
        else:  # default to "word"
            model_file = "word_model.pkl"
            encoder_file = "word_label_encoder.pkl"

        model_path = os.path.join("models", model_file)
        encoder_path = os.path.join("models", encoder_file)

        try:
            if os.path.exists(model_path) and os.path.exists(encoder_path):
                self.model = joblib.load(model_path)
                self.label_encoder = joblib.load(encoder_path)
                print(f"[GestureDetector] Loaded {model_file} successfully.")
            else:
                print(f"[GestureDetector] Model files not found: {model_file}")
        except Exception as e:
            print(f"[GestureDetector] Error loading model: {e}")
            self.model = None
            self.label_encoder = None

    def predict(self, sequence):
        """
        Predict gesture from a sequence of hand landmark frames.

        Args:
            sequence: list of landmark vectors (each 63-dim).
                      Must have at least `self.sequence_length` frames.

        Returns:
            (label, confidence) or (None, confidence)
        """
        if self.model is None or self.label_encoder is None:
            return None, 0.0

        if len(sequence) < self.sequence_length:
            return None, 0.0

        # Take the last `sequence_length` frames and flatten into a single vector
        seq = sequence[-self.sequence_length:]
        input_data = np.array(seq, dtype=np.float32).flatten().reshape(1, -1)

        expected_features = self.sequence_length * FEATURES_PER_FRAME
        if input_data.shape[1] != expected_features:
            return None, 0.0

        try:
            pred = self.model.predict(input_data)[0]
            label = self.label_encoder.inverse_transform([pred])[0]

            # Get confidence from predict_proba if available
            if hasattr(self.model, "predict_proba"):
                proba = self.model.predict_proba(input_data)[0]
                confidence = float(max(proba))
            else:
                confidence = 1.0

            if confidence >= self.confidence_threshold:
                return label, confidence

            return None, confidence

        except Exception:
            return None, 0.0