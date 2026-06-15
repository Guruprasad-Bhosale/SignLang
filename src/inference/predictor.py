import os
import time
import joblib
import numpy as np
from collections import deque, Counter
from src.utils.logger import logger
from src.utils.features import extract_features

class GesturePredictor:
    """
    Handles model loading, feature extraction, real-time prediction,
    and temporal smoothing with majority voting.
    """
    def __init__(self, model_path="models/best_model.joblib", window_size=10, confidence_threshold=0.85):
        self.model_path = model_path
        self.window_size = window_size
        self.confidence_threshold = confidence_threshold
        
        self.model = None
        self.label_encoder = None
        self.model_name = None
        
        # Sliding window for predictions: stores tuples of (letter, confidence)
        self.history = deque(maxlen=window_size)
        
        self.load_model()

    def load_model(self):
        """Loads the saved classifier package from disk."""
        if not os.path.exists(self.model_path):
            logger.warning(f"Model file not found at: {self.model_path}. Predictor will run in dummy/uninitialized mode.")
            return False
            
        try:
            payload = joblib.load(self.model_path)
            self.model = payload["model"]
            self.label_encoder = payload["label_encoder"]
            self.model_name = payload["model_name"]
            logger.info(f"Successfully loaded model '{self.model_name}' from: {self.model_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load trained model: {str(e)}")
            return False

    def predict(self, landmarks, handedness="Right"):
        """
        Predicts the letter sign from landmarks. Applies scale-rotation invariants,
        inference, and temporal smoothing.
        
        Parameters:
        -----------
        landmarks : list or np.ndarray
            The 21 hand landmarks.
        handedness : str
            "Left" or "Right"
            
        Returns:
        --------
        dict : Contains:
            - "smoothed_prediction": letter or "Unknown Gesture"
            - "smoothed_confidence": float
            - "raw_prediction": letter
            - "raw_confidence": float
            - "inference_time_ms": float
        """
        t0 = time.time()
        
        if self.model is None or self.label_encoder is None:
            # Fallback when model is not yet trained/loaded
            return {
                "smoothed_prediction": "Model Missing",
                "smoothed_confidence": 0.0,
                "raw_prediction": "None",
                "raw_confidence": 0.0,
                "inference_time_ms": 0.0
            }
            
        try:
            # 1. Feature Engineering
            feats = extract_features(landmarks, handedness=handedness)
            feats_batch = feats.reshape(1, -1)
            
            # 2. Classifier Prediction
            raw_idx = self.model.predict(feats_batch)[0]
            raw_letter = self.label_encoder.inverse_transform([raw_idx])[0]
            
            # Get prediction probabilities
            if hasattr(self.model, "predict_proba"):
                probs = self.model.predict_proba(feats_batch)[0]
                raw_conf = float(probs[raw_idx])
            else:
                raw_conf = 1.0  # fallback if model doesn't support probability
                
            # If raw confidence is below threshold, treat it as unknown for this frame
            frame_pred = raw_letter if raw_conf >= self.confidence_threshold else "?"
            frame_conf = raw_conf if raw_conf >= self.confidence_threshold else 0.0
            
            # 3. Add to sliding window history
            self.history.append((frame_pred, frame_conf))
            
            # 4. Majority Voting over window
            votes = [item[0] for item in self.history]
            vote_counts = Counter(votes)
            majority_char, count = vote_counts.most_common(1)[0]
            
            # Calculate average confidence for the majority voted character in the window
            majority_confs = [item[1] for item in self.history if item[0] == majority_char]
            avg_conf = np.mean(majority_confs) if majority_confs else 0.0
            
            # Determine output
            if majority_char == "?" or avg_conf < self.confidence_threshold:
                smoothed_pred = "Unknown Gesture"
            else:
                smoothed_pred = majority_char
                
            inference_time_ms = (time.time() - t0) * 1000.0
            
            return {
                "smoothed_prediction": smoothed_pred,
                "smoothed_confidence": avg_conf,
                "raw_prediction": raw_letter,
                "raw_confidence": raw_conf,
                "inference_time_ms": inference_time_ms
            }
            
        except Exception as e:
            logger.error(f"Prediction failure: {str(e)}")
            return {
                "smoothed_prediction": "Error",
                "smoothed_confidence": 0.0,
                "raw_prediction": "None",
                "raw_confidence": 0.0,
                "inference_time_ms": 0.0
            }

    def clear_history(self):
        """Clears the temporal history buffer."""
        self.history.clear()
