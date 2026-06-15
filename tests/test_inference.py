import unittest
import numpy as np
from sklearn.preprocessing import LabelEncoder
from src.inference.predictor import GesturePredictor

class MockClassifier:
    def __init__(self, pred_idx=0, prob_val=0.95):
        self.pred_idx = pred_idx
        self.prob_val = prob_val

    def predict(self, X):
        return np.array([self.pred_idx])

    def predict_proba(self, X):
        # 3 classes: class 0, 1, 2
        probs = np.zeros((1, 3))
        probs[0, self.pred_idx] = self.prob_val
        # split remainder
        rem = (1.0 - self.prob_val) / 2.0
        for i in range(3):
            if i != self.pred_idx:
                probs[0, i] = rem
        return probs

class TestPredictionPipeline(unittest.TestCase):
    def setUp(self):
        # Create mock hand landmarks
        self.mock_lms = np.zeros((21, 3), dtype=np.float32)
        self.mock_lms[9] = [0.0, 0.2, 0.0]  # Middle MCP
        
        # Instantiate GesturePredictor with empty path (runs uninitialized)
        self.predictor = GesturePredictor(model_path="non_existent.joblib", window_size=5, confidence_threshold=0.80)
        
        # Manually wire a mock model and label encoder
        self.mock_model = MockClassifier(pred_idx=0, prob_val=0.90)  # A with 90% confidence
        self.encoder = LabelEncoder()
        self.encoder.fit(['A', 'B', 'C'])
        
        self.predictor.model = self.mock_model
        self.predictor.label_encoder = self.encoder
        self.predictor.model_name = "MockClassifier"

    def test_single_prediction(self):
        res = self.predictor.predict(self.mock_lms, handedness="Right")
        self.assertEqual(res["raw_prediction"], "A")
        self.assertAlmostEqual(res["raw_confidence"], 0.90)
        self.assertEqual(res["smoothed_prediction"], "A")

    def test_low_confidence_unknown(self):
        # If confidence is below threshold, it should output "Unknown Gesture"
        self.predictor.confidence_threshold = 0.95  # set threshold higher than 0.90
        res = self.predictor.predict(self.mock_lms, handedness="Right")
        self.assertEqual(res["raw_prediction"], "A")
        self.assertEqual(res["smoothed_prediction"], "Unknown Gesture")

    def test_sliding_window_voting(self):
        # We test how the temporal smoothing handles a transition.
        # Window size is 5.
        # First 4 frames: Mock A (90% confidence)
        self.predictor.model = MockClassifier(pred_idx=0, prob_val=0.90)
        for _ in range(4):
            res = self.predictor.predict(self.mock_lms, handedness="Right")
            self.assertEqual(res["smoothed_prediction"], "A")
            
        # Frame 5: Mock B (90% confidence)
        # B is in minority in the window (A:4, B:1), so output should still be A
        self.predictor.model = MockClassifier(pred_idx=1, prob_val=0.90) # B
        res = self.predictor.predict(self.mock_lms, handedness="Right")
        self.assertEqual(res["raw_prediction"], "B")
        self.assertEqual(res["smoothed_prediction"], "A")  # majority voted output remains A
        
        # Frame 6, 7: Mock B (90% confidence)
        # Window contents: A:2, B:3. B becomes majority. Output should switch to B!
        self.predictor.predict(self.mock_lms, handedness="Right")
        res = self.predictor.predict(self.mock_lms, handedness="Right")
        self.assertEqual(res["smoothed_prediction"], "B")

if __name__ == '__main__':
    unittest.main()
