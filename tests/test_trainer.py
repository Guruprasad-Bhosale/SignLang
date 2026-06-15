import os
import unittest
import tempfile
import numpy as np
import joblib
from src.data.collector import DatasetCollector
from src.training.trainer import ModelTrainer

class TestModelTrainer(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dataset_path = os.path.join(self.temp_dir.name, "test_hand_gestures.csv")
        self.model_dir = os.path.join(self.temp_dir.name, "models")
        self.log_dir = os.path.join(self.temp_dir.name, "logs")

        # Create a small dataset with 10 samples of 'A' and 10 samples of 'B'
        collector = DatasetCollector(filepath=self.dataset_path)
        
        # Generate base palm coordinates with slight noise
        base_coords = np.zeros((21, 3), dtype=np.float32)
        base_coords[9] = [0.0, 0.2, 0.0]  # Middle MCP
        
        for _ in range(10):
            # Class A
            landmarks_a = base_coords + np.random.normal(scale=0.005, size=(21, 3))
            collector.add_sample("A", "Right", 0.95, landmarks_a.tolist())
            
            # Class B
            landmarks_b = base_coords + np.random.normal(scale=0.005, size=(21, 3))
            # tweak B slightly to make it distinct
            landmarks_b[8] = [0.05, 0.3, 0.0]  
            collector.add_sample("B", "Left", 0.90, landmarks_b.tolist())

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_trainer_pipeline(self):
        # Initialize trainer
        trainer = ModelTrainer(
            filepath=self.dataset_path,
            model_dir=self.model_dir,
            log_dir=self.log_dir
        )
        
        # Run loading and preprocessing
        X_train, X_test, y_train, y_test, label_encoder = trainer.load_and_preprocess_data()
        
        # Assert split sizes and encoding
        self.assertEqual(len(X_train), 16)  # 20 samples * 0.8 = 16
        self.assertEqual(len(X_test), 4)    # 20 samples * 0.2 = 4
        self.assertEqual(len(label_encoder.classes_), 2)
        self.assertIn("A", label_encoder.classes_)
        self.assertIn("B", label_encoder.classes_)
        
        # Run full train and evaluate
        best_model_name, evaluation_results = trainer.train_and_evaluate()
        
        # Verify best model was selected and saved
        self.assertIn(best_model_name, ["RandomForest", "SVM", "XGBoost", "MLP"])
        self.assertIn("RandomForest", evaluation_results)
        self.assertIn("SVM", evaluation_results)
        
        best_model_path = os.path.join(self.model_dir, "best_model.joblib")
        self.assertTrue(os.path.exists(best_model_path))
        
        # Load and verify saved joblib payload keys
        payload = joblib.load(best_model_path)
        self.assertIn("model", payload)
        self.assertIn("label_encoder", payload)
        self.assertEqual(payload["model_name"], best_model_name)
        
        # Verify log files (plots and reports) were written
        self.assertTrue(os.path.exists(os.path.join(self.log_dir, "confusion_matrix.png")))
        self.assertTrue(os.path.exists(os.path.join(self.log_dir, "training_report.md")))
        
        # Read the training report to ensure content is correct
        with open(os.path.join(self.log_dir, "training_report.md"), "r", encoding="utf-8") as f:
            report_content = f.read()
        self.assertIn("GestureVerse Model Training Report", report_content)
        self.assertIn(best_model_name, report_content)

if __name__ == '__main__':
    unittest.main()
