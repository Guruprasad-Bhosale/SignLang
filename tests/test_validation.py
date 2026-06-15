import os
import csv
import unittest
import tempfile
import pandas as pd
from src.data.validator import DatasetValidator

class TestDatasetValidator(unittest.TestCase):
    def setUp(self):
        # Create a temporary file
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_csv_path = os.path.join(self.temp_dir.name, "test_gestures.csv")
        
        # Define headers
        self.headers = ["class", "handedness", "confidence", "timestamp"]
        for i in range(21):
            self.headers.extend([f"x_{i}", f"y_{i}", f"z_{i}"])

    def tearDown(self):
        self.temp_dir.cleanup()

    def write_csv(self, rows):
        with open(self.temp_csv_path, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(self.headers)
            writer.writerows(rows)

    def test_missing_file(self):
        validator = DatasetValidator(filepath="non_existent_file.csv")
        res = validator.validate(report_dir=self.temp_dir.name)
        self.assertFalse(res["file_exists"])
        self.assertFalse(res["is_valid"])
        self.assertIn("not found", res["errors"][0])

    def test_empty_dataset(self):
        # Write only headers
        self.write_csv([])
        validator = DatasetValidator(filepath=self.temp_csv_path)
        res = validator.validate(report_dir=self.temp_dir.name)
        self.assertTrue(res["file_exists"])
        self.assertEqual(res["total_samples"], 0)
        self.assertFalse(res["is_valid"])

    def test_invalid_coordinates_and_imbalance(self):
        # Write some rows with bad coordinates, and class imbalance (e.g. only A and B, count < 1000)
        row1 = ["A", "Right", 0.95, 123456789.0] + [0.0] * 63 # all zeros (invalid coordinates warning)
        row2 = ["B", "Left", 0.90, 123456790.0] + [99.0] * 63 # out-of-bounds (invalid coordinates warning)
        
        self.write_csv([row1, row2])
        
        validator = DatasetValidator(filepath=self.temp_csv_path)
        res = validator.validate(report_dir=self.temp_dir.name)
        
        self.assertEqual(res["total_samples"], 2)
        # Verify invalid coordinates count
        self.assertGreater(res["invalid_coordinates"], 0)
        # Verify class imbalance is detected (A and B have count 1, other letters have count 0)
        self.assertEqual(res["class_imbalance"]["A"], 1)
        self.assertEqual(res["class_imbalance"]["C"], 0)
        
        # A and B are < 1000, C-Z are missing, so warnings/errors will occur
        self.assertFalse(res["is_valid"])
        self.assertGreater(len(res["warnings"]), 0)
        self.assertGreater(len(res["errors"]), 0) # missing classes is a critical error

if __name__ == '__main__':
    unittest.main()
