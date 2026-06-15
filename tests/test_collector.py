import os
import csv
import unittest
import tempfile
from src.data.collector import DatasetCollector

class TestDatasetCollector(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_csv_path = os.path.join(self.temp_dir.name, "subdir", "gestures.csv")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_initialization(self):
        # Initializing collector should create target directory and write headers
        collector = DatasetCollector(filepath=self.temp_csv_path)
        self.assertTrue(os.path.exists(self.temp_csv_path))
        
        with open(self.temp_csv_path, mode='r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            headers = next(reader)
            
        self.assertEqual(headers[0], "class")
        self.assertEqual(headers[1], "handedness")
        self.assertEqual(headers[2], "confidence")
        self.assertEqual(headers[3], "timestamp")
        # 21 landmarks * 3 coords = 63 + 4 headers = 67 total columns
        self.assertEqual(len(headers), 67)

    def test_add_sample_formats(self):
        collector = DatasetCollector(filepath=self.temp_csv_path)
        
        # Format 1: List of lists [x, y, z]
        landmarks_list = [[float(i), float(i+1), float(i+2)] for i in range(21)]
        res1 = collector.add_sample("A", "Right", 0.99, landmarks_list)
        self.assertTrue(res1)
        
        # Format 2: List of dicts {'x': ..., 'y': ..., 'z': ...}
        landmarks_dict = [{'x': float(i), 'y': float(i+1), 'z': float(i+2)} for i in range(21)]
        res2 = collector.add_sample("B", "Left", 0.95, landmarks_dict)
        self.assertTrue(res2)
        
        # Format 3: Mock object with x, y, z attributes
        class MockLandmark:
            def __init__(self, x, y, z):
                self.x = x
                self.y = y
                self.z = z
        landmarks_obj = [MockLandmark(float(i), float(i+1), float(i+2)) for i in range(21)]
        res3 = collector.add_sample("c", "Right", 0.90, landmarks_obj)
        self.assertTrue(res3)

        # Check total counts
        self.assertEqual(collector.get_sample_count(), 3)
        self.assertEqual(collector.get_sample_count("A"), 1)
        self.assertEqual(collector.get_sample_count("B"), 1)
        self.assertEqual(collector.get_sample_count("C"), 1)  # letter is normalized to uppercase
        self.assertEqual(collector.get_sample_count("D"), 0)

        # Check all class counts dictionary
        counts = collector.get_all_class_counts()
        self.assertEqual(counts, {"A": 1, "B": 1, "C": 1})

    def test_invalid_landmark_length(self):
        collector = DatasetCollector(filepath=self.temp_csv_path)
        
        # Landmarker list with only 20 landmarks instead of 21
        invalid_landmarks = [[0.0, 0.0, 0.0] for _ in range(20)]
        res = collector.add_sample("A", "Right", 0.99, invalid_landmarks)
        self.assertFalse(res)
        
        # Landmarker list containing invalid format
        invalid_format = [0.0] * 63
        res2 = collector.add_sample("A", "Right", 0.99, invalid_format)
        self.assertFalse(res2)

if __name__ == '__main__':
    unittest.main()
