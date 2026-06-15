import os
import csv
import time
from src.utils.logger import logger

class DatasetCollector:
    """
    Manages collection of static gesture dataset and saves raw landmark coordinates
    with metadata into a CSV format.
    """
    def __init__(self, filepath="datasets/hand_gestures.csv"):
        self.filepath = filepath
        self.headers = [
            "class", "handedness", "confidence", "timestamp"
        ]
        # Append 21 landmarks (x, y, z)
        for i in range(21):
            self.headers.extend([f"x_{i}", f"y_{i}", f"z_{i}"])
            
        self._ensure_file_exists()

    def _ensure_file_exists(self):
        """Creates the directory and CSV file with headers if it does not exist."""
        dirname = os.path.dirname(self.filepath)
        if dirname and not os.path.exists(dirname):
            os.makedirs(dirname, exist_ok=True)
            logger.info(f"Created directory: {dirname}")
            
        if not os.path.exists(self.filepath):
            try:
                with open(self.filepath, mode='w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(self.headers)
                logger.info(f"Initialized CSV file with headers at: {self.filepath}")
            except Exception as e:
                logger.error(f"Failed to initialize CSV dataset file: {str(e)}")
                raise e

    def add_sample(self, target_letter, handedness, confidence, landmarks):
        """
        Appends a single hand landmark sample to the CSV file.
        
        Parameters:
        -----------
        target_letter : str
            The class label (A-Z)
        handedness : str
            "Left" or "Right"
        confidence : float
            MediaPipe detection confidence
        landmarks : list or np.ndarray
            The 21 hand landmarks, each with x, y, z
        """
        try:
            timestamp = time.time()
            row = [target_letter.upper(), handedness, confidence, timestamp]
            
            # Format landmarks
            for lm in landmarks:
                if hasattr(lm, 'x'):
                    row.extend([lm.x, lm.y, lm.z])
                elif isinstance(lm, dict) and 'x' in lm:
                    row.extend([lm['x'], lm['y'], lm['z']])
                elif isinstance(lm, (list, tuple)) and len(lm) == 3:
                    row.extend(lm)
                else:
                    raise ValueError(f"Invalid landmark format in sample: {lm}")
                    
            if len(row) != len(self.headers):
                raise ValueError(f"Sample length ({len(row)}) does not match header length ({len(self.headers)})")
                
            with open(self.filepath, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(row)
                
            return True
        except Exception as e:
            logger.error(f"Failed to write sample to CSV: {str(e)}")
            return False

    def get_sample_count(self, target_letter=None):
        """
        Returns the number of samples in the dataset.
        If target_letter is specified, returns count for that class only.
        """
        if not os.path.exists(self.filepath):
            return 0
            
        try:
            count = 0
            class_counts = {}
            with open(self.filepath, mode='r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    lbl = row.get("class", "").upper()
                    class_counts[lbl] = class_counts.get(lbl, 0) + 1
                    count += 1
                    
            if target_letter:
                return class_counts.get(target_letter.upper(), 0)
            return count
        except Exception as e:
            logger.error(f"Error reading sample counts: {str(e)}")
            return 0

    def get_all_class_counts(self):
        """Returns a dictionary mapping class labels to sample counts."""
        if not os.path.exists(self.filepath):
            return {}
            
        try:
            class_counts = {}
            with open(self.filepath, mode='r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    lbl = row.get("class", "").upper()
                    if lbl:
                        class_counts[lbl] = class_counts.get(lbl, 0) + 1
            return class_counts
        except Exception as e:
            logger.error(f"Error reading class counts: {str(e)}")
            return {}
