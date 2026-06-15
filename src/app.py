import os
import sys

# Ensure the project root directory is in the path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from PyQt6.QtWidgets import QApplication
from src.utils.logger import logger
from src.data.collector import DatasetCollector
from src.data.validator import DatasetValidator
from src.training.trainer import ModelTrainer
from src.inference.predictor import GesturePredictor
from src.ui.camera_thread import CameraThread
from src.ui.main_window import MainWindow

def main():
    logger.info("Initializing GestureVerse Application...")
    
    # Define file paths
    dataset_path = "datasets/hand_gestures.csv"
    model_path = "models/best_model.joblib"
    log_dir = "logs"
    model_dir = "models"
    
    # Ensure directories exist
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs("datasets", exist_ok=True)
    
    # 1. Initialize Components
    collector = DatasetCollector(filepath=dataset_path)
    validator = DatasetValidator(filepath=dataset_path)
    trainer = ModelTrainer(filepath=dataset_path, model_dir=model_dir, log_dir=log_dir)
    predictor = GesturePredictor(model_path=model_path, window_size=10, confidence_threshold=0.85)
    
    # 2. Setup threads
    camera_thread = CameraThread(camera_index=0)
    camera_thread.set_collector(collector)
    
    # 3. Launch UI
    logger.info("Starting PyQt6 Window...")
    app = QApplication(sys.argv)
    
    window = MainWindow(
        camera_thread=camera_thread,
        collector=collector,
        validator=validator,
        trainer=trainer,
        predictor=predictor
    )
    window.show()
    
    # Run the application event loop
    sys_exit_code = app.exec()
    
    # Graceful shutdown support for terminal
    logger.info(f"GestureVerse application exited with status code: {sys_exit_code}")
    sys.exit(sys_exit_code)

if __name__ == "__main__":
    main()
