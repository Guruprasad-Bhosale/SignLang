import os
import queue
import time
from PyQt6.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QImage, QPixmap, QFont
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QComboBox, QProgressBar, QTabWidget,
    QTextEdit, QGroupBox, QScrollArea, QFrame
)

from src.utils.logger import logger
from src.data.collector import DatasetCollector
from src.data.validator import DatasetValidator
from src.training.trainer import ModelTrainer
from src.inference.predictor import GesturePredictor

# QSS Stylesheet for premium look and feel
DARK_THEME_STYLE = """
QMainWindow {
    background-color: #121212;
}
QWidget {
    color: #e4e4e7;
    font-family: 'Segoe UI', -apple-system, Roboto, sans-serif;
    font-size: 13px;
}
QTabWidget::panel {
    border: 1px solid #27272a;
    background-color: #1c1c1e;
    border-radius: 8px;
    padding: 10px;
}
QTabBar::tab {
    background: #27272a;
    border: 1px solid #3f3f46;
    padding: 8px 16px;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    margin-right: 2px;
}
QTabBar::tab:selected {
    background: #14b8a6;
    color: #09090b;
    border-color: #14b8a6;
    font-weight: bold;
}
QTabBar::tab:hover:!selected {
    background: #3f3f46;
}
QGroupBox {
    border: 1px solid #27272a;
    border-radius: 8px;
    margin-top: 12px;
    padding-top: 16px;
    font-weight: bold;
    color: #14b8a6;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 12px;
    padding: 0 4px;
}
QPushButton {
    background-color: #27272a;
    border: 1px solid #3f3f46;
    border-radius: 6px;
    padding: 8px 16px;
    font-weight: bold;
    min-height: 20px;
}
QPushButton:hover {
    background-color: #3f3f46;
    border-color: #52525b;
}
QPushButton:pressed {
    background-color: #18181b;
}
QPushButton#recordBtn {
    background-color: #b91c1c;
    border-color: #991b1b;
    color: #ffffff;
}
QPushButton#recordBtn:hover {
    background-color: #dc2626;
}
QPushButton#recordBtn:checked {
    background-color: #166534;
    border-color: #14532d;
}
QPushButton#actionBtn {
    background-color: #14b8a6;
    color: #09090b;
    border-color: #14b8a6;
}
QPushButton#actionBtn:hover {
    background-color: #0d9488;
}
QComboBox {
    background-color: #27272a;
    border: 1px solid #3f3f46;
    border-radius: 6px;
    padding: 6px 12px;
    min-width: 60px;
}
QComboBox:hover {
    border-color: #14b8a6;
}
QComboBox QAbstractItemView {
    background-color: #1c1c1e;
    border: 1px solid #27272a;
    selection-background-color: #14b8a6;
    selection-color: #09090b;
}
QProgressBar {
    border: 1px solid #27272a;
    border-radius: 6px;
    text-align: center;
    background-color: #09090b;
}
QProgressBar::chunk {
    background-color: #14b8a6;
    border-radius: 4px;
}
QTextEdit {
    background-color: #09090b;
    border: 1px solid #27272a;
    border-radius: 6px;
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 12px;
    color: #a1a1aa;
}
QLabel#charDisplay {
    background-color: #09090b;
    border: 2px solid #14b8a6;
    border-radius: 12px;
    color: #ffffff;
    font-weight: bold;
    font-size: 96px;
    qproperty-alignment: 'AlignCenter';
}
QLabel#statLabel {
    color: #a1a1aa;
    font-size: 12px;
}
QLabel#statValue {
    font-weight: bold;
    font-size: 12px;
    color: #ffffff;
}
QFrame#videoBorder {
    border: 2px solid #27272a;
    border-radius: 12px;
    background-color: #000000;
}
"""

class InferenceThread(QThread):
    """Asynchronous thread for running model predictions without blocking UI."""
    prediction_ready = pyqtSignal(dict)
    
    def __init__(self, predictor: GesturePredictor):
        super().__init__()
        self.predictor = predictor
        self.queue = queue.Queue(maxsize=1)
        self.running = True
        
    def process_landmarks(self, landmarks, handedness):
        """Pushes landmarks to queue for inference."""
        if landmarks is None:
            return
            
        try:
            # If queue has an item, discard it to ensure we only predict the latest frame
            if self.queue.full():
                self.queue.get_nowait()
            self.queue.put_nowait((landmarks, handedness))
        except Exception:
            pass

    def run(self):
        while self.running:
            try:
                # Wait for data with timeout to periodically check if thread is stopping
                item = self.queue.get(timeout=0.05)
                landmarks, handedness = item
                res = self.predictor.predict(landmarks, handedness)
                self.prediction_ready.emit(res)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in InferenceThread loop: {str(e)}")
                
    def stop(self):
        self.running = False
        self.wait()

class ValidationThread(QThread):
    """Asynchronous dataset validation thread."""
    validation_finished = pyqtSignal(dict)

    def __init__(self, validator: DatasetValidator):
        super().__init__()
        self.validator = validator

    def run(self):
        res = self.validator.validate()
        self.validation_finished.emit(res)

class TrainingThread(QThread):
    """Asynchronous model training thread."""
    training_finished = pyqtSignal(str, dict)
    training_failed = pyqtSignal(str)

    def __init__(self, trainer: ModelTrainer):
        super().__init__()
        self.trainer = trainer

    def run(self):
        try:
            best_model_name, evaluation_results = self.trainer.train_and_evaluate()
            self.training_finished.emit(best_model_name, evaluation_results)
        except Exception as e:
            self.training_failed.emit(str(e))

class MainWindow(QMainWindow):
    def __init__(self, camera_thread, collector, validator, trainer, predictor):
        super().__init__()
        self.setWindowTitle("GestureVerse - Sign Language Interpreter")
        self.resize(1100, 700)
        self.setStyleSheet(DARK_THEME_STYLE)
        
        self.camera_thread = camera_thread
        self.collector = collector
        self.validator = validator
        self.trainer = trainer
        self.predictor = predictor
        
        # Start Inference Thread
        self.infer_thread = InferenceThread(self.predictor)
        self.infer_thread.start()
        
        # Connect signals
        self.camera_thread.frame_processed.connect(self.on_frame_processed)
        self.camera_thread.status_changed.connect(self.on_status_changed)
        self.camera_thread.fps_updated.connect(self.on_fps_updated)
        self.camera_thread.collection_progress.connect(self.on_collection_progress)
        self.infer_thread.prediction_ready.connect(self.on_prediction_ready)
        
        self.setup_ui()
        self.update_collector_counts()
        
        # Start camera thread
        self.camera_thread.start()

    def setup_ui(self):
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main Layout (Horizontal)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(12)
        
        # ================= LEFT PANEL (Webcam Feed & Performance) =================
        left_layout = QVBoxLayout()
        left_layout.setSpacing(8)
        
        # Webcam container
        video_container = QFrame()
        video_container.setObjectName("videoBorder")
        video_container_layout = QVBoxLayout(video_container)
        video_container_layout.setContentsMargins(4, 4, 4, 4)
        
        self.video_label = QLabel()
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setText("Starting camera feed...")
        self.video_label.setStyleSheet("background-color: #09090b; border-radius: 8px;")
        video_container_layout.addWidget(self.video_label)
        
        left_layout.addWidget(video_container)
        
        # Performance/Status Bar
        status_box = QGroupBox("System Performance & Status")
        status_layout = QGridLayout(status_box)
        status_layout.setContentsMargins(12, 8, 12, 8)
        
        lbl_sys = QLabel("System Status:")
        lbl_sys.setObjectName("statLabel")
        self.val_sys = QLabel("Initialized")
        self.val_sys.setObjectName("statValue")
        
        lbl_cam_fps = QLabel("Webcam FPS:")
        lbl_cam_fps.setObjectName("statLabel")
        self.val_cam_fps = QLabel("0.0")
        self.val_cam_fps.setObjectName("statValue")
        
        lbl_proc_fps = QLabel("Processing FPS:")
        lbl_proc_fps.setObjectName("statLabel")
        self.val_proc_fps = QLabel("0.0")
        self.val_proc_fps.setObjectName("statValue")
        
        lbl_latency = QLabel("Inference Latency:")
        lbl_latency.setObjectName("statLabel")
        self.val_latency = QLabel("0.0 ms")
        self.val_latency.setObjectName("statValue")
        
        status_layout.addWidget(lbl_sys, 0, 0)
        status_layout.addWidget(self.val_sys, 0, 1)
        status_layout.addWidget(lbl_cam_fps, 0, 2)
        status_layout.addWidget(self.val_cam_fps, 0, 3)
        status_layout.addWidget(lbl_proc_fps, 1, 0)
        status_layout.addWidget(self.val_proc_fps, 1, 1)
        status_layout.addWidget(lbl_latency, 1, 2)
        status_layout.addWidget(self.val_latency, 1, 3)
        
        left_layout.addWidget(status_box)
        main_layout.addLayout(left_layout, stretch=3)
        
        # ================= RIGHT PANEL (Interpreter, Collection, Training) =================
        right_layout = QVBoxLayout()
        right_layout.setSpacing(12)
        
        self.tabs = QTabWidget()
        
        # Tab 1: Interpreter
        self.interpreter_tab = QWidget()
        self.setup_interpreter_tab()
        self.tabs.addTab(self.interpreter_tab, "Interpreter")
        
        # Tab 2: Dataset Manager
        self.dataset_tab = QWidget()
        self.setup_dataset_tab()
        self.tabs.addTab(self.dataset_tab, "Dataset Manager")
        
        # Tab 3: Model Training
        self.training_tab = QWidget()
        self.setup_training_tab()
        self.tabs.addTab(self.training_tab, "Model Training")
        
        right_layout.addWidget(self.tabs)
        main_layout.addLayout(right_layout, stretch=2)

    def setup_interpreter_tab(self):
        layout = QVBoxLayout(self.interpreter_tab)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(16)
        
        # Character Display
        display_group = QGroupBox("Smoothed Gesture Prediction")
        display_layout = QVBoxLayout(display_group)
        
        self.char_display = QLabel("?")
        self.char_display.setObjectName("charDisplay")
        display_layout.addWidget(self.char_display)
        
        layout.addWidget(display_group, stretch=2)
        
        # Prediction Metadata
        meta_group = QGroupBox("Prediction Confidence & Info")
        meta_layout = QGridLayout(meta_group)
        meta_layout.setSpacing(12)
        
        lbl_conf = QLabel("Confidence:")
        lbl_conf.setObjectName("statLabel")
        self.conf_bar = QProgressBar()
        self.conf_bar.setRange(0, 100)
        self.conf_bar.setValue(0)
        
        lbl_hand = QLabel("Handedness:")
        lbl_hand.setObjectName("statLabel")
        self.val_hand = QLabel("No Hand Detected")
        self.val_hand.setObjectName("statValue")
        
        lbl_raw = QLabel("Raw Model Predict:")
        lbl_raw.setObjectName("statLabel")
        self.val_raw = QLabel("None")
        self.val_raw.setObjectName("statValue")
        
        meta_layout.addWidget(lbl_conf, 0, 0)
        meta_layout.addWidget(self.conf_bar, 0, 1)
        meta_layout.addWidget(lbl_hand, 1, 0)
        meta_layout.addWidget(self.val_hand, 1, 1)
        meta_layout.addWidget(lbl_raw, 2, 0)
        meta_layout.addWidget(self.val_raw, 2, 1)
        
        layout.addWidget(meta_group, stretch=1)
        
        # Add a placeholder for models active status
        model_lbl = QLabel(f"Active Model: {self.predictor.model_name or 'None (Not Trained)'}")
        model_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        model_lbl.setStyleSheet("font-weight: bold; color: #14b8a6; font-size: 11px;")
        self.active_model_label = model_lbl
        layout.addWidget(model_lbl)
        
    def setup_dataset_tab(self):
        layout = QVBoxLayout(self.dataset_tab)
        layout.setSpacing(12)
        
        # Control Group
        ctrl_group = QGroupBox("Record Hand Landmarks")
        ctrl_layout = QGridLayout(ctrl_group)
        ctrl_layout.setSpacing(10)
        
        ctrl_layout.addWidget(QLabel("Target Letter:"), 0, 0)
        self.letter_combo = QComboBox()
        for i in range(ord('A'), ord('Z') + 1):
            self.letter_combo.addItem(chr(i))
        ctrl_layout.addWidget(self.letter_combo, 0, 1)
        
        self.record_btn = QPushButton("Start Capture")
        self.record_btn.setObjectName("recordBtn")
        self.record_btn.setCheckable(True)
        self.record_btn.clicked.connect(self.toggle_recording)
        ctrl_layout.addWidget(self.record_btn, 0, 2)
        
        ctrl_layout.addWidget(QLabel("Progress:"), 1, 0)
        self.collect_progress_bar = QProgressBar()
        self.collect_progress_bar.setRange(0, 1000)
        self.collect_progress_bar.setValue(0)
        ctrl_layout.addWidget(self.collect_progress_bar, 1, 1, 1, 2)
        
        layout.addWidget(ctrl_group)
        
        # Dataset Summary Table
        summary_group = QGroupBox("Current Dataset Statistics")
        summary_layout = QVBoxLayout(summary_group)
        
        # Scroll Area for stats
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("background-color: #09090b; border: none;")
        
        scroll_content = QWidget()
        self.grid_stats_layout = QGridLayout(scroll_content)
        self.grid_stats_layout.setSpacing(6)
        
        scroll.setWidget(scroll_content)
        summary_layout.addWidget(scroll)
        
        layout.addWidget(summary_group, stretch=1)

    def setup_training_tab(self):
        layout = QVBoxLayout(self.training_tab)
        layout.setSpacing(12)
        
        actions_layout = QHBoxLayout()
        
        self.btn_validate = QPushButton("Validate Dataset")
        self.btn_validate.setObjectName("actionBtn")
        self.btn_validate.clicked.connect(self.run_validation)
        actions_layout.addWidget(self.btn_validate)
        
        self.btn_train = QPushButton("Train Classifier Models")
        self.btn_train.setObjectName("actionBtn")
        self.btn_train.clicked.connect(self.run_training)
        actions_layout.addWidget(self.btn_train)
        
        layout.addLayout(actions_layout)
        
        # Console output for training
        layout.addWidget(QLabel("Validation & Training Reports:"))
        self.training_console = QTextEdit()
        self.training_console.setReadOnly(True)
        self.training_console.setPlaceholderText("Click 'Validate Dataset' or 'Train Classifier Models' to output report summaries.")
        layout.addWidget(self.training_console, stretch=1)

    # ================= EVENT HANDLERS & SLOTS =================

    @pyqtSlot(QImage, object, str, float)
    def on_frame_processed(self, q_img, landmarks, handedness, confidence):
        # 1. Update Video Feed
        pixmap = QPixmap.fromImage(q_img)
        self.video_label.setPixmap(
            pixmap.scaled(self.video_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        )
        
        # 2. If landmarks found, send to inference thread if we are in inference mode (i.e. tab index 0 active)
        if landmarks is not None:
            self.val_hand.setText(f"{handedness} Hand ({confidence:.1%})")
            if self.tabs.currentIndex() == 0:
                self.infer_thread.process_landmarks(landmarks, handedness)
        else:
            if self.tabs.currentIndex() == 0:
                self.char_display.setText("?")
                self.conf_bar.setValue(0)
                self.val_hand.setText("No Hand Detected")
                self.val_raw.setText("None")

    @pyqtSlot(dict)
    def on_prediction_ready(self, res):
        # Check if tab 0 is still active
        if self.tabs.currentIndex() != 0:
            return
            
        smoothed_pred = res["smoothed_prediction"]
        smoothed_conf = res["smoothed_confidence"]
        raw_pred = res["raw_prediction"]
        raw_conf = res["raw_confidence"]
        latency = res["inference_time_ms"]
        
        # Update UI Labels
        self.char_display.setText(smoothed_pred if len(smoothed_pred) == 1 else "?")
        self.conf_bar.setValue(int(smoothed_conf * 100))
        self.val_raw.setText(f"{raw_pred} ({raw_conf:.1%})")
        self.val_latency.setText(f"{latency:.1f} ms")
        
        if smoothed_pred == "Unknown Gesture":
            self.char_display.setText("?")
            self.char_display.setStyleSheet("background-color: #09090b; border: 2px solid #b91c1c; color: #b91c1c; font-size: 64px;")
        else:
            self.char_display.setStyleSheet("background-color: #09090b; border: 2px solid #14b8a6; color: #ffffff; font-size: 96px;")

    @pyqtSlot(str)
    def on_status_changed(self, msg):
        self.val_sys.setText(msg)

    @pyqtSlot(float, float)
    def on_fps_updated(self, camera_fps, process_fps):
        self.val_cam_fps.setText(f"{camera_fps:.1f}")
        self.val_proc_fps.setText(f"{process_fps:.1f}")

    @pyqtSlot(int, int, str)
    def on_collection_progress(self, current, target, letter):
        self.collect_progress_bar.setValue(current)
        self.collect_progress_bar.setFormat(f"{current} / {target} ({current/target:.1%})")
        
        # If collection reached target, toggle button back
        if current >= target:
            self.record_btn.setChecked(False)
            self.record_btn.setText("Start Capture")
            self.camera_thread.stop_collection()
            self.update_collector_counts()

    def toggle_recording(self, checked):
        if checked:
            letter = self.letter_combo.currentText()
            self.record_btn.setText("Stop Recording")
            self.camera_thread.start_collection(letter)
        else:
            self.record_btn.setText("Start Capture")
            self.camera_thread.stop_collection()
            self.update_collector_counts()

    def update_collector_counts(self):
        """Refreshes the grid showing the count of collected samples for A-Z."""
        # Clear layout first
        for i in range(self.grid_stats_layout.count()):
            widget = self.grid_stats_layout.itemAt(i).widget()
            if widget is not None:
                widget.deleteLater()
                
        counts = self.collector.get_all_class_counts()
        
        # Build A-Z grid (4 columns)
        cols = 4
        letters = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
        
        for idx, char in enumerate(letters):
            cnt = counts.get(char, 0)
            row = idx // cols
            col = idx % cols
            
            lbl_item = QLabel(f"<b>{char}</b>: {cnt}")
            if cnt >= 1000:
                lbl_item.setStyleSheet("color: #14b8a6; background-color: #1c1c1e; padding: 4px; border-radius: 4px;")
            elif cnt > 0:
                lbl_item.setStyleSheet("color: #fbbf24; background-color: #1c1c1e; padding: 4px; border-radius: 4px;")
            else:
                lbl_item.setStyleSheet("color: #71717a; background-color: #1c1c1e; padding: 4px; border-radius: 4px;")
                
            self.grid_stats_layout.addWidget(lbl_item, row, col)

    def run_validation(self):
        self.training_console.setText("Running dataset validation check...\n")
        self.btn_validate.setEnabled(False)
        self.btn_train.setEnabled(False)
        
        self.val_thread = ValidationThread(self.validator)
        self.val_thread.validation_finished.connect(self.on_validation_finished)
        self.val_thread.start()

    @pyqtSlot(dict)
    def on_validation_finished(self, res):
        self.btn_validate.setEnabled(True)
        self.btn_train.setEnabled(True)
        
        self.update_collector_counts()
        
        out = []
        out.append(f"DATASET VALIDATION SUMMARY:")
        out.append(f"===========================")
        out.append(f"File exists: {res['file_exists']}")
        out.append(f"Total samples: {res['total_samples']}")
        out.append(f"Missing values: {res['missing_values']}")
        out.append(f"Duplicate samples: {res['duplicate_samples']}")
        out.append(f"Invalid coordinate rows: {res['invalid_coordinates']}")
        out.append(f"Is training ready: {'YES (Ready!)' if res['is_valid'] else 'NO'}")
        
        if res["errors"]:
            out.append("\nERRORS DETECTED:")
            for err in res["errors"]:
                out.append(f"- {err}")
                
        if res["warnings"]:
            out.append("\nWARNINGS:")
            for wrn in res["warnings"]:
                out.append(f"- {wrn}")
                
        self.training_console.setText("\n".join(out))
        self.val_sys.setText("Validation check complete.")

    def run_training(self):
        self.training_console.setText("Starting classifier models training pipeline...\n"
                                     "Extracting features, scaling, rotating hand landmarks, and splitting dataset.\n"
                                     "Training Random Forest, SVM, XGBoost, and MLP. Please wait...\n")
        self.btn_validate.setEnabled(False)
        self.btn_train.setEnabled(False)
        self.val_sys.setText("Training models in background...")
        
        self.train_thread = TrainingThread(self.trainer)
        self.train_thread.training_finished.connect(self.on_training_finished)
        self.train_thread.training_failed.connect(self.on_training_failed)
        self.train_thread.start()

    @pyqtSlot(str, dict)
    def on_training_finished(self, best_model_name, results):
        self.btn_validate.setEnabled(True)
        self.btn_train.setEnabled(True)
        self.val_sys.setText("Model training complete.")
        
        # Reload the model in predictor
        reload_success = self.predictor.load_model()
        if reload_success:
            self.active_model_label.setText(f"Active Model: {self.predictor.model_name}")
            
        out = []
        out.append(f"TRAINING PIPELINE COMPLETE!")
        out.append(f"===========================")
        out.append(f"Best classifier: {best_model_name} (Saved to models/best_model.joblib)\n")
        out.append("MODELS COMPARISON TABLE:")
        out.append("| Model | Accuracy | Weighted F1 | Train Time |")
        out.append("|---|---|---|---|")
        
        for name, metrics in results.items():
            out.append(f"| {name} | {metrics['accuracy']:.2%} | {metrics['f1']:.4f} | {metrics['train_time_sec']:.2f}s |")
            
        out.append(f"\nSaved confusion matrix plot to logs/confusion_matrix.png")
        out.append(f"Saved full evaluation report to logs/training_report.md")
        
        self.training_console.setText("\n".join(out))

    @pyqtSlot(str)
    def on_training_failed(self, err_msg):
        self.btn_validate.setEnabled(True)
        self.btn_train.setEnabled(True)
        self.val_sys.setText("Training failed.")
        self.training_console.setText(f"TRAINING PIPELINE FAILED:\n{err_msg}")

    def closeEvent(self, event):
        """Ensures all running threads stop before closing the app window."""
        logger.info("Closing application window. Stopping threads...")
        self.camera_thread.stop()
        self.infer_thread.stop()
        
        # Stop background training or validation threads if running
        if hasattr(self, 'val_thread') and self.val_thread.isRunning():
            logger.info("Terminating active validation thread...")
            self.val_thread.terminate()
            self.val_thread.wait()
            
        if hasattr(self, 'train_thread') and self.train_thread.isRunning():
            logger.info("Terminating active training thread...")
            self.train_thread.terminate()
            self.train_thread.wait()
            
        event.accept()
