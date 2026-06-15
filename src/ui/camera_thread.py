import time
import cv2
import numpy as np
import mediapipe as mp
from PyQt6.QtCore import QThread, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QImage

from src.utils.logger import logger
from src.data.collector import DatasetCollector

class CameraThread(QThread):
    """
    QThread that handles webcam frame capture, MediaPipe Hand tracking via Tasks API,
    renders skeletons using OpenCV, and optionally saves samples for data collection.
    """
    # Signal emitted when a new frame is processed
    # Emits: (QImage of the frame, hand_landmarks_list_or_None, handedness_str, detection_confidence)
    frame_processed = pyqtSignal(QImage, object, str, float)
    
    # Signal emitted to update status messages
    status_changed = pyqtSignal(str)
    
    # Signal emitted to update FPS info (camera_fps, process_fps)
    fps_updated = pyqtSignal(float, float)
    
    # Signal emitted when sample collection progresses: (current_count, target_count, target_letter)
    collection_progress = pyqtSignal(int, int, str)

    def __init__(self, camera_index=0):
        super().__init__()
        self.camera_index = camera_index
        self.running = False
        
        # Data Collection parameters
        self.collector = None
        self.is_collecting = False
        self.target_letter = ""
        self.collect_count = 0
        self.collect_target = 1000
        
        # MediaPipe detector instance
        self.detector = None
        self.last_timestamp_ms = 0
        
    def set_collector(self, collector: DatasetCollector):
        self.collector = collector

    def start_collection(self, letter, target=1000):
        """Starts dataset collection for a specific letter."""
        if self.collector is None:
            logger.error("Cannot start collection: DatasetCollector is not initialized.")
            return
        self.target_letter = letter.upper()
        self.collect_target = target
        self.collect_count = self.collector.get_sample_count(self.target_letter)
        self.is_collecting = True
        logger.info(f"Started collecting samples for letter '{self.target_letter}'. Current count: {self.collect_count}/{self.collect_target}")
        self.collection_progress.emit(self.collect_count, self.collect_target, self.target_letter)

    def stop_collection(self):
        """Stops dataset collection."""
        self.is_collecting = False
        logger.info("Dataset collection stopped.")

    def draw_skeleton(self, frame, landmarks):
        """Draws hand skeleton connections and joints using OpenCV."""
        h, w, c = frame.shape
        # Hand joint connectivity
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),      # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),      # Index
            (9, 10), (10, 11), (11, 12),         # Middle
            (13, 14), (14, 15), (15, 16),        # Ring
            (0, 17), (17, 18), (18, 19), (19, 20),# Pinky
            (5, 9), (9, 13), (13, 17)            # Palm base knuckles
        ]
        
        # Convert all landmarks to pixel coordinates
        pts = []
        for lm in landmarks:
            px = int(lm.x * w)
            py = int(lm.y * h)
            pts.append((px, py))
            
        # Draw connection lines (DodgerBlue: BGR(255, 144, 30))
        for start_idx, end_idx in connections:
            if start_idx < len(pts) and end_idx < len(pts):
                cv2.line(frame, pts[start_idx], pts[end_idx], (255, 144, 30), 2, cv2.LINE_AA)
                
        # Draw joint nodes (Crimson: BGR(60, 20, 220))
        for pt in pts:
            cv2.circle(frame, pt, 4, (60, 20, 220), -1, cv2.LINE_AA)

    def run(self):
        self.running = True
        self.status_changed.emit("Initializing camera and MediaPipe Tasks...")
        
        # Initialize MediaPipe Tasks HandLandmarker
        try:
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
            
            base_options = python.BaseOptions(model_asset_path='models/hand_landmarker.task')
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.VIDEO,
                num_hands=1,
                min_hand_detection_confidence=0.75,
                min_hand_presence_confidence=0.75
            )
            self.detector = vision.HandLandmarker.create_from_options(options)
        except Exception as e:
            logger.error(f"Failed to initialize MediaPipe HandLandmarker: {str(e)}")
            self.status_changed.emit("MediaPipe initialization failed!")
            self.running = False
            return
            
        # Initialize OpenCV Camera
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            logger.error(f"Failed to open video source at index {self.camera_index}")
            self.status_changed.emit("Error: Camera not found!")
            self.running = False
            return
            
        # Set camera parameters for higher speed
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 60)
        
        self.status_changed.emit("System Ready. Tracking Hand...")
        
        prev_time = time.time()
        fps_prev_time = time.time()
        frame_counter = 0
        process_counter = 0
        
        camera_fps = 0.0
        process_fps = 0.0
        
        while self.running:
            current_time = time.time()
            
            # Read Frame
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to grab frame. Camera disconnected?")
                self.status_changed.emit("Error: Camera disconnected!")
                time.sleep(0.5)
                # Attempt to reopen camera
                cap.release()
                cap = cv2.VideoCapture(self.camera_index)
                continue
                
            frame_counter += 1
            
            # Flip frame horizontally for natural mirror display
            frame = cv2.flip(frame, 1)
            h, w, c = frame.shape
            
            # Convert frame to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # MediaPipe tasks expects mp.Image wrapper
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            # Detect
            timestamp_ms = int(current_time * 1000)
            if timestamp_ms <= self.last_timestamp_ms:
                timestamp_ms = self.last_timestamp_ms + 1
            self.last_timestamp_ms = timestamp_ms
            
            results = self.detector.detect_for_video(mp_image, timestamp_ms)
            process_counter += 1
            
            hand_landmarks = None
            handedness = "Right"
            confidence = 0.0
            
            # Draw skeletons & extract landmarks
            if results.hand_landmarks:
                # We only track the first detected hand
                hand_landmarks = results.hand_landmarks[0]
                
                # Get handedness label
                if results.handedness:
                    handedness_category = results.handedness[0][0]
                    handedness = handedness_category.category_name
                    # MediaPipe handedness label is from camera view; flip it for correct display view
                    handedness = "Right" if handedness == "Left" else "Left"
                    confidence = handedness_category.score
                    
                # Draw landmarks on frame
                self.draw_skeleton(frame, hand_landmarks)
                
                # If collecting is active, write sample to dataset
                if self.is_collecting and self.collector:
                    # Convert landmark structure to list of coordinates
                    coords_list = []
                    for lm in hand_landmarks:
                        coords_list.append([lm.x, lm.y, lm.z])
                        
                    success = self.collector.add_sample(
                        target_letter=self.target_letter,
                        handedness=handedness,
                        confidence=confidence,
                        landmarks=coords_list
                    )
                    
                    if success:
                        self.collect_count += 1
                        self.collection_progress.emit(self.collect_count, self.collect_target, self.target_letter)
                        if self.collect_count >= self.collect_target:
                            self.is_collecting = False
                            logger.info(f"Dataset target reached for letter '{self.target_letter}'!")
                            self.status_changed.emit(f"Successfully collected {self.collect_target} samples for {self.target_letter}!")
                            
            # Convert BGR frame back to RGB (for QImage)
            rgb_render_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Create QImage from RGB buffer
            bytes_per_line = c * w
            q_img = QImage(
                rgb_render_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888
            )
            
            # Emit QImage and landmark data
            # Note: q_img points to rgb_render_frame.data buffer, which is about to be overwritten,
            # so we copy it to ensure thread safety
            self.frame_processed.emit(q_img.copy(), hand_landmarks, handedness, confidence)
            
            # Compute FPS every 1.0 seconds
            t_diff = current_time - fps_prev_time
            if t_diff >= 1.0:
                camera_fps = frame_counter / t_diff
                process_fps = process_counter / t_diff
                self.fps_updated.emit(camera_fps, process_fps)
                frame_counter = 0
                process_counter = 0
                fps_prev_time = current_time
                
            # Frame rate control to match target (~60 FPS -> ~16.6ms per frame)
            elapsed = time.time() - current_time
            sleep_time = max(0.001, (1.0 / 60.0) - elapsed)
            time.sleep(sleep_time)
            
        # Cleanup
        cap.release()
        if self.detector:
            try:
                self.detector.close()
            except Exception:
                pass
        logger.info("Camera Thread Stopped.")

    def stop(self):
        """Stops the thread loop."""
        self.running = False
        self.wait()
