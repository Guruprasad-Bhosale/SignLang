import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from src.data.collector import DatasetCollector
from src.utils.logger import logger

def main():
    dataset_dir = r"C:\Users\Gurup\.cache\kagglehub\datasets\jeyasrisenthil\hand-signs-asl-hand-sign-data\versions\1\DATASET"
    output_csv = "datasets/hand_gestures.csv"
    
    # 1. Initialize detector
    base_options = python.BaseOptions(model_asset_path='models/hand_landmarker.task')
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1
    )
    detector = vision.HandLandmarker.create_from_options(options)
    
    # 2. Initialize collector (overwrite existing file)
    if os.path.exists(output_csv):
        try:
            os.remove(output_csv)
            logger.info(f"Removed old dataset file: {output_csv}")
        except Exception as e:
            logger.error(f"Failed to remove old dataset file: {str(e)}")
            
    collector = DatasetCollector(filepath=output_csv)
    
    classes = [chr(i) for i in range(ord('A'), ord('Z') + 1)] # A-Z
    samples_per_class_target = 1000
    
    logger.info("Starting conversion of Kaggle ASL images to landmark dataset...")
    
    total_images_processed = 0
    total_landmarks_extracted = 0
    total_samples_saved = 0
    
    for char in classes:
        char_dir = os.path.join(dataset_dir, char)
        if not os.path.isdir(char_dir):
            logger.warning(f"Directory for class {char} not found: {char_dir}")
            continue
            
        image_files = [f for f in os.listdir(char_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        logger.info(f"Processing {len(image_files)} images for letter '{char}'...")
        
        # We want to extract base landmarks
        base_landmarks_list = []
        
        for img_name in image_files:
            img_path = os.path.join(char_dir, img_name)
            try:
                cv_img = cv2.imread(img_path)
                if cv_img is None:
                    continue
                total_images_processed += 1
                
                rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)
                
                results = detector.detect(mp_image)
                if results.hand_landmarks:
                    hand_landmarks = results.hand_landmarks[0]
                    # Convert to list of [x, y, z]
                    coords = [[lm.x, lm.y, lm.z] for lm in hand_landmarks]
                    
                    # Detect handedness
                    handedness = "Right"
                    if results.handedness:
                        handedness = results.handedness[0][0].category_name
                        
                    base_landmarks_list.append((coords, handedness))
                    total_landmarks_extracted += 1
            except Exception as e:
                logger.error(f"Error processing {img_path}: {str(e)}")
                
        num_base = len(base_landmarks_list)
        if num_base == 0:
            logger.warning(f"No landmarks extracted for letter '{char}'!")
            continue
            
        logger.info(f"Extracted {num_base} base landmarks for '{char}'. Generating augmentations...")
        
        # We need to generate around samples_per_class_target samples
        # Each base sample will be augmented target // num_base times
        aug_factor = max(1, int(np.ceil(samples_per_class_target / num_base)))
        
        for coords, handedness in base_landmarks_list:
            coords_arr = np.array(coords, dtype=np.float32)
            
            for _ in range(aug_factor):
                # 1. Add noise
                noise = np.random.normal(scale=0.005, size=(21, 3)).astype(np.float32)
                noisy_coords = coords_arr + noise
                
                # 2. Add random scale (between 0.8 and 1.3)
                scale = np.random.uniform(0.8, 1.3)
                noisy_coords *= scale
                
                # 3. Add random rotation (+/- 15 degrees roll)
                theta = np.random.uniform(-np.pi/12, np.pi/12)
                c, s = np.cos(theta), np.sin(theta)
                R = np.array([
                    [c, -s, 0],
                    [s, c, 0],
                    [0, 0, 1]
                ])
                noisy_coords = np.dot(noisy_coords, R.T)
                
                # 4. Add random translation
                translation = np.random.uniform(-0.3, 0.3, size=(3,))
                noisy_coords += translation
                
                # Random handedness mirroring (augment left/right hands)
                hand_label = handedness
                if np.random.rand() > 0.5:
                    hand_label = "Left" if handedness == "Right" else "Right"
                    # Mirror X coordinate
                    noisy_coords[:, 0] = -noisy_coords[:, 0]
                    
                collector.add_sample(
                    target_letter=char,
                    handedness=hand_label,
                    confidence=0.98,
                    landmarks=noisy_coords.tolist()
                )
                total_samples_saved += 1
                
    detector.close()
    logger.info("Landmark dataset generation complete!")
    logger.info(f"Processed images: {total_images_processed}")
    logger.info(f"Extracted hand coordinates: {total_landmarks_extracted}")
    logger.info(f"Saved total augmented samples: {total_samples_saved} across {len(classes)} classes.")

if __name__ == "__main__":
    main()
