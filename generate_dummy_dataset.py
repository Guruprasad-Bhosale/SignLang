import os
import sys
import numpy as np
import time

# Ensure root in python path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.logger import logger
from src.data.collector import DatasetCollector

def generate_base_hand():
    """Generates coordinates for a standard flat hand (open palm)."""
    lms = np.zeros((21, 3), dtype=np.float32)
    # Wrist is at (0, 0, 0)
    # Middle MCP (9) is at (0, 0.2, 0)
    lms[9] = [0.0, 0.2, 0.0]
    lms[5] = [0.05, 0.18, 0.0]   # Index MCP
    lms[17] = [-0.05, 0.16, 0.0]  # Pinky MCP
    lms[13] = [-0.02, 0.17, 0.0]  # Ring MCP
    lms[1] = [0.03, 0.05, 0.0]   # Thumb CMC
    
    # Thumb: 1->2->3->4
    lms[2] = [0.06, 0.08, 0.0]
    lms[3] = [0.08, 0.11, 0.0]
    lms[4] = [0.10, 0.13, 0.0]
    
    # Index: 5->6->7->8
    lms[6] = [0.06, 0.24, 0.0]
    lms[7] = [0.06, 0.29, 0.0]
    lms[8] = [0.06, 0.33, 0.0]
    
    # Middle: 9->10->11->12
    lms[10] = [0.0, 0.27, 0.0]
    lms[11] = [0.0, 0.33, 0.0]
    lms[12] = [0.0, 0.38, 0.0]
    
    # Ring: 13->14->15->16
    lms[14] = [-0.03, 0.23, 0.0]
    lms[15] = [-0.03, 0.28, 0.0]
    lms[16] = [-0.03, 0.32, 0.0]
    
    # Pinky: 17->18->19->20
    lms[18] = [-0.06, 0.20, 0.0]
    lms[19] = [-0.06, 0.24, 0.0]
    lms[20] = [-0.06, 0.28, 0.0]
    
    return lms

def modify_hand_for_class(base_hand, char_code):
    """
    Modifies base hand coordinates based on letter to create realistic
    ASL sign language gestures synthetically.
    """
    hand = base_hand.copy()
    char = chr(char_code)
    
    # helper: fold finger (scale coordinates of joints above MCP)
    def fold_finger(finger_name, scale=0.35):
        # Finger indices:
        # thumb: 1-4
        # index: 5-8
        # middle: 9-12
        # ring: 13-16
        # pinky: 17-20
        idxs = {
            'thumb': [2, 3, 4],
            'index': [6, 7, 8],
            'middle': [10, 11, 12],
            'ring': [14, 15, 16],
            'pinky': [18, 19, 20]
        }[finger_name]
        hand[idxs] *= scale

    # Default: open hand (all fingers extended)
    
    if char == 'A':
        # Fist: fold all fingers, thumb held at the side
        fold_finger('index', 0.35)
        fold_finger('middle', 0.35)
        fold_finger('ring', 0.35)
        fold_finger('pinky', 0.35)
        # Thumb remains extended or pressed against the side of index
        hand[1:5] *= np.array([1.1, 0.8, 0.7, 0.7]).reshape(-1, 1)
    elif char == 'B':
        # Flat hand: fold thumb across palm, others extended and close
        fold_finger('thumb', 0.35)
        # Narrow the fingers to represent flat hand
        hand[[5,6,7,8], 0] *= 0.5
        hand[[9,10,11,12], 0] *= 0.5
        hand[[13,14,15,16], 0] *= 0.5
        hand[[17,18,19,20], 0] *= 0.5
    elif char == 'C':
        # Curved shape: all fingers partially folded
        fold_finger('thumb', 0.7)
        fold_finger('index', 0.7)
        fold_finger('middle', 0.7)
        fold_finger('ring', 0.7)
        fold_finger('pinky', 0.7)
    elif char == 'D':
        # Pointer: index extended, others folded
        fold_finger('thumb', 0.4)
        fold_finger('middle', 0.35)
        fold_finger('ring', 0.35)
        fold_finger('pinky', 0.35)
    elif char == 'E':
        # Folded claws: all fingers folded tightly
        fold_finger('thumb', 0.4)
        fold_finger('index', 0.3)
        fold_finger('middle', 0.3)
        fold_finger('ring', 0.3)
        fold_finger('pinky', 0.3)
    elif char == 'F':
        # OK sign: thumb and index touch (folded), others extended
        fold_finger('index', 0.4)
        fold_finger('thumb', 0.5)
        # move tips of index and thumb close together
        hand[4] = hand[8] = (hand[4] + hand[8]) / 2.0
    elif char == 'I':
        # Pinky extended, others folded
        fold_finger('thumb', 0.4)
        fold_finger('index', 0.35)
        fold_finger('middle', 0.35)
        fold_finger('ring', 0.35)
    elif char == 'L':
        # L-shape: thumb and index extended, others folded
        fold_finger('middle', 0.35)
        fold_finger('ring', 0.35)
        fold_finger('pinky', 0.35)
    elif char in ['U', 'V']:
        # Index and middle extended, others folded
        fold_finger('thumb', 0.4)
        fold_finger('ring', 0.35)
        fold_finger('pinky', 0.35)
    elif char == 'W':
        # Index, middle, ring extended, thumb and pinky folded
        fold_finger('thumb', 0.4)
        fold_finger('pinky', 0.35)
    elif char == 'Y':
        # Thumb and pinky extended, index, middle, ring folded
        fold_finger('index', 0.35)
        fold_finger('middle', 0.35)
        fold_finger('ring', 0.35)
    else:
        # Fallback binary pattern for other letters to maintain mathematical separability
        val = char_code - ord('A')
        if val & 1: fold_finger('thumb', 0.4)
        if val & 2: fold_finger('index', 0.4)
        if val & 4: fold_finger('middle', 0.4)
        if val & 8: fold_finger('ring', 0.4)
        if val & 16: fold_finger('pinky', 0.4)
        
    return hand

def main():
    logger.info("Generating realistic mock dataset for sign letters A-Z...")
    
    collector = DatasetCollector(filepath="datasets/hand_gestures.csv")
    
    letters = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
    samples_per_class = 1000
    
    base_hand = generate_base_hand()
    
    t0 = time.time()
    total_added = 0
    
    for char in letters:
        char_code = ord(char)
        logger.info(f"Generating 1000 samples for letter '{char}'...")
        
        # Modify the base hand configuration uniquely for this letter
        class_base = modify_hand_for_class(base_hand, char_code)
        
        for _ in range(samples_per_class):
            # 1. Add Gaussian noise to simulate camera jitter and user variation
            noise = np.random.normal(scale=0.008, size=(21, 3)).astype(np.float32)
            # Maintain wrist at 0
            noise[0] = [0, 0, 0]
            
            noisy_hand = class_base + noise
            
            # 2. Add random hand translation/rotation/scale in raw frame (before extractor cleans it!)
            # This verifies the feature extractor's scale and rotation invariance!
            # Scale
            rand_scale = np.random.uniform(0.7, 1.8)
            noisy_hand *= rand_scale
            # Rotation
            theta = np.random.uniform(-np.pi/4, np.pi/4) # +/- 45 deg roll
            c, s = np.cos(theta), np.sin(theta)
            R = np.array([
                [c, -s, 0],
                [s, c, 0],
                [0, 0, 1]
            ])
            noisy_hand = np.dot(noisy_hand, R.T)
            # Translation
            translation = np.random.uniform(-1.0, 1.0, size=(3,))
            noisy_hand += translation
            
            # Metadata
            hand = "Right" if np.random.rand() > 0.3 else "Left"
            conf = float(np.random.uniform(0.85, 0.99))
            
            # Save
            collector.add_sample(
                target_letter=char,
                handedness=hand,
                confidence=conf,
                landmarks=noisy_hand.tolist()
            )
            total_added += 1
            
    elapsed = time.time() - t0
    logger.info(f"Dataset generation complete! Added {total_added} samples across 26 classes in {elapsed:.2f}s.")
    logger.info("Dataset saved at 'datasets/hand_gestures.csv'. Ready to validate and train.")

if __name__ == "__main__":
    main()
