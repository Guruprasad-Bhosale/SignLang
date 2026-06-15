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
    Modifies base hand coordinates based on letter ASCII value to create
    consistent, mathematically separable classes.
    """
    hand = base_hand.copy()
    val = char_code - ord('A')
    
    # We can control finger extensions based on bits of val
    # Bit 0: Thumb fold
    if val & 1:
        # Fold thumb: pull tips towards center
        hand[2:5] *= 0.6
    # Bit 1: Index fold
    if val & 2:
        # Fold index
        hand[6:9] *= 0.5
    # Bit 2: Middle fold
    if val & 4:
        # Fold middle
        hand[10:13] *= 0.5
    # Bit 3: Ring fold
    if val & 8:
        # Fold ring
        hand[14:17] *= 0.5
    # Bit 4: Pinky fold
    if val & 16:
        # Fold pinky
        hand[18:21] *= 0.5
        
    # Add unique class signature offsets
    signature_offset = (val + 1) * 0.015
    hand[8] += [signature_offset, 0, 0] # modify index tip x
    hand[12] += [0, signature_offset, 0] # modify middle tip y
    hand[16] += [0, 0, signature_offset] # modify ring tip z
    hand[20] -= [signature_offset, 0, 0] # modify pinky tip x
    
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
