import cv2
import numpy as np

def debug_dots(path):
    img = cv2.imread(path)
    if img is None:
        print("Failed to read image")
        return
        
    # 1. Detect red dots
    # Red color mask in BGR (high R, low G, low B)
    # The circles are red, so R should be high, G and B should be low.
    r = img[:, :, 2].astype(float)
    g = img[:, :, 1].astype(float)
    b = img[:, :, 0].astype(float)
    
    # Red metric: R - Max(G, B)
    red_metric = r - np.maximum(g, b)
    _, red_mask = cv2.threshold(red_metric, 80, 255, cv2.THRESH_BINARY)
    red_mask = red_mask.astype(np.uint8)
    
    # Find connected components for the dots
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(red_mask)
    
    # Filter centroids (remove background label 0 and small noise)
    dots = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= 2: # filter out single pixel noise
            dots.append(centroids[i])
            
    print(f"Detected {len(dots)} red dots.")
    for idx, dot in enumerate(dots):
        print(f"  Dot {idx}: {dot}")
        
    # 2. Check connections (grey lines)
    # The lines are grey (R, G, B close to each other, and intermediate brightness)
    # Let's see if we can find them
    gray_mask = (np.abs(r - g) < 15) & (np.abs(r - b) < 15) & (r > 120) & (r < 240)
    print(f"Number of gray pixels: {np.sum(gray_mask)}")

if __name__ == "__main__":
    debug_dots(r"C:\Users\Gurup\.gemini\antigravity-ide\brain\98b263d2-3a21-4c1f-8be6-764afbb10cdb\hand_a.jpg")
