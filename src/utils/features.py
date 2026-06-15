import numpy as np
from src.utils.logger import logger

def landmarks_to_numpy(landmarks):
    """
    Converts MediaPipe landmarks object to a numpy array of shape (21, 3).
    Accepts list of landmark objects, dict list, or numpy array.
    """
    if isinstance(landmarks, (list, tuple)) and len(landmarks) == 63:
        landmarks = np.array(landmarks, dtype=np.float32)

    if isinstance(landmarks, np.ndarray):
        if landmarks.shape == (21, 3):
            return landmarks
        elif landmarks.shape == (63,):
            return landmarks.reshape(21, 3)
        else:
            raise ValueError(f"Invalid numpy shape: {landmarks.shape}")
            
    coords = []
    for lm in landmarks:
        if hasattr(lm, 'x'):
            coords.append([lm.x, lm.y, lm.z])
        elif isinstance(lm, dict) and 'x' in lm:
            coords.append([lm['x'], lm['y'], lm['z']])
        elif isinstance(lm, (list, tuple)) and len(lm) == 3:
            coords.append(lm)
        else:
            raise ValueError(f"Unknown landmark format: {lm}")
            
    coords = np.array(coords, dtype=np.float32)
    if coords.shape != (21, 3):
        raise ValueError(f"Expected 21 landmarks, got shape {coords.shape}")
    return coords

def extract_features(landmarks, handedness="Right"):
    """
    Extracts a highly robust, translation, scale, and rotation invariant feature vector
    from 21 hand landmarks.
    
    Parameters:
    -----------
    landmarks : list or numpy array
        The 21 hand landmarks.
    handedness : str
        "Left" or "Right" to handle mirroring.
        
    Returns:
    --------
    np.ndarray : 1D array of invariant features.
    """
    try:
        # Convert to numpy array of shape (21, 3)
        coords = landmarks_to_numpy(landmarks)
        
        # 1. Translate wrist (0) to origin
        wrist = coords[0]
        translated = coords - wrist
        
        # 2. Scale by palm size
        # Palm size defined as distance between Wrist (0) and Middle finger MCP (9)
        middle_mcp = translated[9]
        palm_size = np.linalg.norm(middle_mcp)
        if palm_size < 1e-6:
            palm_size = 1e-6
        scaled = translated / palm_size
        
        # 3. Rotate to canonical orientation
        # Up vector (Y-axis) points from wrist (0) to middle finger MCP (9)
        y_axis = scaled[9]  # scaled[9] - scaled[0] (which is (0,0,0))
        y_norm = np.linalg.norm(y_axis)
        y_axis = y_axis / (y_norm if y_norm > 1e-6 else 1.0)
        
        # Lateral vector (X-axis) points from Index MCP (5) to Pinky MCP (17)
        lateral = scaled[17] - scaled[5]
        # Project lateral vector onto the plane orthogonal to Y-axis
        x_axis = lateral - np.dot(lateral, y_axis) * y_axis
        x_norm = np.linalg.norm(x_axis)
        x_axis = x_axis / (x_norm if x_norm > 1e-6 else 1.0)
        
        # Z-axis (palm normal) is orthogonal to X and Y
        z_axis = np.cross(x_axis, y_axis)
        z_norm = np.linalg.norm(z_axis)
        z_axis = z_axis / (z_norm if z_norm > 1e-6 else 1.0)
        
        # Rotation matrix (rows are the local axes)
        R = np.vstack([x_axis, y_axis, z_axis])
        
        # Rotate coordinates
        canonical_coords = np.dot(scaled, R.T)
        
        # Mirror along X-axis if it's Left Hand to maintain handedness invariance
        if handedness == "Left":
            canonical_coords[:, 0] = -canonical_coords[:, 0]
            
        # --- Feature Engineering ---
        features = []
        
        # Feature 1: Canonical Coordinates (flat 63 features)
        features.extend(canonical_coords.flatten())
        
        # Helper: vector angle in radians
        def get_angle(v1, v2):
            v1_norm = np.linalg.norm(v1)
            v2_norm = np.linalg.norm(v2)
            if v1_norm < 1e-6 or v2_norm < 1e-6:
                return 0.0
            cos_val = np.dot(v1, v2) / (v1_norm * v2_norm)
            return np.arccos(np.clip(cos_val, -1.0, 1.0))
            
        # Define finger bone sequences
        # Thumb: 0->1->2->3->4
        # Index: 0->5->6->7->8
        # Middle: 0->9->10->11->12
        # Ring: 0->13->14->15->16
        # Pinky: 0->17->18->19->20
        fingers = {
            'thumb': [0, 1, 2, 3, 4],
            'index': [0, 5, 6, 7, 8],
            'middle': [0, 9, 10, 11, 12],
            'ring': [0, 13, 14, 15, 16],
            'pinky': [0, 17, 18, 19, 20]
        }
        
        # Feature 2: Finger extension values (5 features)
        # Ratio of tip-to-MCP distance divided by sum of bone segments
        for name, idxs in fingers.items():
            mcp = canonical_coords[idxs[1]]
            tip = canonical_coords[idxs[-1]]
            tip_dist = np.linalg.norm(tip - mcp)
            
            # Sum of bone lengths
            bone_sum = 0.0
            for k in range(1, len(idxs) - 1):
                bone_sum += np.linalg.norm(canonical_coords[idxs[k+1]] - canonical_coords[idxs[k]])
            if bone_sum < 1e-6:
                bone_sum = 1e-6
            features.append(tip_dist / bone_sum)
            
        # Feature 3: Finger States (5 features, 0 or 1)
        # Check if tip is further from wrist than PIP/MCP joints
        # Index, Middle, Ring, Pinky
        for name in ['index', 'middle', 'ring', 'pinky']:
            idxs = fingers[name]
            tip_wrist = np.linalg.norm(canonical_coords[idxs[-1]])
            pip_wrist = np.linalg.norm(canonical_coords[idxs[-3]])
            features.append(1.0 if tip_wrist > pip_wrist else 0.0)
        # Thumb
        thumb_tip = canonical_coords[4]
        index_mcp = canonical_coords[5]
        thumb_ip = canonical_coords[2]
        features.append(1.0 if np.linalg.norm(thumb_tip - index_mcp) > np.linalg.norm(thumb_ip - index_mcp) else 0.0)
        
        # Feature 4: Joint angles (14 angles)
        # For each finger, compute angles between adjacent bone segments
        # Thumb: (1-0) to (2-1), (2-1) to (3-2), (3-2) to (4-3) -> 3 angles
        # Index: (5-0) to (6-5), (6-5) to (7-6), (7-6) to (8-7) -> 3 angles
        # Middle: (9-0) to (10-9), etc. -> 3 angles
        # Ring: (13-0) to (14-13), etc. -> 3 angles
        # Pinky: (17-0) to (18-17), etc. -> 3 angles
        # Total angles = 15 angles
        for name, idxs in fingers.items():
            # Segment vectors
            segs = []
            for k in range(len(idxs) - 1):
                segs.append(canonical_coords[idxs[k+1]] - canonical_coords[idxs[k]])
            # Compute angles between consecutive segments
            for k in range(len(segs) - 1):
                features.append(get_angle(segs[k], segs[k+1]))
                
        # Feature 5: Relative fingertip distances (4 features)
        # Distances between adjacent tips: Thumb-Index, Index-Middle, Middle-Ring, Ring-Pinky
        tips = [4, 8, 12, 16, 20]
        for k in range(len(tips) - 1):
            dist = np.linalg.norm(canonical_coords[tips[k]] - canonical_coords[tips[k+1]])
            features.append(dist)
            
        # Feature 6: Fingertips to wrist distance (5 features)
        for tip in tips:
            features.append(np.linalg.norm(canonical_coords[tip]))
            
        # Feature 7: Relative ratios (3 features)
        # Ratios between adjacent fingertip distances
        for k in range(3):
            d1 = np.linalg.norm(canonical_coords[tips[k]] - canonical_coords[tips[k+1]])
            d2 = np.linalg.norm(canonical_coords[tips[k+1]] - canonical_coords[tips[k+2]])
            features.append(d1 / (d2 + 1e-6))
            
        # Feature 8: Palm Orientation normal vector in raw space (3 features)
        features.extend(z_axis.tolist())
        
        # Feature 9: Hand Rotation Euler angles (3 features)
        # Derive roll, pitch, yaw from rotation matrix R
        pitch = np.arctan2(-R[2, 0], np.sqrt(R[2, 1]**2 + R[2, 2]**2))
        yaw = np.arctan2(R[1, 0], R[0, 0])
        roll = np.arctan2(R[2, 1], R[2, 2])
        features.extend([pitch, yaw, roll])
        
        # Feature 10: Hand Openness (1 feature)
        # Sum of fingertip to wrist distances
        openness = sum(np.linalg.norm(canonical_coords[tip]) for tip in tips)
        features.append(openness)
        
        return np.array(features, dtype=np.float32)
        
    except Exception as e:
        logger.error(f"Error in feature extraction: {str(e)}")
        raise e
