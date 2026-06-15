import unittest
import numpy as np
from src.utils.features import landmarks_to_numpy, extract_features

class TestFeatureExtraction(unittest.TestCase):
    def setUp(self):
        # Create a mock open hand (21 landmarks)
        # We start with wrist at (0, 0, 0)
        self.mock_lms = np.zeros((21, 3), dtype=np.float32)
        # Middle MCP (9) is at (0, 0.2, 0) - this is our palm reference
        self.mock_lms[9] = [0.0, 0.2, 0.0]
        # Index MCP (5)
        self.mock_lms[5] = [0.05, 0.18, 0.0]
        # Pinky MCP (17)
        self.mock_lms[17] = [-0.05, 0.16, 0.0]
        
        # Populate other finger joints to have valid geometries
        # Thumb: 1, 2, 3, 4
        self.mock_lms[1] = [0.04, 0.05, 0.0]
        self.mock_lms[2] = [0.07, 0.08, 0.0]
        self.mock_lms[3] = [0.09, 0.11, 0.0]
        self.mock_lms[4] = [0.11, 0.13, 0.0]
        
        # Index: 5, 6, 7, 8
        self.mock_lms[6] = [0.06, 0.24, 0.0]
        self.mock_lms[7] = [0.06, 0.29, 0.0]
        self.mock_lms[8] = [0.06, 0.33, 0.0]
        
        # Middle: 9, 10, 11, 12
        self.mock_lms[10] = [0.0, 0.28, 0.0]
        self.mock_lms[11] = [0.0, 0.34, 0.0]
        self.mock_lms[12] = [0.0, 0.39, 0.0]
        
        # Ring: 13, 14, 15, 16
        self.mock_lms[13] = [-0.03, 0.17, 0.0]
        self.mock_lms[14] = [-0.04, 0.23, 0.0]
        self.mock_lms[15] = [-0.04, 0.28, 0.0]
        self.mock_lms[16] = [-0.04, 0.32, 0.0]
        
        # Pinky: 17, 18, 19, 20
        self.mock_lms[18] = [-0.06, 0.21, 0.0]
        self.mock_lms[19] = [-0.06, 0.25, 0.0]
        self.mock_lms[20] = [-0.06, 0.29, 0.0]

    def test_landmarks_to_numpy(self):
        # Test shape conversion
        arr = landmarks_to_numpy(self.mock_lms)
        self.assertEqual(arr.shape, (21, 3))
        
        # Test flat format list
        flat_list = self.mock_lms.flatten().tolist()
        arr_flat = landmarks_to_numpy(flat_list)
        self.assertEqual(arr_flat.shape, (21, 3))
        self.assertTrue(np.allclose(arr, arr_flat))
        
        # Test dict format list
        dict_list = [{'x': lm[0], 'y': lm[1], 'z': lm[2]} for lm in self.mock_lms]
        arr_dict = landmarks_to_numpy(dict_list)
        self.assertEqual(arr_dict.shape, (21, 3))

    def test_extract_features_shape(self):
        # Extract features should return a 1D feature vector
        feats = extract_features(self.mock_lms, handedness="Right")
        self.assertEqual(feats.ndim, 1)
        # Let's ensure it has the correct length (e.g. 108 features as calculated)
        self.assertGreater(len(feats), 63)

    def test_translation_invariance(self):
        # Apply random 3D translation
        translation = np.array([2.5, -1.2, 5.8], dtype=np.float32)
        translated_lms = self.mock_lms + translation
        
        feats_orig = extract_features(self.mock_lms, handedness="Right")
        feats_trans = extract_features(translated_lms, handedness="Right")
        
        # Features should be identical regardless of coordinate translation
        np.testing.assert_array_almost_equal(feats_orig, feats_trans, decimal=4)

    def test_scale_invariance(self):
        # Apply scale factor
        scale = 3.5
        scaled_lms = self.mock_lms * scale
        
        feats_orig = extract_features(self.mock_lms, handedness="Right")
        feats_scaled = extract_features(scaled_lms, handedness="Right")
        
        # Features should be identical regardless of hand scale/distance
        np.testing.assert_array_almost_equal(feats_orig, feats_scaled, decimal=4)

if __name__ == '__main__':
    unittest.main()
