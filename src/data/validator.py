import os
import pandas as pd
import numpy as np
from src.utils.logger import logger

class DatasetValidator:
    """
    Validates the gesture dataset for:
    - Missing values
    - Corrupt samples
    - Duplicate samples
    - Class imbalance (checks A-Z, minimum 1000 samples per class)
    - Invalid coordinates
    """
    def __init__(self, filepath="datasets/hand_gestures.csv"):
        self.filepath = filepath

    def validate(self, report_dir="logs"):
        """
        Validates the dataset and generates a markdown report.
        
        Returns:
        --------
        dict : Summary of validation results and a boolean indicating if dataset is training-ready.
        """
        report_path = os.path.join(report_dir, "validation_report.md")
        os.makedirs(report_dir, exist_ok=True)
        
        results = {
            "file_exists": False,
            "total_samples": 0,
            "missing_values": 0,
            "corrupt_samples": 0,
            "duplicate_samples": 0,
            "invalid_coordinates": 0,
            "class_imbalance": {},
            "warnings": [],
            "errors": [],
            "is_valid": False
        }
        
        if not os.path.exists(self.filepath):
            msg = f"Dataset file not found at: {self.filepath}"
            results["errors"].append(msg)
            self._write_report(results, report_path)
            return results
            
        results["file_exists"] = True
        
        try:
            # Load dataset
            df = pd.read_csv(self.filepath)
            results["total_samples"] = len(df)
            
            if len(df) == 0:
                results["errors"].append("Dataset is empty.")
                self._write_report(results, report_path)
                return results
                
            # 1. Check columns
            landmark_cols = []
            for i in range(21):
                landmark_cols.extend([f"x_{i}", f"y_{i}", f"z_{i}"])
            required_cols = ["class", "handedness", "confidence", "timestamp"] + landmark_cols
            
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                results["errors"].append(f"Missing required columns in CSV: {missing_cols}")
                results["corrupt_samples"] = len(df)
                self._write_report(results, report_path)
                return results
                
            # 2. Check missing values
            missing_val_count = df[required_cols].isnull().sum().sum()
            results["missing_values"] = int(missing_val_count)
            if missing_val_count > 0:
                results["warnings"].append(f"Found {missing_val_count} missing values in the dataset.")
                
            # 3. Check duplicate coordinate sequences
            dup_coords = df.duplicated(subset=landmark_cols).sum()
            results["duplicate_samples"] = int(dup_coords)
            if dup_coords > 0:
                results["warnings"].append(f"Found {dup_coords} duplicate coordinate samples.")
                
            # 4. Check invalid/out-of-bounds coordinates
            # MediaPipe normalized landmarks should generally be in [-2, 2] (relative to frame/hand).
            # Extreme outliers (e.g. > 10.0 or <-10.0) indicate potential tracking noise or corrupt rows.
            coords_df = df[landmark_cols]
            out_of_bounds = ((coords_df > 10.0) | (coords_df < -10.0)).any(axis=1).sum()
            results["invalid_coordinates"] = int(out_of_bounds)
            if out_of_bounds > 0:
                results["warnings"].append(f"Found {out_of_bounds} samples with out-of-bounds coordinates (>10 or <-10).")
                
            # Check for completely flat (all zero) landmarks
            all_zeros = (coords_df == 0.0).all(axis=1).sum()
            if all_zeros > 0:
                results["invalid_coordinates"] += int(all_zeros)
                results["warnings"].append(f"Found {all_zeros} samples with all landmarks set to exactly 0.0.")
                
            # 5. Check class counts & imbalance
            expected_classes = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
            class_counts = df["class"].value_counts().to_dict()
            
            imbalanced_classes = []
            missing_classes = []
            
            for char in expected_classes:
                cnt = class_counts.get(char, 0)
                results["class_imbalance"][char] = cnt
                if cnt == 0:
                    missing_classes.append(char)
                elif cnt < 1000:
                    imbalanced_classes.append((char, cnt))
                    
            if missing_classes:
                results["errors"].append(f"Missing samples for classes: {', '.join(missing_classes)}")
            if imbalanced_classes:
                detail_str = ", ".join([f"{char} ({cnt}/1000)" for char, cnt in imbalanced_classes])
                results["warnings"].append(f"Class imbalance detected (minimum 1000 samples required): {detail_str}")
                
            # Decide if dataset is valid for robust training
            # We want at least some samples of A-Z, no critical column errors, and missing values handled
            has_errors = len(results["errors"]) > 0
            has_extreme_imbalance = len(missing_classes) > 0 or any(cnt < 100 for cnt in class_counts.values())
            
            if not has_errors and not has_extreme_imbalance:
                results["is_valid"] = True
            else:
                results["is_valid"] = False
                
        except Exception as e:
            msg = f"Exception occurred during validation: {str(e)}"
            logger.error(msg)
            results["errors"].append(msg)
            results["is_valid"] = False
            
        self._write_report(results, report_path)
        return results

    def _write_report(self, res, report_path):
        """Generates a structured validation report and writes it to disk."""
        try:
            lines = [
                "# GestureVerse Dataset Validation Report",
                f"Generated at: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
                "",
                "## Summary",
                f"- **Dataset Path**: `{self.filepath}`",
                f"- **Status**: {'PASS ✅' if res['is_valid'] else 'FAIL ❌'}",
                f"- **Total Samples**: {res['total_samples']}",
                f"- **Missing Values**: {res['missing_values']}",
                f"- **Corrupt Samples**: {res['corrupt_samples']}",
                f"- **Duplicate Samples**: {res['duplicate_samples']}",
                f"- **Invalid Coordinates**: {res['invalid_coordinates']}",
                ""
            ]
            
            # Errors
            if res["errors"]:
                lines.append("## Errors 🔴")
                for err in res["errors"]:
                    lines.append(f"- {err}")
                lines.append("")
                
            # Warnings
            if res["warnings"]:
                lines.append("## Warnings ⚠️")
                for wrn in res["warnings"]:
                    lines.append(f"- {wrn}")
                lines.append("")
                
            # Class Distribution Table
            lines.extend([
                "## Class Distribution",
                "| Class | Count | Target (1000) | Status |",
                "|---|---|---|---|"
            ])
            
            for char, count in sorted(res["class_imbalance"].items()):
                status = "✅ Met" if count >= 1000 else ("⚠️ Low" if count > 0 else "❌ Missing")
                lines.append(f"| **{char}** | {count} | 1000 | {status} |")
                
            report_content = "\n".join(lines)
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(report_content)
            logger.info(f"Dataset validation report successfully written to {report_path}")
            
        except Exception as e:
            logger.error(f"Failed to write validation report: {str(e)}")
