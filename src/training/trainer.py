import os
import time
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from src.utils.logger import logger
from src.utils.features import extract_features

class ModelTrainer:
    """
    Handles feature engineering processing of the dataset, training of multiple models
    (Random Forest, SVM, XGBoost, MLP), model evaluation, and automatic selection
    of the best model.
    """
    def __init__(self, filepath="datasets/hand_gestures.csv", model_dir="models", log_dir="logs"):
        self.filepath = filepath
        self.model_dir = model_dir
        self.log_dir = log_dir
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

    def load_and_preprocess_data(self):
        """
        Loads landmarks from CSV, processes them through the feature extraction pipeline,
        and returns train/test splits.
        """
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"Dataset CSV not found at: {self.filepath}")
            
        logger.info("Loading dataset for preprocessing...")
        df = pd.read_csv(self.filepath)
        
        # 1. Extract raw landmarks and labels
        labels = df["class"].values
        handedness_list = df["handedness"].values
        
        # Extract landmark columns (x_0, y_0, z_0 ... x_20, y_20, z_20)
        landmark_cols = []
        for i in range(21):
            landmark_cols.extend([f"x_{i}", f"y_{i}", f"z_{i}"])
            
        raw_landmarks = df[landmark_cols].values
        
        logger.info(f"Extracting features for {len(df)} samples...")
        t0 = time.time()
        
        X = []
        y = []
        errors = 0
        
        for idx in range(len(df)):
            try:
                row_landmarks = raw_landmarks[idx].reshape(21, 3)
                hand = handedness_list[idx]
                feat = extract_features(row_landmarks, handedness=hand)
                X.append(feat)
                y.append(labels[idx])
            except Exception as e:
                errors += 1
                
        if errors > 0:
            logger.warning(f"Failed to extract features for {errors} samples.")
            
        X = np.array(X, dtype=np.float32)
        y = np.array(y)
        
        logger.info(f"Feature extraction completed in {time.time() - t0:.2f}s. Feature shape: {X.shape}")
        
        # 2. Encode Labels
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        
        # 3. Stratified Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
        )
        
        return X_train, X_test, y_train, y_test, label_encoder

    def train_and_evaluate(self):
        """
        Trains RF, SVM, XGBoost, and MLP. Evaluates metrics and selects the best model.
        """
        X_train, X_test, y_train, y_test, label_encoder = self.load_and_preprocess_data()
        
        models = {
            "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            "SVM": SVC(probability=True, random_state=42, C=10.0, gamma="scale"),
            "XGBoost": XGBClassifier(
                n_estimators=150, learning_rate=0.1, max_depth=6,
                random_state=42, n_jobs=-1, eval_metric="mlogloss"
            ),
            "MLP": MLPClassifier(
                hidden_layer_sizes=(128, 64), max_iter=500, activation="relu",
                random_state=42, early_stopping=True, validation_fraction=0.1
            )
        }
        
        best_f1 = -1.0
        best_model_name = ""
        best_model = None
        
        evaluation_results = {}
        
        for name, clf in models.items():
            logger.info(f"Training model: {name}...")
            t0 = time.time()
            clf.fit(X_train, y_train)
            train_time = time.time() - t0
            
            # Predict
            y_pred = clf.predict(X_test)
            
            # Metrics
            acc = accuracy_score(y_test, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted")
            
            evaluation_results[name] = {
                "accuracy": acc,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "train_time_sec": train_time,
                "model_instance": clf
            }
            
            logger.info(f"{name} Results -> Accuracy: {acc:.4f}, F1-Score: {f1:.4f}, Time: {train_time:.2f}s")
            
            # Track best model based on F1 Score
            if f1 > best_f1:
                best_f1 = f1
                best_model_name = name
                best_model = clf
                
        logger.info(f"Best model selected: {best_model_name} with F1-Score: {best_f1:.4f}")
        
        # Save Best Model along with classes mapping
        model_payload = {
            "model": best_model,
            "label_encoder": label_encoder,
            "model_name": best_model_name,
            "features_version": "1.0",
            "timestamp": time.time()
        }
        
        best_model_path = os.path.join(self.model_dir, "best_model.joblib")
        joblib.dump(model_payload, best_model_path)
        logger.info(f"Saved best model package to: {best_model_path}")
        
        # Perform detailed evaluation on the best model
        self.generate_reports(best_model, X_test, y_test, label_encoder, best_model_name, evaluation_results)
        
        return best_model_name, evaluation_results

    def generate_reports(self, model, X_test, y_test, label_encoder, model_name, eval_res):
        """Generates plots, confusion matrix, and markdown evaluation reports."""
        y_pred = model.predict(X_test)
        classes = label_encoder.classes_
        
        # 1. Confusion Matrix
        cm = confusion_matrix(y_test, y_pred, labels=range(len(classes)))
        plt.figure(figsize=(14, 11))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=classes, yticklabels=classes
        )
        plt.title(f"Confusion Matrix - {model_name}")
        plt.ylabel("Actual Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        cm_path = os.path.join(self.log_dir, "confusion_matrix.png")
        plt.savefig(cm_path, dpi=300)
        plt.close()
        logger.info(f"Saved confusion matrix plot to {cm_path}")
        
        # 2. Per-class Accuracy & Misclassifications
        per_class_acc = {}
        misclassifications = []
        for i, char in enumerate(classes):
            total_samples = np.sum(y_test == i)
            correct_samples = np.sum((y_test == i) & (y_pred == i))
            acc = correct_samples / total_samples if total_samples > 0 else 0.0
            per_class_acc[char] = acc
            
            # Find misclassifications for this class
            mis_idx = np.where((y_test == i) & (y_pred != i))[0]
            for idx in mis_idx:
                actual = classes[y_test[idx]]
                predicted = classes[y_pred[idx]]
                misclassifications.append((actual, predicted))
                
        # Group misclassifications
        mis_df = pd.DataFrame(misclassifications, columns=["Actual", "Predicted"])
        mis_summary = []
        if not mis_df.empty:
            mis_counts = mis_df.groupby(["Actual", "Predicted"]).size().reset_index(name="count")
            mis_counts = mis_counts.sort_values(by="count", ascending=False)
            for _, row in mis_counts.head(15).iterrows():
                mis_summary.append(f"- Letter **{row['Actual']}** misclassified as **{row['Predicted']}** ({row['count']} times)")
                
        # 3. Feature Importance (Random Forest or XGBoost)
        feature_importance_lines = []
        if model_name in ["RandomForest", "XGBoost"]:
            importances = model.feature_importances_
            # We have around 108 features. Let's list the top 15.
            # Create generic names for features
            feature_names = []
            for i in range(21):
                feature_names.extend([f"lm_{i}_x", f"lm_{i}_y", f"lm_{i}_z"])
            for f in ['thumb', 'index', 'middle', 'ring', 'pinky']:
                feature_names.append(f"{f}_extension")
            for f in ['index', 'middle', 'ring', 'pinky']:
                feature_names.append(f"{f}_state")
            feature_names.append("thumb_state")
            # Joint angles
            for f in ['thumb', 'index', 'middle', 'ring', 'pinky']:
                for k in range(3 if f != 'thumb' else 3): # 3 angles per finger
                    feature_names.append(f"{f}_joint_angle_{k}")
            # Ensure we match lengths or truncate
            tips = ["thumb", "index", "middle", "ring", "pinky"]
            for k in range(len(tips)-1):
                feature_names.append(f"dist_{tips[k]}_{tips[k+1]}")
            for k in range(len(tips)):
                feature_names.append(f"dist_{tips[k]}_wrist")
            for k in range(3):
                feature_names.append(f"ratio_{k}")
            feature_names.extend(["normal_x", "normal_y", "normal_z", "pitch", "yaw", "roll", "openness"])
            
            # Trim or pad feature names to match exact importance array size
            if len(feature_names) < len(importances):
                feature_names += [f"engineered_feat_{i}" for i in range(len(importances) - len(feature_names))]
            else:
                feature_names = feature_names[:len(importances)]
                
            indices = np.argsort(importances)[::-1]
            feature_importance_lines.append("## Feature Importance (Top 15 Invariant Features)")
            feature_importance_lines.append("| Rank | Feature | Importance |")
            feature_importance_lines.append("|---|---|---|")
            for r in range(min(15, len(importances))):
                idx = indices[r]
                feature_importance_lines.append(f"| {r+1} | `{feature_names[idx]}` | {importances[idx]:.5f} |")
            feature_importance_lines.append("")
            
        # 4. Generate Markdown Training Report
        report_path = os.path.join(self.log_dir, "training_report.md")
        report_lines = [
            "# GestureVerse Model Training Report",
            f"Generated at: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Model Comparison",
            "| Model | Accuracy | Weighted Precision | Weighted Recall | Weighted F1 | Train Time (s) |",
            "|---|---|---|---|---|---|",
        ]
        
        for name, metrics in eval_res.items():
            mark = "⭐️ (Best)" if name == model_name else ""
            report_lines.append(
                f"| **{name}** {mark} | {metrics['accuracy']:.4%}| {metrics['precision']:.4f} | "
                f"{metrics['recall']:.4f} | {metrics['f1']:.4f} | {metrics['train_time_sec']:.2f}s |"
            )
            
        report_lines.extend([
            "",
            "## Best Model Performance Details",
            f"- **Selected Classifier**: `{model_name}`",
            f"- **Test Accuracy**: `{eval_res[model_name]['accuracy']:.4%}`",
            f"- **Confusion Matrix**: ![Confusion Matrix](confusion_matrix.png)",
            "",
            "### Per-Letter Accuracy",
            "| Letter | Accuracy | | Letter | Accuracy |",
            "|---|---|---|---|---|",
        ])
        
        # Format per-letter accuracy in 2-column table
        letter_list = sorted(per_class_acc.items())
        for idx in range(0, len(letter_list), 2):
            l1, a1 = letter_list[idx]
            l2, a2 = letter_list[idx+1] if idx+1 < len(letter_list) else ("", 0.0)
            a2_str = f"{a2:.2%}" if l2 else ""
            report_lines.append(f"| **{l1}** | {a1:.2%} | | **{l2}** | {a2_str} |")
            
        report_lines.extend([
            "",
            "### Top Misclassifications"
        ])
        
        if mis_summary:
            report_lines.extend(mis_summary)
        else:
            report_lines.append("No misclassifications! Perfect 100% test score reached.")
            
        report_lines.append("")
        
        if feature_importance_lines:
            report_lines.extend(feature_importance_lines)
            
        report_content = "\n".join(report_lines)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_content)
        logger.info(f"Detailed training report written to: {report_path}")
