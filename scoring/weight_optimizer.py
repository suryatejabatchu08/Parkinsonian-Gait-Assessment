"""
weight_optimizer.py — Optimise PGSI weights against UPDRS labels.

1. Compute all 5 sub-scores on a labelled dataset.
2. Run Pearson correlation between each sub-score and UPDRS gait item.
3. Fit weights via LinearRegression to predict UPDRS total.
4. Validate with 5-fold cross-validation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SEVERITY_BINS


class WeightOptimizer:
    """Optimize PGSI weights against clinical UPDRS labels."""

    def __init__(self):
        self.weights: Dict[str, float] = {}
        self.correlations: Dict[str, Tuple[float, float]] = {}  # name → (r, p-value)
        self.model: LinearRegression = LinearRegression()

    def compute_correlations(
        self, sub_scores_df: pd.DataFrame, updrs_scores: np.ndarray
    ) -> Dict[str, Tuple[float, float]]:
        """Pearson correlation between each sub-score column and UPDRS labels."""
        self.correlations = {}
        for col in ["stride", "posture", "symmetry", "variability", "armswing"]:
            if col in sub_scores_df.columns:
                r, p = pearsonr(sub_scores_df[col].values, updrs_scores)
                self.correlations[col] = (float(r), float(p))
        return self.correlations

    def fit_weights(
        self, sub_scores_df: pd.DataFrame, updrs_scores: np.ndarray
    ) -> Dict[str, float]:
        """Fit linear regression weights: UPDRS ≈ Σ wᵢ · Sᵢ."""
        feature_cols = ["stride", "posture", "symmetry", "variability", "armswing"]
        X = sub_scores_df[feature_cols].values
        y = updrs_scores

        self.model.fit(X, y)
        raw_weights = np.abs(self.model.coef_)
        total = raw_weights.sum()
        if total > 0:
            normalized = raw_weights / total
        else:
            normalized = np.ones(5) / 5.0

        self.weights = dict(zip(feature_cols, normalized.tolist()))
        return self.weights

    def cross_validate(
        self, sub_scores_df: pd.DataFrame, updrs_scores: np.ndarray, n_folds: int = 5
    ) -> Dict[str, float]:
        """5-fold cross-validation of weight regression."""
        feature_cols = ["stride", "posture", "symmetry", "variability", "armswing"]
        X = sub_scores_df[feature_cols].values
        y = updrs_scores

        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        scores = cross_val_score(LinearRegression(), X, y, cv=kf, scoring="r2")

        return {
            "mean_r2": float(np.mean(scores)),
            "std_r2": float(np.std(scores)),
            "per_fold_r2": scores.tolist(),
        }


class SeverityClassifier:
    """Train and evaluate severity classification model."""

    def __init__(self, model_type: str = "svm"):
        self.scaler = StandardScaler()
        if model_type == "svm":
            self.model = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42)
        else:
            self.model = RandomForestClassifier(
                n_estimators=100, random_state=42
            )
        self.model_type = model_type

    def train(self, X: np.ndarray, y: np.ndarray):
        """Train the severity classifier.
        X: (n_samples, 6) — [pgsi_score, stride, posture, symmetry, variability, armswing]
        y: severity labels (0-3)"""
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict:
        y_pred = self.predict(X)
        return {
            "accuracy": float(accuracy_score(y, y_pred)),
            "f1_macro": float(f1_score(y, y_pred, average="macro", zero_division=0)),
            "classification_report": classification_report(
                y, y_pred, target_names=list(SEVERITY_BINS.keys()), zero_division=0
            ),
            "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
        }

    @staticmethod
    def severity_label_to_int(label: str) -> int:
        mapping = {"Normal": 0, "Mild": 1, "Moderate": 2, "Severe": 3}
        return mapping.get(label, -1)

    @staticmethod
    def int_to_severity_label(val: int) -> str:
        mapping = {0: "Normal", 1: "Mild", 2: "Moderate", 3: "Severe"}
        return mapping.get(val, "Unknown")
