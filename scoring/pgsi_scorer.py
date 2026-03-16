"""
pgsi_scorer.py — Stage 4: PGSI Computation & Severity Classification

Computes the Parkinsonian Gait Severity Index:
    PGSI = w₁·S_stride + w₂·S_posture + w₃·S_symmetry + w₄·S_variability + w₅·S_armswing

Each sub-score is normalised to [0, 100] where 100 = most impaired.
"""

import json
import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PGSI_WEIGHTS, SEVERITY_BINS, FALL_RISK_POSTURE_THRESHOLD, FALL_RISK_VARIABILITY_THRESHOLD
from features.gait_features import GaitFeatures


# ─────────────────────────────────────────────────
# Reference ranges for min-max normalization
# (derived from literature / healthy baselines)
# ─────────────────────────────────────────────────
REFERENCE_RANGES = {
    "stride": {
        "healthy_min": 1.0,    # normalised stride length (good)
        "impaired_max": 0.2,   # short festinating strides (bad)
    },
    "posture": {
        "healthy_min": 0.0,    # upright (degrees)
        "impaired_max": 45.0,  # severe camptocormia (degrees)
    },
    "symmetry": {
        "healthy_min": 0.0,    # perfect symmetry (SI %)
        "impaired_max": 50.0,  # severe asymmetry
    },
    "variability": {
        "healthy_min": 0.0,    # no variability (CV %)
        "impaired_max": 30.0,  # high variability
    },
    "armswing": {
        "healthy_min": 40.0,   # full swing (degrees)
        "impaired_max": 5.0,   # almost no swing
    },
}


@dataclass
class PGSIResult:
    """Full PGSI assessment output."""
    sub_scores: Dict[str, float]      # each in [0, 100]
    weights: Dict[str, float]
    pgsi_score: float                  # weighted composite [0, 100]
    severity_label: str                # Normal / Mild / Moderate / Severe
    fall_risk: bool
    fall_risk_reasons: list
    raw_features: Dict[str, float]    # original feature values before normalization


class PGSIScorer:
    """Computes PGSI score from extracted gait features."""

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or dict(PGSI_WEIGHTS)
        # Ensure weights sum to 1.0
        total = sum(self.weights.values())
        if abs(total - 1.0) > 1e-6:
            self.weights = {k: v / total for k, v in self.weights.items()}

    # ── normalization ─────────────────────────────

    @staticmethod
    def _normalize_sub_score(
        value: float, healthy_val: float, impaired_val: float
    ) -> float:
        """Min-max normalize a raw feature to [0, 100].
        0 = healthy, 100 = most impaired.
        Handles both increasing and decreasing features."""
        if healthy_val == impaired_val:
            return 0.0

        # Determine direction
        if healthy_val < impaired_val:
            # Higher value = more impaired (e.g., posture angle, symmetry, variability)
            score = (value - healthy_val) / (impaired_val - healthy_val) * 100
        else:
            # Lower value = more impaired (e.g., stride length, arm swing)
            score = (healthy_val - value) / (healthy_val - impaired_val) * 100

        return float(np.clip(score, 0, 100))

    def compute_sub_scores(self, features: GaitFeatures) -> Dict[str, float]:
        """Normalize each feature to a [0, 100] sub-score."""
        raw = self._extract_raw_values(features)

        sub_scores = {}
        for key, ref in REFERENCE_RANGES.items():
            sub_scores[key] = self._normalize_sub_score(
                raw[key], ref["healthy_min"], ref["impaired_max"]
            )
        return sub_scores

    @staticmethod
    def _extract_raw_values(features: GaitFeatures) -> Dict[str, float]:
        return {
            "stride": features.mean_stride_length,
            "posture": features.mean_posture_angle,
            "symmetry": features.mean_symmetry_index,
            "variability": features.step_timing_cv,
            "armswing": features.mean_arm_swing,
        }

    # ── PGSI formula ──────────────────────────────

    def compute_pgsi(self, sub_scores: Dict[str, float]) -> float:
        """PGSI = Σ wᵢ · Sᵢ"""
        pgsi = sum(
            self.weights[key] * sub_scores[key] for key in self.weights
        )
        return float(np.clip(pgsi, 0, 100))

    # ── severity classification ───────────────────

    @staticmethod
    def classify_severity(pgsi: float) -> str:
        for label, (lo, hi) in SEVERITY_BINS.items():
            if lo <= pgsi <= hi:
                return label
        return "Severe"  # fallback if score > 100 somehow

    # ── fall risk ─────────────────────────────────

    @staticmethod
    def assess_fall_risk(sub_scores: Dict[str, float]) -> Tuple[bool, list]:
        reasons = []
        if sub_scores.get("posture", 0) >= FALL_RISK_POSTURE_THRESHOLD:
            reasons.append(
                f"Posture sub-score ({sub_scores['posture']:.1f}) exceeds threshold "
                f"({FALL_RISK_POSTURE_THRESHOLD})"
            )
        if sub_scores.get("variability", 0) >= FALL_RISK_VARIABILITY_THRESHOLD:
            reasons.append(
                f"Variability sub-score ({sub_scores['variability']:.1f}) exceeds threshold "
                f"({FALL_RISK_VARIABILITY_THRESHOLD})"
            )
        return len(reasons) > 0, reasons

    # ── full assessment ───────────────────────────

    def assess(self, features: GaitFeatures) -> PGSIResult:
        """Run full PGSI assessment from extracted features."""
        raw = self._extract_raw_values(features)
        sub_scores = self.compute_sub_scores(features)
        pgsi = self.compute_pgsi(sub_scores)
        severity = self.classify_severity(pgsi)
        fall_risk, fall_reasons = self.assess_fall_risk(sub_scores)

        return PGSIResult(
            sub_scores=sub_scores,
            weights=dict(self.weights),
            pgsi_score=pgsi,
            severity_label=severity,
            fall_risk=fall_risk,
            fall_risk_reasons=fall_reasons,
            raw_features=raw,
        )

    # ── weight I/O ────────────────────────────────

    def save_weights(self, path: str):
        with open(path, "w") as f:
            json.dump(self.weights, f, indent=2)

    def load_weights(self, path: str):
        with open(path, "r") as f:
            self.weights = json.load(f)
        total = sum(self.weights.values())
        if abs(total - 1.0) > 1e-6:
            self.weights = {k: v / total for k, v in self.weights.items()}
