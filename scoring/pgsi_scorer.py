"""
pgsi_scorer.py — Stage 4: PGSI Computation & Severity Classification

Computes the Parkinsonian Gait Severity Index:
    PGSI = w₁·S_stride + w₂·S_posture + w₃·S_variability

Only 3 features are used — symmetry and arm swing are unreliable from
sagittal monocular video and are excluded from the composite score.

Each sub-score is normalised to [0, 100] where 100 = most impaired.
"""

import json
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    PGSI_WEIGHTS,
    PGSI_HEALTHY_REF,
    PGSI_IMPAIRED_REF,
    PGSI_HIGHER_IS_WORSE,
    SEVERITY_BINS,
    FALL_RISK_POSTURE_THRESHOLD,
    FALL_RISK_VARIABILITY_THRESHOLD,
)
from features.gait_features import GaitFeatures


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
    def _normalize(value: float, feature: str) -> float:
        """Map raw feature value to 0-100 sub-score.
        0 = healthy, 100 = maximally impaired."""
        h = PGSI_HEALTHY_REF[feature]
        i = PGSI_IMPAIRED_REF[feature]
        if PGSI_HIGHER_IS_WORSE[feature]:
            raw = (value - h) / (i - h) * 100.0
        else:
            raw = (h - value) / (h - i) * 100.0
        return float(np.clip(raw, 0.0, 100.0))

    def compute_sub_scores(self, features: GaitFeatures) -> Dict[str, float]:
        """Normalize each feature to a [0, 100] sub-score."""
        raw = self._extract_raw_values(features)

        sub_scores = {}
        # Only normalize the 3 active features
        for key in PGSI_WEIGHTS:
            sub_scores[key] = self._normalize(raw[key], key)

        # Keep symmetry and armswing in dict for display (dashboard radar chart)
        # but set to 0.0 since they are excluded from the PGSI composite
        sub_scores["symmetry"] = 0.0
        sub_scores["armswing"] = 0.0

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
        """PGSI = Σ wᵢ · Sᵢ (only 3 active features)"""
        pgsi = sum(
            self.weights[key] * sub_scores[key] for key in self.weights
        )
        return float(np.clip(pgsi, 0, 100))

    # ── severity classification ───────────────────

    @staticmethod
    def classify_severity(pgsi: float) -> str:
        if pgsi <= 33:
            return "Normal"
        elif pgsi <= 58:
            return "Mild"
        elif pgsi <= 78:
            return "Moderate"
        else:
            return "Severe"

    # ── fall risk ─────────────────────────────────

    @staticmethod
    def assess_fall_risk(
        sub_scores: Dict[str, float],
        severity: str = "Unknown",
    ) -> Tuple[bool, List[str]]:
        """Rule-based fall risk flag.

        Only triggers for Moderate/Severe severity to avoid
        false positives on Mild cases with noisy sub-scores.
        """
        # Normal and Mild patients should not be flagged for fall risk
        if severity in ("Normal", "Mild"):
            return False, []

        reasons: List[str] = []
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
        fall_risk, fall_reasons = self.assess_fall_risk(sub_scores, severity)

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
