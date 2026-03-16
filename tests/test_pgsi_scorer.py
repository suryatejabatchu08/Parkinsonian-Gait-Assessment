"""
test_pgsi_scorer.py — Unit tests for the PGSI scoring module.
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scoring.pgsi_scorer import PGSIScorer, PGSIResult, REFERENCE_RANGES
from features.gait_features import GaitFeatures, GaitCycleInfo


# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────

def _make_features(
    stride=0.6,
    posture=15.0,
    symmetry=10.0,
    variability=8.0,
    armswing=25.0,
) -> GaitFeatures:
    """Create a GaitFeatures instance with specified mean values."""
    return GaitFeatures(
        stride_lengths=np.array([stride]),
        mean_stride_length=stride,
        posture_angles=np.array([posture]),
        mean_posture_angle=posture,
        symmetry_index_step_length=symmetry,
        symmetry_index_step_time=symmetry,
        mean_symmetry_index=symmetry,
        step_timing_cv=variability,
        arm_swing_left=armswing,
        arm_swing_right=armswing,
        mean_arm_swing=armswing,
        gait_cycles=GaitCycleInfo(
            left_heel_strikes=np.array([0, 30, 60]),
            right_heel_strikes=np.array([15, 45, 75]),
        ),
    )


# ─────────────────────────────────────────────
# Tests: Sub-Score Normalization
# ─────────────────────────────────────────────

class TestSubScoreNormalization:
    """Test that sub-scores normalize correctly to [0, 100]."""

    def test_healthy_person_scores_low(self):
        """Healthy feature values → sub-scores near 0."""
        features = _make_features(
            stride=1.0,       # full stride (healthy)
            posture=0.0,      # upright (healthy)
            symmetry=0.0,     # perfect symmetry
            variability=0.0,  # no variability
            armswing=40.0,    # full swing (healthy)
        )
        scorer = PGSIScorer()
        sub = scorer.compute_sub_scores(features)
        for key, val in sub.items():
            assert val == pytest.approx(0.0, abs=1.0), f"{key} should be ~0 for healthy"

    def test_impaired_person_scores_high(self):
        """Severely impaired feature values → sub-scores near 100."""
        features = _make_features(
            stride=0.2,       # festinating (impaired)
            posture=45.0,     # severe tilt
            symmetry=50.0,    # severe asymmetry
            variability=30.0, # high variability
            armswing=5.0,     # barely any swing
        )
        scorer = PGSIScorer()
        sub = scorer.compute_sub_scores(features)
        for key, val in sub.items():
            assert val == pytest.approx(100.0, abs=1.0), f"{key} should be ~100 for impaired"

    def test_sub_scores_clamped(self):
        """Values beyond reference range should be clamped to [0, 100]."""
        features = _make_features(stride=0.0, posture=90.0, symmetry=100.0, variability=60.0, armswing=0.0)
        scorer = PGSIScorer()
        sub = scorer.compute_sub_scores(features)
        for key, val in sub.items():
            assert 0.0 <= val <= 100.0, f"{key} = {val}, should be in [0, 100]"


# ─────────────────────────────────────────────
# Tests: PGSI Computation
# ─────────────────────────────────────────────

class TestPGSIComputation:

    def test_pgsi_in_range(self):
        features = _make_features()
        scorer = PGSIScorer()
        result = scorer.assess(features)
        assert 0.0 <= result.pgsi_score <= 100.0

    def test_pgsi_weights_sum_to_one(self):
        scorer = PGSIScorer()
        assert sum(scorer.weights.values()) == pytest.approx(1.0, abs=1e-6)

    def test_pgsi_with_equal_weights(self):
        """With equal weights and all sub-scores = 50, PGSI should be 50."""
        features = _make_features()
        scorer = PGSIScorer()
        # Manually set all sub-scores to 50
        sub = {k: 50.0 for k in scorer.weights}
        pgsi = scorer.compute_pgsi(sub)
        assert pgsi == pytest.approx(50.0, abs=0.1)

    def test_custom_weights(self):
        """Custom weights should be normalized and used."""
        scorer = PGSIScorer(weights={"stride": 1, "posture": 0, "symmetry": 0, "variability": 0, "armswing": 0})
        assert scorer.weights["stride"] == pytest.approx(1.0)
        assert scorer.weights["posture"] == pytest.approx(0.0)


# ─────────────────────────────────────────────
# Tests: Severity Classification
# ─────────────────────────────────────────────

class TestSeverityClassification:

    @pytest.mark.parametrize("score,expected", [
        (0, "Normal"),
        (10, "Normal"),
        (25, "Normal"),
        (26, "Mild"),
        (50, "Mild"),
        (51, "Moderate"),
        (75, "Moderate"),
        (76, "Severe"),
        (100, "Severe"),
    ])
    def test_severity_bins(self, score, expected):
        assert PGSIScorer.classify_severity(score) == expected


# ─────────────────────────────────────────────
# Tests: Fall Risk
# ─────────────────────────────────────────────

class TestFallRisk:

    def test_no_fall_risk(self):
        sub = {"posture": 30, "variability": 30, "stride": 30, "symmetry": 30, "armswing": 30}
        risk, reasons = PGSIScorer.assess_fall_risk(sub)
        assert risk is False
        assert len(reasons) == 0

    def test_fall_risk_posture(self):
        sub = {"posture": 70, "variability": 30, "stride": 30, "symmetry": 30, "armswing": 30}
        risk, reasons = PGSIScorer.assess_fall_risk(sub)
        assert risk is True
        assert any("Posture" in r for r in reasons)

    def test_fall_risk_variability(self):
        sub = {"posture": 30, "variability": 70, "stride": 30, "symmetry": 30, "armswing": 30}
        risk, reasons = PGSIScorer.assess_fall_risk(sub)
        assert risk is True
        assert any("Variability" in r for r in reasons)


# ─────────────────────────────────────────────
# Tests: Full Assessment Pipeline
# ─────────────────────────────────────────────

class TestFullAssessment:

    def test_assess_returns_pgsi_result(self):
        features = _make_features()
        scorer = PGSIScorer()
        result = scorer.assess(features)
        assert isinstance(result, PGSIResult)
        assert isinstance(result.sub_scores, dict)
        assert isinstance(result.severity_label, str)
        assert isinstance(result.fall_risk, bool)

    def test_assess_consistency(self):
        """Same input should produce same output."""
        features = _make_features(stride=0.5, posture=20.0, symmetry=15.0, variability=10.0, armswing=20.0)
        scorer = PGSIScorer()
        r1 = scorer.assess(features)
        r2 = scorer.assess(features)
        assert r1.pgsi_score == r2.pgsi_score
        assert r1.severity_label == r2.severity_label
