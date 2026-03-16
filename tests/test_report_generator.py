"""
test_report_generator.py — Unit tests for the report generation module.
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reporting.report_generator import ReportGenerator
from scoring.pgsi_scorer import PGSIResult


# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────

def _make_result(pgsi=45.0, severity="Mild", fall_risk=False) -> PGSIResult:
    """Create a PGSIResult for testing."""
    return PGSIResult(
        sub_scores={"stride": 30.0, "posture": 50.0, "symmetry": 40.0, "variability": 55.0, "armswing": 45.0},
        weights={"stride": 0.20, "posture": 0.20, "symmetry": 0.20, "variability": 0.20, "armswing": 0.20},
        pgsi_score=pgsi,
        severity_label=severity,
        fall_risk=fall_risk,
        fall_risk_reasons=["Posture sub-score (50.0) exceeds threshold (60)"] if fall_risk else [],
        raw_features={"stride": 0.65, "posture": 22.5, "symmetry": 20.0, "variability": 16.5, "armswing": 22.5},
    )


# ─────────────────────────────────────────────
# Tests: Clinical Interpretation
# ─────────────────────────────────────────────

class TestClinicalInterpretation:

    def test_interpretation_not_empty(self):
        result = _make_result()
        text = ReportGenerator.generate_interpretation(result)
        assert len(text) > 0

    def test_interpretation_contains_pgsi_score(self):
        result = _make_result(pgsi=45.0)
        text = ReportGenerator.generate_interpretation(result)
        assert "45.0" in text

    def test_interpretation_contains_severity(self):
        result = _make_result(severity="Mild")
        text = ReportGenerator.generate_interpretation(result)
        assert "Mild" in text

    @pytest.mark.parametrize("severity,expected_text", [
        ("Normal", "within normal range"),
        ("Mild", "Mild Parkinsonian"),
        ("Moderate", "Moderate Parkinsonian"),
        ("Severe", "Severe gait impairment"),
    ])
    def test_severity_specific_text(self, severity, expected_text):
        result = _make_result(severity=severity)
        text = ReportGenerator.generate_interpretation(result)
        assert expected_text in text

    def test_fall_risk_warning(self):
        result = _make_result(fall_risk=True)
        text = ReportGenerator.generate_interpretation(result)
        assert "fall risk" in text.lower()

    def test_no_fall_risk_no_warning(self):
        result = _make_result(fall_risk=False)
        text = ReportGenerator.generate_interpretation(result)
        assert "Elevated fall risk" not in text

    def test_worst_feature_highlighted(self):
        """Should mention the most impaired feature when > 50."""
        result = _make_result()
        # variability is 55.0 (highest sub-score > 50)
        text = ReportGenerator.generate_interpretation(result)
        assert "variability" in text.lower() or "irregular" in text.lower()


# ─────────────────────────────────────────────
# Tests: HTML Report Rendering
# ─────────────────────────────────────────────

class TestHTMLRendering:

    def test_render_html_returns_string(self):
        report_gen = ReportGenerator()
        result = _make_result()
        html = report_gen.render_html(result, patient_id="TEST-001")
        assert isinstance(html, str)
        assert len(html) > 0

    def test_html_contains_patient_id(self):
        report_gen = ReportGenerator()
        result = _make_result()
        html = report_gen.render_html(result, patient_id="PAT-42")
        assert "PAT-42" in html

    def test_html_contains_pgsi_score(self):
        report_gen = ReportGenerator()
        result = _make_result(pgsi=72.5)
        html = report_gen.render_html(result)
        assert "72.5" in html

    def test_html_contains_severity_label(self):
        report_gen = ReportGenerator()
        result = _make_result(severity="Moderate")
        html = report_gen.render_html(result)
        assert "Moderate" in html

    def test_html_contains_all_features(self):
        report_gen = ReportGenerator()
        result = _make_result()
        html = report_gen.render_html(result)
        for name in ["Stride Length", "Posture Angle", "Gait Symmetry", "Step Timing Variability", "Arm Swing"]:
            assert name in html, f"Expected '{name}' in HTML report"

    def test_html_contains_skeleton_when_provided(self):
        report_gen = ReportGenerator()
        result = _make_result()
        fake_png = b"\x89PNG\r\n\x1a\nfake_image_data"
        html = report_gen.render_html(result, skeleton_image_bytes=fake_png)
        assert "data:image/png;base64," in html

    def test_html_no_skeleton_when_not_provided(self):
        report_gen = ReportGenerator()
        result = _make_result()
        html = report_gen.render_html(result)
        assert "data:image/png;base64," not in html

    def test_html_with_longitudinal_data(self):
        report_gen = ReportGenerator()
        result = _make_result()
        longitudinal = [
            {"label": "Session 1", "date": "2026-01-01", "pgsi": 40.0, "severity": "Mild"},
            {"label": "Session 2", "date": "2026-02-01", "pgsi": 35.0, "severity": "Mild"},
        ]
        html = report_gen.render_html(result, longitudinal_data=longitudinal)
        assert "Session 1" in html
        assert "Session 2" in html


# ─────────────────────────────────────────────
# Tests: File Output
# ─────────────────────────────────────────────

class TestFileOutput:

    def test_generate_html_file(self, tmp_path):
        report_gen = ReportGenerator()
        result = _make_result()
        html_path = str(tmp_path / "test_report.html")
        output_path = report_gen.generate_html_file(result, html_path, patient_id="TEST")
        assert os.path.isfile(output_path)
        with open(output_path, "r", encoding="utf-8") as f:
            content = f.read()
        assert "PGSI" in content

    def test_generate_pdf_fallback(self, tmp_path):
        """generate_pdf should produce a file (PDF or HTML fallback)."""
        report_gen = ReportGenerator()
        result = _make_result()
        pdf_path = str(tmp_path / "test_report.pdf")
        output_path = report_gen.generate_pdf(result, pdf_path, patient_id="TEST")
        assert os.path.isfile(output_path)
