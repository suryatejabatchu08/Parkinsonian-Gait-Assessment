"""
report_generator.py — Stage 6: Automated Clinical PDF Report Generation

Uses Jinja2 for HTML templating and ReportLab (or WeasyPrint) for PDF conversion.
"""

import os
import base64
from datetime import datetime
from typing import Dict, List, Optional
from jinja2 import Environment, FileSystemLoader

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import REPORT_TEMPLATE_DIR, OUTPUT_DIR, SEVERITY_BINS
from scoring.pgsi_scorer import PGSIResult


class ReportGenerator:
    """Generates HTML and PDF clinical reports from PGSI results."""

    def __init__(self):
        self.jinja_env = Environment(
            loader=FileSystemLoader(REPORT_TEMPLATE_DIR),
            autoescape=True,
        )
        self.template = self.jinja_env.get_template("report_template.html")

    # ── clinical interpretation text ──────────────

    @staticmethod
    def generate_interpretation(result: PGSIResult) -> str:
        """Auto-generate clinical interpretation text based on PGSI result."""
        lines = []

        lines.append(
            f"The patient's overall Parkinsonian Gait Severity Index (PGSI) is "
            f"<strong>{result.pgsi_score:.1f}/100</strong>, classified as "
            f"<strong>{result.severity_label}</strong>."
        )

        # Identify most impaired feature
        worst = max(result.sub_scores, key=result.sub_scores.get)
        worst_score = result.sub_scores[worst]
        feature_descriptions = {
            "stride": "reduced stride length (festination pattern)",
            "posture": "increased forward trunk tilt (stooped posture)",
            "symmetry": "left-right gait asymmetry",
            "variability": "high step timing variability (irregular cadence)",
            "armswing": "reduced arm swing amplitude",
        }
        if worst_score > 50:
            lines.append(
                f"The most prominent impairment is {feature_descriptions.get(worst, worst)} "
                f"with a sub-score of {worst_score:.1f}/100."
            )

        # Fall risk
        if result.fall_risk:
            lines.append(
                "<strong>Elevated fall risk</strong> was detected based on postural instability "
                "and/or high step timing variability. Close monitoring is recommended."
            )

        # Severity-specific notes
        if result.severity_label == "Normal":
            lines.append("Gait parameters are within normal range. No Parkinsonian gait features detected.")
        elif result.severity_label == "Mild":
            lines.append(
                "Mild Parkinsonian gait features are present. Regular monitoring is recommended "
                "to track progression."
            )
        elif result.severity_label == "Moderate":
            lines.append(
                "Moderate Parkinsonian gait impairment detected. Therapeutic intervention and "
                "periodic reassessment are advised."
            )
        elif result.severity_label == "Severe":
            lines.append(
                "Severe gait impairment is present across multiple domains. Immediate clinical "
                "attention and fall prevention measures are strongly recommended."
            )

        return "<br><br>".join(lines)

    # ── HTML report ───────────────────────────────

    def render_html(
        self,
        result: PGSIResult,
        patient_id: str = "N/A",
        session_date: str = "",
        session_tag: str = "",
        video_filename: str = "",
        video_duration: str = "",
        skeleton_image_bytes: Optional[bytes] = None,
        longitudinal_data: Optional[List[Dict]] = None,
    ) -> str:
        """Render the HTML report from PGSI result."""

        if not session_date:
            session_date = datetime.now().strftime("%Y-%m-%d")

        feature_names = {
            "stride": "Stride Length",
            "posture": "Posture Angle",
            "symmetry": "Gait Symmetry",
            "variability": "Step Timing Variability",
            "armswing": "Arm Swing",
        }

        features_list = []
        for key in ["stride", "posture", "symmetry", "variability", "armswing"]:
            features_list.append({
                "name": feature_names.get(key, key),
                "raw_value": f"{result.raw_features.get(key, 0):.4f}",
                "sub_score": f"{result.sub_scores.get(key, 0):.1f}",
                "weight": f"{result.weights.get(key, 0):.2f}",
            })

        skeleton_b64 = ""
        if skeleton_image_bytes:
            skeleton_b64 = base64.b64encode(skeleton_image_bytes).decode("utf-8")

        html = self.template.render(
            patient_id=patient_id,
            session_date=session_date,
            session_tag=session_tag,
            video_filename=video_filename,
            video_duration=video_duration,
            analysis_timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            pgsi_score=f"{result.pgsi_score:.1f}",
            severity_label=result.severity_label,
            features=features_list,
            fall_risk=result.fall_risk,
            fall_risk_reasons=result.fall_risk_reasons,
            skeleton_image_b64=skeleton_b64,
            interpretation=self.generate_interpretation(result),
            longitudinal_data=longitudinal_data,
        )
        return html

    # ── PDF report ────────────────────────────────

    def generate_pdf(
        self,
        result: PGSIResult,
        output_path: str,
        **kwargs,
    ) -> str:
        """Generate a PDF report. Tries WeasyPrint first, falls back to simple HTML save."""
        html_content = self.render_html(result, **kwargs)

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        try:
            from weasyprint import HTML
            HTML(string=html_content).write_pdf(output_path)
            return output_path
        except (ImportError, OSError):
            pass

        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.pdfgen import canvas
            from reportlab.lib.units import inch

            c = canvas.Canvas(output_path, pagesize=A4)
            width, height = A4
            y = height - 50

            c.setFont("Helvetica-Bold", 18)
            c.drawCentredString(width / 2, y, "PGSI Clinical Assessment Report")
            y -= 40

            c.setFont("Helvetica", 11)
            lines = [
                f"Patient ID: {kwargs.get('patient_id', 'N/A')}",
                f"Session Date: {kwargs.get('session_date', '')}",
                f"Video: {kwargs.get('video_filename', '')}",
                "",
                f"PGSI Score: {result.pgsi_score:.1f} / 100",
                f"Severity: {result.severity_label}",
                f"Fall Risk: {'YES' if result.fall_risk else 'No'}",
                "",
                "Sub-Scores:",
            ]
            feature_names = {
                "stride": "Stride Length",
                "posture": "Posture Angle",
                "symmetry": "Gait Symmetry",
                "variability": "Step Timing Variability",
                "armswing": "Arm Swing",
            }
            for key in ["stride", "posture", "symmetry", "variability", "armswing"]:
                val = result.sub_scores.get(key, 0)
                lines.append(f"  {feature_names[key]}: {val:.1f} / 100  (weight: {result.weights.get(key, 0):.2f})")

            for line in lines:
                c.drawString(50, y, line)
                y -= 18
                if y < 50:
                    c.showPage()
                    y = height - 50

            c.save()
            return output_path
        except (ImportError, OSError):
            # Last resort: save as HTML
            html_path = output_path.replace(".pdf", ".html")
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(html_content)
            return html_path

    def generate_html_file(self, result: PGSIResult, output_path: str, **kwargs) -> str:
        """Save the rendered HTML report to a file."""
        html_content = self.render_html(result, **kwargs)
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        return output_path
