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
            from reportlab.lib import colors
            from reportlab.lib.units import inch, cm
            from reportlab.lib.utils import ImageReader
            import io

            c = canvas.Canvas(output_path, pagesize=A4)
            width, height = A4
            
            # --- Fonts & Colors ---
            # Using standard fonts available everywhere
            FONT_TITLE = ("Helvetica-Bold", 20)
            FONT_HEADING = ("Helvetica-Bold", 14)
            FONT_SUBHEADING = ("Helvetica-Bold", 12)
            FONT_BODY = ("Helvetica", 11)
            FONT_BODY_BOLD = ("Helvetica-Bold", 11)
            FONT_SMALL = ("Helvetica", 9)
            
            COLOR_PRIMARY = colors.HexColor("#2C3E50")     # Dark Blue
            COLOR_SECONDARY = colors.HexColor("#34495E")   # Lighter Blue
            COLOR_ACCENT = colors.HexColor("#3498DB")      # Bright Blue
            COLOR_BG_GRAY = colors.HexColor("#F8F9F9")
            COLOR_TEXT = colors.HexColor("#212F3D")
            COLOR_TEXT_LIGHT = colors.HexColor("#7F8C8D")
            
            # Severity Colors
            sev_colors = {
                "Normal": colors.HexColor("#27AE60"),    # Green
                "Mild": colors.HexColor("#F1C40F"),      # Yellow
                "Moderate": colors.HexColor("#E67E22"),  # Orange
                "Severe": colors.HexColor("#E74C3C")     # Red
            }
            color_sev = sev_colors.get(result.severity_label, COLOR_TEXT)
            
            # Margins
            margin_x = 1 * inch
            y = height - 1 * inch
            
            # ==========================================
            # HEADER SECTION
            # ==========================================
            c.setFillColor(COLOR_PRIMARY)
            c.setFont(*FONT_TITLE)
            c.drawString(margin_x, y, "PGSI Clinical Assessment Report")
            
            c.setFillColor(COLOR_TEXT_LIGHT)
            c.setFont(*FONT_SMALL)
            c.drawRightString(width - margin_x, y, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
            
            y -= 0.4 * inch
            c.setStrokeColor(COLOR_PRIMARY)
            c.setLineWidth(2)
            c.line(margin_x, y, width - margin_x, y)
            
            y -= 0.6 * inch
            
            # ==========================================
            # PATIENT & SESSION DETAILS
            # ==========================================
            # Draw an elegant box for patient info
            box_height = 1.2 * inch
            c.setFillColor(COLOR_BG_GRAY)
            c.setStrokeColor(colors.lightgrey)
            c.setLineWidth(0.5)
            c.roundRect(margin_x, y - box_height + 0.2*inch, width - 2*margin_x, box_height, radius=6, fill=1, stroke=1)
            
            c.setFillColor(COLOR_TEXT)
            c.setFont(*FONT_BODY_BOLD)
            c.drawString(margin_x + 0.2*inch, y, "Patient Information & Session Profile")
            y -= 0.3 * inch
            
            c.setFont(*FONT_BODY)
            col1_x = margin_x + 0.3*inch
            col2_x = width/2 + 0.1*inch
            
            # Row 1
            c.setFillColor(COLOR_TEXT_LIGHT)
            c.drawString(col1_x, y, "Patient ID:")
            c.setFillColor(COLOR_TEXT)
            c.setFont(*FONT_BODY_BOLD)
            c.drawString(col1_x + 1.2*inch, y, str(kwargs.get('patient_id', 'N/A')))
            
            c.setFont(*FONT_BODY)
            c.setFillColor(COLOR_TEXT_LIGHT)
            c.drawString(col2_x, y, "Session Date:")
            c.setFillColor(COLOR_TEXT)
            c.setFont(*FONT_BODY_BOLD)
            c.drawString(col2_x + 1.2*inch, y, str(kwargs.get('session_date', 'N/A')))
            
            y -= 0.25 * inch
            
            # Row 2
            c.setFont(*FONT_BODY)
            c.setFillColor(COLOR_TEXT_LIGHT)
            c.drawString(col1_x, y, "Video Source:")
            c.setFillColor(COLOR_TEXT)
            c.drawString(col1_x + 1.2*inch, y, str(kwargs.get('video_filename', 'N/A')))
            
            c.setFillColor(COLOR_TEXT_LIGHT)
            c.drawString(col2_x, y, "Video Duration:")
            c.setFillColor(COLOR_TEXT)
            c.drawString(col2_x + 1.2*inch, y, str(kwargs.get('video_duration', 'N/A')))
            
            y -= 0.8 * inch
            
            # ==========================================
            # OVERALL SCORE HIGHLIGHT
            # ==========================================
            c.setFillColor(color_sev)
            # Big filled box for the score
            c.roundRect(margin_x, y - 0.8*inch, 2.5*inch, 1*inch, radius=6, fill=1, stroke=0)
            
            # Text inside box
            c.setFillColor(colors.white)
            c.setFont(*FONT_SUBHEADING)
            c.drawCentredString(margin_x + 1.25*inch, y - 0.2*inch, "PGSI SCORE")
            
            c.setFont("Helvetica-Bold", 24)
            c.drawCentredString(margin_x + 1.25*inch, y - 0.6*inch, f"{result.pgsi_score:.1f} / 100")
            
            # Severity text next to box
            c.setFillColor(COLOR_TEXT)
            c.setFont(*FONT_HEADING)
            c.drawString(margin_x + 2.8*inch, y - 0.2*inch, f"Severity: {result.severity_label}")
            
            # Fall risk badge
            y -= 0.5 * inch
            if result.fall_risk:
                c.setFillColor(colors.HexColor("#E74C3C"))  # Red text
                c.setFont(*FONT_BODY_BOLD)
                c.drawString(margin_x + 2.8*inch, y, "⚠️ ELEVATED FALL RISK DETECTED")
                c.setFont(*FONT_BODY)
                c.setFillColor(COLOR_TEXT)
                c.drawString(margin_x + 2.8*inch, y - 0.2*inch, "Close monitoring recommended due to postural/timing issues.")
            else:
                c.setFillColor(colors.HexColor("#27AE60"))  # Green text
                c.setFont(*FONT_BODY_BOLD)
                c.drawString(margin_x + 2.8*inch, y, "✓ No Elevated Fall Risk")
            
            y -= 0.9 * inch
            
            # ==========================================
            # BIOMECHANICAL SUB-SCORES
            # ==========================================
            c.setFillColor(COLOR_PRIMARY)
            c.setFont(*FONT_HEADING)
            c.drawString(margin_x, y, "Biomechanical Sub-Scores (0 = Healthy, 100 = Impaired)")
            y -= 0.1 * inch
            c.setStrokeColor(colors.lightgrey)
            c.setLineWidth(1)
            c.line(margin_x, y, width - margin_x, y)
            
            y -= 0.4 * inch
            
            feature_names = {
                "stride": "Stride Length",
                "posture": "Posture Angle",
                "symmetry": "Gait Symmetry",
                "variability": "Step Timing Variability",
                "armswing": "Arm Swing Amplitude",
            }
            
            # Draw rows with progress bars
            bar_width = 3.0 * inch
            bar_height = 0.15 * inch
            
            for key in ["stride", "posture", "symmetry", "variability", "armswing"]:
                val = result.sub_scores.get(key, 0)
                raw_val = result.raw_features.get(key, 0)
                
                # Feature Name
                c.setFillColor(COLOR_TEXT)
                c.setFont(*FONT_BODY_BOLD)
                c.drawString(margin_x, y, feature_names.get(key, key))
                
                # Raw Value underneath
                c.setFont(*FONT_SMALL)
                c.setFillColor(COLOR_TEXT_LIGHT)
                c.drawString(margin_x, y - 0.15*inch, f"Raw measurement: {raw_val:.2f}")
                
                # Progress Bar Background (Grey)
                c.setFillColor(colors.HexColor("#EBEDEF"))
                c.roundRect(margin_x + 2.0*inch, y - 0.05*inch, bar_width, bar_height, radius=3, fill=1, stroke=0)
                
                # Progress Bar Foreground (Color based on score)
                fill_width = (val / 100.0) * bar_width
                if val <= 25: bar_color = sev_colors["Normal"]
                elif val <= 50: bar_color = sev_colors["Mild"]
                elif val <= 75: bar_color = sev_colors["Moderate"]
                else: bar_color = sev_colors["Severe"]
                
                c.setFillColor(bar_color)
                if fill_width > 0:
                    c.roundRect(margin_x + 2.0*inch, y - 0.05*inch, fill_width, bar_height, radius=3, fill=1, stroke=0)
                
                # Score Text
                c.setFillColor(COLOR_TEXT)
                c.setFont(*FONT_BODY_BOLD)
                c.drawString(margin_x + 2.1*inch + bar_width, y, f"{val:.1f}")
                
                y -= 0.5 * inch
                
            y -= 0.2 * inch
            
            # ==========================================
            # CLINICAL INTERPRETATION & OVERLAY
            # ==========================================
            # Left column: Interpretation text
            # Right column: Skeleton Overlay
            
            c.setFillColor(COLOR_PRIMARY)
            c.setFont(*FONT_HEADING)
            c.drawString(margin_x, y, "Clinical Interpretation")
            y -= 0.1 * inch
            c.setStrokeColor(colors.lightgrey)
            c.setLineWidth(1)
            c.line(margin_x, y, width - margin_x, y)
            y -= 0.3 * inch
            
            # Break interpretation into wrapped lines
            import textwrap
            interp_text = self.generate_interpretation(result).replace("<strong>", "").replace("</strong>", "").split("<br><br>")
            
            c.setFillColor(COLOR_TEXT)
            c.setFont(*FONT_BODY)
            text_y = y
            text_width_chars = 60 # approx for half page
            
            for para in interp_text:
                wrapped_lines = textwrap.wrap(para, width=text_width_chars)
                for line in wrapped_lines:
                    c.drawString(margin_x, text_y, line)
                    text_y -= 0.2 * inch
                text_y -= 0.1 * inch # spacing between paragraphs
            
            # Draw Skeleton Image if available
            if kwargs.get('skeleton_image_bytes'):
                try:
                    img_data = io.BytesIO(kwargs['skeleton_image_bytes'])
                    img = ImageReader(img_data)
                    
                    # Calculate aspect ratio
                    img_w, img_h = img.getSize()
                    aspect = img_h / float(img_w)
                    
                    draw_w = 3.0 * inch
                    draw_h = draw_w * aspect
                    
                    # Position on the right side
                    img_x = width - margin_x - draw_w
                    img_y = y - draw_h + 0.1*inch
                    
                    # Draw subtle border
                    c.setStrokeColor(colors.lightgrey)
                    c.setLineWidth(0.5)
                    c.rect(img_x - 2, img_y - 2, draw_w + 4, draw_h + 4)
                    
                    c.drawImage(img, img_x, img_y, width=draw_w, height=draw_h)
                except Exception as e:
                    print(f"[PDF] Error drawing skeleton image: {e}")
            
            # ==========================================
            # FOOTER
            # ==========================================
            c.setFillColor(COLOR_TEXT_LIGHT)
            c.setFont(*FONT_SMALL)
            c.drawCentredString(width/2, 0.5*inch, "Automated assessment by Parkinsonian Gait Severity Index (PGSI) software. Not for primary diagnosis.")
            
            c.save()
            return output_path
        except Exception as e:
            print(f"[PDF] ReportLab generation failed: {e}. Falling back to HTML.")
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
