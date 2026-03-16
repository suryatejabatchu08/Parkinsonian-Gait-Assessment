"""
charts.py — Visualization helpers for the PGSI system.

Provides Plotly interactive charts and Matplotlib static figures for:
  • PGSI radar plot (sub-score breakdown)
  • Severity gauge
  • Feature waveforms (gait cycle)
  • Pose skeleton overlay
  • Longitudinal PGSI trend
"""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import List, Dict, Optional
from io import BytesIO
import base64

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SEVERITY_BINS


# ═══════════════════════════════════════════════════
# 1. PGSI Radar Plot (Sub-Score Breakdown)
# ═══════════════════════════════════════════════════

def create_radar_chart(sub_scores: Dict[str, float], title: str = "PGSI Sub-Score Breakdown") -> go.Figure:
    """Create a radar (spider) chart showing all 5 PGSI sub-scores."""
    categories = list(sub_scores.keys())
    values = list(sub_scores.values())
    # Close the polygon
    categories += [categories[0]]
    values += [values[0]]

    fig = go.Figure(
        data=[
            go.Scatterpolar(
                r=values,
                theta=[c.replace("_", " ").title() for c in categories],
                fill="toself",
                name="Sub-Scores",
                line_color="#EF553B",
                fillcolor="rgba(239, 85, 59, 0.3)",
            )
        ]
    )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=False,
        title=dict(text=title, x=0.5),
        height=450,
        margin=dict(l=60, r=60, t=60, b=60),
    )
    return fig


# ═══════════════════════════════════════════════════
# 2. Severity Gauge
# ═══════════════════════════════════════════════════

def create_severity_gauge(pgsi_score: float, severity_label: str) -> go.Figure:
    """Create a gauge chart showing PGSI score and severity band."""
    color_map = {
        "Normal": "#2ECC71",
        "Mild": "#F1C40F",
        "Moderate": "#E67E22",
        "Severe": "#E74C3C",
    }

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=pgsi_score,
            title={"text": f"PGSI Score — {severity_label}"},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": color_map.get(severity_label, "#3498DB")},
                "steps": [
                    {"range": [0, 25], "color": "#D5F5E3"},
                    {"range": [25, 50], "color": "#FCF3CF"},
                    {"range": [50, 75], "color": "#FDEBD0"},
                    {"range": [75, 100], "color": "#FADBD8"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 4},
                    "thickness": 0.8,
                    "value": pgsi_score,
                },
            },
        )
    )
    fig.update_layout(height=350, margin=dict(l=30, r=30, t=60, b=30))
    return fig


# ═══════════════════════════════════════════════════
# 3. Sub-Score Bar Chart
# ═══════════════════════════════════════════════════

def create_subscore_bar_chart(sub_scores: Dict[str, float]) -> go.Figure:
    """Horizontal bar chart of sub-scores with severity coloring."""
    names = [k.replace("_", " ").title() for k in sub_scores.keys()]
    values = list(sub_scores.values())

    colors = []
    for v in values:
        if v <= 25:
            colors.append("#2ECC71")
        elif v <= 50:
            colors.append("#F1C40F")
        elif v <= 75:
            colors.append("#E67E22")
        else:
            colors.append("#E74C3C")

    fig = go.Figure(
        go.Bar(
            x=values,
            y=names,
            orientation="h",
            marker_color=colors,
            text=[f"{v:.1f}" for v in values],
            textposition="auto",
        )
    )
    fig.update_layout(
        title="Sub-Score Breakdown",
        xaxis=dict(title="Score (0–100)", range=[0, 100]),
        yaxis=dict(autorange="reversed"),
        height=350,
        margin=dict(l=120, r=30, t=50, b=40),
    )
    return fig


# ═══════════════════════════════════════════════════
# 4. Gait Cycle Waveform Plot
# ═══════════════════════════════════════════════════

def create_waveform_plot(
    signal: np.ndarray,
    fps: float = 30.0,
    title: str = "Gait Feature Waveform",
    ylabel: str = "Value",
    heel_strikes: Optional[np.ndarray] = None,
) -> go.Figure:
    """Time-domain waveform plot for a gait feature signal."""
    time = np.arange(len(signal)) / fps

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=time, y=signal, mode="lines", name=ylabel, line_color="#3498DB")
    )

    if heel_strikes is not None and len(heel_strikes) > 0:
        hs_time = heel_strikes / fps
        hs_vals = signal[heel_strikes] if max(heel_strikes) < len(signal) else []
        if len(hs_vals) > 0:
            fig.add_trace(
                go.Scatter(
                    x=hs_time,
                    y=hs_vals,
                    mode="markers",
                    name="Heel Strikes",
                    marker=dict(color="red", size=8, symbol="triangle-down"),
                )
            )

    fig.update_layout(
        title=title,
        xaxis_title="Time (s)",
        yaxis_title=ylabel,
        height=350,
        margin=dict(l=60, r=30, t=50, b=40),
    )
    return fig


# ═══════════════════════════════════════════════════
# 5. Longitudinal PGSI Trend
# ═══════════════════════════════════════════════════

def create_longitudinal_chart(
    session_labels: List[str],
    pgsi_scores: List[float],
    title: str = "PGSI Longitudinal Trend",
) -> go.Figure:
    """Line chart showing PGSI progression across sessions."""
    colors = []
    for s in pgsi_scores:
        if s <= 25:
            colors.append("#2ECC71")
        elif s <= 50:
            colors.append("#F1C40F")
        elif s <= 75:
            colors.append("#E67E22")
        else:
            colors.append("#E74C3C")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=session_labels,
            y=pgsi_scores,
            mode="lines+markers",
            marker=dict(color=colors, size=12),
            line=dict(color="#3498DB", width=2),
            name="PGSI",
        )
    )

    # Severity bands
    fig.add_hrect(y0=0, y1=25, fillcolor="#D5F5E3", opacity=0.2, line_width=0)
    fig.add_hrect(y0=25, y1=50, fillcolor="#FCF3CF", opacity=0.2, line_width=0)
    fig.add_hrect(y0=50, y1=75, fillcolor="#FDEBD0", opacity=0.2, line_width=0)
    fig.add_hrect(y0=75, y1=100, fillcolor="#FADBD8", opacity=0.2, line_width=0)

    fig.update_layout(
        title=title,
        xaxis_title="Session",
        yaxis_title="PGSI Score",
        yaxis=dict(range=[0, 100]),
        height=400,
        margin=dict(l=60, r=30, t=50, b=40),
    )
    return fig


# ═══════════════════════════════════════════════════
# 6. Before-vs-After Comparison
# ═══════════════════════════════════════════════════

def create_comparison_chart(
    pre_sub_scores: Dict[str, float],
    post_sub_scores: Dict[str, float],
    pre_pgsi: float,
    post_pgsi: float,
) -> go.Figure:
    """Grouped bar chart comparing pre- and post-therapy sub-scores."""
    categories = [k.replace("_", " ").title() for k in pre_sub_scores.keys()]
    categories.append("PGSI Total")

    pre_vals = list(pre_sub_scores.values()) + [pre_pgsi]
    post_vals = list(post_sub_scores.values()) + [post_pgsi]

    fig = go.Figure(
        data=[
            go.Bar(name="Pre-Therapy", x=categories, y=pre_vals, marker_color="#E74C3C"),
            go.Bar(name="Post-Therapy", x=categories, y=post_vals, marker_color="#2ECC71"),
        ]
    )
    fig.update_layout(
        barmode="group",
        title="Pre- vs Post-Therapy Comparison",
        yaxis=dict(title="Score (0–100)", range=[0, 100]),
        height=400,
        margin=dict(l=60, r=30, t=50, b=40),
    )
    return fig


# ═══════════════════════════════════════════════════
# 7. Matplotlib skeleton overlay (for static export)
# ═══════════════════════════════════════════════════

def render_skeleton_overlay(frame_bgr: np.ndarray, keypoint_frame) -> bytes:
    """Render skeleton on a frame and return PNG bytes.
    Uses matplotlib for static image export."""
    import cv2

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w = frame_rgb.shape[:2]

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.imshow(frame_rgb)

    if keypoint_frame is not None:
        # Draw keypoints
        for name, (x, y, z, vis) in keypoint_frame.landmarks.items():
            if vis >= 0.5:
                px_x, px_y = x * w, y * h
                ax.plot(px_x, px_y, "go", markersize=5)
                ax.annotate(
                    name.replace("_", " "),
                    (px_x, px_y),
                    fontsize=5,
                    color="lime",
                    ha="center",
                    va="bottom",
                )

        # Draw connections (simplified: shoulder-hip, hip-knee, knee-ankle, shoulder-elbow-wrist)
        connections = [
            ("left_shoulder", "right_shoulder"),
            ("left_shoulder", "left_hip"),
            ("right_shoulder", "right_hip"),
            ("left_hip", "right_hip"),
            ("left_hip", "left_knee"),
            ("right_hip", "right_knee"),
            ("left_knee", "left_ankle"),
            ("right_knee", "right_ankle"),
            ("left_shoulder", "left_elbow"),
            ("right_shoulder", "right_elbow"),
            ("left_elbow", "left_wrist"),
            ("right_elbow", "right_wrist"),
        ]
        for a, b in connections:
            lm_a = keypoint_frame.landmarks.get(a)
            lm_b = keypoint_frame.landmarks.get(b)
            if lm_a and lm_b and lm_a[3] >= 0.5 and lm_b[3] >= 0.5:
                ax.plot(
                    [lm_a[0] * w, lm_b[0] * w],
                    [lm_a[1] * h, lm_b[1] * h],
                    "g-",
                    linewidth=1.5,
                )

    ax.axis("off")
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ═══════════════════════════════════════════════════
# 8. Session Sub-Score Correlation Heatmap (Seaborn)
# ═══════════════════════════════════════════════════

def create_correlation_heatmap(sessions: List[Dict]) -> bytes:
    """Create a seaborn heatmap showing correlation between sub-scores across sessions.

    Args:
        sessions: List of session dicts, each with a 'sub_scores' dict containing
                  keys: stride, posture, symmetry, variability, armswing.

    Returns:
        PNG image bytes of the heatmap.
    """
    import seaborn as sns
    import pandas as pd

    feature_names = {
        "stride": "Stride Length",
        "posture": "Posture Angle",
        "symmetry": "Gait Symmetry",
        "variability": "Step Timing CV",
        "armswing": "Arm Swing",
    }

    # Build DataFrame from sessions
    rows = []
    for s in sessions:
        sub = s.get("sub_scores", {})
        row = {feature_names.get(k, k): v for k, v in sub.items() if k in feature_names}
        if row:
            rows.append(row)

    if len(rows) < 2:
        # Need at least 2 sessions for correlation
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "Need ≥ 2 sessions\nwith sub-scores",
                ha="center", va="center", fontsize=14, color="#7f8c8d")
        ax.axis("off")
    else:
        df = pd.DataFrame(rows)
        corr = df.corr()

        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(
            corr,
            annot=True,
            fmt=".2f",
            cmap="RdYlBu_r",
            vmin=-1,
            vmax=1,
            center=0,
            square=True,
            linewidths=0.5,
            ax=ax,
        )
        ax.set_title("Sub-Score Correlation Across Sessions", fontsize=13, pad=12)

    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    plt.close(fig)
    buf.seek(0)
    return buf.read()

