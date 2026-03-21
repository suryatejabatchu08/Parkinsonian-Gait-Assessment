"""
app.py — Streamlit Dashboard for the PGSI System

Entry point: streamlit run app.py

Pages:
  1. Upload — Video upload + patient metadata
  2. Pose Viewer — Skeleton overlay preview
  3. Feature Analysis — Per-feature waveforms, stride-by-stride breakdown
  4. PGSI Report — Severity gauge, radar chart, classification
  5. Longitudinal — PGSI trend over sessions, before-vs-after comparison
"""

import streamlit as st
import cv2
import os
import json
import tempfile
import numpy as np
from datetime import datetime

from config import TARGET_FPS, OUTPUT_DIR, SUPPORTED_VIDEO_EXTENSIONS, SESSIONS_HISTORY_PATH
from preprocessing.video_processor import VideoProcessor
from pose.pose_estimator import PoseEstimator, KeypointFrame
from features.gait_features import GaitFeatureExtractor
from scoring.pgsi_scorer import PGSIScorer, PGSIResult
from visualization.charts import (
    create_radar_chart,
    create_severity_gauge,
    create_subscore_bar_chart,
    create_waveform_plot,
    create_longitudinal_chart,
    create_comparison_chart,
    render_skeleton_overlay,
    create_correlation_heatmap,
)
from reporting.report_generator import ReportGenerator

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="PGSI — Parkinsonian Gait Assessment",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Session state initialization
# ─────────────────────────────────────────────
def _load_sessions_history():
    """Load persisted session history from JSON file."""
    if os.path.isfile(SESSIONS_HISTORY_PATH):
        try:
            with open(SESSIONS_HISTORY_PATH, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return []


def _save_sessions_history(history):
    """Persist session history to JSON file."""
    os.makedirs(os.path.dirname(SESSIONS_HISTORY_PATH), exist_ok=True)
    with open(SESSIONS_HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)


defaults = {
    "video_path": None,
    "frames": None,
    "keypoints": None,
    "features": None,
    "pgsi_result": None,
    "patient_id": "",
    "session_tag": "",
    "session_date": datetime.now().strftime("%Y-%m-%d"),
    "sessions_history": _load_sessions_history(),
    "analysis_done": False,
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val


# ─────────────────────────────────────────────
# Sidebar navigation
# ─────────────────────────────────────────────
st.sidebar.title("🧠 PGSI System")
st.sidebar.markdown("**Vision-Based Parkinsonian Gait Assessment**")
page = st.sidebar.radio(
    "Navigation",
    ["📤 Upload", "🦴 Pose Viewer", "📊 Feature Analysis", "🎯 PGSI Report", "📈 Longitudinal"],
)

st.sidebar.markdown("---")


# ═════════════════════════════════════════════
# PAGE 1: Upload
# ═════════════════════════════════════════════
if page == "📤 Upload":
    st.title("📤 Video Upload & Analysis")
    st.markdown("Upload a walking video to begin Parkinsonian gait assessment.")

    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_file = st.file_uploader(
            "Upload Walking Video",
            type=["mp4", "avi", "mov", "mkv"],
            help="Supported formats: MP4, AVI, MOV, MKV",
        )

    with col2:
        st.session_state.patient_id = st.text_input("Patient ID", value=st.session_state.patient_id)
        st.session_state.session_date = st.text_input("Session Date", value=st.session_state.session_date)
        st.session_state.session_tag = st.text_input(
            "Session Tag", value=st.session_state.session_tag,
            help="e.g., 'Pre-Therapy', 'Post-Therapy Week 4'"
        )

    if uploaded_file is not None:
        # Save to temp file
        suffix = os.path.splitext(uploaded_file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        st.session_state.video_path = tmp_path
        st.success(f"✅ Video uploaded: **{uploaded_file.name}**")

        # Show video info
        vp = VideoProcessor(tmp_path)
        st.info(
            f"**Duration:** {vp.duration_seconds:.1f}s  |  "
            f"**FPS:** {vp.original_fps:.1f}  |  "
            f"**Resolution:** {vp.width}×{vp.height}  |  "
            f"**Frames:** {vp.frame_count}"
        )
        vp.release()

    st.markdown("---")

    # Run analysis button
    if st.button("🚀 Run PGSI Analysis", type="primary", use_container_width=True):
        if st.session_state.video_path is None:
            st.error("Please upload a video first.")
        else:
            with st.spinner("Processing video..."):
                progress = st.progress(0, "Initializing...")

                # Stage 1: Preprocessing
                progress.progress(10, "Stage 1/5: Preprocessing video...")
                vp = VideoProcessor(st.session_state.video_path)
                frames = vp.preprocess_all(use_bg_subtraction=False)
                st.session_state.frames = frames
                vp.release()

                # Stage 2: Pose Estimation
                progress.progress(30, "Stage 2/5: Estimating pose...")
                estimator = PoseEstimator()
                keypoints = estimator.process_video_frames(frames)
                st.session_state.keypoints = keypoints

                valid_count = sum(1 for kf in keypoints if kf is not None)
                st.session_state._valid_keypoints = valid_count

                # Stage 3: Feature Extraction
                progress.progress(60, "Stage 3/5: Extracting gait features...")
                extractor = GaitFeatureExtractor(fps=TARGET_FPS)
                features = extractor.extract_all(estimator, keypoints)
                st.session_state.features = features

                # Stage 4: PGSI Scoring
                progress.progress(80, "Stage 4/5: Computing PGSI score...")
                scorer = PGSIScorer()
                result = scorer.assess(features)
                st.session_state.pgsi_result = result

                estimator.close()

                # Stage 5: Done
                progress.progress(100, "✅ Analysis complete!")
                st.session_state.analysis_done = True

                # Add to session history and persist
                st.session_state.sessions_history.append({
                    "label": st.session_state.session_tag or f"Session {len(st.session_state.sessions_history) + 1}",
                    "date": st.session_state.session_date,
                    "pgsi": result.pgsi_score,
                    "severity": result.severity_label,
                    "sub_scores": dict(result.sub_scores),
                })
                _save_sessions_history(st.session_state.sessions_history)

            st.success(
                f"**PGSI Score: {result.pgsi_score:.1f}** — {result.severity_label}  |  "
                f"Pose detected in {valid_count}/{len(keypoints)} frames"
            )


# ═════════════════════════════════════════════
# PAGE 2: Pose Viewer
# ═════════════════════════════════════════════
elif page == "🦴 Pose Viewer":
    st.title("🦴 Pose Skeleton Viewer")

    if not st.session_state.analysis_done:
        st.warning("Run analysis first from the Upload page.")
    else:
        frames = st.session_state.frames
        keypoints = st.session_state.keypoints
        n_frames = len(frames)

        frame_idx = st.slider("Frame", 0, n_frames - 1, n_frames // 2)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original Frame")
            frame_rgb = cv2.cvtColor(frames[frame_idx], cv2.COLOR_BGR2RGB)
            st.image(frame_rgb, use_container_width=True)

        with col2:
            st.subheader("Skeleton Overlay")
            kf = keypoints[frame_idx]
            if kf is not None:
                estimator = PoseEstimator(static_image_mode=True)
                annotated = estimator.draw_skeleton(frames[frame_idx], kf)
                annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                st.image(annotated_rgb, use_container_width=True)
                estimator.close()

                # Show keypoint confidence
                st.markdown(f"**Average Visibility:** {kf.avg_visibility:.2f}")
                with st.expander("Keypoint Details"):
                    for name, (x, y, z, vis) in kf.landmarks.items():
                        status = "✅" if vis >= 0.5 else "❌"
                        st.text(f"{status} {name:20s}  x={x:.3f}  y={y:.3f}  vis={vis:.2f}")
            else:
                st.error("No pose detected in this frame.")


# ═════════════════════════════════════════════
# PAGE 3: Feature Analysis
# ═════════════════════════════════════════════
elif page == "📊 Feature Analysis":
    st.title("📊 Biomechanical Feature Analysis")

    if not st.session_state.analysis_done:
        st.warning("Run analysis first from the Upload page.")
    else:
        features = st.session_state.features
        result = st.session_state.pgsi_result

        # Summary metrics
        cols = st.columns(3)
        metric_data = [
            ("Stride Length", f"{features.mean_stride_length:.3f}", "normalized by leg length"),
            ("Posture Angle", f"{features.mean_posture_angle:.1f}°", "forward trunk tilt"),
            ("Timing CV", f"{features.step_timing_cv:.1f}%", "step timing variability"),
        ]
        for col, (label, value, help_text) in zip(cols, metric_data):
            col.metric(label, value, help=help_text)

        st.markdown("---")

        # Posture angle waveform
        st.subheader("Posture Angle (Trunk Tilt Over Time)")
        fig_posture = create_waveform_plot(
            features.posture_angles,
            fps=TARGET_FPS,
            title="Forward Trunk Tilt",
            ylabel="Angle (degrees)",
        )
        st.plotly_chart(fig_posture, use_container_width=True)

        # Stride breakdown
        st.subheader("Stride Lengths (Per-Stride)")
        if len(features.stride_lengths) > 1:
            fig_strides = create_waveform_plot(
                features.stride_lengths,
                fps=1.0,  # stride index, not time
                title="Stride Length Distribution",
                ylabel="Normalized Stride Length",
            )
            st.plotly_chart(fig_strides, use_container_width=True)
        else:
            st.info("Insufficient gait cycles detected for stride-by-stride breakdown.")

        # Gait cycle info
        gc = features.gait_cycles
        col1, col2 = st.columns(2)
        col1.metric("Left Heel Strikes", len(gc.left_heel_strikes))
        col2.metric("Right Heel Strikes", len(gc.right_heel_strikes))


# ═════════════════════════════════════════════
# PAGE 4: PGSI Report
# ═════════════════════════════════════════════
elif page == "🎯 PGSI Report":
    st.title("🎯 PGSI Assessment Report")

    if not st.session_state.analysis_done:
        st.warning("Run analysis first from the Upload page.")
    else:
        result = st.session_state.pgsi_result

        # Severity gauge
        fig_gauge = create_severity_gauge(result.pgsi_score, result.severity_label)
        st.plotly_chart(fig_gauge, use_container_width=True)

        col1, col2 = st.columns(2)

        with col1:
            # Radar chart
            fig_radar = create_radar_chart(result.sub_scores)
            st.plotly_chart(fig_radar, use_container_width=True)

        with col2:
            # Bar chart
            fig_bar = create_subscore_bar_chart(result.sub_scores)
            st.plotly_chart(fig_bar, use_container_width=True)

        # Fall risk
        st.markdown("---")
        if result.fall_risk:
            st.error("⚠️ **ELEVATED FALL RISK DETECTED**")
            for reason in result.fall_risk_reasons:
                st.warning(reason)
        else:
            st.success("✅ Fall risk within normal range.")

        # Clinical interpretation
        st.markdown("---")
        st.subheader("Clinical Interpretation")
        report_gen = ReportGenerator()
        interpretation = report_gen.generate_interpretation(result)
        st.markdown(interpretation, unsafe_allow_html=True)

        # Download report
        st.markdown("---")
        st.subheader("Download Report")

        col_dl1, col_dl2 = st.columns(2)

        with col_dl1:
            if st.button("📄 Generate PDF Report"):
                os.makedirs(OUTPUT_DIR, exist_ok=True)
                pdf_path = os.path.join(
                    OUTPUT_DIR,
                    f"PGSI_Report_{st.session_state.patient_id}_{st.session_state.session_date}.pdf",
                )
                report_gen.generate_pdf(
                    result,
                    pdf_path,
                    patient_id=st.session_state.patient_id,
                    session_date=st.session_state.session_date,
                    session_tag=st.session_state.session_tag,
                    video_filename=os.path.basename(st.session_state.video_path or ""),
                )
                if os.path.exists(pdf_path):
                    with open(pdf_path, "rb") as f:
                        st.download_button("⬇️ Download PDF", f, file_name=os.path.basename(pdf_path))
                else:
                    # Might have been saved as HTML
                    html_path = pdf_path.replace(".pdf", ".html")
                    if os.path.exists(html_path):
                        with open(html_path, "rb") as f:
                            st.download_button("⬇️ Download HTML Report", f, file_name=os.path.basename(html_path))

        with col_dl2:
            if st.button("📊 Generate HTML Report"):
                os.makedirs(OUTPUT_DIR, exist_ok=True)
                html_path = os.path.join(
                    OUTPUT_DIR,
                    f"PGSI_Report_{st.session_state.patient_id}_{st.session_state.session_date}.html",
                )
                report_gen.generate_html_file(
                    result,
                    html_path,
                    patient_id=st.session_state.patient_id,
                    session_date=st.session_state.session_date,
                    session_tag=st.session_state.session_tag,
                    video_filename=os.path.basename(st.session_state.video_path or ""),
                )
                with open(html_path, "rb") as f:
                    st.download_button("⬇️ Download HTML", f, file_name=os.path.basename(html_path))


# ═════════════════════════════════════════════
# PAGE 5: Longitudinal Tracking
# ═════════════════════════════════════════════
elif page == "📈 Longitudinal":
    st.title("📈 Longitudinal Tracking")

    history = st.session_state.sessions_history

    if len(history) == 0:
        st.info("No sessions recorded yet. Run at least one analysis from the Upload page.")
    else:
        # PGSI trend chart
        labels = [s["label"] for s in history]
        scores = [s["pgsi"] for s in history]

        fig_trend = create_longitudinal_chart(labels, scores)
        st.plotly_chart(fig_trend, use_container_width=True)

        # Session table
        st.subheader("Session History")
        for i, session in enumerate(history):
            st.markdown(
                f"**{session['label']}** ({session['date']}) — "
                f"PGSI: **{session['pgsi']:.1f}** — {session['severity']}"
            )

        # Sub-Score Correlation Heatmap (requires ≥ 3 sessions with sub-scores)
        sessions_with_scores = [s for s in history if s.get("sub_scores") and any(v != 0 for v in s["sub_scores"].values())]
        if len(sessions_with_scores) >= 3:
            st.markdown("---")
            st.subheader("Sub-Score Correlation Heatmap")
            heatmap_bytes = create_correlation_heatmap(sessions_with_scores)
            st.image(heatmap_bytes, caption="Correlation between sub-scores across sessions", use_container_width=True)

        # Before-vs-After comparison
        if len(history) >= 2:
            st.markdown("---")
            st.subheader("Before vs After Comparison")

            col1, col2 = st.columns(2)
            with col1:
                pre_idx = st.selectbox("Pre-Therapy Session", range(len(history)),
                                       format_func=lambda i: history[i]["label"])
            with col2:
                post_idx = st.selectbox("Post-Therapy Session", range(len(history)),
                                        index=len(history) - 1,
                                        format_func=lambda i: history[i]["label"])

            if pre_idx != post_idx:
                pre = history[pre_idx]
                post = history[post_idx]

                fig_comp = create_comparison_chart(
                    pre["sub_scores"], post["sub_scores"],
                    pre["pgsi"], post["pgsi"],
                )
                st.plotly_chart(fig_comp, use_container_width=True)

                delta = post["pgsi"] - pre["pgsi"]
                if delta < 0:
                    st.success(f"PGSI improved by **{abs(delta):.1f}** points ({pre['severity']} → {post['severity']})")
                elif delta > 0:
                    st.warning(f"PGSI worsened by **{delta:.1f}** points ({pre['severity']} → {post['severity']})")
                else:
                    st.info("No change in PGSI score between sessions.")

    # Manual session data entry
    st.markdown("---")
    with st.expander("➕ Add Manual Session Entry"):
        m_label = st.text_input("Session Label", key="manual_label")
        m_date = st.text_input("Date", value=datetime.now().strftime("%Y-%m-%d"), key="manual_date")
        m_pgsi = st.number_input("PGSI Score", 0.0, 100.0, 50.0, key="manual_pgsi")
        if st.button("Add Session"):
            from scoring.pgsi_scorer import PGSIScorer
            severity = PGSIScorer.classify_severity(m_pgsi)
            st.session_state.sessions_history.append({
                "label": m_label or f"Manual Session {len(history) + 1}",
                "date": m_date,
                "pgsi": m_pgsi,
                "severity": severity,
                "sub_scores": {"stride": 0, "posture": 0, "variability": 0, "symmetry": 0, "armswing": 0},
            })
            _save_sessions_history(st.session_state.sessions_history)
            st.success("Session added!")
            st.rerun()

    # Clear history button
    if len(history) > 0:
        st.markdown("---")
        if st.button("🗑️ Clear All Session History", type="secondary"):
            st.session_state.sessions_history = []
            _save_sessions_history([])
            st.rerun()
