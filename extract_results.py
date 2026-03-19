"""
extract_results.py
──────────────────────────────────────────────────────────────────
PGSI Results Extractor — Research Paper Edition
Group 2, Section J

Processes a labeled video dataset and produces all tables, figures,
and statistics needed for the Results section of a research paper.

OUTPUTS (saved to results/):
  ┌─────────────────────────────────────────┬────────────────────────────────────┐
  │ File                                    │ Maps to                            │
  ├─────────────────────────────────────────┼────────────────────────────────────┤
  │ subject_scores.csv                      │ Supplementary material             │
  │ group_summary.csv                       │ Table 1 — Group-level features     │
  │ statistics.json                         │ Section 5.2 — Statistical tests    │
  │ classification_report.json              │ Table 2 — Classification metrics   │
  │ confusion_matrix.csv                    │ Figure — Confusion matrix          │
  │ single_feature_baselines.csv            │ Table 3 — Ablation / baselines     │
  │ feature_distributions.png               │ Figure — Box plots per feature     │
  │ pgsi_by_group.png                       │ Figure — PGSI distribution         │
  │ correlation_heatmap.png                 │ Figure — Feature correlation        │
  │ confusion_matrix.png                    │ Figure — Confusion matrix heatmap  │
  │ roc_curves.png                          │ Figure — ROC curves (OvR)          │
  │ results_summary.txt                     │ Quick-paste text for paper         │
  └─────────────────────────────────────────┴────────────────────────────────────┘

Usage:
    python extract_results.py --dataset_dir path/to/dataset

Dataset structure (subfolder-based):
    dataset_dir/
        Normal/           (or NM/)
        PD_Mild/          (or Mild/)
        PD_Moderate/      (or Moderate/)
        PD_Severe/        (or Severe/)

Or flat structure (filename-based):
    dataset_dir/
        001_NM_01_SV.MOV          ← contains "NM"
        001_PD_01_ML.MOV          ← mapped via --default-label
        001_PD_02_SV.MOV
        ...

Dependencies:
    pip install opencv-python mediapipe numpy scipy pandas scikit-learn matplotlib seaborn
"""

import cv2
import numpy as np
import mediapipe as mp
import argparse
import math
import os
import json
import csv
from collections import deque
from pathlib import Path
import pandas as pd
from scipy import stats
from scipy.stats import kruskal
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ── CONFIG ────────────────────────────────────────────────────────────────────

HEALTHY_REF = {
    "stride_length": 1.0,
    "posture_angle": 0.0,
    "gait_symmetry": 0.0,
    "step_timing_cv": 0.0,
    "arm_swing": 80.0,
}

IMPAIRED_REF = {
    "stride_length": 0.2,
    "posture_angle": 45.0,
    "gait_symmetry": 50.0,   # Aligned with pgsi_scorer.py REFERENCE_RANGES
    "step_timing_cv": 30.0,
    "arm_swing": 5.0,
}

HIGHER_IS_WORSE = {
    "stride_length": False,
    "posture_angle": True,
    "gait_symmetry": True,
    "step_timing_cv": True,
    "arm_swing": False,
}

WEIGHTS = {
    "stride_length": 0.20,
    "posture_angle": 0.20,
    "gait_symmetry": 0.20,
    "step_timing_cv": 0.20,
    "arm_swing": 0.20,
}

# Map folder names to severity labels
# Supports the actual dataset structure: data/NM/, data/PD/PD_ML/, etc.
FOLDER_LABEL_MAP = {
    # Top-level folders
    "NM":          "Normal",
    "Normal":      "Normal",
    # PD subfolders
    "PD_ML":       "Mild",
    "PD_Mild":     "Mild",
    "Mild":        "Mild",
    "PD_MD":       "Moderate",
    "PD_Moderate": "Moderate",
    "Moderate":    "Moderate",
    "PD_SV":       "Severe",
    "PD_Severe":   "Severe",
    "Severe":      "Severe",
    # Other
    "KOA":         "Mild",
}

# Filename keyword → label (for flat directory structure)
FILENAME_LABEL_MAP = {
    "NM": "Normal",
    "KOA": "Mild",
    "PD": None,  # needs severity suffix
}

FALL_RISK_THRESHOLDS = {
    "posture_angle": 70,
    "step_timing_cv": 70,
}

# ── GEOMETRY ──────────────────────────────────────────────────────────────────

def dist2d(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def angle_3pts(a, b, c):
    ba = (a[0]-b[0], a[1]-b[1])
    bc = (c[0]-b[0], c[1]-b[1])
    denom = max(math.hypot(*ba), 1e-6) * max(math.hypot(*bc), 1e-6)
    cos_a = (ba[0]*bc[0] + ba[1]*bc[1]) / denom
    return math.degrees(math.acos(np.clip(cos_a, -1, 1)))

def lm_px(landmarks, idx, w, h):
    p = landmarks[idx]
    return (p.x * w, p.y * h)

# ── NORMALIZATION / SCORING ───────────────────────────────────────────────────

def normalize(value, feature):
    h = HEALTHY_REF[feature]
    i = IMPAIRED_REF[feature]
    if HIGHER_IS_WORSE[feature]:
        raw = (value - h) / (i - h) * 100
    else:
        raw = (h - value) / (h - i) * 100
    return float(np.clip(raw, 0.0, 100.0))

def compute_pgsi(sub_scores):
    return sum(WEIGHTS[k] * sub_scores[k] for k in WEIGHTS)

def classify_pgsi(pgsi):
    """Classify severity with no gaps between bins."""
    if pgsi <= 25:
        return "Normal"
    elif pgsi <= 50:
        return "Mild"
    elif pgsi <= 75:
        return "Moderate"
    else:
        return "Severe"

def fall_risk_flag(sub_scores, severity):
    if severity not in ("Moderate", "Severe"):
        return False
    return (sub_scores["posture_angle"] >= FALL_RISK_THRESHOLDS["posture_angle"] or
            sub_scores["step_timing_cv"] >= FALL_RISK_THRESHOLDS["step_timing_cv"])

# ── FEATURE EXTRACTION FROM VIDEO ─────────────────────────────────────────────

def extract_features_from_video(video_path):
    """
    Opens a video file, runs MediaPipe Pose on every frame,
    and returns a dict of 5 raw feature values.
    Returns None if the video cannot be processed.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [!] Cannot open: {video_path}")
        return None

    # Use MediaPipe Tasks API (compatible with mediapipe >= 0.10.9)
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import (
        PoseLandmarker,
        PoseLandmarkerOptions,
        RunningMode,
    )

    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "pose_landmarker.task")
    if not os.path.isfile(model_path):
        print(f"  [!] Model not found: {model_path}")
        return None

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.3,
        min_pose_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )

    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # Per-frame accumulators
    posture_angles    = []
    l_elbow_angles    = []
    r_elbow_angles    = []
    l_ankle_x         = []
    r_ankle_x         = []
    l_ankle_y         = []
    r_ankle_y         = []
    leg_lengths       = []

    frame_idx         = 0
    frames_processed  = 0
    frames_skipped    = 0

    REQUIRED_LMS = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
    MIN_VISIBILITY = 0.3  # Lowered for side-view

    landmarker = PoseLandmarker.create_from_options(options)
    timestamp_ms = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocessing: denoise + sharpen for clearer frames
        frame = cv2.fastNlMeansDenoisingColored(frame, None, 3, 3, 7, 21)
        blur = cv2.GaussianBlur(frame, (0, 0), 3)
        frame = cv2.addWeighted(frame, 1.5, blur, -0.5, 0)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms += int(1000 / fps)
        results = landmarker.detect_for_video(mp_image, timestamp_ms)
        frame_idx += 1

        if not results.pose_landmarks or len(results.pose_landmarks) == 0:
            frames_skipped += 1
            continue

        lms = results.pose_landmarks[0]  # First person

        # Quality filter — skip frame if any required landmark is low confidence
        if any(getattr(lms[i], 'visibility', 0) < MIN_VISIBILITY for i in REQUIRED_LMS):
            frames_skipped += 1
            continue

        frames_processed += 1

        # Extract keypoints in pixel coords (Tasks API gives normalized coords)
        def _px(idx):
            return (lms[idx].x * fw, lms[idx].y * fh)

        ls = _px(11); rs = _px(12)
        lh = _px(23); rh = _px(24)
        le = _px(13); re = _px(14)
        lw_pt = _px(15); rw_pt = _px(16)
        la = _px(27); ra = _px(28)

        # Leg length proxy (hip to ankle), normalized by frame height
        ll = dist2d(lh, la) / fh
        rl = dist2d(rh, ra) / fh
        leg_len = (ll + rl) / 2
        if leg_len > 0.01:
            leg_lengths.append(leg_len)

        # Posture angle (trunk forward tilt)
        smid = ((ls[0]+rs[0])/2, (ls[1]+rs[1])/2)
        hmid = ((lh[0]+rh[0])/2, (lh[1]+rh[1])/2)
        dx = hmid[0] - smid[0]
        dy = hmid[1] - smid[1]
        p_angle = abs(math.degrees(math.atan2(abs(dx), max(dy, 1e-6))))
        posture_angles.append(p_angle)

        # Elbow angles (arm swing)
        l_elbow_angles.append(angle_3pts(lw_pt, le, ls))
        r_elbow_angles.append(angle_3pts(rw_pt, re, rs))

        # Ankle positions (for stride and timing)
        l_ankle_x.append(la[0] / fw)
        r_ankle_x.append(ra[0] / fw)
        l_ankle_y.append(la[1] / fh)
        r_ankle_y.append(ra[1] / fh)

    cap.release()
    landmarker.close()

    if frames_processed < 10:
        print(f"  [!] Too few valid frames ({frames_processed}): {Path(video_path).name}")
        return None

    quality_pct = frames_processed / max(frame_idx, 1) * 100
    print(f"  [✓] {Path(video_path).name}: {frames_processed}/{frame_idx} frames ({quality_pct:.1f}% valid)")

    # ── FEATURE COMPUTATION ────────────────────────────────────────────────────

    # F2: Posture angle
    posture_angle = float(np.mean(posture_angles))

    # F5: Arm swing — 5th to 95th percentile range
    l_swing = np.percentile(l_elbow_angles, 95) - np.percentile(l_elbow_angles, 5)
    r_swing = np.percentile(r_elbow_angles, 95) - np.percentile(r_elbow_angles, 5)
    arm_swing = float((l_swing + r_swing) / 2)

    # Gait cycle detection via ankle vertical oscillation peaks
    l_y = np.array(l_ankle_y)
    r_y = np.array(r_ankle_y)

    from scipy.signal import find_peaks, savgol_filter

    window = min(15, len(l_y) - 1 if len(l_y) % 2 == 0 else len(l_y))
    if window >= 5:
        l_y_sm = savgol_filter(l_y, window_length=window, polyorder=2)
        r_y_sm = savgol_filter(r_y, window_length=window, polyorder=2)
    else:
        l_y_sm = l_y
        r_y_sm = r_y

    # Heel strikes = local maxima in ankle y
    min_dist = max(int(fps * 0.3), 5)
    l_peaks, _ = find_peaks(l_y_sm, distance=min_dist, prominence=0.01)
    r_peaks, _ = find_peaks(r_y_sm, distance=min_dist, prominence=0.01)

    mean_leg_len = float(np.mean(leg_lengths)) if leg_lengths else 0.15

    # F1: Stride length
    l_x = np.array(l_ankle_x)
    r_x = np.array(r_ankle_x)

    stride_lengths = []
    if len(l_peaks) >= 2:
        for i in range(len(l_peaks) - 1):
            sl = abs(l_x[l_peaks[i+1]] - l_x[l_peaks[i]])
            stride_lengths.append(sl / max(mean_leg_len, 0.01))
    if len(r_peaks) >= 2:
        for i in range(len(r_peaks) - 1):
            sl = abs(r_x[r_peaks[i+1]] - r_x[r_peaks[i]])
            stride_lengths.append(sl / max(mean_leg_len, 0.01))

    if stride_lengths:
        stride_length = float(np.mean(stride_lengths))
    else:
        # Fallback: estimate from total displacement
        total_left = abs(l_x[-1] - l_x[0]) if len(l_x) > 1 else 0
        total_right = abs(r_x[-1] - r_x[0]) if len(r_x) > 1 else 0
        avg_disp = (total_left + total_right) / 2.0
        n_frames = len(l_x)
        est_steps = max(1, n_frames / (fps * 0.5))
        stride_length = (avg_disp / est_steps) / max(mean_leg_len, 0.01) if mean_leg_len > 0.01 else 0.5

    # F3: Gait symmetry
    all_peaks = sorted(list(l_peaks) + list(r_peaks))
    l_steps, r_steps = [], []
    for i in range(len(all_peaks) - 1):
        s = abs(l_x[all_peaks[i+1]] - l_x[all_peaks[i]])
        if all_peaks[i] in l_peaks:
            l_steps.append(s)
        else:
            r_steps.append(s)

    if l_steps and r_steps:
        L = np.mean(l_steps)
        R = np.mean(r_steps)
        denom = 0.5 * (L + R)
        gait_symmetry = float(abs(L - R) / max(denom, 1e-6) * 100)
    else:
        gait_symmetry = 10.0

    # F4: Step timing CV
    if len(all_peaks) >= 4:
        intervals = np.diff(all_peaks) / fps
        step_timing_cv = float(np.std(intervals) / max(np.mean(intervals), 1e-6) * 100)
    else:
        step_timing_cv = 5.0

    return {
        "stride_length": stride_length,
        "posture_angle": posture_angle,
        "gait_symmetry": gait_symmetry,
        "step_timing_cv": step_timing_cv,
        "arm_swing": arm_swing,
        "frames_processed": frames_processed,
        "frames_total": frame_idx,
        "quality_pct": quality_pct,
    }

# ── LABEL INFERENCE FROM FILENAME ─────────────────────────────────────────────

def infer_label_from_filename(filename, default_label="Mild"):
    """Infer severity label from filename like 001_PD_02_SV.MOV or 001_NM_01_SV.MOV."""
    name_upper = filename.upper()
    if "_NM_" in name_upper or "_NM." in name_upper or name_upper.startswith("NM"):
        return "Normal"
    if "_KOA_" in name_upper or name_upper.startswith("KOA"):
        return "Mild"
    if "_SV_" in name_upper or "_SV." in name_upper or "SEVERE" in name_upper:
        return "Severe"
    if "_MD_" in name_upper or "_MD." in name_upper or "MODERATE" in name_upper:
        return "Moderate"
    if "_ML_" in name_upper or "_ML." in name_upper or "MILD" in name_upper:
        return "Mild"
    if "_PD_" in name_upper:
        return default_label
    return default_label

# ── STATISTICAL TESTS ─────────────────────────────────────────────────────────

def run_statistics(df):
    """Run Kruskal-Wallis + pairwise Mann-Whitney U tests."""
    results = {}
    features = ["stride_length_sub", "posture_angle_sub", "gait_symmetry_sub",
                "step_timing_cv_sub", "arm_swing_sub", "pgsi"]
    groups = ["Normal", "Mild", "Moderate", "Severe"]

    for feat in features:
        group_data = [df[df["true_label"] == g][feat].values for g in groups if g in df["true_label"].values]
        group_data = [g for g in group_data if len(g) > 0]

        if len(group_data) >= 2:
            h_stat, p_val = kruskal(*group_data)
            results[feat] = {
                "H_statistic": round(float(h_stat), 3),
                "p_value": float(p_val),
                "p_value_formatted": f"{'<0.001' if p_val < 0.001 else f'{p_val:.4f}'}",
                "significant": bool(p_val < 0.05),
            }

    # Pairwise Mann-Whitney U (Bonferroni corrected)
    pgsi_groups = {g: df[df["true_label"] == g]["pgsi"].values for g in groups if g in df["true_label"].values}
    pairwise = {}
    group_list = list(pgsi_groups.keys())
    n_comparisons = len(group_list) * (len(group_list) - 1) // 2
    for i in range(len(group_list)):
        for j in range(i+1, len(group_list)):
            g1, g2 = group_list[i], group_list[j]
            if len(pgsi_groups[g1]) > 0 and len(pgsi_groups[g2]) > 0:
                stat, p = stats.mannwhitneyu(pgsi_groups[g1], pgsi_groups[g2], alternative="two-sided")
                p_bonf = min(p * n_comparisons, 1.0)
                pairwise[f"{g1}_vs_{g2}"] = {
                    "U_statistic": round(float(stat), 3),
                    "p_value_uncorrected": float(p),
                    "p_value_bonferroni": float(p_bonf),
                    "p_value_formatted": f"{'<0.001' if p_bonf < 0.001 else f'{p_bonf:.4f}'}",
                    "significant_bonferroni": bool(p_bonf < 0.05),
                    "effect_size_r": round(float(stat / (len(pgsi_groups[g1]) * len(pgsi_groups[g2]))), 3),
                }

    results["pairwise_pgsi"] = pairwise
    return results

def run_classification(df):
    """Evaluate PGSI threshold classification against ground truth labels."""
    label_order = ["Normal", "Mild", "Moderate", "Severe"]

    y_true = df["true_label"].values
    y_pred = df["predicted_label"].values

    acc = accuracy_score(y_true, y_pred)
    report = classification_report(
        y_true, y_pred,
        labels=label_order,
        output_dict=True,
        zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=label_order)

    return {
        "overall_accuracy": round(acc * 100, 1),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "label_order": label_order,
    }

def single_feature_baselines(df):
    """
    For each feature sub-score, find the best threshold for binary PD vs Normal
    and compute multi-class accuracy using that single feature alone.
    """
    features = {
        "stride_length": "stride_length_sub",
        "posture_angle": "posture_angle_sub",
        "gait_symmetry": "gait_symmetry_sub",
        "step_timing_cv": "step_timing_cv_sub",
        "arm_swing": "arm_swing_sub",
    }
    results = {}
    df_bin = df.copy()
    df_bin["is_pd"] = (df_bin["true_label"] != "Normal").astype(int)

    for name, col in features.items():
        best_acc = 0
        best_thresh = 0
        for thresh in np.linspace(0, 100, 200):
            pred = (df_bin[col] >= thresh).astype(int)
            acc = accuracy_score(df_bin["is_pd"], pred)
            if acc > best_acc:
                best_acc = acc
                best_thresh = thresh
        results[name] = {
            "best_threshold": round(best_thresh, 1),
            "binary_accuracy_pct": round(best_acc * 100, 1),
        }

    # Also compute multi-class "PGSI composite" accuracy for comparison
    pgsi_acc = accuracy_score(df["true_label"], df["predicted_label"])
    results["pgsi_composite"] = {
        "best_threshold": "N/A (multi-class)",
        "binary_accuracy_pct": round(pgsi_acc * 100, 1),
    }

    return results

# ── FIGURE GENERATION ─────────────────────────────────────────────────────────

def generate_figures(df, out_dir):
    """Generate all publication-quality figures."""
    label_order = ["Normal", "Mild", "Moderate", "Severe"]
    colors = {"Normal": "#2ecc71", "Mild": "#f1c40f", "Moderate": "#e67e22", "Severe": "#e74c3c"}

    # ── Figure 1: PGSI by Group (box + swarm) ──────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    existing_labels = [l for l in label_order if l in df["true_label"].values]
    palette = [colors[l] for l in existing_labels]
    sns.boxplot(data=df, x="true_label", y="pgsi", order=existing_labels,
                palette=palette, width=0.5, ax=ax)
    sns.stripplot(data=df, x="true_label", y="pgsi", order=existing_labels,
                  color="black", alpha=0.5, size=4, ax=ax)
    ax.set_xlabel("True Severity Group", fontsize=12)
    ax.set_ylabel("PGSI Score (0–100)", fontsize=12)
    ax.set_title("PGSI Score Distribution by Severity Group", fontsize=14)
    ax.axhline(y=25, color="#95a5a6", linestyle="--", alpha=0.5, label="Normal/Mild threshold")
    ax.axhline(y=50, color="#95a5a6", linestyle=":", alpha=0.5, label="Mild/Moderate threshold")
    ax.axhline(y=75, color="#95a5a6", linestyle="-.", alpha=0.5, label="Moderate/Severe threshold")
    ax.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    fig.savefig(out_dir / "pgsi_by_group.png", dpi=300)
    plt.close(fig)
    print(f"  [fig] pgsi_by_group.png")

    # ── Figure 2: Feature distribution box plots ───────────────────────────
    feature_cols = [
        ("stride_length_raw", "Stride Length\n(normalized)"),
        ("posture_angle_raw", "Posture Angle\n(degrees)"),
        ("gait_symmetry_raw", "Symmetry Index\n(%)"),
        ("step_timing_cv_raw", "Step Timing CV\n(%)"),
        ("arm_swing_raw", "Arm Swing\n(degrees)"),
    ]
    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    for ax, (col, title) in zip(axes, feature_cols):
        sns.boxplot(data=df, x="true_label", y=col, order=existing_labels,
                    palette=palette, width=0.5, ax=ax)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=30, labelsize=9)
    fig.suptitle("Raw Feature Distributions by Severity Group", fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(out_dir / "feature_distributions.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig] feature_distributions.png")

    # ── Figure 3: Sub-score correlation heatmap ────────────────────────────
    sub_cols = ["stride_length_sub", "posture_angle_sub", "gait_symmetry_sub",
                "step_timing_cv_sub", "arm_swing_sub"]
    sub_labels = ["Stride", "Posture", "Symmetry", "Variability", "Arm Swing"]

    corr = df[sub_cols].corr()
    corr.index = sub_labels
    corr.columns = sub_labels

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlBu_r", vmin=-1, vmax=1,
                center=0, square=True, linewidths=0.5, ax=ax)
    ax.set_title("Sub-Score Correlation Matrix", fontsize=14, pad=12)
    plt.tight_layout()
    fig.savefig(out_dir / "correlation_heatmap.png", dpi=300)
    plt.close(fig)
    print(f"  [fig] correlation_heatmap.png")

    # ── Figure 4: Confusion matrix ─────────────────────────────────────────
    cm = confusion_matrix(df["true_label"], df["predicted_label"], labels=existing_labels)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=existing_labels,
                yticklabels=existing_labels, ax=ax)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title("Confusion Matrix — PGSI Classification", fontsize=14, pad=12)
    plt.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=300)
    plt.close(fig)
    print(f"  [fig] confusion_matrix.png")

    # ── Figure 5: ROC curves (One-vs-Rest) ─────────────────────────────────
    try:
        y_true_bin = label_binarize(df["true_label"], classes=existing_labels)
        y_score = df[[c for c in sub_cols]].values  # Use sub-scores as proxy
        # Use PGSI as main score
        pgsi_vals = df["pgsi"].values

        fig, ax = plt.subplots(figsize=(7, 6))
        for idx, label in enumerate(existing_labels):
            if y_true_bin.shape[1] > idx:
                y_class = y_true_bin[:, idx]
                fpr, tpr, _ = roc_curve(y_class, pgsi_vals if label != "Normal" else -pgsi_vals)
                roc_auc = auc(fpr, tpr)
                ax.plot(fpr, tpr, label=f"{label} (AUC = {roc_auc:.2f})", color=colors[label], linewidth=2)

        ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
        ax.set_xlabel("False Positive Rate", fontsize=12)
        ax.set_ylabel("True Positive Rate", fontsize=12)
        ax.set_title("ROC Curves — One-vs-Rest", fontsize=14, pad=12)
        ax.legend(loc="lower right")
        plt.tight_layout()
        fig.savefig(out_dir / "roc_curves.png", dpi=300)
        plt.close(fig)
        print(f"  [fig] roc_curves.png")
    except Exception as e:
        print(f"  [!] ROC curves skipped: {e}")

    # ── Save confusion matrix as CSV too ───────────────────────────────────
    cm_df = pd.DataFrame(cm, index=existing_labels, columns=existing_labels)
    cm_df.to_csv(out_dir / "confusion_matrix.csv")

# ── RESULTS SUMMARY TEXT ──────────────────────────────────────────────────────

def generate_summary_text(df, stats_results, clf_results, baselines, out_dir):
    """Generate a copy-pasteable text summary for the paper's Results section."""
    lines = []
    lines.append("="*70)
    lines.append(" PGSI RESULTS — Research Paper Summary")
    lines.append("="*70)

    # Sample description
    groups = df.groupby("true_label").size()
    lines.append(f"\n1. DATASET")
    lines.append(f"   Total subjects: N = {len(df)}")
    for label in ["Normal", "Mild", "Moderate", "Severe"]:
        if label in groups.index:
            lines.append(f"   {label}: n = {groups[label]}")

    # Group PGSI
    lines.append(f"\n2. PGSI SCORES (mean ± SD)")
    for label in ["Normal", "Mild", "Moderate", "Severe"]:
        g = df[df["true_label"] == label]["pgsi"]
        if len(g) > 0:
            lines.append(f"   {label:12s}: {g.mean():.1f} ± {g.std():.1f}  (range: {g.min():.1f}–{g.max():.1f})")

    # Statistical tests
    lines.append(f"\n3. STATISTICAL SIGNIFICANCE")
    kw = stats_results.get("pgsi", {})
    if kw:
        lines.append(f"   Kruskal-Wallis (PGSI): H = {kw['H_statistic']}, p {kw['p_value_formatted']}")

    pairwise = stats_results.get("pairwise_pgsi", {})
    if pairwise:
        lines.append(f"   Pairwise Mann-Whitney U (Bonferroni corrected):")
        for pair, vals in pairwise.items():
            sig = "***" if vals["significant_bonferroni"] else "ns"
            lines.append(f"     {pair:25s}: U = {vals['U_statistic']}, p = {vals['p_value_formatted']} {sig}")

    # Classification
    lines.append(f"\n4. CLASSIFICATION PERFORMANCE")
    lines.append(f"   Overall accuracy: {clf_results['overall_accuracy']}%")
    report = clf_results["classification_report"]
    lines.append(f"   {'Class':12s} {'Precision':>10s} {'Recall':>10s} {'F1-Score':>10s} {'Support':>10s}")
    for label in ["Normal", "Mild", "Moderate", "Severe"]:
        if label in report:
            r = report[label]
            lines.append(f"   {label:12s} {r['precision']:10.2f} {r['recall']:10.2f} {r['f1-score']:10.2f} {int(r['support']):10d}")

    # Feature baselines
    lines.append(f"\n5. SINGLE-FEATURE vs COMPOSITE COMPARISON")
    lines.append(f"   {'Feature':25s} {'Accuracy':>10s}")
    for feat, vals in baselines.items():
        lines.append(f"   {feat:25s} {vals['binary_accuracy_pct']:>9.1f}%")

    # Feature means by group
    lines.append(f"\n6. RAW FEATURES BY GROUP (mean ± SD)")
    feature_raw_cols = ["stride_length_raw", "posture_angle_raw", "gait_symmetry_raw",
                        "step_timing_cv_raw", "arm_swing_raw"]
    feature_names = ["Stride Length", "Posture Angle (°)", "Symmetry Index (%)",
                     "Step Timing CV (%)", "Arm Swing (°)"]

    header = f"   {'Feature':25s}"
    for label in ["Normal", "Mild", "Moderate", "Severe"]:
        if label in df["true_label"].values:
            header += f"  {label:>18s}"
    lines.append(header)

    for col, name in zip(feature_raw_cols, feature_names):
        row = f"   {name:25s}"
        for label in ["Normal", "Mild", "Moderate", "Severe"]:
            g = df[df["true_label"] == label][col]
            if len(g) > 0:
                row += f"  {g.mean():>7.2f} ± {g.std():<6.2f}"
        lines.append(row)

    # Fall risk
    lines.append(f"\n7. FALL RISK")
    mod_sev = df[df["true_label"].isin(["Moderate", "Severe"])]
    flagged = df["fall_risk"].sum()
    lines.append(f"   Flagged: {flagged}/{len(mod_sev)} Moderate+Severe patients")

    # Quality
    lines.append(f"\n8. DATA QUALITY")
    lines.append(f"   Mean frame detection rate: {df['quality_pct'].mean():.1f}% ± {df['quality_pct'].std():.1f}%")

    lines.append("\n" + "="*70)

    text = "\n".join(lines)
    with open(out_dir / "results_summary.txt", "w") as f:
        f.write(text)
    print(f"\n{text}")
    return text

# ── MAIN ──────────────────────────────────────────────────────────────────────

def main(dataset_dir, default_label="Mild"):
    dataset_dir = Path(dataset_dir)
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)

    print("\n" + "="*60)
    print(" PGSI Results Extractor — Research Paper Edition")
    print("="*60)

    rows = []
    video_extensions = {".mov", ".mp4", ".avi", ".MOV", ".MP4", ".AVI", ".mkv"}

    # Recursively find all labeled video folders
    # Handles both flat (data/NM/) and nested (data/PD/PD_ML/) structures
    labeled_folders = []  # list of (folder_path, label)

    def _scan_dir(d):
        """Recursively scan for folders that match FOLDER_LABEL_MAP."""
        for child in sorted(d.iterdir()):
            if not child.is_dir():
                continue
            if child.name in FOLDER_LABEL_MAP:
                labeled_folders.append((child, FOLDER_LABEL_MAP[child.name]))
            elif child.name == "PD" or child.name.startswith("PD"):
                # Recurse into PD/ parent folder
                _scan_dir(child)
            else:
                # Try recursing one more level for unknown folders
                _scan_dir(child)

    _scan_dir(dataset_dir)

    if labeled_folders:
        # ── Subfolder mode ──
        print(f"\n[mode] Found {len(labeled_folders)} labeled folders:")
        for fp, lbl in labeled_folders:
            print(f"  {fp.relative_to(dataset_dir)} → {lbl}")

        for folder, label in labeled_folders:
            videos = [f for f in folder.iterdir()
                      if f.suffix.lower() in {e.lower() for e in video_extensions}
                      and not f.name.startswith(".")]
            print(f"\n[{label}] Found {len(videos)} videos in '{folder.relative_to(dataset_dir)}'")

            for video_path in sorted(videos):
                raw = extract_features_from_video(video_path)
                if raw is None:
                    continue
                sub = {k: normalize(raw[k], k) for k in WEIGHTS}
                pgsi = compute_pgsi(sub)
                severity = classify_pgsi(pgsi)
                fr = fall_risk_flag(sub, severity)
                rows.append(_make_row(video_path, folder.name, label, raw, sub, pgsi, severity, fr))
    else:
        # ── Flat mode — infer labels from filenames ──
        videos = [f for f in dataset_dir.iterdir() if f.suffix.lower() in {e.lower() for e in video_extensions}]
        print(f"\n[mode] Flat directory — inferring labels from filenames ({len(videos)} videos)")

        for video_path in sorted(videos):
            label = infer_label_from_filename(video_path.name, default_label)
            raw = extract_features_from_video(video_path)
            if raw is None:
                continue
            sub = {k: normalize(raw[k], k) for k in WEIGHTS}
            pgsi = compute_pgsi(sub)
            severity = classify_pgsi(pgsi)
            fr = fall_risk_flag(sub, severity)
            rows.append(_make_row(video_path, "flat", label, raw, sub, pgsi, severity, fr))

    if not rows:
        print("\n[ERROR] No subjects processed. Check dataset folder structure.")
        return

    df = pd.DataFrame(rows)

    # ── Save per-subject CSV ───────────────────────────────────────────────────
    subject_csv = out_dir / "subject_scores.csv"
    df.to_csv(subject_csv, index=False)
    print(f"\n[saved] {subject_csv}")

    # ── Group summary ──────────────────────────────────────────────────────────
    sub_cols = ["stride_length_sub", "posture_angle_sub", "gait_symmetry_sub",
                "step_timing_cv_sub", "arm_swing_sub", "pgsi"]
    summary_rows = []
    for label in ["Normal", "Mild", "Moderate", "Severe"]:
        g = df[df["true_label"] == label]
        if len(g) == 0:
            continue
        row_s = {"group": label, "n": len(g)}
        for col in sub_cols:
            row_s[f"{col}_mean"] = round(g[col].mean(), 2)
            row_s[f"{col}_std"]  = round(g[col].std(), 2)
        # Also include raw feature means
        for col in ["stride_length_raw", "posture_angle_raw", "gait_symmetry_raw",
                     "step_timing_cv_raw", "arm_swing_raw"]:
            row_s[f"{col}_mean"] = round(g[col].mean(), 2)
            row_s[f"{col}_std"]  = round(g[col].std(), 2)
        summary_rows.append(row_s)

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = out_dir / "group_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"[saved] {summary_csv}")

    # ── Statistical tests ──────────────────────────────────────────────────────
    stats_results = run_statistics(df)
    stats_json = out_dir / "statistics.json"
    with open(stats_json, "w") as f:
        json.dump(stats_results, f, indent=2)
    print(f"[saved] {stats_json}")

    # ── Classification report ──────────────────────────────────────────────────
    clf_results = run_classification(df)
    clf_json = out_dir / "classification_report.json"
    with open(clf_json, "w") as f:
        json.dump(clf_results, f, indent=2)
    print(f"[saved] {clf_json}")

    # ── Single feature baselines ───────────────────────────────────────────────
    baselines = single_feature_baselines(df)
    baselines_csv = out_dir / "single_feature_baselines.csv"
    with open(baselines_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["feature", "best_threshold", "binary_accuracy_pct"])
        writer.writeheader()
        for feat, vals in baselines.items():
            writer.writerow({"feature": feat, **vals})
    print(f"[saved] {baselines_csv}")

    # ── Generate figures ───────────────────────────────────────────────────────
    print(f"\n[figs] Generating publication-quality figures...")
    generate_figures(df, out_dir)

    # ── Generate summary text ──────────────────────────────────────────────────
    generate_summary_text(df, stats_results, clf_results, baselines, out_dir)

    print(f"\n{'='*60}")
    print(f" All results saved to: {out_dir.resolve()}/")
    print(f"{'='*60}")
    print(f"\n  PAPER MAPPING:")
    print(f"  ├── subject_scores.csv           → Supplementary Table S1")
    print(f"  ├── group_summary.csv            → Table 1 (Group-level features)")
    print(f"  ├── statistics.json              → Section 5.2 (Statistical tests)")
    print(f"  ├── classification_report.json   → Table 2 (Classification metrics)")
    print(f"  ├── single_feature_baselines.csv → Table 3 (Ablation / baselines)")
    print(f"  ├── pgsi_by_group.png            → Figure 3 (PGSI distribution)")
    print(f"  ├── feature_distributions.png    → Figure 4 (Feature box plots)")
    print(f"  ├── correlation_heatmap.png      → Figure 5 (Sub-score correlations)")
    print(f"  ├── confusion_matrix.png         → Figure 6 (Confusion matrix)")
    print(f"  ├── roc_curves.png               → Figure 7 (ROC curves)")
    print(f"  └── results_summary.txt          → Copy-paste for Results section\n")


def _make_row(video_path, folder, label, raw, sub, pgsi, severity, fr):
    return {
        "subject_id": Path(video_path).stem,
        "folder": folder,
        "true_label": label,
        "predicted_label": severity,
        # Raw features
        "stride_length_raw": round(raw["stride_length"], 4),
        "posture_angle_raw": round(raw["posture_angle"], 2),
        "gait_symmetry_raw": round(raw["gait_symmetry"], 2),
        "step_timing_cv_raw": round(raw["step_timing_cv"], 2),
        "arm_swing_raw": round(raw["arm_swing"], 2),
        # Sub-scores
        "stride_length_sub": round(sub["stride_length"], 2),
        "posture_angle_sub": round(sub["posture_angle"], 2),
        "gait_symmetry_sub": round(sub["gait_symmetry"], 2),
        "step_timing_cv_sub": round(sub["step_timing_cv"], 2),
        "arm_swing_sub": round(sub["arm_swing"], 2),
        # PGSI
        "pgsi": round(pgsi, 2),
        "fall_risk": fr,
        # Quality
        "frames_processed": raw["frames_processed"],
        "frames_total": raw["frames_total"],
        "quality_pct": round(raw["quality_pct"], 1),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PGSI Results Extractor — Research Paper Edition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Subfolder-based dataset
  python extract_results.py --dataset_dir path/to/KOA-PD-NM/

  # Flat directory (filenames contain labels like NM, ML, MD, SV)
  python extract_results.py --dataset_dir data/

  # Flat directory with custom default label
  python extract_results.py --dataset_dir data/ --default-label Moderate
        """,
    )
    parser.add_argument(
        "--dataset_dir", type=str, required=True,
        help="Path to dataset folder (subfolders or flat)"
    )
    parser.add_argument(
        "--default-label", type=str, default="Mild",
        help="Default severity label when filename is ambiguous (default: Mild)"
    )
    args = parser.parse_args()
    main(args.dataset_dir, args.default_label)
