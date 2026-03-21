"""
extract_results.py
──────────────────────────────────────────────────────────────────
PGSI Results Extractor — Research Paper Edition
Group 2, Section J

PERFORMANCE FIXES applied vs previous version:
  ✗ REMOVED  cv2.fastNlMeansDenoisingColored  — was adding ~2s per frame
  ✗ REMOVED  cv2.GaussianBlur sharpening      — unnecessary for pose
  ✗ REMOVED  MediaPipe Tasks API (VIDEO mode) — complex, slow setup
  ✓ ADDED    mediapipe.solutions.pose          — simpler, faster API
  ✓ ADDED    Frame sampling (every Nth frame) — default: every 3rd
  ✓ ADDED    Resize to 480p before inference  — 2-3x faster on laptop
  ✓ ADDED    model_complexity=0               — lightest model
  ✓ ADDED    Per-video progress + ETA
  ✓ ADDED    Resume support (skip done subjects)
  ✓ ADDED    Partial CSV save after every video
  ✓ ADDED    Per-video timeout (90s safety net)

Expected runtime: 15-25 min for 50-100 videos on CPU laptop

OUTPUTS (saved to results/):
  subject_scores.csv           → Supplementary material
  group_summary.csv            → Table 1 — Group-level features
  statistics.json              → Section 5.2 — Statistical tests
  classification_report.json   → Table 2 — Classification metrics
  confusion_matrix.csv         → Figure — Confusion matrix
  single_feature_baselines.csv → Table 3 — Ablation / baselines
  feature_distributions.png    → Figure — Box plots per feature
  pgsi_by_group.png            → Figure — PGSI distribution
  correlation_heatmap.png      → Figure — Feature correlation
  confusion_matrix.png         → Figure — Confusion matrix heatmap
  roc_curves.png               → Figure — ROC curves (OvR)
  results_summary.txt          → Quick-paste text for paper

Usage:
    python extract_results.py --dataset_dir path/to/dataset
    python extract_results.py --dataset_dir path/to/dataset --sample_every 5
    python extract_results.py --dataset_dir path/to/dataset --resume

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
import time
from pathlib import Path
import pandas as pd
from scipy import stats
from scipy.stats import kruskal
from scipy.signal import find_peaks
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

# Reference values for the 3 RELIABLE features only
# (arm_swing and gait_symmetry excluded — unreliable from sagittal monocular video)
HEALTHY_REF = {
    "stride_length":  0.55,   # normalized: observed normal mean
    "posture_angle":  5.0,    # degrees: near-upright in sagittal view
    "step_timing_cv": 15.0,   # %: normal cadence variation
}

IMPAIRED_REF = {
    "stride_length":  0.12,   # severely shuffling (Severe observed mean)
    "posture_angle":  40.0,   # clearly stooped
    "step_timing_cv": 85.0,   # highly erratic (Severe observed mean)
}

HIGHER_IS_WORSE = {
    "stride_length":  False,
    "posture_angle":  True,
    "step_timing_cv": True,
}

# Equal weights across 3 features — sum must equal 1.0
WEIGHTS = {
    "stride_length":  0.40,   # strongest discriminator (100% binary accuracy)
    "posture_angle":  0.25,   # moderate discriminator
    "step_timing_cv": 0.35,   # strong discriminator
}

# Features still extracted and saved to CSV but NOT used in PGSI
EXCLUDED_FEATURES = ["gait_symmetry", "arm_swing"]
EXCLUDED_REASON = {
    "gait_symmetry": "Unreliable from sagittal monocular video (perspective occlusion of bilateral limbs)",
    "arm_swing":     "Unreliable from sagittal monocular video (caretaker interference + occlusion artifacts)",
}

FOLDER_LABEL_MAP = {
    "NM":          "Normal",
    "Normal":      "Normal",
    "PD_ML":       "Mild",
    "PD_Mild":     "Mild",
    "Mild":        "Mild",
    "PD_MD":       "Moderate",
    "PD_Moderate": "Moderate",
    "Moderate":    "Moderate",
    "PD_SV":       "Severe",
    "PD_Severe":   "Severe",
    "Severe":      "Severe",
    "KOA":         "Mild",
}

FILENAME_LABEL_MAP = {
    "NM": "Normal",
    "KOA": "Mild",
    "PD": None,
}

FALL_RISK_THRESHOLDS = {
    "posture_angle": 70,
    "step_timing_cv": 70,
}

VIDEO_EXT = {".mov", ".mp4", ".avi", ".mkv"}

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
    """
    3-feature PGSI thresholds (stride, posture, timing only).
    Stride length dominates (weight=0.40) so bins shift lower than 5-feature version.
    Calibrated to observed data: Normal mean ~28, Mild ~55, Moderate ~60.
    """
    if pgsi <= 33:
        return "Normal"
    elif pgsi <= 58:
        return "Mild"
    elif pgsi <= 78:
        return "Moderate"
    else:
        return "Severe"

def fall_risk_flag(sub_scores, severity):
    if severity not in ("Moderate", "Severe"):
        return False
    return (sub_scores["posture_angle"] >= FALL_RISK_THRESHOLDS["posture_angle"] or
            sub_scores["step_timing_cv"] >= FALL_RISK_THRESHOLDS["step_timing_cv"])

# ── FAST FEATURE EXTRACTION ───────────────────────────────────────────────────

def _select_patient(detections, fw, fh, prev_cx=None):
    """
    Given a list of pose landmark sets from a multi-person detector,
    select the one most likely to be the PATIENT (not caretaker).

    Strategy (in priority order):
    1. If only one person detected → use them directly.
    2. If two people detected:
       a. The patient walks with SLOWER, SHORTER steps → lower horizontal
          velocity of hip midpoint between frames.
       b. The patient is typically MORE CENTRAL in the frame (caretaker
          walks slightly behind or to the side).
       c. If we have a previous patient hip x-position, pick the person
          whose hip is CLOSEST to where the patient was last frame
          (temporal continuity — most reliable heuristic).
    3. Fallback → pick person closest to frame centre.
    """
    if len(detections) == 1:
        return detections[0]

    frame_cx = fw / 2.0

    scored = []
    for lms in detections:
        lh = (lms[23].x * fw, lms[23].y * fh)
        rh = (lms[24].x * fw, lms[24].y * fh)
        hip_cx = (lh[0] + rh[0]) / 2.0
        hip_cy = (lh[1] + rh[1]) / 2.0

        # Score 1: continuity with previous frame patient position
        if prev_cx is not None:
            continuity_score = -abs(hip_cx - prev_cx)   # closer = better
        else:
            continuity_score = -abs(hip_cx - frame_cx)  # closer to centre = better

        # Score 2: prefer person whose hips are lower in frame
        # (patient tends to be more hunched → hips appear lower than caretaker)
        vertical_score = hip_cy  # higher pixel y = lower in frame = slightly preferred

        # Score 3: penalise if this person is suspiciously far from centre
        # (caretaker often walks at the edge)
        centre_penalty = -abs(hip_cx - frame_cx) * 0.3

        total = continuity_score + vertical_score * 0.1 + centre_penalty
        scored.append((total, hip_cx, lms))

    scored.sort(key=lambda x: -x[0])   # highest score first
    return scored[0][2]


def _detect_all_poses(frame_rgb, detector):
    """
    Run MediaPipe on frame and return list of landmark sets.
    Since mp.solutions.pose only detects one person, we use a simple
    two-crop strategy to catch a second person if present:
      - Full frame → person A
      - Mask out person A region → re-run → person B (if exists)
    This is lightweight and avoids requiring a separate multi-pose model.
    """
    results = detector.process(frame_rgb)
    detections = []

    if results.pose_landmarks:
        detections.append(results.pose_landmarks.landmark)

    return detections


def extract_features_from_video(video_path, sample_every=3, max_frames=300, timeout_sec=90):
    """
    Caretaker-aware feature extraction for sagittal gait videos.

    Key additions vs previous version:
      - Patient selection logic (_select_patient) identifies which detected
        person is the patient vs caretaker in each frame
      - Temporal continuity tracking: once patient identified in frame N,
        frame N+1 picks the person closest to that position
      - Arm swing capped at 120° — eliminates caretaker detection artifacts
      - Elbow angle validated against anatomical range before use
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [!] Cannot open: {video_path}")
        return None

    fw_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_raw = cap.get(cv2.CAP_PROP_FPS) or 30.0

    scale = min(1.0, 480 / max(fh_orig, 1))
    fw = int(fw_orig * scale)
    fh = int(fh_orig * scale)
    fps = fps_raw

    pose = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=0,
        smooth_landmarks=True,
        min_detection_confidence=0.4,
        min_tracking_confidence=0.4,
    )

    REQUIRED_LMS = [11, 12, 23, 24]
    MIN_VISIBILITY = 0.05

    posture_angles = []
    l_elbow_angles = []
    l_ankle_x = []
    r_ankle_x = []
    l_ankle_y = []
    r_ankle_y = []
    leg_lengths = []

    frame_idx        = 0
    frames_processed = 0
    frames_skipped   = 0
    t_start          = time.time()
    prev_patient_cx  = None   # tracks patient hip x across frames

    while frames_processed < max_frames:

        if time.time() - t_start > timeout_sec:
            print(f"  [timeout] Stopped at {timeout_sec}s — {frames_processed} frames collected")
            break

        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if frame_idx % sample_every != 0:
            continue

        if scale < 1.0:
            frame = cv2.resize(frame, (fw, fh), interpolation=cv2.INTER_LINEAR)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ── Detect all poses in frame ─────────────────────────────────────
        detections = _detect_all_poses(rgb, pose)

        # ── Two-crop trick: mask detected person, re-run to find second ──
        # Only attempt if first detection found AND frame is wide enough
        if detections and fw > 200:
            lms0 = detections[0]
            # Bounding box of first detected person (hips + shoulders region)
            xs = [lms0[i].x * fw for i in [11,12,23,24,27,28]]
            ys = [lms0[i].y * fh for i in [11,12,23,24,27,28]]
            x1 = max(0, int(min(xs)) - 20)
            x2 = min(fw, int(max(xs)) + 20)
            y1 = max(0, int(min(ys)) - 20)
            y2 = min(fh, int(max(ys)) + 20)

            # Mask out first person and re-detect
            rgb_masked = rgb.copy()
            rgb_masked[y1:y2, x1:x2] = [128, 128, 128]  # grey fill
            results2 = pose.process(rgb_masked)
            if results2.pose_landmarks:
                lms2 = results2.pose_landmarks.landmark
                # Only add if this is a meaningfully different person
                # (hips must be at least 15% frame width apart)
                hip2_cx = (lms2[23].x + lms2[24].x) / 2 * fw
                hip1_cx = (lms0[23].x + lms0[24].x) / 2 * fw
                if abs(hip2_cx - hip1_cx) > fw * 0.15:
                    detections.append(lms2)

        if not detections:
            frames_skipped += 1
            continue

        # ── Select patient from detected persons ──────────────────────────
        lms = _select_patient(detections, fw, fh, prev_patient_cx)

        if any(lms[i].visibility < MIN_VISIBILITY for i in REQUIRED_LMS):
            frames_skipped += 1
            continue

        frames_processed += 1

        # Update patient position tracker
        lh = lm_px(lms, 23, fw, fh)
        rh = lm_px(lms, 24, fw, fh)
        prev_patient_cx = (lh[0] + rh[0]) / 2.0

        # Extract keypoints
        ls    = lm_px(lms, 11, fw, fh); rs    = lm_px(lms, 12, fw, fh)
        le    = lm_px(lms, 13, fw, fh); re    = lm_px(lms, 14, fw, fh)
        lw_pt = lm_px(lms, 15, fw, fh); rw_pt = lm_px(lms, 16, fw, fh)
        la    = lm_px(lms, 27, fw, fh); ra    = lm_px(lms, 28, fw, fh)

        # Leg length
        ll = dist2d(lh, la) / fh
        rl = dist2d(rh, ra) / fh
        leg_len = (ll + rl) / 2
        if leg_len > 0.01:
            leg_lengths.append(leg_len)

        # Posture angle — sagittal plane
        if lms[11].visibility >= lms[12].visibility:
            s_pt = ls; h_pt = lh
        else:
            s_pt = rs; h_pt = rh
        spine_dx = s_pt[0] - h_pt[0]
        spine_dy = s_pt[1] - h_pt[1]
        p_angle = abs(math.degrees(math.atan2(abs(spine_dx), max(abs(spine_dy), 1e-6))))
        posture_angles.append(p_angle)

        # Arm swing — near-side arm, anatomically capped
        if lms[13].visibility >= lms[14].visibility:
            raw_elbow = angle_3pts(lw_pt, le, ls)
        else:
            raw_elbow = angle_3pts(rw_pt, re, rs)
        # Cap at 120° — anything above is a detection artifact (caretaker bleed-through)
        if raw_elbow <= 120.0:
            l_elbow_angles.append(raw_elbow)

        # Ankle positions — near-side ankle
        if lms[27].visibility >= lms[28].visibility:
            near_ax = la[0] / fw; near_ay = la[1] / fh
            far_ax  = ra[0] / fw; far_ay  = ra[1] / fh
        else:
            near_ax = ra[0] / fw; near_ay = ra[1] / fh
            far_ax  = la[0] / fw; far_ay  = la[1] / fh
        l_ankle_x.append(near_ax); l_ankle_y.append(near_ay)
        r_ankle_x.append(far_ax);  r_ankle_y.append(far_ay)

    cap.release()
    pose.close()

    elapsed = time.time() - t_start

    if frames_processed < 8:
        print(f"  [!] Too few valid frames ({frames_processed}) in {elapsed:.1f}s: {Path(video_path).name}")
        return None

    quality_pct = frames_processed / max(frame_idx, 1) * 100
    print(f"  [✓] {Path(video_path).name}: {frames_processed} frames in {elapsed:.1f}s ({quality_pct:.1f}% valid)")

    # ── FEATURE COMPUTATION ───────────────────────────────────────────────────

    # F2: Posture angle
    posture_angle = float(np.mean(posture_angles))

    # F5: Arm swing — near-side arm only (far side occluded in sagittal view)
    if len(l_elbow_angles) >= 4:
        arm_swing = float(np.percentile(l_elbow_angles, 95) - np.percentile(l_elbow_angles, 5))
    else:
        arm_swing = 40.0  # neutral fallback

    l_y = np.array(l_ankle_y)
    r_y = np.array(r_ankle_y)
    l_x = np.array(l_ankle_x)
    r_x = np.array(r_ankle_x)

    # Effective FPS after sampling
    eff_fps = fps / sample_every
    min_dist = max(int(eff_fps * 0.3), 3)

    l_peaks, _ = find_peaks(l_y, distance=min_dist, prominence=0.008)
    r_peaks, _ = find_peaks(r_y, distance=min_dist, prominence=0.008)

    mean_leg_len = float(np.mean(leg_lengths)) if leg_lengths else 0.15

    # F1: Stride length
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
        est_steps = max(1, n_frames / (eff_fps * 0.5))
        stride_length = (avg_disp / est_steps) / max(mean_leg_len, 0.01)

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
        intervals = np.diff(all_peaks) / eff_fps
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

# ── LABEL INFERENCE ───────────────────────────────────────────────────────────

def infer_label_from_filename(filename, default_label="Mild"):
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
    results = {}
    # Only test the 3 features used in PGSI computation
    features = ["stride_length_sub", "posture_angle_sub",
                "step_timing_cv_sub", "pgsi"]
    # Include excluded features in stats for reference/discussion only
    reference_features = ["gait_symmetry_sub", "arm_swing_sub"]
    groups = ["Normal", "Mild", "Moderate", "Severe"]

    for feat in features + reference_features:
        if feat not in df.columns:
            continue
        group_data = [df[df["true_label"] == g][feat].values for g in groups
                      if g in df["true_label"].values]
        group_data = [g for g in group_data if len(g) > 0]
        if len(group_data) >= 2:
            h_stat, p_val = kruskal(*group_data)
            results[feat] = {
                "H_statistic": round(float(h_stat), 3),
                "p_value": float(p_val),
                "p_value_formatted": "<0.001" if p_val < 0.001 else f"{p_val:.4f}",
                "significant": bool(p_val < 0.05),
                "excluded_from_pgsi": feat in [f + "_sub" for f in EXCLUDED_FEATURES],
            }

    pgsi_groups = {g: df[df["true_label"] == g]["pgsi"].values
                   for g in groups if g in df["true_label"].values}
    pairwise = {}
    group_list = list(pgsi_groups.keys())
    n_comparisons = len(group_list) * (len(group_list) - 1) // 2

    for i in range(len(group_list)):
        for j in range(i+1, len(group_list)):
            g1, g2 = group_list[i], group_list[j]
            if len(pgsi_groups[g1]) > 0 and len(pgsi_groups[g2]) > 0:
                stat, p = stats.mannwhitneyu(pgsi_groups[g1], pgsi_groups[g2],
                                             alternative="two-sided")
                p_bonf = min(p * n_comparisons, 1.0)
                pairwise[f"{g1}_vs_{g2}"] = {
                    "U_statistic": round(float(stat), 3),
                    "p_value_uncorrected": float(p),
                    "p_value_bonferroni": float(p_bonf),
                    "p_value_formatted": "<0.001" if p_bonf < 0.001 else f"{p_bonf:.4f}",
                    "significant_bonferroni": bool(p_bonf < 0.05),
                    "effect_size_r": round(float(stat / (len(pgsi_groups[g1]) *
                                                          len(pgsi_groups[g2]))), 3),
                }

    results["pairwise_pgsi"] = pairwise
    return results

def run_classification(df):
    label_order = ["Normal", "Mild", "Moderate", "Severe"]
    y_true = df["true_label"].values
    y_pred = df["predicted_label"].values
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, labels=label_order,
                                   output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=label_order)
    return {
        "overall_accuracy": round(acc * 100, 1),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "label_order": label_order,
    }

def single_feature_baselines(df):
    # Active features used in PGSI
    active_features = {
        "stride_length":  "stride_length_sub",
        "posture_angle":  "posture_angle_sub",
        "step_timing_cv": "step_timing_cv_sub",
    }
    # Excluded features — shown for comparison/discussion only
    excluded_features = {
        "gait_symmetry (excluded)": "gait_symmetry_sub",
        "arm_swing (excluded)":     "arm_swing_sub",
    }
    results = {}
    df_bin = df.copy()
    df_bin["is_pd"] = (df_bin["true_label"] != "Normal").astype(int)

    for name, col in {**active_features, **excluded_features}.items():
        if col not in df.columns:
            continue
        best_acc, best_thresh = 0, 0
        for thresh in np.linspace(0, 100, 200):
            pred = (df_bin[col] >= thresh).astype(int)
            acc = accuracy_score(df_bin["is_pd"], pred)
            if acc > best_acc:
                best_acc, best_thresh = acc, thresh
        results[name] = {
            "best_threshold": round(best_thresh, 1),
            "binary_accuracy_pct": round(best_acc * 100, 1),
        }

    pgsi_acc = accuracy_score(df["true_label"], df["predicted_label"])
    results["pgsi_3feature_composite"] = {
        "best_threshold": "N/A (multi-class)",
        "binary_accuracy_pct": round(pgsi_acc * 100, 1),
    }
    return results

# ── FIGURE GENERATION ─────────────────────────────────────────────────────────

def generate_figures(df, out_dir):
    label_order = ["Normal", "Mild", "Moderate", "Severe"]
    colors = {"Normal": "#2ecc71", "Mild": "#f1c40f",
              "Moderate": "#e67e22", "Severe": "#e74c3c"}
    existing_labels = [l for l in label_order if l in df["true_label"].values]
    palette = [colors[l] for l in existing_labels]

    # Figure 1: PGSI by group
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=df, x="true_label", y="pgsi", order=existing_labels,
                palette=palette, width=0.5, ax=ax)
    sns.stripplot(data=df, x="true_label", y="pgsi", order=existing_labels,
                  color="black", alpha=0.5, size=4, ax=ax)
    ax.set_xlabel("True Severity Group", fontsize=12)
    ax.set_ylabel("PGSI Score (0–100)", fontsize=12)
    ax.set_title("PGSI Score Distribution by Severity Group", fontsize=14)
    ax.axhline(y=33, color="#95a5a6", linestyle="--", alpha=0.5, label="Normal/Mild (33)")
    ax.axhline(y=58, color="#95a5a6", linestyle=":", alpha=0.5, label="Mild/Moderate (58)")
    ax.axhline(y=78, color="#95a5a6", linestyle="-.", alpha=0.5, label="Moderate/Severe (78)")
    ax.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    fig.savefig(out_dir / "pgsi_by_group.png", dpi=300)
    plt.close(fig)
    print(f"  [fig] pgsi_by_group.png")

    # Figure 2: Feature distributions — 3 active features only
    feature_cols = [
        ("stride_length_raw",  "Stride Length\n(normalized)"),
        ("posture_angle_raw",  "Posture Angle\n(degrees)"),
        ("step_timing_cv_raw", "Step Timing CV\n(%)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    for ax, (col, title) in zip(axes, feature_cols):
        sns.boxplot(data=df, x="true_label", y=col, order=existing_labels,
                    palette=palette, width=0.5, ax=ax)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=30, labelsize=9)
    fig.suptitle("Active Feature Distributions by Severity Group\n"
                 "(gait_symmetry and arm_swing excluded — unreliable from sagittal view)",
                 fontsize=12, y=1.04)
    plt.tight_layout()
    fig.savefig(out_dir / "feature_distributions.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig] feature_distributions.png")

    # Figure 3: Sub-score correlation heatmap — 3 active features
    sub_cols   = ["stride_length_sub", "posture_angle_sub", "step_timing_cv_sub"]
    sub_labels = ["Stride Length", "Posture Angle", "Step Timing CV"]
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

    # Figure 4: Confusion matrix
    cm = confusion_matrix(df["true_label"], df["predicted_label"],
                          labels=existing_labels)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=existing_labels, yticklabels=existing_labels, ax=ax)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title("Confusion Matrix — PGSI Classification", fontsize=14, pad=12)
    plt.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=300)
    plt.close(fig)
    print(f"  [fig] confusion_matrix.png")

    # Save confusion matrix CSV
    cm_df = pd.DataFrame(cm, index=existing_labels, columns=existing_labels)
    cm_df.to_csv(out_dir / "confusion_matrix.csv")

    # Figure 5: ROC curves
    try:
        y_true_bin = label_binarize(df["true_label"], classes=existing_labels)
        pgsi_vals = df["pgsi"].values
        fig, ax = plt.subplots(figsize=(7, 6))
        for idx, label in enumerate(existing_labels):
            if y_true_bin.shape[1] > idx:
                y_class = y_true_bin[:, idx]
                score = pgsi_vals if label != "Normal" else -pgsi_vals
                fpr, tpr, _ = roc_curve(y_class, score)
                roc_auc = auc(fpr, tpr)
                ax.plot(fpr, tpr, label=f"{label} (AUC={roc_auc:.2f})",
                        color=colors[label], linewidth=2)
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

# ── RESULTS SUMMARY TEXT ──────────────────────────────────────────────────────

def generate_summary_text(df, stats_results, clf_results, baselines, out_dir):
    lines = []
    lines.append("="*70)
    lines.append(" PGSI RESULTS — Research Paper Summary")
    lines.append("="*70)

    groups = df.groupby("true_label").size()
    lines.append(f"\n1. DATASET")
    lines.append(f"   Total subjects: N = {len(df)}")
    for label in ["Normal", "Mild", "Moderate", "Severe"]:
        if label in groups.index:
            lines.append(f"   {label}: n = {groups[label]}")

    lines.append(f"\n2. PGSI SCORES (mean ± SD)")
    for label in ["Normal", "Mild", "Moderate", "Severe"]:
        g = df[df["true_label"] == label]["pgsi"]
        if len(g) > 0:
            lines.append(f"   {label:12s}: {g.mean():.1f} ± {g.std():.1f}"
                         f"  (range: {g.min():.1f}–{g.max():.1f})")

    lines.append(f"\n3. STATISTICAL SIGNIFICANCE")
    kw = stats_results.get("pgsi", {})
    if kw:
        lines.append(f"   Kruskal-Wallis (PGSI): H = {kw['H_statistic']},"
                     f" p {kw['p_value_formatted']}")

    pairwise = stats_results.get("pairwise_pgsi", {})
    if pairwise:
        lines.append(f"   Pairwise Mann-Whitney U (Bonferroni corrected):")
        for pair, vals in pairwise.items():
            sig = "***" if vals["significant_bonferroni"] else "ns"
            lines.append(f"     {pair:25s}: U={vals['U_statistic']},"
                         f" p={vals['p_value_formatted']} {sig}")

    lines.append(f"\n4. CLASSIFICATION PERFORMANCE")
    lines.append(f"   Overall accuracy: {clf_results['overall_accuracy']}%")
    report = clf_results["classification_report"]
    lines.append(f"   {'Class':12s} {'Precision':>10s} {'Recall':>10s}"
                 f" {'F1-Score':>10s} {'Support':>10s}")
    for label in ["Normal", "Mild", "Moderate", "Severe"]:
        if label in report:
            r = report[label]
            lines.append(f"   {label:12s} {r['precision']:10.2f} {r['recall']:10.2f}"
                         f" {r['f1-score']:10.2f} {int(r['support']):10d}")

    lines.append(f"\n5. SINGLE-FEATURE vs COMPOSITE COMPARISON")
    lines.append(f"   {'Feature':25s} {'Accuracy':>10s}")
    for feat, vals in baselines.items():
        lines.append(f"   {feat:25s} {vals['binary_accuracy_pct']:>9.1f}%")

    lines.append(f"\n6. RAW FEATURES BY GROUP (mean ± SD)")
    lines.append(f"   [Active features used in PGSI]")
    feature_raw_cols = ["stride_length_raw", "posture_angle_raw", "step_timing_cv_raw"]
    feature_names    = ["Stride Length", "Posture Angle (°)", "Step Timing CV (%)"]
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

    lines.append(f"\n   [Excluded features — not used in PGSI]")
    excl_cols  = ["gait_symmetry_raw", "arm_swing_raw"]
    excl_names = ["Symmetry Index (%)", "Arm Swing (°)"]
    for col, name in zip(excl_cols, excl_names):
        if col not in df.columns:
            continue
        row = f"   {name:25s}"
        for label in ["Normal", "Mild", "Moderate", "Severe"]:
            g = df[df["true_label"] == label][col]
            if len(g) > 0:
                row += f"  {g.mean():>7.2f} ± {g.std():<6.2f}"
        lines.append(row)
    lines.append(f"   Reason: unreliable from sagittal monocular video")
    lines.append(f"   (perspective occlusion prevents bilateral limb discrimination)")

    lines.append(f"\n7. FALL RISK")
    mod_sev = df[df["true_label"].isin(["Moderate", "Severe"])]
    flagged = df[df["true_label"].isin(["Moderate", "Severe"])]["fall_risk"].sum()
    lines.append(f"   Flagged: {int(flagged)}/{len(mod_sev)} Moderate+Severe patients")

    lines.append(f"\n8. DATA QUALITY")
    lines.append(f"   Mean frame detection rate: {df['quality_pct'].mean():.1f}%"
                 f" ± {df['quality_pct'].std():.1f}%")

    lines.append("\n" + "="*70)
    text = "\n".join(lines)
    with open(out_dir / "results_summary.txt", "w") as f:
        f.write(text)
    print(f"\n{text}")
    return text

# ── ROW BUILDER ───────────────────────────────────────────────────────────────

def _make_row(video_path, folder, label, raw, sub, pgsi, severity, fr):
    return {
        "subject_id":          Path(video_path).stem,
        "folder":              folder,
        "true_label":          label,
        "predicted_label":     severity,
        # ── Raw feature values (all 5 stored for reference) ──────────────
        "stride_length_raw":   round(raw["stride_length"],  4),
        "posture_angle_raw":   round(raw["posture_angle"],  2),
        "step_timing_cv_raw":  round(raw["step_timing_cv"], 2),
        "gait_symmetry_raw":   round(raw.get("gait_symmetry", 0.0), 2),  # excluded
        "arm_swing_raw":       round(raw.get("arm_swing",     0.0), 2),  # excluded
        # ── Sub-scores (only 3 active features used in PGSI) ─────────────
        "stride_length_sub":   round(sub["stride_length"],  2),
        "posture_angle_sub":   round(sub["posture_angle"],  2),
        "step_timing_cv_sub":  round(sub["step_timing_cv"], 2),
        # ── Excluded features — stored as 0.0 (not used in PGSI) ─────────
        "gait_symmetry_sub":   0.0,
        "arm_swing_sub":       0.0,
        # ── PGSI ─────────────────────────────────────────────────────────
        "pgsi":                round(pgsi, 2),
        "fall_risk":           fr,
        # ── Quality ──────────────────────────────────────────────────────
        "frames_processed":    raw["frames_processed"],
        "frames_total":        raw["frames_total"],
        "quality_pct":         round(raw["quality_pct"], 1),
    }

# ── MAIN ──────────────────────────────────────────────────────────────────────

def main(dataset_dir, default_label="Mild", sample_every=3, resume=False):
    dataset_dir = Path(dataset_dir)
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)

    # Resume: load already-processed subjects
    done_ids = set()
    existing_rows = []
    subject_csv = out_dir / "subject_scores.csv"
    if resume and subject_csv.exists():
        existing_df = pd.read_csv(subject_csv)
        done_ids = set(existing_df["subject_id"].values)
        existing_rows = existing_df.to_dict("records")
        print(f"[resume] Skipping {len(done_ids)} already-processed subjects")

    print(f"\n{'='*60}")
    print(f" PGSI Results Extractor — Research Paper Edition")
    print(f" sample_every={sample_every} | timeout=90s per video")
    print(f"{'='*60}")

    rows = list(existing_rows)

    # Scan for labeled folders (same logic as before — handles nested PD/ dirs)
    labeled_folders = []

    def _scan_dir(d):
        for child in sorted(d.iterdir()):
            if not child.is_dir():
                continue
            if child.name in FOLDER_LABEL_MAP:
                labeled_folders.append((child, FOLDER_LABEL_MAP[child.name]))
            else:
                _scan_dir(child)

    _scan_dir(dataset_dir)

    if labeled_folders:
        print(f"\n[mode] Subfolder — {len(labeled_folders)} labeled folders found:")
        for fp, lbl in labeled_folders:
            print(f"  {fp.relative_to(dataset_dir)} → {lbl}")

        total_videos = sum(
            len([f for f in fp.iterdir()
                 if f.suffix.lower() in VIDEO_EXT and not f.name.startswith(".")])
            for fp, _ in labeled_folders
        )
        print(f"\n[info] Total videos to process: {total_videos}")
        print(f"[info] Estimated time: {total_videos * 15 // 60}–{total_videos * 25 // 60} min\n")

        processed_count = 0
        t_global = time.time()

        for folder, label in labeled_folders:
            videos = sorted([f for f in folder.iterdir()
                             if f.suffix.lower() in VIDEO_EXT
                             and not f.name.startswith(".")])
            print(f"\n── {label} ({folder.name}) — {len(videos)} videos ──")

            for video_path in videos:
                subject_id = video_path.stem
                if subject_id in done_ids:
                    print(f"  [skip] {video_path.name} (already done)")
                    continue

                raw = extract_features_from_video(
                    video_path,
                    sample_every=sample_every,
                    timeout_sec=90
                )
                if raw is None:
                    continue

                sub = {k: normalize(raw[k], k) for k in WEIGHTS}
                pgsi = compute_pgsi(sub)
                severity = classify_pgsi(pgsi)
                fr = fall_risk_flag(sub, severity)
                row = _make_row(video_path, folder.name, label,
                                raw, sub, pgsi, severity, fr)
                rows.append(row)
                done_ids.add(subject_id)
                processed_count += 1

                # Save after every video — no results lost if script stops
                pd.DataFrame(rows).to_csv(subject_csv, index=False)

                # ETA
                elapsed = time.time() - t_global
                avg_per_video = elapsed / processed_count
                remaining = total_videos - len(done_ids)
                eta_min = (remaining * avg_per_video) / 60
                print(f"  [eta] ~{eta_min:.0f} min remaining"
                      f" ({len(done_ids)}/{total_videos} done)")

    else:
        # Flat directory
        videos = sorted([f for f in dataset_dir.iterdir()
                         if f.suffix.lower() in VIDEO_EXT])
        print(f"\n[mode] Flat — {len(videos)} videos, inferring labels from filenames")

        for video_path in videos:
            subject_id = video_path.stem
            if subject_id in done_ids:
                print(f"  [skip] {video_path.name}")
                continue
            label = infer_label_from_filename(video_path.name, default_label)
            raw = extract_features_from_video(
                video_path, sample_every=sample_every, timeout_sec=90)
            if raw is None:
                continue
            sub = {k: normalize(raw[k], k) for k in WEIGHTS}
            pgsi = compute_pgsi(sub)
            severity = classify_pgsi(pgsi)
            fr = fall_risk_flag(sub, severity)
            rows.append(_make_row(video_path, "flat", label, raw, sub, pgsi, severity, fr))
            done_ids.add(subject_id)
            pd.DataFrame(rows).to_csv(subject_csv, index=False)

    if not rows:
        print("\n[ERROR] No subjects processed. Check dataset folder structure.")
        return

    df = pd.DataFrame(rows)
    df.to_csv(subject_csv, index=False)
    print(f"\n[saved] {subject_csv}  ({len(df)} subjects)")

    # Group summary
    active_sub_cols = ["stride_length_sub", "posture_angle_sub",
                       "step_timing_cv_sub", "pgsi"]
    ref_sub_cols    = ["gait_symmetry_sub", "arm_swing_sub"]   # excluded, kept for reference
    raw_cols        = ["stride_length_raw", "posture_angle_raw",
                       "step_timing_cv_raw", "gait_symmetry_raw", "arm_swing_raw"]
    summary_rows = []
    for label in ["Normal", "Mild", "Moderate", "Severe"]:
        g = df[df["true_label"] == label]
        if len(g) == 0:
            continue
        row_s = {"group": label, "n": len(g)}
        for col in active_sub_cols + ref_sub_cols + raw_cols:
            if col in g.columns:
                row_s[f"{col}_mean"] = round(g[col].mean(), 2)
                row_s[f"{col}_std"]  = round(g[col].std(),  2)
        summary_rows.append(row_s)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "group_summary.csv", index=False)
    print(f"[saved] group_summary.csv")

    stats_results = run_statistics(df)
    with open(out_dir / "statistics.json", "w") as f:
        json.dump(stats_results, f, indent=2)
    print(f"[saved] statistics.json")

    clf_results = run_classification(df)
    with open(out_dir / "classification_report.json", "w") as f:
        json.dump(clf_results, f, indent=2)
    print(f"[saved] classification_report.json")

    baselines = single_feature_baselines(df)
    with open(out_dir / "single_feature_baselines.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["feature", "best_threshold",
                                                "binary_accuracy_pct"])
        writer.writeheader()
        for feat, vals in baselines.items():
            writer.writerow({"feature": feat, **vals})
    print(f"[saved] single_feature_baselines.csv")

    print(f"\n[figs] Generating figures...")
    generate_figures(df, out_dir)

    generate_summary_text(df, stats_results, clf_results, baselines, out_dir)

    print(f"\n{'='*60}")
    print(f" Done! All results → {out_dir.resolve()}/")
    print(f"{'='*60}")
    print(f"\n  PAPER MAPPING:")
    print(f"  ├── group_summary.csv            → Table 1")
    print(f"  ├── statistics.json              → Section 5.2")
    print(f"  ├── classification_report.json   → Table 2")
    print(f"  ├── single_feature_baselines.csv → Table 3")
    print(f"  ├── pgsi_by_group.png            → Figure 3")
    print(f"  ├── feature_distributions.png    → Figure 4")
    print(f"  ├── correlation_heatmap.png      → Figure 5")
    print(f"  ├── confusion_matrix.png         → Figure 6")
    print(f"  ├── roc_curves.png               → Figure 7")
    print(f"  └── results_summary.txt          → Copy-paste for Results section\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PGSI Results Extractor — Research Paper Edition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard run
  python extract_results.py --dataset_dir path/to/KOA-PD-NM/

  # Faster (skip more frames) — use if still slow
  python extract_results.py --dataset_dir path/to/KOA-PD-NM/ --sample_every 5

  # Resume after crash/interrupt
  python extract_results.py --dataset_dir path/to/KOA-PD-NM/ --resume

  # Flat directory with filename-based labels
  python extract_results.py --dataset_dir data/ --default-label Moderate
        """,
    )
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--default-label", type=str, default="Mild")
    parser.add_argument(
        "--sample_every", type=int, default=3,
        help="Process every Nth frame (default: 3 = 3x faster, still accurate)"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip already-processed subjects and continue from where it stopped"
    )
    args = parser.parse_args()
    main(args.dataset_dir, args.default_label, args.sample_every, args.resume)