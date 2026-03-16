"""
run_pgsi.py — CLI entry point for batch PGSI processing.

Usage:
    python run_pgsi.py --video path/to/video.mp4
    python run_pgsi.py --video path/to/video.mp4 --patient-id P001 --output output/
    python run_pgsi.py --batch data/videos/ --output output/
"""

import argparse
import os
import sys
import json
import time
from datetime import datetime

from config import TARGET_FPS, OUTPUT_DIR, SUPPORTED_VIDEO_EXTENSIONS
from preprocessing.video_processor import VideoProcessor
from pose.pose_estimator import PoseEstimator
from features.gait_features import GaitFeatureExtractor
from scoring.pgsi_scorer import PGSIScorer
from reporting.report_generator import ReportGenerator
from visualization.charts import render_skeleton_overlay


def process_single_video(
    video_path: str,
    patient_id: str = "N/A",
    session_tag: str = "",
    output_dir: str = OUTPUT_DIR,
    generate_report: bool = True,
    verbose: bool = True,
    weights_path: str = "",
    use_bg_sub: bool = False,
) -> dict:
    """Run the full PGSI pipeline on a single video."""

    start_time = time.time()

    if verbose:
        print(f"\n{'='*60}")
        print(f"  PGSI — Parkinsonian Gait Severity Index")
        print(f"  Video: {video_path}")
        print(f"{'='*60}\n")

    # ── Stage 1: Video Preprocessing ──
    if verbose:
        print("[1/5] Preprocessing video...")
    vp = VideoProcessor(video_path)
    frames = vp.preprocess_all(use_bg_subtraction=use_bg_sub)
    if verbose:
        bg_note = " (with background subtraction)" if use_bg_sub else ""
        print(f"      → {len(frames)} frames at {TARGET_FPS} FPS "
              f"(duration: {vp.duration_seconds:.1f}s){bg_note}")
    vp.release()

    # ── Stage 2: Pose Estimation ──
    if verbose:
        print("[2/5] Estimating body pose...")
    estimator = PoseEstimator()
    keypoints = estimator.process_video_frames(frames)
    valid = sum(1 for kf in keypoints if kf is not None)
    if verbose:
        print(f"      → Pose detected in {valid}/{len(keypoints)} frames "
              f"({valid/len(keypoints)*100:.1f}%)")

    if valid < 10:
        print("ERROR: Too few valid pose frames. Cannot compute PGSI.")
        estimator.close()
        return {"error": "Insufficient pose detections"}

    # Capture skeleton overlay from a mid-video frame with valid keypoints
    skeleton_image_bytes = None
    valid_indices = [i for i, kf in enumerate(keypoints) if kf is not None]
    if valid_indices:
        mid_idx = valid_indices[len(valid_indices) // 2]
        skeleton_image_bytes = render_skeleton_overlay(frames[mid_idx], keypoints[mid_idx])

    # ── Stage 3: Feature Extraction ──
    if verbose:
        print("[3/5] Extracting biomechanical features...")
    extractor = GaitFeatureExtractor(fps=TARGET_FPS)
    features = extractor.extract_all(estimator, keypoints)

    if verbose:
        print(f"      → Stride Length (mean): {features.mean_stride_length:.4f}")
        print(f"      → Posture Angle (mean): {features.mean_posture_angle:.1f}°")
        print(f"      → Symmetry Index:       {features.mean_symmetry_index:.1f}%")
        print(f"      → Step Timing CV:       {features.step_timing_cv:.1f}%")
        print(f"      → Arm Swing (mean):     {features.mean_arm_swing:.1f}°")

    # ── Stage 4: PGSI Scoring ──
    if verbose:
        print("[4/5] Computing PGSI score...")

    # Load custom weights if provided
    custom_weights = None
    if weights_path and os.path.isfile(weights_path):
        with open(weights_path, "r") as f:
            custom_weights = json.load(f)
        if verbose:
            print(f"      → Loaded custom weights from {weights_path}")

    scorer = PGSIScorer(weights=custom_weights)
    result = scorer.assess(features)

    if verbose:
        print(f"\n      ╔══════════════════════════════════╗")
        print(f"      ║  PGSI Score:  {result.pgsi_score:6.1f} / 100     ║")
        print(f"      ║  Severity:    {result.severity_label:>18s}  ║")
        print(f"      ║  Fall Risk:   {'YES ⚠️' if result.fall_risk else 'No  ✓':>18s}  ║")
        print(f"      ╚══════════════════════════════════╝\n")

        print("      Sub-Scores:")
        for key, val in result.sub_scores.items():
            bar = "█" * int(val / 5) + "░" * (20 - int(val / 5))
            print(f"        {key:15s}  [{bar}] {val:.1f}")

    # ── Stage 5: Report Generation ──
    if generate_report:
        if verbose:
            print("\n[5/5] Generating report...")

        os.makedirs(output_dir, exist_ok=True)
        base_name = f"PGSI_{patient_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # JSON output
        json_path = os.path.join(output_dir, f"{base_name}.json")
        json_data = {
            "patient_id": patient_id,
            "session_tag": session_tag,
            "video_file": os.path.basename(video_path),
            "analysis_timestamp": datetime.now().isoformat(),
            "pgsi_score": round(result.pgsi_score, 2),
            "severity": result.severity_label,
            "fall_risk": result.fall_risk,
            "sub_scores": {k: round(v, 2) for k, v in result.sub_scores.items()},
            "weights": {k: round(v, 4) for k, v in result.weights.items()},
            "raw_features": {k: round(v, 6) for k, v in result.raw_features.items()},
        }
        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=2)

        # PDF/HTML report
        report_gen = ReportGenerator()
        pdf_path = os.path.join(output_dir, f"{base_name}.pdf")
        actual_path = report_gen.generate_pdf(
            result,
            pdf_path,
            patient_id=patient_id,
            session_date=datetime.now().strftime("%Y-%m-%d"),
            session_tag=session_tag,
            video_filename=os.path.basename(video_path),
            video_duration=f"{vp.duration_seconds:.1f}s" if hasattr(vp, 'duration_seconds') else "N/A",
            skeleton_image_bytes=skeleton_image_bytes,
        )

        if verbose:
            print(f"      → JSON:   {json_path}")
            print(f"      → Report: {actual_path}")

    estimator.close()

    elapsed = time.time() - start_time
    if verbose:
        print(f"\n✅ Analysis complete in {elapsed:.1f}s")

    return json_data if generate_report else {
        "pgsi_score": result.pgsi_score,
        "severity": result.severity_label,
        "sub_scores": dict(result.sub_scores),
    }


def batch_process(
    video_dir: str,
    output_dir: str = OUTPUT_DIR,
    patient_id: str = "Batch",
):
    """Process all supported videos in a directory."""
    video_files = [
        os.path.join(video_dir, f)
        for f in os.listdir(video_dir)
        if os.path.splitext(f)[1].lower() in SUPPORTED_VIDEO_EXTENSIONS
    ]

    if not video_files:
        print(f"No supported video files found in {video_dir}")
        return

    print(f"Found {len(video_files)} video(s) to process.\n")
    results = []

    for i, vpath in enumerate(video_files, 1):
        print(f"\n[{i}/{len(video_files)}] Processing: {os.path.basename(vpath)}")
        try:
            res = process_single_video(
                vpath,
                patient_id=patient_id,
                session_tag=f"Batch {i}",
                output_dir=output_dir,
                verbose=True,
            )
            results.append(res)
        except Exception as e:
            print(f"  ❌ Error: {e}")
            results.append({"error": str(e), "video": vpath})

    # Summary
    print(f"\n{'='*60}")
    print(f"  Batch Processing Complete: {len(results)} videos")
    print(f"{'='*60}")
    for r in results:
        if "error" in r:
            print(f"  ❌ {r.get('video', 'unknown')}: {r['error']}")
        else:
            print(f"  ✅ {r.get('video_file', '?')}: PGSI={r['pgsi_score']:.1f} ({r['severity']})")


def main():
    parser = argparse.ArgumentParser(
        description="PGSI — Parkinsonian Gait Severity Index (CLI)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pgsi.py --video walking_video.mp4
  python run_pgsi.py --video video.mp4 --patient-id P001 --tag "Pre-Therapy"
  python run_pgsi.py --batch data/videos/ --output results/
        """,
    )

    parser.add_argument("--video", type=str, help="Path to a single walking video")
    parser.add_argument("--batch", type=str, help="Directory of videos for batch processing")
    parser.add_argument("--patient-id", type=str, default="N/A", help="Patient identifier")
    parser.add_argument("--tag", type=str, default="", help="Session tag (e.g., Pre-Therapy)")
    parser.add_argument("--output", type=str, default=OUTPUT_DIR, help="Output directory for reports")
    parser.add_argument("--weights", type=str, default="", help="Path to custom weights JSON file")
    parser.add_argument("--bg-sub", action="store_true", help="Enable background subtraction")
    parser.add_argument("--no-report", action="store_true", help="Skip report generation")
    parser.add_argument("--quiet", action="store_true", help="Minimal output")

    args = parser.parse_args()

    if args.video:
        process_single_video(
            args.video,
            patient_id=args.patient_id,
            session_tag=args.tag,
            output_dir=args.output,
            generate_report=not args.no_report,
            verbose=not args.quiet,
            weights_path=args.weights,
            use_bg_sub=args.bg_sub,
        )
    elif args.batch:
        batch_process(args.batch, output_dir=args.output, patient_id=args.patient_id)
    else:
        parser.print_help()
        print("\nError: Provide --video or --batch argument.")
        sys.exit(1)


if __name__ == "__main__":
    main()
