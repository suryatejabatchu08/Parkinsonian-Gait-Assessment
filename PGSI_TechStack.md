# Tech Stack Document
## Vision-Based Quantitative Parkinsonian Gait Severity Assessment (PGSI)

| Field | Details |
|---|---|
| **Project** | Parkinsonian Gait Severity Index (PGSI) System |
| **Document Type** | Tech Stack Document |
| **Version** | 1.0 |
| **Date** | February 20, 2026 |
| **Team** | Group 2 — Section J |

---

## 1. Overview

This document describes the complete technology stack selected for the PGSI system — a six-stage video-to-clinical-score pipeline. Technologies are chosen to maximise accuracy, interpretability, and portability while remaining runnable on standard CPU-based hardware without specialized clinical infrastructure.

---

## 2. Architecture at a Glance

```
┌─────────────────────────────────────────────────────────────┐
│                        INPUT LAYER                          │
│           Walking Video (MP4 / AVI / MOV)                   │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│                   VIDEO PREPROCESSING                        │
│          OpenCV  ·  FFmpeg  ·  NumPy                        │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│                    POSE ESTIMATION                           │
│              MediaPipe Pose  /  OpenPose                     │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│              BIOMECHANICAL FEATURE EXTRACTION                │
│                  NumPy  ·  SciPy                             │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│              PGSI COMPUTATION & CLASSIFICATION               │
│              scikit-learn  ·  SciPy  ·  NumPy               │
└──────────────────┬────────────────────┬─────────────────────┘
                   │                    │
┌──────────────────▼──────┐  ┌──────────▼──────────────────── ┐
│   VISUALIZATION &        │  │   REPORT GENERATION             │
│   DASHBOARD              │  │                                 │
│   Streamlit · Plotly     │  │   ReportLab / WeasyPrint        │
│   Matplotlib             │  │   Jinja2                        │
└─────────────────────────┘  └─────────────────────────────────┘
```

---

## 3. Language & Runtime

| Component | Choice | Rationale |
|---|---|---|
| **Primary Language** | Python 3.10+ | Dominant language in computer vision and ML; rich ecosystem; cross-platform |
| **Runtime** | CPython | Standard interpreter; broad library compatibility |
| **Package Manager** | pip + virtualenv / conda | Reproducible environment management |
| **Version Pinning** | `requirements.txt` | Ensures consistent builds across machines |

---

## 4. Stage-by-Stage Tech Stack

### 4.1 Video Preprocessing

| Library | Version | Purpose |
|---|---|---|
| **OpenCV** (`cv2`) | 4.9+ | Frame extraction, resizing, grayscale/contrast normalization, background subtraction |
| **FFmpeg** | 6.x (via `ffmpeg-python`) | Format conversion, frame rate normalization, codec handling |
| **NumPy** | 1.26+ | Frame array manipulation, pixel-level operations |

Key operations performed at this stage:
- Decode video to per-frame NumPy arrays
- Normalize frame rate to 30 FPS
- Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) for lighting correction
- Background subtraction using `cv2.createBackgroundSubtractorMOG2`
- Crop and pad frames to subject bounding box

---

### 4.2 Pose Estimation

| Library | Version | Purpose |
|---|---|---|
| **MediaPipe Pose** | 0.10+ | Primary: real-time, CPU-compatible 33-keypoint body landmark detection |
| **OpenPose** | 1.7 *(fallback)* | Alternative: higher accuracy on clinical-quality video; GPU-optional |

**Primary choice — MediaPipe Pose:**

- Detects 33 full-body landmarks per frame (shoulders, elbows, wrists, hips, knees, ankles, feet)
- Runs on CPU at real-time speed (~30 FPS on standard laptop)
- Returns normalized `(x, y, z, visibility)` per keypoint
- No hardware dependency; installable via `pip install mediapipe`

**Fallback — OpenPose:**

- Higher landmark accuracy on lower-quality or occluded video
- Requires additional setup (Caffe/TensorFlow backend)
- Used if MediaPipe confidence scores fall below threshold

**Keypoints used for PGSI features:**

| Keypoint Indices (MediaPipe) | Body Part |
|---|---|
| 11, 12 | Left / Right Shoulder |
| 13, 14 | Left / Right Elbow |
| 15, 16 | Left / Right Wrist |
| 23, 24 | Left / Right Hip |
| 25, 26 | Left / Right Knee |
| 27, 28 | Left / Right Ankle |
| 31, 32 | Left / Right Foot Index |

---

### 4.3 Biomechanical Feature Extraction

| Library | Version | Purpose |
|---|---|---|
| **NumPy** | 1.26+ | Joint angle calculation, vector operations, trajectory arrays |
| **SciPy** | 1.13+ | Peak detection (`find_peaks`), signal smoothing, gait cycle segmentation |
| **Pandas** | 2.2+ | Per-frame feature tabulation, time-series storage, session management |

**Feature computation methods:**

| Feature | Method |
|---|---|
| **Stride Length** | Euclidean distance between successive ipsilateral heel-strike keypoints, normalized by subject height proxy (hip-to-ankle distance) |
| **Posture Angle** | Angle between shoulder midpoint → hip midpoint vector and vertical axis, computed via `np.arctan2` |
| **Gait Symmetry** | Symmetry Index: `SI = |L - R| / (0.5 * (L + R)) * 100` applied to step length and step time |
| **Step Timing Variability** | Coefficient of Variation (CV) of inter-heel-strike intervals: `CV = std / mean * 100` |
| **Arm Swing** | Angular excursion of elbow joint across gait cycle, computed from wrist–elbow–shoulder angle trajectory |

---

### 4.4 PGSI Computation & Classification

| Library | Version | Purpose |
|---|---|---|
| **NumPy** | 1.26+ | Feature normalization (min-max scaling), weighted sum computation |
| **scikit-learn** | 1.5+ | Weight optimization (linear regression against UPDRS labels), severity classifier |
| **SciPy** | 1.13+ | Pearson correlation (`pearsonr`), statistical significance testing |

**PGSI formula:**

```
PGSI = w₁·S_stride + w₂·S_posture + w₃·S_symmetry + w₄·S_variability + w₅·S_armswing
```

**Weight optimization pipeline:**

1. Compute all 5 sub-scores on labelled dataset
2. Run `scipy.stats.pearsonr` between each sub-score and UPDRS gait item
3. Use `sklearn.linear_model.LinearRegression` to fit weights to UPDRS total
4. Validate with 5-fold cross-validation (`sklearn.model_selection.KFold`)

**Severity classifier:**

- Model: `sklearn.svm.SVC` with RBF kernel (or `RandomForestClassifier` as fallback)
- Classes: Normal (0–25) · Mild (26–50) · Moderate (51–75) · Severe (76–100)
- Input: PGSI scalar + raw sub-scores
- Evaluation: accuracy, F1-score, confusion matrix via `sklearn.metrics`

---

### 4.5 Visualization & Dashboard

| Library | Version | Purpose |
|---|---|---|
| **Streamlit** | 1.35+ | Interactive web dashboard; session upload, results display, longitudinal charts |
| **Plotly** | 5.22+ | Interactive charts: PGSI radar plot, sub-score bar chart, gait cycle waveforms |
| **Matplotlib** | 3.9+ | Pose skeleton overlay rendering on video frames; static figure export |
| **Seaborn** | 0.13+ | Heatmaps for correlation matrices; distribution plots for validation results |

**Dashboard pages:**

| Page | Contents |
|---|---|
| **Upload** | Video file upload, patient metadata input, session tagging |
| **Pose Viewer** | Skeleton overlay preview, keypoint confidence scores |
| **Feature Analysis** | Per-feature waveform plots, stride-by-stride breakdown |
| **PGSI Report** | Severity gauge, sub-score radar chart, classification label |
| **Longitudinal** | PGSI trend over sessions, before-vs-after therapy comparison |

---

### 4.6 Report Generation

| Library | Version | Purpose |
|---|---|---|
| **ReportLab** | 4.2+ | Programmatic PDF generation (clinical summary report) |
| **Jinja2** | 3.1+ | HTML templating for report content rendering |
| **WeasyPrint** | 62+ *(optional)* | HTML-to-PDF conversion as alternative to ReportLab |

**Report contents:**

- Patient session metadata
- PGSI score with severity classification
- Sub-score breakdown table
- Pose skeleton snapshot
- Feature waveform charts
- Longitudinal trend (if multiple sessions)
- Fall risk flag
- Auto-generated clinical interpretation text

---

## 5. Data Management

| Tool | Purpose |
|---|---|
| **Pandas** | Per-session feature DataFrames, CSV export/import |
| **JSON / YAML** | Session configuration, weight storage, metadata |
| **SQLite** *(optional)* | Lightweight local database for multi-session longitudinal tracking |
| **PhysioNet Dataset** | Primary labelled Parkinson's gait data for training and validation |

**Data flow:**

```
Raw Video → Preprocessed Frames → Keypoint JSON → Feature CSV → PGSI Score → Report PDF
```

---

## 6. Development & Testing Tools

| Tool | Version | Purpose |
|---|---|---|
| **Python** | 3.10+ | Primary development language |
| **pytest** | 8.x | Unit testing for feature extractors, PGSI formula, classifiers |
| **pytest-cov** | 5.x | Code coverage reporting |
| **Black** | 24.x | Code formatting |
| **Flake8** | 7.x | Linting |
| **Jupyter Notebook** | 7.x | Exploratory data analysis, weight optimization experiments |
| **Git** | 2.x | Version control |
| **GitHub** | — | Repository hosting, issue tracking |

---

## 7. Environment & Deployment

| Aspect | Choice |
|---|---|
| **OS Support** | Windows 10+, macOS 12+, Ubuntu 20.04+ |
| **Hardware Requirement** | Standard CPU laptop (no GPU required for MediaPipe inference) |
| **Python Environment** | `virtualenv` or `conda` environment |
| **Packaging** | `requirements.txt` with pinned versions |
| **Entry Point** | `streamlit run app.py` for dashboard; `python run_pgsi.py` for CLI batch mode |
| **GPU (optional)** | CUDA 11.x for OpenPose fallback only |

### requirements.txt (core)

```
opencv-python>=4.9.0
mediapipe>=0.10.0
ffmpeg-python>=0.2.0
numpy>=1.26.0
scipy>=1.13.0
pandas>=2.2.0
scikit-learn>=1.5.0
matplotlib>=3.9.0
plotly>=5.22.0
seaborn>=0.13.0
streamlit>=1.35.0
reportlab>=4.2.0
jinja2>=3.1.0
weasyprint>=62.0
pytest>=8.0.0
pytest-cov>=5.0.0
```

---

## 8. Technology Decision Summary

| Stage | Selected Tool | Key Reason |
|---|---|---|
| Video Preprocessing | OpenCV + FFmpeg | Industry standard; highly optimised for frame-level operations |
| Pose Estimation | MediaPipe Pose | CPU-compatible; no hardware dependency; 33-keypoint body model |
| Feature Extraction | NumPy + SciPy | Numerically efficient; well-documented signal processing functions |
| PGSI Computation | NumPy + scikit-learn | Lightweight; reproducible; correlation-based weight fitting |
| Classification | scikit-learn SVC | Proven performance on small clinical datasets; interpretable |
| Dashboard | Streamlit + Plotly | Rapid deployment of interactive clinical UI without frontend expertise |
| Report Generation | ReportLab + Jinja2 | Programmatic, template-driven PDF output |
| Testing | pytest | Python standard; integrates with CI pipelines |

---

## 9. Known Constraints & Trade-offs

| Constraint | Impact | Mitigation |
|---|---|---|
| No GPU requirement | Inference speed limited to MediaPipe CPU throughput | Acceptable for pre-recorded video; real-time deferred to Phase 2 |
| MediaPipe keypoint accuracy | May degrade on baggy clothing or extreme camera angles | Preprocessing filters + visibility confidence thresholds |
| Small labelled dataset | Weight optimization may overfit | 5-fold cross-validation; regularization in linear regression |
| Local-only processing | No cloud sync or multi-device access | Deliberate for patient data privacy; network mode in Phase 2 |

---

## Appendix: Document History

| Version | Date | Changes | Author |
|---|---|---|---|
| 1.0 | Feb 20, 2026 | Initial tech stack document | Group 2 |
