# Product Requirements Document
## Vision-Based Quantitative Parkinsonian Gait Severity Assessment
### Using a Novel Pose-Derived Clinical Index (PGSI)

| Field | Details |
|---|---|
| **Project** | Parkinsonian Gait Severity Index (PGSI) System |
| **Document Type** | Product Requirements Document (PRD) |
| **Version** | 1.0 |
| **Date** | February 20, 2026 |
| **Team** | Group 2 — Section J |
| **Team Lead** | P V Sai Tejaswi (AP23110010532) |
| **Members** | Shiva Nagesh (AP23110010496) · B. Surya Teja (AP23110010549) · N. Sri Nithya (AP23110010427) |
| **Status** | Draft — Pending Review |

---

## 1. Executive Summary

Parkinson's disease (PD) is a progressive neurodegenerative disorder that impairs motor control, posture, and gait. Clinical evaluation of gait severity is currently performed through subjective observation and rating scales such as the Unified Parkinson's Disease Rating Scale (UPDRS), which are time-intensive, expert-dependent, and difficult to administer remotely.

This project delivers an end-to-end, computer vision-powered system that captures patient walking video, extracts markerless skeletal pose data, computes clinically meaningful biomechanical features, and produces a novel composite **Parkinsonian Gait Severity Index (PGSI)**. The system enables objective, reproducible, and remotely accessible gait assessment to support neurologists, physiotherapists, and rehabilitation specialists.

---

## 2. Problem Statement

### 2.1 Clinical Gap

Current gait assessment methods for Parkinson's disease rely on subjective clinician observation and manual scoring. This introduces inter-rater variability and limits the frequency of assessments outside clinic settings. There is no widely adopted quantitative, vision-based system specifically calibrated for Parkinsonian movement characteristics.

### 2.2 Limitations of Existing Systems

- Produce raw joint measurements without clinical interpretation or disease-specific context
- Lack a validated, composite severity score tied to Parkinsonian movement patterns
- Cannot track longitudinal therapy effectiveness from video data alone
- Are not designed to operate in low-resource or home-based settings
- Require expensive hardware (e.g., motion capture suits, force plates)

---

## 3. Objectives

- Develop a markerless, video-based pose estimation pipeline for gait analysis
- Extract and compute five core Parkinsonian biomechanical gait features from pose keypoints
- Introduce and validate a novel composite metric — the **Parkinsonian Gait Severity Index (PGSI)**
- Enable before-vs-after therapy comparison and longitudinal disease progression tracking
- Provide interpretable clinical output including severity classification and automated reports
- Demonstrate potential for remote patient monitoring via standard camera hardware

---

## 4. Scope

### 4.1 In Scope

- Video input processing (pre-recorded clinical walking videos and Parkinson's gait datasets)
- Pose estimation using a markerless skeleton detection model
- Biomechanical feature extraction: stride length, posture angle, gait symmetry, step timing variability, arm swing
- PGSI computation and normalization
- Severity classification (Normal / Mild / Moderate / Severe)
- Fall risk assessment module
- Before-vs-after therapy comparison
- Longitudinal disease progression tracking
- Automated clinical report generation
- Visualization dashboard

### 4.2 Out of Scope

- Real-time video streaming or live camera feed integration (Phase 1)
- Electronic Health Record (EHR) system integration
- FDA/CE regulatory submission and clinical trial validation
- Wearable sensor fusion
- Multi-person simultaneous tracking

---

## 5. Stakeholders

| Stakeholder | Role | Interest | Influence |
|---|---|---|---|
| Neurologists | Primary User | Objective gait scoring | High |
| Physiotherapists | Primary User | Therapy tracking | High |
| Patients | End Beneficiary | Non-invasive monitoring | Medium |
| Clinical Researchers | Validator | Dataset correlation | High |
| Software Team (Group 2) | Developer | System delivery | High |
| Faculty Supervisor | Reviewer | Academic oversight | Medium |

---

## 6. Functional Requirements

### 6.1 Data Acquisition

- **FR-1.1:** The system shall accept walking video in common formats (MP4, AVI, MOV) as input
- **FR-1.2:** The system shall support both pre-therapy and post-therapy video recordings
- **FR-1.3:** The system shall allow batch processing of multiple patient recordings

### 6.2 Video Preprocessing

- **FR-2.1:** The system shall normalize video frame rate and resolution prior to analysis
- **FR-2.2:** The system shall apply background stabilization and contrast enhancement
- **FR-2.3:** The system shall isolate the walking subject from background elements

### 6.3 Pose Estimation

- **FR-3.1:** The system shall extract full-body skeletal keypoints from each video frame using a markerless pose estimation model
- **FR-3.2:** The system shall track temporal joint trajectories across the full walking sequence
- **FR-3.3:** The pose model shall operate on standard video without specialized capture hardware

### 6.4 Biomechanical Feature Extraction

- **FR-4.1 — Stride Length:** Compute stride length from foot keypoint displacement per gait cycle
- **FR-4.2 — Posture Angle:** Measure forward trunk tilt angle relative to the vertical axis
- **FR-4.3 — Gait Symmetry:** Quantify left-right step length and timing asymmetry
- **FR-4.4 — Step Timing Variability:** Calculate coefficient of variation in inter-step intervals
- **FR-4.5 — Arm Swing:** Measure elbow and wrist angular excursion amplitude per gait cycle

### 6.5 PGSI Computation

- **FR-5.1:** The system shall compute normalized sub-scores for each of the five feature domains
- **FR-5.2:** The system shall compute PGSI as a weighted composite score:
  > `PGSI = w₁·S_stride + w₂·S_posture + w₃·S_symmetry + w₄·S_variability + w₅·S_armswing`
- **FR-5.3:** Weights (w₁–w₅) shall be optimized via statistical correlation with clinical UPDRS scores
- **FR-5.4:** PGSI shall output a scalar value on a normalized scale (0–100)

### 6.6 Severity Classification

- **FR-6.1:** The system shall map PGSI scores to four severity classes: Normal, Mild, Moderate, Severe
- **FR-6.2:** The system shall include a fall risk assessment output based on postural and variability features

### 6.7 Longitudinal Monitoring

- **FR-7.1:** The system shall compare PGSI scores from pre-therapy and post-therapy recordings
- **FR-7.2:** The system shall generate trend charts illustrating disease progression across multiple sessions

### 6.8 Reporting and Output

- **FR-8.1:** The system shall generate an automated clinical report summarizing PGSI, sub-scores, classification, and trend
- **FR-8.2:** The system shall produce a visualization dashboard rendering pose overlays, feature plots, and PGSI timeline
- **FR-8.3:** Reports shall be exportable in PDF format

---

## 7. Non-Functional Requirements

| Category | Requirement |
|---|---|
| **Performance** | System shall process a 60-second walking video and produce PGSI output within 3 minutes on standard hardware |
| **Accuracy** | PGSI shall achieve a Pearson correlation ≥ 0.75 with clinician-assigned UPDRS sub-scores on validation dataset |
| **Usability** | Dashboard shall be operable by clinical staff with no specialized technical training; task completion < 5 minutes |
| **Reliability** | System shall produce consistent PGSI output (±5% variance) across repeated analyses of the same video |
| **Scalability** | Architecture shall support batch processing of ≥ 50 recordings without manual intervention |
| **Portability** | System shall run on standard desktop/laptop hardware without GPU requirement for inference |
| **Privacy & Security** | All patient video data shall be processed locally; no data transmitted externally without explicit consent |
| **Maintainability** | Codebase shall be modular with documented APIs for each processing stage |

---

## 8. System Architecture Overview

The system is organized as a six-stage sequential pipeline:

1. **Patient Walking Video Capture** — Input video ingestion and metadata tagging
2. **Video Preprocessing** — Frame normalization, stabilization, subject isolation
3. **Pose Estimation Model** — Markerless skeleton detection yielding joint keypoints per frame
4. **Biomechanical Feature Computation** — Stride length, posture angle, symmetry, variability, arm swing
5. **PGSI Computation & Classification** — Feature normalization, weighted aggregation, severity mapping
6. **Clinical Decision Support** — Report generation, dashboard, longitudinal tracking, remote monitoring

---

## 9. Core Research Contribution: PGSI Definition

The Parkinsonian Gait Severity Index (PGSI) is the primary research contribution of this project. It is a composite, normalized severity score defined as:

```
PGSI = w₁·S_stride + w₂·S_posture + w₃·S_symmetry + w₄·S_variability + w₅·S_armswing
```

| Component | Feature | Clinical Significance |
|---|---|---|
| **S_stride** | Stride Length Reduction | Shorter strides are a hallmark of Parkinsonian gait (festination) |
| **S_posture** | Forward Trunk Tilt | Camptocormia and stooped posture indicate disease severity |
| **S_symmetry** | Gait Asymmetry | Left-right asymmetry reflects differential limb involvement |
| **S_variability** | Step Timing Variability | High variability correlates with freezing risk and cognitive load |
| **S_armswing** | Arm Swing Reduction | Reduced arm swing is an early and sensitive PD biomarker |

Weights w₁–w₅ will be determined through Pearson correlation analysis between each sub-score and clinical UPDRS gait sub-items on a labelled dataset, with statistical significance testing (p < 0.05) to confirm validity.

---

## 10. Implementation Plan

| Phase | Module | Deliverable | Timeline |
|---|---|---|---|
| 1 | Data & Preprocessing | Dataset collection, video preprocessing pipeline, pose model integration | Weeks 1–3 |
| 2 | Feature Extraction | Implementation of all 5 biomechanical feature extractors with unit tests | Weeks 4–6 |
| 3 | PGSI Engine | Feature normalization, weighted PGSI formula, weight optimization via correlation | Weeks 7–8 |
| 4 | Classification & Reporting | Severity classifier, fall risk module, automated PDF report generator | Weeks 9–10 |
| 5 | Dashboard & Evaluation | Visualization dashboard, longitudinal tracking, system validation on labelled data | Weeks 11–13 |
| 6 | Final Review | Bug fixes, documentation, final presentation and demo | Week 14 |

---

## 11. Evaluation Plan

### 11.1 Validation Metrics

- Pearson correlation coefficient between PGSI and UPDRS gait sub-scores (target: r ≥ 0.75)
- Classification accuracy across Normal / Mild / Moderate / Severe categories (target: ≥ 80%)
- Sensitivity to therapy-induced change: statistically significant PGSI delta pre- vs. post-therapy (p < 0.05)
- Intra-session repeatability: PGSI variance < ±5% across repeated analyses of the same video

### 11.2 Comparison Baseline

PGSI performance will be benchmarked against conventional single-feature gait metrics (e.g., stride length alone, step variability alone) to demonstrate the added discriminative value of the composite index.

---

## 12. Tools and Technologies

| Category | Tool / Library |
|---|---|
| **Pose Estimation** | MediaPipe Pose or OpenPose (markerless, CPU-compatible) |
| **Numerical Computing** | Python · NumPy · SciPy |
| **ML / Optimization** | scikit-learn (weight optimization, classification) |
| **Statistical Validation** | SciPy stats (Pearson r, p-values, confidence intervals) |
| **Visualization** | Matplotlib · Plotly · Streamlit |
| **Report Generation** | ReportLab or WeasyPrint (automated PDF output) |
| **Dataset** | PhysioNet Parkinson's gait dataset or equivalent labelled recordings |

---

## 13. Risks and Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Insufficient labelled Parkinson's gait data | Medium | Use publicly available PhysioNet datasets; supplement with simulated severity variations |
| Pose estimation fails in low-quality video | Medium | Implement preprocessing filters; define minimum video quality thresholds |
| PGSI correlation with UPDRS falls below target | Low–Medium | Tune weights iteratively; expand feature set or recalibrate normalization |
| Scope creep beyond available timeline | Medium | Strictly phase deliverables; defer real-time streaming to Phase 2 |
| Ethical / privacy concerns with patient video | Low | Use de-identified public datasets; document data handling protocols |

---

## 14. Success Criteria

The project will be considered successful when **all** of the following criteria are met:

- PGSI is computed end-to-end from walking video with no manual feature labelling
- Pearson correlation r ≥ 0.75 with UPDRS gait scores on held-out validation set
- Severity classification accuracy ≥ 80% across four classes
- Statistically significant improvement in PGSI detected between pre- and post-therapy recordings
- Automated clinical report generated within 3 minutes of video input
- Visualization dashboard renders pose overlay, feature trends, and PGSI timeline

---

## 15. Expected Impact

- Provides low-cost, non-invasive diagnostic support accessible with standard cameras
- Reduces clinician burden and assessment subjectivity in Parkinson's disease monitoring
- Enables remote and home-based gait assessment for elderly and mobility-limited patients
- Creates a reusable, extensible framework adaptable to other gait-related conditions
- Produces interpretable, AI-assisted clinical outputs aligned with existing rating scale conventions

---

## Appendix: Document History

| Version | Date | Changes | Author |
|---|---|---|---|
| 1.0 | Feb 20, 2026 | Initial PRD created from project proposal | Group 2 |
