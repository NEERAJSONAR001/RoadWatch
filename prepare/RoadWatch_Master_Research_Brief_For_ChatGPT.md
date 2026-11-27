# RoadWatch: Complete Research Dossier, Experimental Analysis & Phase-2 Prompt Brief

> **How to use this file:** Copy-paste or attach this single file into ChatGPT (alongside your `.docx` report if needed). It contains **100% of the verified facts, dataset numbers, code architecture, experimental settings, measured results, limitations, and target goals** of the RoadWatch M.Sc. research project so far.

---

## Prompt / Instruction for AI Reviewer

I am an M.Sc. Computer Science student (HSNC University / K.C. College, Mumbai) working on my research project **RoadWatch: AI-Assisted Pothole Detection, Severity Scoring & Repair Prioritisation (targeted at Mumbai streets)**.

Below is the complete, verified summary of everything I have implemented, measured, and documented so far (Phase 1 / Research Proposal + Pilot Baseline).

Using ONLY the verified facts below (without inventing numbers):
1. **Critique my current research & methodology** from an M.Sc. thesis / academic paper perspective (strengths, gaps, statistical/experimental weaknesses).
2. **Design a rigorous Phase-2 research & experimental plan** to turn this pilot baseline into a strong M.Sc. thesis / publishable paper, specifically covering:
   - Moving from the general public dataset to a **Mumbai-specific street dataset** (collection, annotation, class design, day/night/rain/waterlogging conditions, and domain adaptation/fine-tuning from `RDD2022` + custom Mumbai images).
   - **Improving Recall** (currently `0.433` on `YOLOv8n` — missing 115/203 test potholes) without destroying Precision.
   - **Fair multi-family benchmarking** (adding `Faster R-CNN` and `SSD`, or `RT-DETR` / `YOLOv10` / `YOLOv11`, with multi-seed GPU runs and proper statistical reporting).
   - **Research novelty / contribution** beyond "just running YOLO" (e.g., explainable severity scoring, GPS-based spatial deduplication, weather/shadow robustness, or lightweight edge/CPU deployment).
3. **Suggest exact improvements to my current Research Report & Proposal Presentation** (structure, missing sections, literature gaps, and defense Q&A preparation).

---

## 1. Project Identity & Scope

- **Project Title:** RoadWatch: Pothole Detection Dataset, Existing Methods, Comparative Analysis and Best-Method Selection
- **Author:** Neeraj Sonar (Roll No.: `26KFMSCCS26`)
- **Degree:** M.Sc. in Computer Science (Academic Year: 2026–2027)
- **Institution:** Kishinchand Chellaram (K.C.) College, Churchgate, Mumbai — HSNC University, Mumbai
- **Guide:** Dr. Geeta N Brijwani
- **Long-Term End-to-End Vision (Full RoadWatch System):**
  `Citizen Road Photo Upload → AI Pothole Detection → GPS Location Association → Duplicate-Report Merging → Explainable Severity & Priority Score → Municipal Officer Repair Dashboard`
- **Current Implemented Scope (Phase 1 / Half-Project):**
  - Reproducible data ingestion, cleaning, COCO-to-YOLO conversion, and SHA-256 deduplication pipeline (`scripts/roadwatch_baseline.py`).
  - Controlled pilot training & evaluation of two pretrained detector scales (`YOLOv8n` and `YOLOv8s`) on CPU.
  - Automated comparative charting and report generation (`scripts/build_comparative_report.py`).
  - Interactive live inference & analysis web prototype (`demo_app.py` built with Streamlit + OpenCV + Ultralytics).
- **Explicitly NOT Implemented Yet (Future Work):**
  - GPS metadata extraction / geographic duplicate merging.
  - Depth estimation or human-validated severity/priority scoring.
  - User authentication, database backend, React/FastAPI production app, or officer repair dashboard.
  - Training/execution of `SSD` and `Faster R-CNN` (reviewed in literature only).
  - Mumbai-specific road image collection and weather/lighting tagging.

---

## 2. Dataset & Data-Integrity Pipeline

- **Dataset Name:** Road Damage Dataset (originally by `abdulmohsen-fahad` on Roboflow Universe, mirrored on Hugging Face as `manot/pothole-segmentation`)
- **Source URL:** `https://huggingface.co/datasets/manot/pothole-segmentation`
- **License:** `CC BY 4.0` (as stated on the dataset card)
- **Task:** Single-class object detection (`0: pothole`)
- **Annotation Processing:**
  - Spelling variants `"pothole"` and `"potholes"` were unified into one class (`pothole`).
  - Generic `"object"` annotations were excluded rather than relabelled (`3` in train, `2` in val, `1` in test).
  - COCO bounding boxes `[x, y, width, height]` were clipped to image boundaries and normalized to YOLO format `[class_id, center_x, center_y, width, height]`. `0` invalid boxes were skipped.
- **Deduplication (Data-Leakage Prevention):**
  - Method: **SHA-256 byte-exact hashing** across all splits.
  - Findings: `84` duplicate groups found total, of which `40` groups crossed split boundaries.
  - Precedence Rule: `test` kept first > `val` > `train`; lower-priority duplicate copies and their label files were deleted (`84` duplicate images removed total: `75` from train, `8` from val, `1` from test).
  - Known Gap: Byte-exact SHA-256 does not catch visually similar near-duplicates (re-encoded, resized, cropped, or burst frames).

### Exact Dataset Split Counts (Before vs. After Cleaning)

| Split | Source Images | Source Boxes | Ignored `"object"` | Duplicates Removed | Clean Images | Clean Pothole Boxes | Clean Empty Images |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train** | 582 | 1,499 | 3 | 75 | **507** | **1,306** | 4 |
| **Validation** | 157 | 405 | 2 | 8 | **149** | **397** | 3 |
| **Test** | 80 | 204 | 1 | 1 | **79** | **203** | 2 |
| **Total** | **819** | **2,108** | **6** | **84** | **735** | **1,906** | **9** |

---

## 3. Experimental Protocol & Controlled Settings

Both executed models used the **exact same cleaned files, splits, and hyperparameters** to guarantee a fair, reproducible comparison:

| Parameter | Value | Purpose / Control |
| :--- | :--- | :--- |
| **Models Tested** | `yolov8n.pt` (Nano), `yolov8s.pt` (Small) | COCO-pretrained transfer learning |
| **Epochs** | `5` | Matched short pilot baseline |
| **Input Resolution (`imgsz`)** | `416 × 416` | Matched |
| **Batch Size** | `8` | Matched |
| **Device** | `CPU` (`workers=0`) | Hardware constraint (no GPU in Phase 1) |
| **Random Seed** | `42` (`deterministic=True`, `patience=0`) | Reproducibility |
| **Augmentation** | Ultralytics training defaults | Matched |
| **Fixed Eval Confidence Threshold** | `0.25` | Fixed in advance (not tuned on test set) |
| **Box-Match IoU Criterion** | `IoU ≥ 0.50` | Greedy 1-to-1 matching sorted by confidence |
| **Pre-declared Model Selection Rule** | **Highest Validation `mAP@0.5`** | Kept held-out test set unbiased |

### Evaluation Logic Implemented in `scripts/roadwatch_baseline.py`
1. `model.val(split="val")` computes validation `Precision`, `Recall`, `mAP@0.5`, and `mAP@0.5:0.95` for model selection.
2. `model.val(split="test")` computes test `mAP@0.5` and `mAP@0.5:0.95`.
3. `evaluate_error_cases()` runs per-image inference (`conf=0.25`, `iou=0.7` NMS) on all `79` test images, measures CPU inference wall-clock throughput (`images/sec`), performs greedy 1-to-1 matching (`IoU >= 0.50`) against ground truth to count `TP`, `FP`, `FN`, exports `reports/test_error_analysis_<model>.csv`, and saves green/red overlay images in `reports/predictions/<model>/`.
4. `apply_fixed_threshold_metrics()` computes fixed-threshold `Precision = TP / (TP + FP)`, `Recall = TP / (TP + FN)`, and `F1 = 2 * TP / (Predictions + GroundTruth)`.

---

## 4. Complete Measured Results & Comparative Analysis

### A. Validation Split Results (149 Images, 397 Boxes — Used for Model Selection)

| Model | Val Precision | Val Recall | Val `mAP@0.5` (Selection Metric) | Val `mAP@0.5:0.95` |
| :--- | :---: | :---: | :---: | :---: |
| **`yolov8n.pt`** | **0.6334** | **0.6196** | **0.6268 (0.627)** ← **Selected** | **0.3507 (0.351)** |
| `yolov8s.pt` | 0.5882 | 0.5648 | 0.5837 (0.584) | 0.2946 (0.295) |

### B. Held-Out Test Split Results (79 Images, 203 Ground-Truth Potholes)

| Metric | `yolov8n.pt` (Nano) | `yolov8s.pt` (Small) | Winner / Observation |
| :--- | :---: | :---: | :--- |
| **Test Precision** (`conf ≥ 0.25, IoU ≥ 0.50`) | **0.6377 (0.638)** | 0.5444 (0.544) | `YOLOv8n` (+0.093) — fewer false alarms |
| **Test Recall** (`conf ≥ 0.25, IoU ≥ 0.50`) | 0.4335 (0.433) | **0.4532 (0.453)** | `YOLOv8s` (+0.020) — slightly fewer misses |
| **Test F1 Score** | **0.5161 (0.516)** | 0.4946 (0.495) | `YOLOv8n` (+0.022) |
| **Test `mAP@0.5`** | **0.5367 (0.537)** | 0.4945 (0.494) | `YOLOv8n` (+0.042) |
| **Test `mAP@0.5:0.95`** | **0.2814 (0.281)** | 0.2492 (0.249) | `YOLOv8n` (+0.032) |
| **True Positives (TP)** | 88 | **92** | `YOLOv8s` matched 4 more potholes |
| **False Positives (FP)** | **50** | 77 | `YOLOv8s` generated 27 more false alarms |
| **False Negatives / Missed (FN)** | 115 | **111** | Both miss >54% of test potholes |
| **Total Predictions** | 138 | 169 | `88+50=138`; `92+77=169` |
| **CPU Throughput (img/sec)** | **10.41** | 5.75 | `YOLOv8n` is ~1.81× faster on CPU |

### C. Key Analytical Findings
1. **Model Selection:** Under the pre-declared rule (Validation `mAP@0.5`), `YOLOv8n` (`0.627`) beats `YOLOv8s` (`0.584`) by `+0.043`.
2. **Speed-Accuracy Trade-off:** On this 5-epoch CPU pilot, `YOLOv8n` is simultaneously more accurate on `mAP@0.5` (`0.537` vs `0.494`) and `1.81×` faster (`10.41` vs `5.75` img/s).
3. **Why `YOLOv8s` Underperformed `YOLOv8n` (Hypothesis):** `YOLOv8s` has higher parameter capacity and likely under-converged in only `5` epochs on `507` training images. This requires a longer GPU run to verify.
4. **Main Weakness — Low Recall:** At `conf = 0.25`, `YOLOv8n` misses `115 / 203` (`56.7%`) and `YOLOv8s` misses `111 / 203` (`54.7%`) of ground-truth potholes. Neither model is deployment-ready yet.
5. **Note on Metric Comparison:** Validation `P/R` from `model.val()` are Ultralytics curve-summary values, whereas Test `P/R/F1` are fixed-threshold (`0.25`) counts; they should not be compared directly as a before/after pair.

---

## 5. Implemented Codebase & Artifacts

- **`scripts/roadwatch_baseline.py`** (`884` lines): CLI tool (`prepare`, `train`, `validate`, `evaluate`, `report`, `run`) that downloads ZIPs, parses COCO JSON, filters/normalizes labels, runs SHA-256 deduplication, writes `data/processed/roadwatch.yaml`, trains YOLOv8 on CPU, evaluates val & test splits, runs greedy TP/FP/FN error analysis, plots per-model charts, and exports `.docx` + `.json` + `.csv`.
- **`scripts/build_comparative_report.py`** (`625` lines): Validates that both `metrics.json` and `metrics_yolov8s.json` share identical protocol settings and dataset splits, then generates 4 comparison charts (`dataset_split_summary.png`, `comparison_metrics.png`, `comparison_speed_accuracy.png`, `comparison_error_counts.png`), `model_comparison.csv`, and `RoadWatch_Research_Report.docx`.
- **`demo_app.py`** (`567` lines) + **`RUN_DEMO.bat`**: Interactive 3-tab Streamlit web application (`Detect Potholes`, `Analysis & Results`, `About the Project`) loading the trained `best.pt` weights (`runs/roadwatch-yolov8n-20261114-004822/weights/best.pt` and `yolov8s` equivalent) for live road image upload or test-split sampling, adjustable confidence threshold & `imgsz`, inference timing, and heuristic severity badge (`LOW`/`MEDIUM`/`HIGH` — explicitly a UI heuristic, not yet human-validated).

---

## 6. Literature & References Status

### Currently Cited in Report
1. **[1] YOLO (v1):** J. Redmon et al., CVPR 2016 — `https://arxiv.org/abs/1506.02640`
2. **[2] Faster R-CNN:** S. Ren et al., NeurIPS 2015 — `https://arxiv.org/abs/1506.01497` *(Literature review only; not trained)*
3. **[3] SSD:** W. Liu et al., ECCV 2016 — `https://arxiv.org/abs/1512.02325` *(Literature review only; not trained)*
4. **[4] Ultralytics YOLOv8 Docs:** `https://docs.ultralytics.com/models/yolov8/` *(Software documentation, not a peer-reviewed paper)*
5. **[5] Dataset:** `abdulmohsen-fahad`, Roboflow Universe (`manot/pothole-segmentation` on Hugging Face), CC BY 4.0 — `https://huggingface.co/datasets/manot/pothole-segmentation`
6. **[6] RDD2022 Dataset:** D. Arya et al., Figshare 2022 (`13.26 GB`, `47,420` images, includes India) — `https://figshare.com/articles/dataset/21431547`

### Identified Literature Gaps (To Add in Phase 2)
- Pothole-specific deep learning papers & pavement damage surveys (e.g., `arXiv:2204.11609`, `MDPI Sensors 2022` survey).
- Full `RDD2022` peer-reviewed paper (`https://arxiv.org/abs/2209.08538`) rather than just the Figshare dataset link.
- Backbone/architecture theory citations (e.g., `ResNet`, `FPN`/`PANet`, `CIoU` loss).

---

## 7. Honest Limitations & Phase-2 Target Goals

1. **Short Training & Compute:** Only `5` epochs on CPU; larger `YOLOv8s` likely underfit/under-converged.
2. **Single Seed & Small Test Set:** `79` test images (`203` boxes) with `seed=42` once — differences of `0.02–0.04` may fall within run-to-run variance.
3. **Domain Gap for Mumbai Streets:** Current dataset is generic/international, has no verified `day / night / monsoon rain / shadow / waterlogged pothole` labels, and no Indian road validation.
4. **No Cross-Family Benchmark Yet:** Only two scales of `YOLOv8` were executed; `Faster R-CNN` and `SSD` were not trained.
5. **Near-Duplicate Blind Spot:** SHA-256 catches exact byte duplicates only, not perceptual/cropped near-duplicates.
6. **Unvalidated Downstream Pipeline:** GPS deduplication, physical severity (depth/area), and officer prioritization dashboard are planned for Phase 2.
