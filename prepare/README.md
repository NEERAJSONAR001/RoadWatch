# RoadWatch: two-week baseline workflow

This workspace contains a reproducible detector baseline and matched model comparison for the project described in
`RoadWatch_Presentation.pptx` and `RoadWatch Detailed Execution Roadmap.docx`. It focuses on
the first verifiable step: training and evaluating a pothole detector on a real, attributed
dataset. It does **not** claim that the GPS, duplicate merging, severity validation, officer
dashboard, or repair workflow is implemented.

## What the experiment does

1. Downloads the official train, validation, and test archives for
   [manot/pothole-segmentation](https://huggingface.co/datasets/manot/pothole-segmentation).
   The dataset card reports 819 images, COCO bounding-box annotations, and CC BY 4.0.
2. Preserves its train/validation/test assignment and converts COCO bounding boxes to YOLO labels. The
   spelling variants `pothole` and `potholes` are combined into one class. The generic
   `object` annotations are excluded rather than mislabeled as potholes.
   The script checks exact SHA-256 image duplicates and removes copies from the lower-priority
   split (test, then validation, then train) so duplicate photographs cannot leak into both
   training and test evaluation.
3. Fine-tunes COCO-pretrained YOLOv8n and YOLOv8s on the same cleaned splits with the same
   epochs, image size, CPU, seed, batch size, confidence and IoU thresholds.
4. Measures each checkpoint on validation for model selection and on the held-out test split
   for final metrics. Precision, recall and F1 use fixed thresholds; mAP@0.5 and
   mAP@0.5:0.95 are also recorded. The fixed-threshold P/R avoids choosing a confidence
   point from the test split.
5. Writes per-model metrics, error CSVs and prediction images, comparison charts, and a
   consolidated final Word report under `reports/`.

The small dataset and short first run are a proof-of-concept, not deployment evidence or
proof of Indian/all-weather generalization. RDD2022 is a sensible later dataset, but its
official Figshare archive is about 13 GB. Do not mix its data into this baseline or claim
RDD2022 results until it has actually been downloaded and evaluated.

## Run it (Windows PowerShell)

Use Python 3.10+ and install the listed packages:

```powershell
python -m pip install -r requirements.txt
python scripts\roadwatch_baseline.py prepare
python scripts\roadwatch_baseline.py train --model yolov8n.pt --epochs 5 --imgsz 416
python scripts\roadwatch_baseline.py validate --model yolov8n.pt
python scripts\roadwatch_baseline.py train --model yolov8s.pt --epochs 5 --imgsz 416
python scripts\roadwatch_baseline.py validate --model yolov8s.pt
python scripts\build_comparative_report.py
```

The first run downloads about 55 MB of dataset archives and each model's pretrained weights.
Training runs on CPU by default and can take a while. Use the same epoch count and image size
for both candidates. For a quick pipeline smoke test, use `--epochs 1`; do not compare a
one-epoch run against a five-epoch run as a fair model comparison.

### Run stages separately

```powershell
python scripts\roadwatch_baseline.py prepare
python scripts\roadwatch_baseline.py train --model yolov8n.pt --epochs 5 --imgsz 416
python scripts\roadwatch_baseline.py validate --model yolov8n.pt
python scripts\roadwatch_baseline.py train --model yolov8s.pt --epochs 5 --imgsz 416
python scripts\roadwatch_baseline.py validate --model yolov8s.pt
python scripts\roadwatch_baseline.py report
python scripts\build_comparative_report.py
```

`train` expects the dataset to have already been prepared and trains then evaluates its
checkpoint. `evaluate` repeats only the test evaluation of the latest checkpoint without
retraining (useful if evaluation/reporting was interrupted). `validate` evaluates the saved
best checkpoint on the validation split; it does not retrain. The comparative report requires
both model runs and both validation results. Its winner is selected by validation mAP@0.5;
the test split is reported separately.

## Output map

- `data\raw\` — downloaded original ZIP archives.
- `data\processed\` — YOLO-format images, labels, YAML configuration, and split counts.
- `runs\` — Ultralytics training logs, weights, validation plots, and test evaluation.
- `reports\metrics.json` — machine-readable measured metrics and run configuration.
- `reports\metrics_yolov8s.json` — YOLOv8s configuration and results.
- `reports\test_error_analysis_yolov8n.csv` and `reports\test_error_analysis_yolov8s.csv` —
  per-image TP/FP/FN at confidence 0.25 and IoU 0.50.
- `reports\predictions\yolov8n\` and `reports\predictions\yolov8s\` — test images with ground
  truth in green and detections in red.
- `reports\comparison_metrics.png`, `reports\comparison_speed_accuracy.png`,
  `reports\comparison_error_counts.png` — measured model comparison charts.
- `reports\model_comparison.csv` — validation/test metric table for both trained candidates.
- `reports\RoadWatch_Final_Comparative_Report.docx` — consolidated methods, dataset,
  comparative analysis, figures, results, limitations, references and code/run instructions.
- `reports\RoadWatch_2_Week_Progress_Report.docx` — generated submission-ready interim report.

## How to explain the script

The baseline script has six commands:

- `prepare`: obtains the dataset and converts its COCO boxes to normalized YOLO boxes.
- `train`: initializes YOLOv8n from public COCO-pretrained weights, fine-tunes on train,
  then evaluates on the dataset's held-out test split. Select YOLOv8n or YOLOv8s with
  `--model`.
- `validate`: evaluates a trained checkpoint on the validation split for model selection;
  it does not train again.
- `report`: builds the Word report from the saved run metrics; it does not invent or
  regenerate results.
- `run`: performs prepare → train/evaluate → report.

The separate `scripts\build_comparative_report.py` requires both model metrics and validation
scores. Its experimental comparison covers two sizes of one detector family. SSD and Faster
R-CNN are discussed as established methods, but are explicitly marked as literature-only,
not run in this experiment.

The split is kept unchanged. The validation split is available for model selection; the
test split is used for the final reported detector metrics. The severity formula in the
roadmap still needs independent human ratings before it can be called validated. The
system must present severity as an estimate and repair prioritization as decision support.

## Reproducibility and attribution

The script fixes the training seed to 42, records model settings and the best checkpoint
path, and writes the dataset split counts and evaluation output. A CPU run can be slower
than a GPU run and should be compared only with the device stated in the report.

Dataset source: [Hugging Face dataset card](https://huggingface.co/datasets/manot/pothole-segmentation),
which credits the Roboflow Universe road-damage dataset and states CC BY 4.0. Preserve
attribution when redistributing the dataset. Ultralytics YOLO licensing has AGPL-3.0 and
Enterprise terms; check the applicable terms for any distribution or deployment.
