from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
BASELINE_PATH = REPORTS / "metrics.json"
COMPARISON_PATH = REPORTS / "metrics_yolov8s.json"
CODE_PATH = ROOT / "scripts" / "roadwatch_baseline.py"


def load_experiments() -> tuple[dict[str, Any], dict[str, Any]]:
    missing = [path for path in (BASELINE_PATH, COMPARISON_PATH) if not path.exists()]
    if missing:
        paths = ", ".join(str(path.relative_to(ROOT)) for path in missing)
        raise FileNotFoundError(
            f"Missing experiment results: {paths}. Train both models before building the report."
        )
    small = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    nano = json.loads(COMPARISON_PATH.read_text(encoding="utf-8"))
    small.setdefault("model_weights", "yolov8n.pt")
    small.setdefault("experiment_id", "yolov8n")
    if "validation" not in small or "validation" not in nano:
        raise ValueError(
            "Run validation for both checkpoints first: "
            "python scripts\\roadwatch_baseline.py validate --model yolov8n.pt "
            "and --model yolov8s.pt."
        )
    comparable_fields = ("epochs", "image_size", "device", "seed", "confidence_threshold", "iou_threshold")
    mismatches = {
        field: (small.get(field), nano.get(field))
        for field in comparable_fields
        if small.get(field) != nano.get(field)
    }
    if mismatches:
        raise ValueError(f"Experiment settings do not match; comparison would be unfair: {mismatches}")
    if small.get("dataset", {}).get("splits") != nano.get("dataset", {}).get("splits"):
        raise ValueError("The two experiments do not share identical cleaned dataset splits.")
    if small.get("evaluation_split") != "official test split" or nano.get("evaluation_split") != "official test split":
        raise ValueError("Both models must have held-out test results before comparison.")
    return small, nano


def create_comparison_charts(small: dict[str, Any], nano: dict[str, Any]) -> dict[str, Path]:
    REPORTS.mkdir(parents=True, exist_ok=True)
    labels = ["YOLOv8n", "YOLOv8s"]
    colors = ["#3977a8", "#e39b34"]
    paths: dict[str, Path] = {}

    metrics = [
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1", "F1"),
        ("map50", "mAP@0.5"),
        ("map50_95", "mAP@0.5:0.95"),
    ]
    x = np.arange(len(metrics))
    width = 0.34
    figure, axis = plt.subplots(figsize=(10.5, 5.2))
    for offset, (experiment, label, color) in enumerate(zip((small, nano), labels, colors)):
        values = [experiment[key] for key, _ in metrics]
        bars = axis.bar(x + (offset - 0.5) * width, values, width, label=label, color=color)
        axis.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)
    axis.set_xticks(x, [label for _, label in metrics])
    axis.set_ylim(0, 1.12)
    axis.set_ylabel("Score")
    axis.set_title("Held-out test metrics (P/R/F1 at fixed threshold)")
    axis.legend()
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    paths["test_metrics"] = REPORTS / "comparison_metrics.png"
    figure.savefig(paths["test_metrics"], dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.5, 5.2))
    for experiment, label, color in zip((small, nano), labels, colors):
        axis.scatter(
            experiment["test_inference_images_per_second"],
            experiment["map50"],
            s=110,
            label=label,
            color=color,
        )
        axis.annotate(
            label,
            (
                experiment["test_inference_images_per_second"],
                experiment["map50"],
            ),
            xytext=(7, 7),
            textcoords="offset points",
        )
    axis.set_xlabel("CPU inference throughput (images/second; higher is faster)")
    axis.set_ylabel("Test mAP@0.5 (higher is better)")
    axis.set_ylim(0, 1)
    axis.set_title("Accuracy–speed trade-off")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    paths["tradeoff"] = REPORTS / "comparison_speed_accuracy.png"
    figure.savefig(paths["tradeoff"], dpi=180)
    plt.close(figure)

    error_names = ["Matched TP", "False positives", "Missed potholes (FN)"]
    figure, axis = plt.subplots(figsize=(8, 4.8))
    error_colors = ["#4c9f70", "#e39b34", "#d95d39"]
    for index, (experiment, label) in enumerate(zip((small, nano), labels)):
        counts = experiment["test_error_counts"]
        values = [counts["tp"], counts["fp"], counts["fn"]]
        axis.bar(
            np.arange(len(error_names)) + (index - 0.5) * 0.34,
            values,
            width=0.34,
            label=label,
            color=error_colors,
        )
        for metric_index, value in enumerate(values):
            axis.text(
                metric_index + (index - 0.5) * 0.34,
                value + 1,
                str(value),
                ha="center",
                fontsize=8,
            )
    axis.set_xticks(np.arange(len(error_names)), error_names)
    axis.set_ylabel("Number of boxes")
    axis.set_title("Test-set error counts at fixed confidence and IoU")
    axis.legend()
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    paths["errors"] = REPORTS / "comparison_error_counts.png"
    figure.savefig(paths["errors"], dpi=180)
    plt.close(figure)

    split_names = ["Train", "Validation", "Test"]
    split_keys = ["train", "val", "test"]
    image_counts = [small["dataset"]["splits"][key]["images"] for key in split_keys]
    box_counts = [small["dataset"]["splits"][key]["pothole_boxes"] for key in split_keys]
    figure, left = plt.subplots(figsize=(8, 4.6))
    bars = left.bar(split_names, image_counts, color="#3977a8", label="Images")
    left.set_ylabel("Images")
    left.set_title("Cleaned dataset split")
    right = left.twinx()
    right.plot(split_names, box_counts, color="#d95d39", marker="o", linewidth=2, label="Pothole boxes")
    right.set_ylabel("Pothole bounding boxes")
    lines, line_labels = right.get_legend_handles_labels()
    left.legend([bars, *lines], ["Images", *line_labels], loc="upper right")
    left.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    paths["dataset"] = REPORTS / "dataset_split_summary.png"
    figure.savefig(paths["dataset"], dpi=180)
    plt.close(figure)
    return paths


def write_comparison_csv(small: dict[str, Any], nano: dict[str, Any]) -> Path:
    output = REPORTS / "model_comparison.csv"
    fields = [
        "model",
        "epochs",
        "image_size",
        "device",
        "seed",
        "validation_precision",
        "validation_recall",
        "validation_map50",
        "validation_map50_95",
        "test_precision",
        "test_recall",
        "test_f1",
        "test_map50",
        "test_map50_95",
        "test_cpu_images_per_second",
        "test_tp",
        "test_fp",
        "test_fn",
        "checkpoint",
    ]
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for experiment in (small, nano):
            counts = experiment["test_error_counts"]
            writer.writerow(
                {
                    "model": experiment["model_weights"],
                    "epochs": experiment["epochs"],
                    "image_size": experiment["image_size"],
                    "device": experiment["device"],
                    "seed": experiment["seed"],
                    "validation_precision": experiment["validation"]["precision"],
                    "validation_recall": experiment["validation"]["recall"],
                    "validation_map50": experiment["validation"]["map50"],
                    "validation_map50_95": experiment["validation"]["map50_95"],
                    "test_precision": experiment["precision"],
                    "test_recall": experiment["recall"],
                    "test_f1": experiment["f1"],
                    "test_map50": experiment["map50"],
                    "test_map50_95": experiment["map50_95"],
                    "test_cpu_images_per_second": experiment["test_inference_images_per_second"],
                    "test_tp": counts["tp"],
                    "test_fp": counts["fp"],
                    "test_fn": counts["fn"],
                    "checkpoint": experiment["weights"],
                }
            )
    return output


def style_document(document: Document) -> None:
    section = document.sections[0]
    section.top_margin = Inches(0.65)
    section.bottom_margin = Inches(0.65)
    section.left_margin = Inches(0.72)
    section.right_margin = Inches(0.72)
    normal = document.styles["Normal"]
    normal.font.name = "Aptos"
    normal.font.size = Pt(9.5)
    normal.font.color.rgb = RGBColor(42, 52, 62)
    for name, size, color in (
        ("Title", 27, "123B5D"),
        ("Heading 1", 16, "123B5D"),
        ("Heading 2", 12, "246B8E"),
    ):
        style = document.styles[name]
        style.font.name = "Aptos Display"
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = RGBColor.from_string(color)


def add_table(document: Document, headers: list[str], rows: list[list[str]]) -> None:
    table = document.add_table(rows=1, cols=len(headers))
    table.style = "Light Shading Accent 1"
    for cell, value in zip(table.rows[0].cells, headers):
        cell.text = value
    for row in rows:
        cells = table.add_row().cells
        for cell, value in zip(cells, row):
            cell.text = str(value)


def add_figure(document: Document, path: Path, caption: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Expected report chart is missing: {path}")
    document.add_picture(str(path), width=Inches(6.6))
    paragraph = document.add_paragraph(caption)
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER


def add_code_block(document: Document, code: str) -> None:
    for line in code.rstrip().splitlines():
        paragraph = document.add_paragraph()
        paragraph.paragraph_format.space_after = Pt(0)
        run = paragraph.add_run(line or " ")
        run.font.name = "Consolas"
        run.font.size = Pt(8)
        run.font.color.rgb = RGBColor(30, 55, 75)


def extract_training_code() -> str:
    source = CODE_PATH.read_text(encoding="utf-8")
    start = source.index("    model = YOLO(model_weights)", source.index("def train_and_evaluate"))
    end = source.index("    best_weights =", start)
    return source[start:end].rstrip()


def add_references(document: Document) -> None:
    references = [
        "Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016). You Only Look Once: Unified, Real-Time Object Detection. CVPR 2016. https://arxiv.org/abs/1506.02640",
        "Ren, S., He, K., Girshick, R., & Sun, J. (2015). Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks. NeurIPS 2015. https://arxiv.org/abs/1506.01497",
        "Liu, W. et al. (2016). SSD: Single Shot MultiBox Detector. ECCV 2016. https://arxiv.org/abs/1512.02325",
        "Ultralytics. YOLOv8 documentation and model usage. https://docs.ultralytics.com/models/yolov8/",
        "abdulmohsen-fahad. Road Damage Dataset, Roboflow Universe, mirrored as manot/pothole-segmentation. Dataset card and annotations: https://huggingface.co/datasets/manot/pothole-segmentation. Dataset license stated by the card: CC BY 4.0.",
        "Arya, D. et al. (2022). RDD2022 – The multi-national Road Damage Dataset released through CRDDC'2022. Figshare. DOI: 10.6084/m9.figshare.21431547.v1. https://figshare.com/articles/dataset/21431547",
    ]
    for reference in references:
        document.add_paragraph(reference, style="List Number")


def build_report(small: dict[str, Any], nano: dict[str, Any]) -> Path:
    charts = create_comparison_charts(small, nano)
    comparison_csv = write_comparison_csv(small, nano)
    # Select using validation data only; test results remain a final, separately reported check.
    winner = max(
        (small, nano),
        key=lambda experiment: (
            experiment["validation"]["map50"],
            experiment["validation"]["map50_95"],
            experiment["test_inference_images_per_second"],
        ),
    )
    runner_up = nano if winner is small else small
    validation_gap = winner["validation"]["map50"] - runner_up["validation"]["map50"]

    document = Document()
    style_document(document)
    title = document.add_paragraph(style="Title")
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.add_run("ROADWATCH")
    subtitle = document.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    text = subtitle.add_run("Pothole Detection: Comparative Methods and Results")
    text.bold = True
    text.font.size = Pt(17)
    byline = document.add_paragraph()
    byline.alignment = WD_ALIGN_PARAGRAPH.CENTER
    byline.add_run(
        f"Research progress report | Neeraj Sonar | {datetime.now().strftime('%d %B %Y')}"
    )
    document.add_page_break()

    document.add_heading("Abstract", level=1)
    document.add_paragraph(
        "This report presents a reproducible pothole-detection baseline and a controlled "
        "comparison between two pretrained YOLOv8 model scales, nano (YOLOv8n) and small "
        "(YOLOv8s). Both models use the same cleaned, one-class dataset, train/validation/test "
        "partitions, CPU device, image size, epoch count, random seed, confidence threshold, "
        "and IoU matching threshold. Model selection is based on validation mAP@0.5; the "
        "held-out test metrics are reported separately. The report also reviews established "
        "one-stage and two-stage detectors. Faster R-CNN and SSD are literature baselines, "
        "not models trained in this experiment. Results therefore support a fair comparison "
        "within the tested YOLOv8 family, not a claim that YOLO is universally the best detector."
    )

    document.add_heading("1. Problem and project scope", level=1)
    document.add_paragraph(
        "RoadWatch is an AI-assisted road-damage reporting and repair-prioritisation concept. "
        "Its intended product flow is: image report → pothole detection → location association "
        "→ duplicate-report handling → explainable severity/priority → officer repair workflow. "
        "This experimental milestone addresses the detector and its data/evaluation pipeline. "
        "GPS, location-based deduplication, human-validated severity, user management and the "
        "officer dashboard are not implemented or evaluated in this report."
    )
    document.add_paragraph(
        "Research question: under a matched, reproducible short-run protocol, which of the "
        "two tested YOLOv8 model sizes gives the stronger validation detection score, and what "
        "accuracy–speed trade-off appears on the held-out test images?"
    )

    document.add_heading("2. Dataset and data preparation", level=1)
    document.add_paragraph(
        "The experiment uses the public pothole dataset mirrored as manot/pothole-segmentation "
        "on Hugging Face and sourced from a Roboflow Universe road-damage dataset [5]. It provides "
        "road images and COCO bounding-box annotations. The dataset card states CC BY 4.0. "
        "Its three original splits are retained, subject only to removal of byte-identical "
        "copies from lower-priority splits."
    )
    split_rows = []
    for label, key in (("Train", "train"), ("Validation", "val"), ("Test", "test")):
        info = small["dataset"]["splits"][key]
        split_rows.append(
            [
                label,
                f"{info['source_images']}",
                f"{info['images']}",
                f"{info['pothole_boxes']}",
                f"{info['duplicate_images_removed']}",
            ]
        )
    add_table(
        document,
        ["Split", "Source images", "Clean images", "Pothole boxes", "Duplicate copies removed"],
        split_rows,
    )
    add_figure(document, charts["dataset"], "Figure 1. Image and pothole-box counts in the cleaned split.")
    document.add_paragraph(
        "Annotation processing: the two spelling variants “pothole” and “potholes” are mapped "
        "to a single target class. Generic “object” annotations are excluded (3 in train, 2 in "
        "validation, 1 in test), not relabelled as potholes. COCO [x, y, width, height] boxes are "
        "clipped to image bounds and converted to YOLO normalized center-x, center-y, width, "
        "height coordinates."
    )
    duplicate_count = small["dataset"]["deduplication"]["cross_split_duplicate_groups"]
    document.add_paragraph(
        f"Data-integrity check: SHA-256 identified {duplicate_count} exact-image groups that "
        "crossed the source split boundaries. For each exact duplicate group, the test copy was "
        "retained first, then validation, then train; remaining duplicate copies were removed. "
        "This reduces exact-byte leakage but does not detect visually similar re-encoded or "
        "cropped images. It also removes some examples from the publisher’s original partitions, "
        "so both experiment candidates use the exact same cleaned copies."
    )
    document.add_paragraph(
        "Scope limitation: this compact dataset is appropriate for a short baseline, but does not "
        "demonstrate Indian-road or day/night/rain generalization. The report does not claim "
        "RDD2022 [6]; that separate Figshare dataset is a future benchmark."
    )

    document.add_heading("3. Existing object-detection methods", level=1)
    document.add_paragraph(
        "Object detection predicts both a category and a bounding box. Existing approaches "
        "include one-stage detectors, which predict boxes and classes directly from image "
        "features, and two-stage detectors, which first propose regions and then classify/refine "
        "them. The cited methods are established alternatives; only the two YOLOv8 sizes in "
        "the next section were executed in this project."
    )
    add_table(
        document,
        ["Method", "Detection family", "Why it is used / trade-off", "Status in this project"],
        [
            [
                "YOLO family (YOLOv1 [1]; this experiment uses Ultralytics YOLOv8 [4])",
                "One-stage",
                "Direct box/class prediction; practical speed and deployment ecosystem. Performance depends on data, model size and threshold.",
                "Experimentally tested as YOLOv8n and YOLOv8s.",
            ],
            [
                "SSD [3]",
                "One-stage",
                "Predicts boxes at multiple feature-map scales in one pass; useful speed-oriented historical baseline.",
                "Literature review only; not trained/evaluated here.",
            ],
            [
                "Faster R-CNN [2]",
                "Two-stage",
                "Region Proposal Network followed by region classification/refinement; strong established detector but generally heavier than a compact one-stage model.",
                "Literature review only; not trained/evaluated here.",
            ],
        ],
    )
    document.add_paragraph(
        "The experimental comparison is deliberately described as a model-size comparison "
        "within YOLOv8, not as a head-to-head benchmark of YOLO versus SSD or Faster R-CNN. "
        "Future work can add those detectors if compute and a matched implementation protocol "
        "are available."
    )

    document.add_heading("4. Experimental method and reproducibility", level=1)
    if small["dataset"]["dataset_id"] != nano["dataset"]["dataset_id"]:
        raise ValueError("Model candidates must use the same dataset.")
    settings = [
        ["Dataset/splits", small["dataset"]["dataset_id"], "Same cleaned train/val/test files"],
        ["Pretrained weights", "COCO-pretrained YOLOv8n.pt / YOLOv8s.pt", "Transfer-learning initialization"],
        ["Training epochs", str(small["epochs"]), "Matched"],
        ["Input resolution", f"{small['image_size']} × {small['image_size']}", "Matched"],
        ["Device and batch", f"{small['device']}, batch size 8", "Matched"],
        ["Seed", str(small["seed"]), "Matched"],
        ["Data augmentation", "Ultralytics training defaults", "Same implementation defaults"],
        ["P/R/F1 threshold", f"confidence {small['confidence_threshold']:.2f}", "Fixed for both"],
        ["Box-match criterion", f"IoU ≥ {small['iou_threshold']:.2f}", "Fixed for both"],
    ]
    add_table(document, ["Setting", "Value", "Control"], settings)
    document.add_paragraph(
        "Both candidates fine-tune the corresponding COCO-pretrained weights on the same "
        "one-class dataset. Ultralytics training validates on the validation partition; the "
        "best checkpoint is explicitly evaluated on validation for model selection and on the "
        "held-out test split for final reported test results. Validation mAP@0.5 is the primary "
        "selection criterion. Validation mAP@0.5:0.95 and CPU speed are tie-break/context "
        "measures. Test metrics are disclosed but are not used to select the winner. No custom "
        "augmentation values were introduced; both runs use the same Ultralytics training "
        "defaults, recorded in each run's args.yaml."
    )
    document.add_paragraph(
        "Precision is TP/(TP+FP), recall is TP/(TP+FN), and F1 is the harmonic mean of "
        "precision and recall at the fixed threshold. For these fixed-threshold counts, "
        "predictions are sorted by confidence and greedily matched one-to-one to an unmatched "
        "ground-truth box when IoU is at least 0.50. IoU is intersection over union between a "
        "predicted and labelled box. mAP@0.5 summarizes the precision–recall curve at IoU 0.50; "
        "mAP@0.5:0.95 averages across stricter IoU thresholds. A single “accuracy” percentage "
        "is not a sufficient object-detection metric because a correct result requires both "
        "correct classification and adequate box overlap, while background pixels dominate. "
        "Therefore detection accuracy is reported using precision, recall, F1 and mAP."
    )

    document.add_heading("5. Comparative results and best tested method", level=1)
    validation_rows = []
    test_rows = []
    for experiment in (small, nano):
        validation_rows.append(
            [
                experiment["model_weights"],
                f"{experiment['validation']['precision']:.3f}",
                f"{experiment['validation']['recall']:.3f}",
                f"{experiment['validation']['map50']:.3f}",
                f"{experiment['validation']['map50_95']:.3f}",
            ]
        )
        test_rows.append(
            [
                experiment["model_weights"],
                f"{experiment['precision']:.3f}",
                f"{experiment['recall']:.3f}",
                f"{experiment['f1']:.3f}",
                f"{experiment['map50']:.3f}",
                f"{experiment['map50_95']:.3f}",
                f"{experiment['test_inference_images_per_second']:.2f}",
            ]
        )
    add_table(
        document,
        ["Model", "Val P", "Val R", "Val mAP50", "Val mAP50–95"],
        validation_rows,
    )
    add_table(
        document,
        ["Model", "Test P", "Test R", "Test F1", "mAP50", "mAP50–95", "CPU img/s"],
        test_rows,
    )
    add_figure(document, charts["test_metrics"], "Figure 2. Test-set detection scores for both trained model sizes.")
    add_figure(document, charts["tradeoff"], "Figure 3. Test mAP@0.5 versus measured CPU throughput.")
    add_figure(document, charts["errors"], "Figure 4. Matched detections, false positives and missed potholes.")
    document.add_heading("Comparative analysis", level=2)
    document.add_paragraph(
        f"By the predeclared selection rule (validation mAP@0.5), the best tested candidate is "
        f"{winner['model_weights']} with validation mAP@0.5 = {winner['validation']['map50']:.3f}. "
        f"The runner-up {runner_up['model_weights']} scored {runner_up['validation']['map50']:.3f}; "
        f"the difference is {validation_gap:+.3f}. This identifies the best of these two tested "
        "configurations only; it is not evidence that it is the best detector overall."
    )
    document.add_paragraph(
        f"On the held-out test split, the selected candidate has precision {winner['precision']:.3f}, "
        f"recall {winner['recall']:.3f}, F1 {winner['f1']:.3f}, mAP@0.5 "
        f"{winner['map50']:.3f}, mAP@0.5:0.95 {winner['map50_95']:.3f}, and CPU throughput "
        f"{winner['test_inference_images_per_second']:.2f} images/second. "
        f"{'It is also faster than the alternative in this run.' if winner['test_inference_images_per_second'] > runner_up['test_inference_images_per_second'] else 'It is slower than the alternative in this run, so the accuracy–speed trade-off should be considered.'}"
    )
    counts = winner["test_error_counts"]
    document.add_paragraph(
        f"At confidence ≥ {winner['confidence_threshold']:.2f} and IoU ≥ "
        f"{winner['iou_threshold']:.2f}, the selected test run contains {counts['tp']} "
        f"matched detections, {counts['fp']} false positives, and {counts['fn']} missed potholes "
        f"among {counts['ground_truth']} labelled boxes. Recall and the miss count remain "
        "important weaknesses; model selection does not imply deployment readiness."
    )
    document.add_paragraph(
        "The CPU throughput is a local end-to-end per-image prediction measurement and is "
        "machine-specific. The speed chart is descriptive, not a hardware-normalized benchmark."
    )

    document.add_heading("6. Discussion, limitations and next work", level=1)
    for point in (
        "Only two sizes from the same YOLOv8 family were experimentally compared. SSD and Faster R-CNN are documented as existing methods, but no comparative scores are claimed for them.",
        "Five epochs on a small dataset are a short baseline. The absolute scores are pilot results and may change with longer training, augmentation, hardware and data composition.",
        "The dataset does not supply a verified day/night/rain label protocol, location metadata, pothole depth or human severity ratings. Condition-wise robustness, geographic duplicate merging and severity agreement cannot be concluded from this experiment.",
        "Exact-byte deduplication does not catch similar images that were re-encoded, resized or cropped.",
        "The intended RoadWatch GPS workflow, database, authentication and repair-status dashboard remain future implementation work.",
    ):
        document.add_paragraph(point, style="List Bullet")
    document.add_paragraph(
        "Next steps: inspect the per-image false detections and misses; tune confidence on "
        "validation only; run a longer controlled experiment; compare a genuinely different "
        "detector such as SSD/Faster R-CNN if feasible; and evaluate an appropriately licensed "
        "India-focused RDD2022 subset before making regional claims. Validate any explainable "
        "severity score against independent human ratings before describing it as validated."
    )

    document.add_heading("7. Code, run commands and experiment artifacts", level=1)
    document.add_paragraph(
        "The implementation is in scripts/roadwatch_baseline.py. It performs download, "
        "COCO-to-YOLO conversion, SHA-256 split audit, pretrained model fine-tuning, validation "
        "and test evaluation, fixed-threshold error matching, and output writing. The report "
        "generator is scripts/build_comparative_report.py."
    )
    for command in (
        "python -m pip install -r requirements.txt",
        "python scripts\\roadwatch_baseline.py prepare",
        "python scripts\\roadwatch_baseline.py train --model yolov8n.pt --epochs 5 --imgsz 416",
        "python scripts\\roadwatch_baseline.py validate --model yolov8n.pt",
        "python scripts\\roadwatch_baseline.py train --model yolov8s.pt --epochs 5 --imgsz 416",
        "python scripts\\roadwatch_baseline.py validate --model yolov8s.pt",
        "python scripts\\build_comparative_report.py",
    ):
        add_code_block(document, command)
    document.add_paragraph("Core training call, extracted from the implemented training script:")
    add_code_block(document, extract_training_code())
    add_table(
        document,
        ["Artifact", "Purpose"],
        [
            ["reports/metrics.json", "YOLOv8n configuration, validation/test results and counts."],
            ["reports/metrics_yolov8s.json", "YOLOv8s configuration, validation/test results and counts."],
            ["reports/test_error_analysis_yolov8n.csv / yolov8s.csv", "Per-image TP/FP/FN records at the fixed thresholds."],
            ["reports/predictions/yolov8n/ and yolov8s/", "Ground-truth and prediction overlays for error review."],
            ["reports/comparison_metrics.png", "Grouped model-metric comparison."],
            ["reports/comparison_speed_accuracy.png", "Test mAP@0.5 and CPU throughput trade-off."],
            ["reports/comparison_error_counts.png", "Matched detections and error counts."],
            [str(comparison_csv.relative_to(ROOT)), "Tabular validation/test results for both candidates."],
            ["runs/", "Training logs, best checkpoints and Ultralytics evaluation outputs."],
            ["runs/<experiment>/args.yaml", "Saved model/training settings, including the augmentation defaults."],
        ],
    )

    document.add_heading("References", level=1)
    add_references(document)
    document.add_paragraph(
        "Licensing note: retain the dataset attribution and comply with CC BY 4.0. Check the "
        "applicable Ultralytics AGPL-3.0/Enterprise terms before distributing or deploying a "
        "model/application."
    )

    output = REPORTS / "RoadWatch_Final_Comparative_Report.docx"
    document.save(output)
    return output


def main() -> None:
    global BASELINE_PATH, COMPARISON_PATH
    parser = argparse.ArgumentParser(description="Build RoadWatch comparative research report.")
    parser.add_argument("--yolov8n", type=Path, default=BASELINE_PATH)
    parser.add_argument("--yolov8s", type=Path, default=COMPARISON_PATH)
    args = parser.parse_args()
    BASELINE_PATH = args.yolov8n
    COMPARISON_PATH = args.yolov8s
    small, nano = load_experiments()
    report_path = build_report(small, nano)
    print(f"Generated {report_path.relative_to(ROOT)} ({report_path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
