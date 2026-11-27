from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import time
import urllib.request
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
DATASET_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports"
RUNS_DIR = ROOT / "runs"
DATASET_ID = "manot/pothole-segmentation"
DATASET_URL = "https://huggingface.co/datasets/manot/pothole-segmentation"
DATASET_FILES = {
    "train": "https://huggingface.co/datasets/manot/pothole-segmentation/resolve/main/data/train.zip?download=true",
    "val": "https://huggingface.co/datasets/manot/pothole-segmentation/resolve/main/data/valid.zip?download=true",
    "test": "https://huggingface.co/datasets/manot/pothole-segmentation/resolve/main/data/test.zip?download=true",
}
ARCHIVE_NAMES = {"train": "train", "val": "valid", "test": "test"}
POTHOLE_NAMES = {"pothole", "potholes"}
CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.5


def ensure_downloads() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for split, url in DATASET_FILES.items():
        archive = RAW_DIR / f"{ARCHIVE_NAMES[split]}.zip"
        if archive.exists() and zipfile.is_zipfile(archive):
            continue
        partial = archive.with_suffix(".zip.part")
        print(f"Downloading {DATASET_ID} {split} split...")
        request = urllib.request.Request(url, headers={"User-Agent": "RoadWatch-baseline/1.0"})
        try:
            with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as target:
                shutil.copyfileobj(response, target)
        except Exception as exc:
            partial.unlink(missing_ok=True)
            raise RuntimeError(f"Dataset download failed for {split}: {exc}") from exc
        if not zipfile.is_zipfile(partial):
            partial.unlink(missing_ok=True)
            raise RuntimeError(f"Downloaded {split} archive is not a valid ZIP file.")
        partial.replace(archive)
        print(f"Saved {archive.relative_to(ROOT)} ({archive.stat().st_size:,} bytes)")


def prepare_dataset() -> dict[str, Any]:
    dataset_stats: dict[str, Any] = {
        "dataset_id": DATASET_ID,
        "source_url": DATASET_URL,
        "license": "CC BY 4.0",
        "class_mapping": {
            "pothole": "pothole",
            "potholes": "pothole",
            "object": "ignored (not a pothole class)",
        },
        "splits": {},
    }

    for split in ("train", "val", "test"):
        archive = RAW_DIR / f"{ARCHIVE_NAMES[split]}.zip"
        if not archive.exists() or not zipfile.is_zipfile(archive):
            raise FileNotFoundError(f"Missing or invalid archive: {archive}")
        with zipfile.ZipFile(archive) as source:
            annotation_names = [
                name for name in source.namelist()
                if Path(name).name == "_annotations.coco.json"
            ]
            if len(annotation_names) != 1:
                raise ValueError(f"{archive.name} must contain exactly one COCO annotation file.")
            coco = json.loads(source.read(annotation_names[0]))
            categories = {category["id"]: category["name"].strip().lower() for category in coco["categories"]}
            annotations_by_image: dict[int, list[dict[str, Any]]] = {}
            ignored_labels = Counter()
            invalid_boxes = 0

            for annotation in coco["annotations"]:
                category_name = categories.get(annotation["category_id"], "")
                if category_name not in POTHOLE_NAMES:
                    ignored_labels[category_name or "unknown"] += 1
                    continue
                annotations_by_image.setdefault(annotation["image_id"], []).append(annotation)

            output_images = DATASET_DIR / split / "images"
            output_labels = DATASET_DIR / split / "labels"
            output_images.mkdir(parents=True, exist_ok=True)
            output_labels.mkdir(parents=True, exist_ok=True)
            image_members = {Path(name).name: name for name in source.namelist() if Path(name).suffix.lower() in {".jpg", ".jpeg", ".png"}}
            mapped_boxes = 0
            empty_images = 0

            for image in coco["images"]:
                filename = Path(image["file_name"]).name
                member = image_members.get(filename)
                if member is None:
                    raise FileNotFoundError(f"{filename} referenced by annotations is missing from {archive.name}.")
                width, height = float(image["width"]), float(image["height"])
                if width <= 0 or height <= 0:
                    raise ValueError(f"Invalid image dimensions in annotation for {filename}.")
                destination_image = output_images / filename
                destination_image.write_bytes(source.read(member))
                lines = []
                for annotation in annotations_by_image.get(image["id"], []):
                    x, y, box_width, box_height = map(float, annotation["bbox"])
                    x1 = min(max(x, 0.0), width)
                    y1 = min(max(y, 0.0), height)
                    x2 = min(max(x + box_width, 0.0), width)
                    y2 = min(max(y + box_height, 0.0), height)
                    if x2 <= x1 or y2 <= y1:
                        invalid_boxes += 1
                        continue
                    center_x = ((x1 + x2) / 2.0) / width
                    center_y = ((y1 + y2) / 2.0) / height
                    normalized_width = (x2 - x1) / width
                    normalized_height = (y2 - y1) / height
                    lines.append(
                        f"0 {center_x:.6f} {center_y:.6f} "
                        f"{normalized_width:.6f} {normalized_height:.6f}"
                    )
                if not lines:
                    empty_images += 1
                (output_labels / f"{Path(filename).stem}.txt").write_text(
                    "\n".join(lines), encoding="utf-8"
                )
                mapped_boxes += len(lines)

            dataset_stats["splits"][split] = {
                "source_images": len(coco["images"]),
                "source_pothole_boxes": mapped_boxes,
                "source_images_without_potholes": empty_images,
                "ignored_annotations": dict(ignored_labels),
                "invalid_boxes_skipped": invalid_boxes,
            }
            print(
                f"{split}: {len(coco['images'])} images, {mapped_boxes} pothole boxes; "
                f"ignored labels={dict(ignored_labels)}"
            )

    # Keep test copies when exact files leaked across the publisher's supplied splits.
    split_priority = {"test": 0, "val": 1, "train": 2}
    images_by_hash: dict[str, list[tuple[str, Path]]] = {}
    for split in ("train", "val", "test"):
        for image_path in (DATASET_DIR / split / "images").iterdir():
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
            images_by_hash.setdefault(digest, []).append((split, image_path))
    removed_per_split = Counter()
    duplicate_groups = 0
    cross_split_groups = 0
    for copies in images_by_hash.values():
        if len(copies) < 2:
            continue
        duplicate_groups += 1
        if len({split for split, _ in copies}) > 1:
            cross_split_groups += 1
        copies.sort(key=lambda item: (split_priority[item[0]], item[1].name))
        for split, image_path in copies[1:]:
            (DATASET_DIR / split / "labels" / f"{image_path.stem}.txt").unlink(missing_ok=True)
            image_path.unlink()
            removed_per_split[split] += 1
    dataset_stats["deduplication"] = {
        "method": "SHA-256 of exact image bytes; test takes precedence over validation, then training",
        "duplicate_groups_found": duplicate_groups,
        "cross_split_duplicate_groups": cross_split_groups,
        "duplicate_images_removed": sum(removed_per_split.values()),
        "removed_by_split": dict(removed_per_split),
    }
    for split in ("train", "val", "test"):
        label_files = list((DATASET_DIR / split / "labels").glob("*.txt"))
        dataset_stats["splits"][split].update(
            {
                "images": len(label_files),
                "pothole_boxes": sum(
                    len(label.read_text(encoding="utf-8").splitlines()) for label in label_files
                ),
                "images_without_potholes": sum(
                    not label.read_text(encoding="utf-8").strip() for label in label_files
                ),
                "duplicate_images_removed": removed_per_split[split],
            }
        )

    yaml_path = DATASET_DIR / "roadwatch.yaml"
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text(
        yaml.safe_dump(
            {
                "path": DATASET_DIR.as_posix(),
                "train": "train/images",
                "val": "val/images",
                "test": "test/images",
                "names": {0: "pothole"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    DATASET_DIR.joinpath("dataset_stats.json").write_text(
        json.dumps(dataset_stats, indent=2), encoding="utf-8"
    )
    return dataset_stats


def intersection_over_union(box_a: np.ndarray, box_b: np.ndarray) -> float:
    x1 = max(float(box_a[0]), float(box_b[0]))
    y1 = max(float(box_a[1]), float(box_b[1]))
    x2 = min(float(box_a[2]), float(box_b[2]))
    y2 = min(float(box_a[3]), float(box_b[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(box_a[2] - box_a[0])) * max(0.0, float(box_a[3] - box_a[1]))
    area_b = max(0.0, float(box_b[2] - box_b[0])) * max(0.0, float(box_b[3] - box_b[1]))
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def read_ground_truth(label_path: Path, width: int, height: int) -> list[np.ndarray]:
    if not label_path.exists():
        raise FileNotFoundError(f"Ground-truth label file not found: {label_path}")
    boxes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"Malformed YOLO label in {label_path}: {line}")
        _, cx, cy, bw, bh = map(float, parts)
        boxes.append(
            np.array(
                [
                    (cx - bw / 2) * width,
                    (cy - bh / 2) * height,
                    (cx + bw / 2) * width,
                    (cy + bh / 2) * height,
                ],
                dtype=float,
            )
        )
    return boxes


def evaluate_error_cases(
    model: YOLO,
    imgsz: int,
    artifact_slug: str,
) -> tuple[list[dict[str, Any]], dict[str, int], float]:
    image_dir = DATASET_DIR / "test" / "images"
    label_dir = DATASET_DIR / "test" / "labels"
    files = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not files:
        raise FileNotFoundError(f"No test images found in {image_dir}")

    rows = []
    totals = {"tp": 0, "fp": 0, "fn": 0, "ground_truth": 0, "predictions": 0}
    prediction_seconds = 0.0
    for image_path in files:
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"OpenCV could not read test image: {image_path}")
        height, width = image.shape[:2]
        ground_truth = read_ground_truth(label_dir / f"{image_path.stem}.txt", width, height)
        started = time.perf_counter()
        prediction = model.predict(
            source=str(image_path), imgsz=imgsz, conf=CONFIDENCE_THRESHOLD, iou=0.7,
            device="cpu", verbose=False,
        )[0]
        prediction_seconds += time.perf_counter() - started
        if prediction.boxes is None or len(prediction.boxes) == 0:
            predicted_boxes = []
            confidences = []
        else:
            predicted_boxes = prediction.boxes.xyxy.cpu().numpy()
            confidences = prediction.boxes.conf.cpu().numpy()

        used_ground_truth: set[int] = set()
        tp = 0
        fp = 0
        for index in np.argsort(confidences)[::-1]:
            candidates = [
                (intersection_over_union(predicted_boxes[index], gt), gt_index)
                for gt_index, gt in enumerate(ground_truth)
                if gt_index not in used_ground_truth
            ]
            best_iou, best_index = max(candidates, default=(0.0, -1))
            if best_iou >= IOU_THRESHOLD:
                tp += 1
                used_ground_truth.add(best_index)
            else:
                fp += 1
        fn = len(ground_truth) - tp
        totals["tp"] += tp
        totals["fp"] += fp
        totals["fn"] += fn
        totals["ground_truth"] += len(ground_truth)
        totals["predictions"] += len(predicted_boxes)
        rows.append(
            {
                "image": image_path.name,
                "ground_truth": len(ground_truth),
                "predictions": len(predicted_boxes),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "seconds": round(time.perf_counter() - started, 4),
            }
        )
        if tp + fp + fn > 0:
            save_error_visualization(
                image,
                ground_truth,
                predicted_boxes,
                confidences,
                image_path.name,
                REPORT_DIR / "predictions" / artifact_slug,
            )

    fps = len(files) / prediction_seconds if prediction_seconds else 0.0
    return rows, totals, fps


def apply_fixed_threshold_metrics(metrics: dict[str, Any]) -> None:
    counts = metrics["test_error_counts"]
    metrics["confidence_threshold"] = CONFIDENCE_THRESHOLD
    metrics["iou_threshold"] = IOU_THRESHOLD
    metrics.pop("confidence_threshold_for_error_counts", None)
    metrics.pop("iou_threshold_for_error_counts", None)
    predictions = counts["predictions"]
    ground_truth = counts["ground_truth"]
    true_positives = counts["tp"]
    metrics["precision"] = true_positives / predictions if predictions else 0.0
    metrics["recall"] = true_positives / ground_truth if ground_truth else 0.0
    denominator = predictions + ground_truth
    metrics["f1"] = 2 * true_positives / denominator if denominator else 0.0


def save_error_visualization(
    image: np.ndarray,
    ground_truth: list[np.ndarray],
    predictions: list[np.ndarray],
    confidences: list[float],
    filename: str,
    output_dir: Path,
) -> None:
    canvas = image.copy()
    for box in ground_truth:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (60, 200, 60), 2)
    for box, confidence in zip(predictions, confidences):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (30, 30, 230), 2)
        cv2.putText(
            canvas, f"{confidence:.2f}", (x1, max(y1 - 5, 15)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (30, 30, 230), 1, cv2.LINE_AA,
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_dir / filename), canvas):
        raise OSError(f"Could not save prediction visualization: {output_dir / filename}")


def create_analysis_plot(
    metrics: dict[str, Any],
    rows: list[dict[str, Any]],
    output: Path | None = None,
) -> Path:
    if output is None:
        output = ROOT / metrics["analysis_plot"]
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    metric_names = ["Precision", "Recall", "mAP@0.5", "mAP@0.5:0.95"]
    metric_values = [
        metrics["precision"], metrics["recall"],
        metrics["map50"], metrics["map50_95"],
    ]
    axes[0].bar(metric_names, metric_values, color=["#3878a8", "#4c9f70", "#e0a33b", "#8b6bb1"])
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Score")
    axes[0].set_title("Held-out test metrics")
    axes[0].tick_params(axis="x", rotation=20)
    for index, value in enumerate(metric_values):
        axes[0].text(index, value + 0.02, f"{value:.3f}", ha="center", fontsize=9)

    sorted_rows = sorted(rows, key=lambda row: (row["fn"] + row["fp"], row["fn"]), reverse=True)[:10]
    names = [row["image"][:12] for row in sorted_rows]
    positions = np.arange(len(sorted_rows))
    axes[1].barh(positions, [row["fn"] for row in sorted_rows], label="Missed potholes (FN)", color="#d95d39")
    axes[1].barh(
        positions,
        [row["fp"] for row in sorted_rows],
        left=[row["fn"] for row in sorted_rows],
        label="False detections (FP)", color="#e0a33b",
    )
    axes[1].set_yticks(positions, names)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Count at confidence 0.25 / IoU 0.50")
    axes[1].set_title("Largest per-image error counts")
    axes[1].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output, dpi=170)
    plt.close(figure)
    return output


def evaluate_checkpoint(
    best_weights: Path,
    model_weights: str,
    epochs: int,
    imgsz: int,
) -> dict[str, Any]:
    if not best_weights.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {best_weights}")
    artifact_slug = Path(model_weights).stem
    dataset_yaml = DATASET_DIR / "roadwatch.yaml"
    if not dataset_yaml.exists():
        raise FileNotFoundError("Prepared dataset not found. Run the prepare command first.")
    best_model = YOLO(str(best_weights))
    test_metrics = best_model.val(
        data=str(dataset_yaml),
        split="test",
        imgsz=imgsz,
        batch=8,
        device="cpu",
        workers=0,
        plots=True,
        project=str(RUNS_DIR),
        name=f"{best_weights.parent.parent.name}-test-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        exist_ok=False,
        verbose=True,
    )
    rows, counts, fps = evaluate_error_cases(best_model, imgsz, artifact_slug)
    if not rows:
        raise ValueError(f"No test predictions were produced for {best_weights}.")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    error_csv = REPORT_DIR / f"test_error_analysis_{artifact_slug}.csv"
    with error_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    stats_path = DATASET_DIR / "dataset_stats.json"
    dataset_stats = json.loads(stats_path.read_text(encoding="utf-8"))
    metrics = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "model": f"Ultralytics {artifact_slug} (COCO-pretrained fine-tuning)",
        "model_weights": model_weights,
        "experiment_id": artifact_slug,
        "weights": str(best_weights.relative_to(ROOT)),
        "epochs": epochs,
        "image_size": imgsz,
        "device": "CPU",
        "seed": 42,
        "evaluation_split": "official test split",
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "iou_threshold": IOU_THRESHOLD,
        "map50": float(test_metrics.box.map50),
        "map50_95": float(test_metrics.box.map),
        "test_inference_images_per_second": fps,
        "test_error_counts": counts,
        "test_images": len(rows),
        "dataset": dataset_stats,
        "errors_csv": str(error_csv.relative_to(ROOT)),
        "analysis_plot": f"reports/results_analysis_{artifact_slug}.png",
    }
    apply_fixed_threshold_metrics(metrics)
    create_analysis_plot(metrics, rows)
    metrics_filename = "metrics.json" if artifact_slug == "yolov8n" else f"metrics_{artifact_slug}.json"
    REPORT_DIR.joinpath(metrics_filename).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({key: metrics[key] for key in ("precision", "recall", "f1", "map50", "map50_95", "test_inference_images_per_second", "test_error_counts")}, indent=2))
    return metrics


def train_and_evaluate(epochs: int, imgsz: int, model_weights: str) -> dict[str, Any]:
    dataset_yaml = DATASET_DIR / "roadwatch.yaml"
    if not dataset_yaml.exists():
        raise FileNotFoundError("Prepared dataset not found. Run the prepare command first.")
    if model_weights not in {"yolov8n.pt", "yolov8s.pt"}:
        raise ValueError("This controlled comparison supports yolov8n.pt and yolov8s.pt.")
    artifact_slug = Path(model_weights).stem
    run_name = f"roadwatch-{artifact_slug}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    print(f"Loading pretrained {model_weights} weights; training for {epochs} epochs on CPU.")
    model = YOLO(model_weights)
    model.train(
        data=str(dataset_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=8,
        device="cpu",
        workers=0,
        seed=42,
        deterministic=True,
        patience=0,
        project=str(RUNS_DIR),
        name=run_name,
        exist_ok=False,
        plots=True,
        verbose=True,
    )
    best_weights = RUNS_DIR / run_name / "weights" / "best.pt"
    if not best_weights.exists():
        raise FileNotFoundError(f"Training completed without saving expected checkpoint: {best_weights}")
    return evaluate_checkpoint(best_weights, model_weights, epochs, imgsz)


def evaluate_saved_checkpoint(model_weights: str) -> dict[str, Any]:
    artifact_slug = Path(model_weights).stem
    checkpoints = list(RUNS_DIR.glob(f"roadwatch-{artifact_slug}-*/weights/best.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No trained checkpoint found for {model_weights} in {RUNS_DIR}.")
    best_weights = max(checkpoints, key=lambda path: path.stat().st_mtime)
    args_path = best_weights.parent.parent / "args.yaml"
    if not args_path.exists():
        raise FileNotFoundError(f"Training arguments are missing: {args_path}")
    training_args = yaml.safe_load(args_path.read_text(encoding="utf-8"))
    return evaluate_checkpoint(
        best_weights,
        model_weights,
        int(training_args["epochs"]),
        int(training_args["imgsz"]),
    )


def evaluate_validation(model_weights: str) -> dict[str, Any]:
    artifact_slug = Path(model_weights).stem
    metrics_filename = "metrics.json" if artifact_slug == "yolov8n" else f"metrics_{artifact_slug}.json"
    metrics_path = REPORT_DIR / metrics_filename
    if not metrics_path.exists():
        raise FileNotFoundError(f"Run training for {model_weights} before validation.")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    weights_path = ROOT / metrics["weights"]
    if not weights_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {weights_path}")
    model = YOLO(str(weights_path))
    result = model.val(
        data=str(DATASET_DIR / "roadwatch.yaml"),
        split="val",
        imgsz=metrics["image_size"],
        batch=8,
        device="cpu",
        workers=0,
        plots=False,
        project=str(RUNS_DIR),
        name=f"{artifact_slug}-selection-val-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        exist_ok=False,
        verbose=True,
    )
    metrics["validation"] = {
        "split": "validation",
        "precision": float(result.box.mp),
        "recall": float(result.box.mr),
        "map50": float(result.box.map50),
        "map50_95": float(result.box.map),
    }
    metrics["model_selection_metric"] = "validation mAP@0.5 (higher is better)"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics["validation"], indent=2))
    return metrics


def set_document_style(document: Document) -> None:
    section = document.sections[0]
    section.top_margin = Inches(0.65)
    section.bottom_margin = Inches(0.65)
    section.left_margin = Inches(0.75)
    section.right_margin = Inches(0.75)
    normal = document.styles["Normal"]
    normal.font.name = "Aptos"
    normal.font.size = Pt(10)
    normal.font.color.rgb = RGBColor(42, 52, 62)
    for style_name, size, color in [
        ("Title", 28, "123B5D"),
        ("Heading 1", 17, "123B5D"),
        ("Heading 2", 12, "246B8E"),
    ]:
        style = document.styles[style_name]
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


def generate_report(metrics: dict[str, Any]) -> Path:
    document = Document()
    set_document_style(document)
    date_text = datetime.now().strftime("%d %B %Y")

    title = document.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.paragraph_format.space_before = Pt(45)
    run = title.add_run("ROADWATCH")
    run.bold = True
    run.font.size = Pt(31)
    run.font.color.rgb = RGBColor(18, 59, 93)
    subtitle = document.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = subtitle.add_run("Two-Week Baseline Progress Report")
    run.bold = True
    run.font.size = Pt(18)
    subtitle = document.add_paragraph(
        "Real-data pothole detection experiment, evaluation and next-step workflow"
    )
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    document.add_paragraph()
    details = document.add_paragraph()
    details.alignment = WD_ALIGN_PARAGRAPH.CENTER
    details.add_run(f"Prepared for: Neeraj Sonar\nGenerated: {date_text}\n")
    details.add_run("Project: AI-based pothole detection, severity scoring and repair prioritisation")
    document.add_page_break()

    document.add_heading("1. Executive summary", level=1)
    document.add_paragraph(
        "This report turns the existing RoadWatch presentation and execution roadmap into a "
        "small, reproducible implementation milestone. It uses a real public pothole dataset, "
        "converts its published bounding boxes, fine-tunes a lightweight YOLO detector, evaluates "
        "on the dataset's held-out test split and records image-level false positives and misses. "
        "The measured numbers below are generated from this run; they are not mock results."
    )
    document.add_paragraph(
        "This is a baseline experiment, not a claim that the full RoadWatch platform is complete. "
        "GPS, duplicate merging, validated severity scoring, user roles and the repair dashboard "
        "remain planned work and are explicitly not represented as working features. YOLO-based "
        "object detection is an existing method; the defensible project contribution at this stage "
        "is a reproducible implementation and an honest evaluation, not a claim of a new detector."
    )
    document.add_heading("2. Project flow and scope", level=1)
    document.add_paragraph(
        "The proposed end-to-end product flow in the original slides is: citizen photo upload → "
        "pothole detection → GPS tagging → duplicate-report merging → explainable severity and "
        "priority → officer dashboard → repair status. The present implementation covers the "
        "data-preparation and detector-evaluation portion of that flow."
    )
    add_table(
        document,
        ["Stage", "Current evidence / status"],
        [
            ["Dataset and provenance", "Public pothole dataset downloaded; source, licence and split recorded."],
            ["Label conversion", "COCO bounding boxes mapped into one pothole class; generic 'object' labels excluded."],
            ["Detector baseline", f"YOLOv8n fine-tuned for {metrics['epochs']} epochs from COCO-pretrained weights."],
            ["Held-out evaluation", "Precision, recall, F1, mAP and image-level errors are measured on the official test split."],
            ["GPS, merge, severity, dashboard", "Not implemented in this baseline; no sample locations or mock predictions are presented as measured."],
        ],
    )

    document.add_heading("3. Real dataset and preparation", level=1)
    split_stats = metrics["dataset"]["splits"]
    add_table(
        document,
        ["Official split", "Images", "Mapped pothole boxes", "Ignored generic labels"],
        [
            [split, str(info["images"]), str(info["pothole_boxes"]), str(sum(info["ignored_annotations"].values()))]
            for split, info in split_stats.items()
        ],
    )
    document.add_paragraph(
        f"Dataset: {DATASET_ID}, published from the Roboflow Universe road-damage dataset. "
        f"The source dataset contains 819 images in total and is distributed under CC BY 4.0. "
        f"Source: {DATASET_URL}. The category list contains spelling variants ('pothole' and "
        "'potholes'), which are normalized to one class. The few generic 'object' annotations are "
        "excluded instead of being mislabeled as potholes. During data audit, "
        f"{metrics['dataset']['deduplication']['duplicate_groups_found']} exact duplicate image "
        f"groups were found, including {metrics['dataset']['deduplication']['cross_split_duplicate_groups']} "
        "groups crossing split boundaries. Duplicate copies were removed from lower-priority "
        "splits (test is retained before validation, then train) to avoid exact-image leakage; the "
        "unique held-out test images are retained. This check is byte-exact and does not detect "
        "visually similar crops or re-encoded copies."
    )
    document.add_paragraph(
        "Important limitation: this small dataset is suitable for a short proof-of-concept, not "
        "for claiming Indian-road or all-weather generalization. The larger RDD2022 dataset is a "
        "future evaluation target; its public Figshare archive is approximately 13 GB, so it is "
        "not silently substituted or partially represented here."
    )

    document.add_heading("4. Model and evaluation method", level=1)
    document.add_paragraph(
        f"Model: {metrics['model']}. Training used {metrics['epochs']} epochs, image size "
        f"{metrics['image_size']} × {metrics['image_size']}, random seed {metrics['seed']}, and "
        f"{metrics['device']}. The fixed official validation split is available for model "
        "selection; the official test split is used for the final reported evaluation."
    )
    document.add_paragraph(
        f"Precision, recall and F1 are computed from one-to-one greedy detection matches at a "
        f"fixed confidence ≥ {metrics['confidence_threshold']:.2f} and IoU ≥ "
        f"{metrics['iou_threshold']:.2f}; this avoids choosing the confidence threshold from the "
        "test results. mAP@0.5 and mAP@0.5:0.95 are the standard object-detection metrics from "
        "Ultralytics validation. Throughput is measured by running prediction on the "
        "test images on this CPU and includes per-image model inference overhead; it is not a "
        "GPU or production-server speed claim."
    )
    document.add_heading("5. Measured results", level=1)
    add_table(
        document,
        ["Metric", "Measured value"],
        [
            ["Precision", f"{metrics['precision']:.4f}"],
            ["Recall", f"{metrics['recall']:.4f}"],
            ["F1 (from precision and recall)", f"{metrics['f1']:.4f}"],
            ["mAP@0.5", f"{metrics['map50']:.4f}"],
            ["mAP@0.5:0.95", f"{metrics['map50_95']:.4f}"],
            ["Test-set inference throughput (CPU)", f"{metrics['test_inference_images_per_second']:.2f} images/second"],
        ],
    )
    counts = metrics["test_error_counts"]
    document.add_paragraph(
        f"At confidence ≥ {metrics['confidence_threshold']:.2f} and IoU ≥ "
        f"{metrics['iou_threshold']:.2f}, the image-level test pass counted "
        f"{counts['tp']} matched detections, {counts['fp']} false detections and "
        f"{counts['fn']} missed potholes across {metrics['test_images']} test images "
        f"({counts['ground_truth']} annotated potholes). These threshold-specific counts are "
        "provided for error analysis and are not interchangeable with mAP."
    )
    plot_path = ROOT / metrics["analysis_plot"]
    if plot_path.exists():
        document.add_picture(str(plot_path), width=Inches(6.6))
        caption = document.add_paragraph("Figure 1. Test metrics and largest per-image error counts.")
        caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
    document.add_heading("6. Analysis and interpretation", level=1)
    if metrics["recall"] < metrics["precision"]:
        interpretation = (
            "Recall is below precision in this run, indicating that the model misses more "
            "annotated potholes at its selected operating point than it falsely labels background."
        )
    elif metrics["precision"] < metrics["recall"]:
        interpretation = (
            "Precision is below recall in this run, indicating more false detections relative to "
            "the number of detected potholes at the selected operating point."
        )
    else:
        interpretation = (
            "Precision and recall are similar in this run; inspect the saved error examples to "
            "understand the remaining false detections and missed potholes."
        )
    document.add_paragraph(interpretation)
    document.add_paragraph(
        "The test split is small and the training run is deliberately short to establish a "
        "repeatable first baseline. Scores can vary with initialization, data composition and "
        "training duration. They should be reported as pilot results, not as evidence of "
        "deployment readiness. Review reports/test_error_analysis.csv and the generated images "
        "under reports/predictions/ before deciding which failure types to address."
    )
    document.add_heading("7. Two-week work sequence", level=1)
    add_table(
        document,
        ["Timebox", "Work", "Evidence / exit check"],
        [
            ["Days 1–2", "Confirm scope, dataset terms, Python environment and source citation.", "Reproducible environment and dataset source noted."],
            ["Days 3–4", "Download, audit categories, preserve splits and convert labels.", "Split counts and conversion checks pass."],
            ["Days 5–7", "Train a small pretrained YOLO baseline; record settings and checkpoint.", "Weights and training logs saved."],
            ["Days 8–9", "Evaluate held-out test data and inspect false positives/misses.", "Metrics, error CSV and example images saved."],
            ["Days 10–12", "Improve data quality/augmentation and run a controlled comparison.", "Experiment table; no changes to test split."],
            ["Days 13–14", "Validate scoring with human ratings, connect the prototype workflow, update report.", "Human agreement and demo evidence; known limitations listed."],
        ],
    )
    document.add_heading("8. Current limits and next work", level=1)
    for item in [
        "No GPS metadata or location capture is present in the selected dataset; mapping and geographic duplicate merging are not evaluated.",
        "No depth or calibrated human severity ratings are available; severity and repair priority are not claimed as validated.",
        "No login, database, image privacy pipeline, React/FastAPI application or repair-status workflow is implemented in this baseline.",
        "Next: inspect saved mistakes, expand the training comparison, then evaluate on a suitably licensed India-focused subset of RDD2022 before making regional claims.",
        "Only after the detector and data flow are stable, implement explainable severity with human ratings and the dashboard/status workflow.",
    ]:
        document.add_paragraph(item, style="List Bullet")
    document.add_heading("9. Reproduction", level=1)
    document.add_paragraph(
        "From the project directory, install requirements and run "
        "`python scripts/roadwatch_baseline.py run --epochs 5`. The script downloads the three "
        "public splits if needed, prepares COCO-to-YOLO labels, trains, evaluates, writes metrics "
        "and creates this report. Use the README.md in the project for the separate prepare, "
        "train and report commands."
    )
    document.add_paragraph(
        f"Environment recorded for this run: Python {__import__('sys').version.split()[0]}, "
        f"PyTorch {__import__('torch').__version__}, Ultralytics {__import__('ultralytics').__version__}; "
        f"training device: {metrics['device']}."
    )
    document.add_heading("References and licensing", level=1)
    document.add_paragraph(
        "Road damage dataset: abdulmohsen-fahad, Road Damage Dataset, Roboflow Universe "
        "(version 3, 2023), mirrored as manot/pothole-segmentation on Hugging Face; "
        f"{DATASET_URL}; CC BY 4.0. Attribution should be retained when redistributing data."
    )
    document.add_paragraph(
        "RDD2022 (future dataset candidate): Arya et al., RDD2022 – The multi-national Road "
        "Damage Dataset released through CRDDC'2022, Figshare, DOI: 10.6084/m9.figshare.21431547.v1, "
        "CC BY 4.0. Its source page reports 47,420 images and a 13.26 GB archive."
    )
    document.add_paragraph(
        "Model library: Ultralytics YOLO; verify the applicable AGPL-3.0/Enterprise licensing "
        "terms before distributing or deploying a product."
    )
    document.add_paragraph(
        "Prior method: Redmon, J., Divvala, S., Girshick, R., and Farhadi, A. (2016). "
        "You Only Look Once: Unified, Real-Time Object Detection. Proceedings of CVPR 2016. "
        "DOI: 10.1109/CVPR.2016.91."
    )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    output = REPORT_DIR / "RoadWatch_2_Week_Progress_Report.docx"
    document.save(output)
    print(f"Generated report: {output.relative_to(ROOT)}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproducible RoadWatch pothole detection baseline.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare", help="Download and convert the real dataset to YOLO format.")
    train_parser = commands.add_parser("train", help="Fine-tune YOLOv8n and evaluate the held-out test split.")
    train_parser.add_argument("--epochs", type=int, default=5)
    train_parser.add_argument("--imgsz", type=int, default=416)
    train_parser.add_argument("--model", choices=("yolov8n.pt", "yolov8s.pt"), default="yolov8n.pt")
    report_parser = commands.add_parser("report", help="Generate the Word report from the saved measured metrics.")
    report_parser.add_argument("--metrics", type=Path, default=REPORT_DIR / "metrics.json")
    validation_parser = commands.add_parser(
        "validate",
        help="Evaluate a trained checkpoint on validation data for model selection.",
    )
    validation_parser.add_argument("--model", choices=("yolov8n.pt", "yolov8s.pt"), required=True)
    evaluation_parser = commands.add_parser(
        "evaluate",
        help="Evaluate the latest saved checkpoint on the held-out test split without retraining.",
    )
    evaluation_parser.add_argument("--model", choices=("yolov8n.pt", "yolov8s.pt"), required=True)
    run_parser = commands.add_parser("run", help="Run the complete dataset-to-report workflow.")
    run_parser.add_argument("--epochs", type=int, default=5)
    run_parser.add_argument("--imgsz", type=int, default=416)
    run_parser.add_argument("--model", choices=("yolov8n.pt", "yolov8s.pt"), default="yolov8n.pt")
    args = parser.parse_args()

    if args.command in {"prepare", "run"}:
        ensure_downloads()
        prepare_dataset()
        if args.command == "prepare":
            return
    if args.command in {"train", "run"}:
        if args.epochs < 1:
            parser.error("--epochs must be at least 1")
        metrics = train_and_evaluate(args.epochs, args.imgsz, args.model)
        generate_report(metrics)
        return
    if args.command == "report":
        if not args.metrics.exists():
            raise FileNotFoundError(f"Metrics file not found: {args.metrics}. Run the train command first.")
        metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
        apply_fixed_threshold_metrics(metrics)
        args.metrics.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        generate_report(metrics)
    if args.command == "validate":
        evaluate_validation(args.model)
    if args.command == "evaluate":
        evaluate_saved_checkpoint(args.model)


if __name__ == "__main__":
    main()
