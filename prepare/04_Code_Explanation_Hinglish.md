import argparse      # Command line se arguments lene ke liye
import csv           # CSV files likhne ke liye (results save karna)
import hashlib       # SHA-256 hashing ke liye (duplicates check)
import json          # JSON files read/write karne ke liye
import shutil        # Files copy karne ke liye
import time          # Speed measure karne ke liye
import urllib        # Internet se files download karne ke liye
import zipfile       # ZIP files kholne ke liye
from pathlib import Path  # File paths handle karne ke liye

import cv2           # OpenCV — images read/draw karne ke liye
import matplotlib    # Graphs banane ke liye
import numpy as np   # Math/Array operations ke liye
import yaml          # YAML config files read karne ke liye
from docx import Document  # Word document banane ke liye
from ultralytics import YOLO  # YOLOv8 model use karne ke liye
```

**Simple Matlab:** Yeh sab "tools" hain jo humne apne toolbox mein rakhe — jaise ek carpenter ke paas hammer, screwdriver, etc.

---

### 📦 PART 2 — Global Constants (Lines 29–44)

```python
ROOT = Path(__file__).resolve().parents[1]
# Matlab: Project ka root folder dhundo automatically
# __file__ = "scripts/roadwatch_baseline.py"
# .parents[1] = ek upar jaao = "New_Project/"

RAW_DIR = ROOT / "data" / "raw"           # ZIP files yahan save hongi
DATASET_DIR = ROOT / "data" / "processed" # Processed data yahan
REPORT_DIR = ROOT / "reports"             # Reports yahan
RUNS_DIR = ROOT / "runs"                  # Training results yahan

DATASET_ID = "manot/pothole-segmentation"  # Hugging Face dataset name
DATASET_URL = "https://huggingface.co/..."

DATASET_FILES = {
    "train": "https://...train.zip",   # Teen alag ZIP URLs
    "val":   "https://...valid.zip",
    "test":  "https://...test.zip",
}

POTHOLE_NAMES = {"pothole", "potholes"}  # Dono spelling accept karo
CONFIDENCE_THRESHOLD = 0.25              # 25% confident ho tabhi detect
IOU_THRESHOLD = 0.5                      # 50% box overlap ho tabhi sahi
```

**Simple Matlab:** Yeh sab "settings" hain — jaise phone ka Wi-Fi password, volume level etc. Ek jagah define karo, poore code mein use karo.

---

### 🔷 FUNCTION 1: `ensure_downloads()` (Lines 47–66)

```python
def ensure_downloads() -> None:
```

**Kya karta hai:** Internet se dataset ke ZIP files download karta hai.

```python
# Step 1: Folder banao agar nahi hai
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Step 2: Teen splits ke liye loop chalao (train, val, test)
for split, url in DATASET_FILES.items():
    archive = RAW_DIR / f"{split}.zip"
    
    # Agar ZIP pehle se hai toh skip karo
    if archive.exists() and zipfile.is_zipfile(archive):
        continue
    
    # Nahi hai toh download karo
    partial = archive.with_suffix(".zip.part")  # Pehle .part file mein save
    with urllib.request.urlopen(request) as response:
        shutil.copyfileobj(response, target)   # Download karo
    
    # Check karo — sahi ZIP file hai?
    if not zipfile.is_zipfile(partial):
        raise RuntimeError("Invalid ZIP!")
    
    partial.replace(archive)  # .part ko .zip mein rename karo
```

**Flow Diagram:**
```
Start
  ↓
Folder exist karta hai? → Nahi → Banao
  ↓
ZIP already downloaded? → Haan → Skip (time bachao)
  ↓
Nahi → Download karo
  ↓
Valid ZIP hai? → Nahi → Error throw karo
  ↓
.part → .zip rename karo
  ↓
Done ✅
```

**Professor ko batao:** "Humne robust download mechanism banaya — agar file pehle se hai toh dobara download nahi, aur corrupt files automatically detect ho jaati hain."

---

### 🔷 FUNCTION 2: `prepare_dataset()` (Lines 69–224) — **SABSE IMPORTANT**

```python
def prepare_dataset() -> dict[str, Any]:
```

**Kya karta hai:** Raw dataset ko YOLO format mein convert karta hai + clean karta hai.

#### Step A — ZIP kholna aur COCO annotations padhna

```python
for split in ("train", "val", "test"):
    with zipfile.ZipFile(archive) as source:
        # Annotation file dhundo "_annotations.coco.json"
        coco = json.loads(source.read(annotation_names[0]))
        
        # Categories: {1: "pothole", 2: "potholes", 3: "object"}
        categories = {cat["id"]: cat["name"] for cat in coco["categories"]}
```

**COCO format mein annotation kaisi hoti hai:**
```json
{
  "images": [{"id": 1, "file_name": "road1.jpg", "width": 640, "height": 480}],
  "annotations": [
    {"image_id": 1, "category_id": 1, "bbox": [100, 150, 80, 60]}
    // bbox = [x_topleft, y_topleft, width, height]
  ],
  "categories": [{"id": 1, "name": "pothole"}]
}
```

#### Step B — Filtering (Sirf Potholes Rakhna)

```python
for annotation in coco["annotations"]:
    category_name = categories.get(annotation["category_id"], "")
    
    if category_name not in POTHOLE_NAMES:  # {"pothole", "potholes"}
        ignored_labels[category_name] += 1   # Count karo kitne ignore kiye
        continue                              # Skip karo "object" etc.
    
    # Sirf pothole annotations rakhte hain
    annotations_by_image[annotation["image_id"]].append(annotation)
```

#### Step C — COCO → YOLO Format Convert

```python
# COCO format: [x_topleft, y_topleft, width, height] in pixels
x, y, box_width, box_height = map(float, annotation["bbox"])

# Step 1: Box ko image bounds ke andar clip karo
x1 = min(max(x, 0.0), width)      # 0 se image_width ke beech
y1 = min(max(y, 0.0), height)
x2 = min(max(x + box_width, 0.0), width)
y2 = min(max(y + box_height, 0.0), height)

# Step 2: YOLO format mein convert karo
# YOLO chahta hai: center_x, center_y, width, height (0 to 1 normalized)
center_x = ((x1 + x2) / 2.0) / width      # Center X / image width
center_y = ((y1 + y2) / 2.0) / height     # Center Y / image height
norm_width  = (x2 - x1) / width           # Box width / image width
norm_height = (y2 - y1) / height          # Box height / image height

# YOLO label file line: "class_id cx cy w h"
line = f"0 {center_x:.6f} {center_y:.6f} {norm_width:.6f} {norm_height:.6f}"
```

**Visual Conversion:**
```
COCO:                          YOLO:
[100, 150, 80, 60]            [0  0.281  0.437  0.125  0.125]
 x    y    w    h             cls  cx     cy      w      h
 (pixels)                     (normalized 0-1)

Conversion:
x1=100, y1=150, x2=180, y2=210
cx = (100+180)/2 / 640 = 140/640 = 0.219
cy = (150+210)/2 / 480 = 180/480 = 0.375
```

#### Step D — SHA-256 Duplicate Removal

```python
# Har image ka unique fingerprint (hash) banao
for split in ("train", "val", "test"):
    for image_path in images_folder.iterdir():
        digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
        images_by_hash[digest].append((split, image_path))

# Duplicate groups dhundo
for copies in images_by_hash.values():
    if len(copies) < 2:
        continue  # Unique hai, skip
    
    # Priority: test > val > train
    # Test ka copy rakhte hain, baaki hatate hain
    copies.sort(key=lambda item: (split_priority[item[0]], item[1].name))
    
    for split, image_path in copies[1:]:  # Pehle wala rakhte hain
        image_path.unlink()               # Image hatao
        label_path.unlink()               # Uska label bhi hatao
```

**Matlab:** Ek hi photo teen jagah thi — test, val, aur train mein. Test wali rakhte hain, baaki dono hatate hain. Warna model "already seen" data pe test hota — cheating!

#### Step E — YAML Config File Banana

```python
yaml_path.write_text(yaml.safe_dump({
    "path": DATASET_DIR.as_posix(),   # Dataset kahan hai
    "train": "train/images",           # Training images folder
    "val": "val/images",              # Validation images folder
    "test": "test/images",            # Test images folder
    "names": {0: "pothole"},          # Class 0 = "pothole"
}))
```

**Matlab:** YOLO ko ek config file chahiye — "data kahan hai, classes kya hain." Yeh YAML file woh info deti hai.

---

### 🔷 FUNCTION 3: `intersection_over_union()` (Lines 227–236)

```python
def intersection_over_union(box_a, box_b) -> float:
```

**Kya karta hai:** Do boxes ke beech IoU calculate karta hai.

```python
# Dono boxes ka overlap area dhundo
x1 = max(box_a[0], box_b[0])  # Left edge (zyada waala)
y1 = max(box_a[1], box_b[1])  # Top edge (zyada waala)
x2 = min(box_a[2], box_b[2])  # Right edge (kam waala)
y2 = min(box_a[3], box_b[3])  # Bottom edge (kam waala)

intersection = max(0, x2-x1) * max(0, y2-y1)  # Overlap area
area_a = (box_a[2]-box_a[0]) * (box_a[3]-box_a[1])  # Box A area
area_b = (box_b[2]-box_b[0]) * (box_b[3]-box_b[1])  # Box B area
union = area_a + area_b - intersection  # Total area (minus double-counted overlap)

return intersection / union  # IoU = overlap / total
```

**Visual:**
```
Box A: [100, 100, 200, 200]  → 100x100 = 10000 sq px
Box B: [150, 150, 250, 250]  → 100x100 = 10000 sq px

Overlap: x1=150, y1=150, x2=200, y2=200 → 50x50 = 2500 sq px
Union: 10000 + 10000 - 2500 = 17500 sq px

IoU = 2500/17500 = 0.143 (14.3% overlap)
```

---

### 🔷 FUNCTION 4: `evaluate_error_cases()` (Lines 262–338) — **KEY FUNCTION**

```python
def evaluate_error_cases(model, imgsz, artifact_slug):
```

**Kya karta hai:** Test images pe model run karta hai aur TP/FP/FN count karta hai.

```python
for image_path in test_images:
    # 1. Image padho
    image = cv2.imread(str(image_path))
    
    # 2. Ground truth labels padho (actual annotations)
    ground_truth = read_ground_truth(label_path, width, height)
    
    # 3. Model se prediction lao
    prediction = model.predict(
        source=image_path,
        conf=0.25,   # 25% confident ho tabhi show karo
        iou=0.7,     # Overlapping boxes merge karo
    )
    predicted_boxes = prediction.boxes.xyxy  # Predicted box coordinates
    confidences = prediction.boxes.conf      # Confidence scores
    
    # 4. Greedy matching — highest confidence predictions pehle match karo
    for index in np.argsort(confidences)[::-1]:  # High confidence pehle
        # Yeh predicted box kisi ground truth se match karta hai?
        best_iou = max(IoU(predicted_boxes[index], gt) for gt in ground_truth)
        
        if best_iou >= 0.50:    # 50% overlap?
            tp += 1              # ✅ True Positive
        else:
            fp += 1              # ❌ False Positive (galat alarm)
    
    fn = len(ground_truth) - tp  # Baaki sab miss ho gaye = False Negative
```

**Greedy Matching ka Matlab:**
```
Ground Truth:   [GT1] [GT2] [GT3]
Predictions:    [P1:90%] [P2:70%] [P3:30%]

Step 1: P1 (90%) → GT2 se match (IoU=0.8) → TP ✅ GT2 "used" mark karo
Step 2: P2 (70%) → GT1 se match (IoU=0.6) → TP ✅ GT1 "used" mark karo
Step 3: P3 (30%) → GT3 se IoU=0.2 → ❌ FP

FN = GT3 (koi match nahi mila)
```

---

### 🔷 FUNCTION 5: `apply_fixed_threshold_metrics()` (Lines 341–353)

```python
def apply_fixed_threshold_metrics(metrics):
    tp = counts["tp"]
    predictions = counts["predictions"]
    ground_truth = counts["ground_truth"]
    
    metrics["precision"] = tp / predictions  # 88/138 = 0.638
    metrics["recall"] = tp / ground_truth    # 88/203 = 0.433
    
    # F1 = 2*tp / (predictions + ground_truth)
    # = 2*88 / (138 + 203) = 176/341 = 0.516
    metrics["f1"] = 2 * tp / (predictions + ground_truth)
```

**Matlab:** TP, FP, FN counts se Precision, Recall, F1 calculate karo — simple math!

---

### 🔷 FUNCTION 6: `save_error_visualization()` (Lines 356–377)

```python
def save_error_visualization(image, ground_truth, predictions, ...):
```

**Kya karta hai:** Photo pe boxes draw karke save karta hai — green = actual, blue = predicted.

```python
canvas = image.copy()  # Original image copy

# Green boxes = actual potholes (ground truth)
for box in ground_truth:
    cv2.rectangle(canvas, (x1,y1), (x2,y2), (60,200,60), 2)
    # (60,200,60) = RGB Green color, 2 = line thickness

# Blue boxes = model ki predictions
for box, confidence in zip(predictions, confidences):
    cv2.rectangle(canvas, (x1,y1), (x2,y2), (30,30,230), 2)
    # (30,30,230) = RGB Blue color
    cv2.putText(canvas, f"{confidence:.2f}", ...)  # Confidence score likho

cv2.imwrite(output_path, canvas)  # Image save karo
```

**Output:** `reports/predictions/` folder mein images milti hain — green aur blue boxes ke saath.

---

### 🔷 FUNCTION 7: `train_and_evaluate()` (Lines 491–520) — **CORE TRAINING**

```python
def train_and_evaluate(epochs, imgsz, model_weights):
```

**Kya karta hai:** Model load karo, train karo, evaluate karo.

```python
# 1. Pretrained model load karo
model = YOLO(model_weights)  
# "yolov8n.pt" = download hoga agar nahi hai
# Already COCO dataset pe trained hai

# 2. Training shuru karo
model.train(
    data=str(dataset_yaml),   # Hamara pothole dataset
    epochs=5,                 # 5 baar poora dataset dekho
    imgsz=416,               # 416x416 pixels mein resize
    batch=8,                 # Ek baar mein 8 photos
    device="cpu",            # GPU nahi, CPU use karo
    workers=0,               # Background workers (0 = main thread)
    seed=42,                 # Random seed fix = reproducible
    deterministic=True,      # Same result hamesha same seed pe
    patience=0,              # Early stopping band karo
    plots=True,              # Training graphs save karo
)

# 3. Best checkpoint dhundo
best_weights = RUNS_DIR / run_name / "weights" / "best.pt"

# 4. Test pe evaluate karo
return evaluate_checkpoint(best_weights, ...)
```

**Training Flow:**
```
Pretrained YOLOv8n.pt (COCO dataset pe trained)
        ↓
Our pothole dataset pe fine-tune
        ↓ Epoch 1: Galat predictions, high loss
        ↓ Epoch 2: Thoda better
        ↓ Epoch 3: Aur better
        ↓ Epoch 4: ...
        ↓ Epoch 5: Done!
        ↓
Best checkpoint save → "best.pt"
        ↓
Test split pe evaluate
```

---

### 🔷 FUNCTION 8: `evaluate_checkpoint()` (Lines 423–488)

```python
def evaluate_checkpoint(best_weights, model_weights, epochs, imgsz):
```

**Kya karta hai:** Trained model ko test data pe chalata hai aur metrics save karta hai.

```python
# 1. Trained model load karo
best_model = YOLO(str(best_weights))

# 2. Ultralytics ka built-in test evaluation
test_metrics = best_model.val(
    data=str(dataset_yaml),
    split="test",           # Test split pe evaluate karo
    device="cpu",
    plots=True,             # PR curve etc. save karo
)
# Yeh mAP@0.5 aur mAP@0.5:0.95 deta hai

# 3. Hamara custom TP/FP/FN calculation
rows, counts, fps = evaluate_error_cases(best_model, imgsz, artifact_slug)

# 4. Speed calculate karo
fps = len(test_images) / total_prediction_time  # Images per second

# 5. Sab kuch metrics.json mein save karo
metrics = {
    "precision": 0.638,
    "recall": 0.433,
    "f1": 0.516,
    "map50": 0.537,
    "map50_95": 0.198,
    "test_inference_images_per_second": 10.41,
    ...
}
```

---

### 🔷 FUNCTION 9: `main()` (Lines 833–884) — **Entry Point**

```python
def main():
```

**Kya karta hai:** Command line se arguments le ke sahi function call karta hai.

```
Terminal mein kya type karo → Kya hota hai:

python roadwatch_baseline.py prepare
    → ensure_downloads() + prepare_dataset()
    → Dataset download aur clean hota hai

python roadwatch_baseline.py train --model yolov8n.pt --epochs 5
    → train_and_evaluate(5, 416, "yolov8n.pt")
    → Model train hota hai + test pe evaluate

python roadwatch_baseline.py validate --model yolov8n.pt
    → evaluate_validation("yolov8n.pt")
    → Validation mAP check karo (model selection ke liye)

python roadwatch_baseline.py evaluate --model yolov8n.pt
    → Sirf evaluate karo, train nahi

python roadwatch_baseline.py run --epochs 5
    → Sab kuch ek saath: prepare + train + report
```

---

## 🔷 FILE 2: `build_comparative_report.py`

---

### **Kya karta hai poori file:**

Dono models (YOLOv8n + YOLOv8s) ke saved results padh ke comparison graphs aur Word document banata hai.

---

### FUNCTION: `load_experiments()` (Lines 27–56)

```python
def load_experiments():
    # metrics.json = YOLOv8n results
    small = json.loads(BASELINE_PATH.read_text())
    
    # metrics_yolov8s.json = YOLOv8s results  
    nano = json.loads(COMPARISON_PATH.read_text())
    
    # Fair comparison check — same settings honi chahiye
    comparable_fields = ("epochs", "image_size", "device", "seed", ...)
    mismatches = {field: ... for field if small[field] != nano[field]}
    
    if mismatches:
        raise ValueError("Settings match nahi karte — unfair comparison!")
```

**Matlab:** Pehle check karo ki dono experiments same settings pe the — warna comparison fair nahi hogi.

---

### FUNCTION: `create_comparison_charts()` (Lines 59–200ish)

**4 Charts banata hai:**

#### Chart 1 — Test Metrics Bar Chart (Figure 2)
```python
metrics = [("precision","Precision"), ("recall","Recall"), 
           ("f1","F1"), ("map50","mAP@0.5"), ("map50_95","mAP@0.5:0.95")]

# YOLOv8n aur YOLOv8s ke side-by-side bars
axis.bar(x, yolov8n_values, label="YOLOv8n", color="#3977a8")
axis.bar(x, yolov8s_values, label="YOLOv8s", color="#e39b34")
```

#### Chart 2 — Speed vs mAP Scatter (Figure 3)
```python
# X-axis = images per second (speed)
# Y-axis = mAP@0.5 (accuracy)
axis.scatter(experiment["test_inference_images_per_second"], 
             experiment["map50"])
# Top-right = Fast aur Accurate = Best!
```

#### Chart 3 — Dataset Split (Figure 1)
```python
# Train/Val/Test mein kitni images aur boxes hain
axis.bar(["Train","Val","Test"], [515, 141, 79])  # Images
```

#### Chart 4 — TP/FP/FN Stacked Bar (Figure 4)
```python
# Green = TP (sahi detection)
# Red = FP (false alarm)
# Orange = FN (miss)
axis.bar(["YOLOv8n","YOLOv8s"], tp_values, color="green", label="TP")
axis.bar(["YOLOv8n","YOLOv8s"], fp_values, bottom=tp, color="red", label="FP")
axis.bar(["YOLOv8n","YOLOv8s"], fn_values, bottom=..., color="orange", label="FN")
```

---

## 🎯 CODE FLOW — Poora Experiment Ek Nazar Mein

```
Step 1: python roadwatch_baseline.py prepare
├── ensure_downloads()     → Internet se 3 ZIP files download
└── prepare_dataset()      → COCO → YOLO convert + clean + deduplicate

Step 2: python roadwatch_baseline.py train --model yolov8n.pt
├── train_and_evaluate()   → YOLOv8n ko pothole data pe train karo
│   ├── YOLO(model).train() → 5 epochs CPU pe
│   ├── evaluate_checkpoint() → Test pe evaluate
│   │   ├── model.val()       → mAP calculate (Ultralytics built-in)
│   │   └── evaluate_error_cases() → TP/FP/FN manually count
│   └── metrics.json save   → {precision, recall, f1, map50, ...}
└── generate_report()      → Word document banao

Step 3: python roadwatch_baseline.py train --model yolov8s.pt
└── Same as Step 2, metrics_yolov8s.json mein save

Step 4: python roadwatch_baseline.py validate --model yolov8n.pt
└── evaluate_validation()  → Val split pe mAP check (model selection)

Step 5: python scripts/build_comparative_report.py
├── load_experiments()     → Dono JSON files padho
├── create_comparison_charts() → 4 graphs banao
└── build_report_document()   → Comparative Word doc banao
```

---

## 📋 Professor Ko Explain Kaise Karo

### Agar poochhe: "Yeh code kya karta hai?"

> *"Sir, hamare paas 2 scripts hain. Pehla `roadwatch_baseline.py` poora pipeline handle karta hai — dataset download, COCO se YOLO format conversion, SHA-256 se duplicate removal, YOLOv8 training, aur TP/FP/FN evaluation. Dusra `build_comparative_report.py` dono trained models ke results uthata hai, charts banata hai, aur comparative analysis report generate karta hai."*

### Agar poochhe: "COCO to YOLO conversion kyun?"

> *"Sir, downloaded dataset COCO format mein tha — absolute pixel coordinates mein. YOLOv8 normalized coordinates chahiye (0-1 range). Humne manually box clipping aur normalization implement ki — `center_x = ((x1+x2)/2) / image_width`."*

### Agar poochhe: "Duplicate removal kyun kiya?"

> *"Sir, same photo train aur test dono mein thi — isse model 'already seen' data pe test hota jo cheating hai. SHA-256 hashing se exact duplicates identify kiye aur lower-priority splits se hataye — test ka copy preserve kiya."*

### Agar poochhe: "Greedy matching kya hai?"

> *"Sir, evaluation mein hum highest confidence prediction ko pehle match karte hain available ground truth box se (agar IoU ≥ 0.5 ho). Ek ground truth box sirf ek prediction se match ho sakta — yahi greedy matching hai. Yeh standard PASCAL VOC evaluation protocol follow karta hai."*

---

## 🚨 Key Lines Jo Definitely Poochhi Jaayengi

| Line | Code | Matlab |
|------|------|--------|
| 500 | `model = YOLO(model_weights)` | Pretrained model load karo |
| 501-516 | `model.train(...)` | Training shuru karo |
| 508 | `seed=42` | Reproducibility ke liye fixed random |
| 509 | `deterministic=True` | Hamesha same result |
| 134-140 | `center_x = ...` | COCO → YOLO convert |
| 168 | `hashlib.sha256(...)` | Duplicate detection |
| 298-309 | `np.argsort(confidences)[::-1]` | Greedy matching |
| 350-353 | `precision = tp/predictions` | Metric calculation |

---

*Code samajh aa gaya? Koi bhi function aur explain karna ho toh batao! 🚀*
