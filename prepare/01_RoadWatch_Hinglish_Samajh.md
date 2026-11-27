# 🛣️ RoadWatch Report — Hinglish Mein Poori Samajh

**By: Neeraj Sonar | M.Sc. Computer Science | KC College, Mumbai**
**Guide: Dr. Geeta N Brijwani**

---

## 📌 Yeh Report Kya Hai? (Ek Line Mein)

> AI use karke **sadak ke gaddhe (potholes) automatically detect karna** — aur do alag-alag AI models compare karna ki kaun better hai.

---

## 🎯 Objective — Humne Kya Karna Tha?

Humara goal tha:
1. **Ek dataset lao** (sadak ki photos jisme gaddhe hain)
2. **Purane research papers se methods lo** (YOLO, Faster R-CNN, SSD)
3. **Un methods ka code run karo** aur unhe train karo
4. **Compare karo** — kaun sa model zyada accurate hai
5. **Best method select karo** — logical reason ke saath

> 🏗️ **RoadWatch kya hai?** — Ek future app idea jisme public gaddha photo kheench ke report karein, AI detect kare, location GPS se save ho, aur officer ko priority ke saath repair order mile. Is report mein sirf **detector part** banaya gaya hai — baaki future work hai.

---

## 📂 Dataset — Humne Kya Data Liya?

| Cheez | Detail |
|-------|--------|
| **Source** | Hugging Face: `manot/pothole-segmentation` (originally Roboflow se) |
| **Total Images** | **735 photos** sadak ki |
| **Total Boxes** | **1,906 gaddhe** marked (bounding boxes) |
| **License** | CC BY 4.0 (free use with credit) |

### Dataset 3 Hisso Mein Baanta Gaya:

```
Train Set  → 515 images, 1,328 boxes  (model sikhta hai)
Val Set    →  141 images,   375 boxes  (model check hota hai training ke dauran)
Test Set   →   79 images,   203 boxes  (final test, model ne kabhi dekha nahi)
```

### Cleaning Kaise Hua?

- **"pothole"** aur **"potholes"** — dono same mana gaya (ek class)
- Generic **"object"** labels hata diye gaye (3 train + 2 val + 1 test = 6 total)
- **SHA-256 hashing**: Ek hi photo agar 3 sets mein thi, toh duplicate remove kiya
  - 40 aise groups mile jisme same photo alag sets mein thi
  - Rule: Test pehle rakha → fir Val → fir Train → duplicates hataye
- COCO format se **YOLO format** mein convert kiya (box coordinates normalize kiye)
k
---

## 🤖 Methods — Kaunse AI Models Compare Hue?

### 4 Methods Literature Mein Review Hue:

| Method | Kya Hai | Status |
|--------|---------|--------|
| **M1 — YOLOv8n (Nano)** | YOLO ka sabse chhota fast version | ✅ **Trained & Tested** |
| **M2 — YOLOv8s (Small)** | YOLO ka thoda bada version | ✅ **Trained & Tested** |
| **M3 — Faster R-CNN** | 2-stage detector, slow but accurate | 📖 Sirf review, train nahi kiya |
| **M4 — SSD** | Single-shot fast detector | 📖 Sirf review, train nahi kiya |

> ⚠️ M3 aur M4 ke liye koi numbers nahi hain — sirf literature mein bataya gaya hai.

---

## ⚙️ Training Settings — Experiment Kaise Chala?

Dono models ke liye **bilkul same settings**:

| Setting | Value | Matlab |
|---------|-------|--------|
| `epochs = 5` | 5 baar poora dataset dekha | Bahut chhota run hai (pilot) |
| `imgsz = 416` | Photo 416×416 pixels mein resize | Standard size |
| `batch = 8` | Ek baar mein 8 photos | Memory ke hisaab se |
| `device = "cpu"` | GPU nahi, sirf CPU | Slow but accessible |
| `seed = 42` | Random seed fix | Reproducibility ke liye |
| `confidence = 0.25` | 25% se zyada sure ho tabhi detect karo | Fixed threshold |
| `IoU = 0.50` | 50% overlap ho tabhi sahi detection mana | Matching rule |

---

## 📊 Results — Kya Mila?

### Validation Results (Training ke dauran check):

| Model | Val mAP@0.5 | Val mAP@0.5:0.95 |
|-------|-------------|-----------------|
| **YOLOv8n ✅** | **0.627** | **0.351** |
| YOLOv8s | 0.584 | 0.295 |

### Test Results (Final — model ne kabhi nahi dekha tha):

| Model | Precision | Recall | F1 | mAP@0.5 | Speed (img/sec) |
|-------|-----------|--------|----|---------|-----------------|
| **YOLOv8n ✅** | **0.638** | 0.433 | **0.516** | **0.537** | **10.41** |
| YOLOv8s | 0.544 | **0.453** | 0.495 | 0.494 | 5.75 |

### Detection Counts (79 test images, 203 actual gaddhe):

| Model | TP (Sahi Pakde) | FP (Galat Alarm) | FN (Miss Hue) |
|-------|-----------------|------------------|---------------|
| YOLOv8n | 88 | 50 | 115 |
| YOLOv8s | 92 | 77 | 111 |

---

## 📈 Graphs Kya Dikhate Hain?

### Figure 1 — Dataset Split Bar Chart
- **Kya dikhata hai**: Train/Val/Test mein kitni images aur boxes hain
- **Finding**: Train set mein sabse zyada data hai, test set bahut chhota (sirf 79 images)

### Figure 2 — Test Metrics Bar Chart (Precision, Recall, F1, mAP)
- **Kya dikhata hai**: Dono models ka side-by-side comparison
- **Finding**: YOLOv8n precision aur mAP mein aage, YOLOv8s thoda better recall

### Figure 3 — Speed vs mAP@0.5 Scatter Plot
- **Kya dikhata hai**: X-axis = speed (images/sec), Y-axis = test mAP
- **Finding**: YOLOv8n ek hi saath **faster bhi hai aur accurate bhi** — ideal choice!

### Figure 4 — TP / FP / FN Bar Chart
- **Kya dikhata hai**: Kite gaddhe sahi pakde, kitne galat alarm, kitne miss
- **Finding**: Dono models **half se zyada gaddhe miss** kar rahe hain — abhi deployment ke liye ready nahi

---

## 🏆 Best Method Selection

> **Winner: YOLOv8n (M1)**

**Rule pehle declare kiya tha**: Jo validation mAP@0.5 mein highest ho, woh best.

### Kyun YOLOv8n Jeeta?

1. ✅ **Validation mAP@0.5**: 0.627 > 0.584 (YOLOv8s se better)
2. ✅ **Test mAP@0.5**: 0.537 > 0.494
3. ✅ **Precision**: 0.638 > 0.544 (kam galat alarms — 50 vs 77 FP)
4. ✅ **Speed**: 10.41 img/sec vs 5.75 — **almost 2x faster on CPU**
5. ✅ Test results se winner nahi chuna — validation se chuna (unbiased!)

> 💡 **YOLOv8s kyun nahi jeeta?** Large model ko 5 epochs bahut kam hain — usse aur training chahiye. Small dataset pe chhota model better fit karta hai.

---

## ⚠️ Limitations (Kamiyaan)

1. **Sirf 5 epochs** — bahut short training, scores improve hoge zyada epochs se
2. **CPU pe train kiya** — GPU hota toh zyada experiments possible hote
3. **Sirf 2 models compare kiye** — SSD aur Faster R-CNN actually run nahi kiye
4. **Small test set** (79 images) — results reliable nahi ho sakte completely
5. **Indian roads ke liye test nahi kiya** — dataset India-specific nahi hai
6. **Night/Rain/Day labels nahi** — weather conditions ka pata nahi
7. **Duplicate detection partial** — exact copies pakde, visually similar nahi

---

## 🔮 Future Work (Aage Kya Karna Hai)

1. 🔧 **Confidence threshold tune karo** (0.25 se change karke better recall paao)
2. 🖥️ **GPU pe zyada epochs train karo** — real results ke liye
3. 🆚 **SSD aur Faster R-CNN actually compare karo** same protocol se
4. 🇮🇳 **RDD2022 dataset use karo** — Indian roads ke liye
5. 🌦️ **Day/Night/Rain conditions tag karo** dataset mein
6. 📱 **Full RoadWatch app banao**:
   - GPS tagging
   - Duplicate report merge karna
   - Severity score (AI + human validation)
   - Officer dashboard

---

## 🧾 Final Summary (Ek Nazar Mein)

| Poochha Gaya | Jawab |
|-------------|-------|
| Kitna data use kiya? | 735 images, 1906 gaddhe |
| Kitne models compare kiye? | 2 (YOLOv8n vs YOLOv8s), 2 sirf review |
| Training kitni baar? | 5 epochs, CPU pe |
| Best model kaun? | **YOLOv8n** |
| Best ka mAP@0.5? | **0.537 (test), 0.627 (val)** |
| Deployment ready? | ❌ Nahi — baseline pilot hai |
| Next step kya? | GPU training + zyada epochs + SSD/Faster R-CNN |

---

> 📝 **Report ki Quality kaisi hai?**
> Report **academically solid** hai — proper methodology, pre-declared selection rule, honest limitations, aur no inflated claims. Ek genuine pilot study ka structure hai. Well done! 👍

