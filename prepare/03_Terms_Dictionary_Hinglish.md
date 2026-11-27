# 📖 Technical Terms Dictionary — Hinglish Mein Samjho

> Har technical term ko simple Hinglish mein samjhaya gaya hai.
> Report mein aaye **har ek word** ka matlab yahan milega.

---

## 🔤 A se Z — Alphabetical Order

---

### 🅰️ A

**Annotation / Bounding Box Annotation**
> Matlab: Photo mein manually gadda/pothole ke around box kheenchna. Jaise teacher red pen se circle karta hai answer mein — waise hi ek insaan ne har pothole ke around ek rectangle banaya.

**arXiv**
> Matlab: Ek free website (arxiv.org) jahan scientists apne research papers publish karte hain — peer review se pehle bhi. World ka sabse bada free paper store.

---

### 🅱️ B

**Baseline**
> Matlab: Starting point / reference point. "Humara baseline yeh hai" matlab "is level se shuruat ki — aage improve karenge." YOLOv8n hamara baseline model hai.

**Batch / Batch Size**
> Matlab: Ek baar mein kitni photos model dekhta hai training mein. Humara batch=8 matlab ek baar mein 8 photos process kiye. Zyada batch = zyada memory chahiye.

**Bounding Box**
> Matlab: Photo mein object ke around kheechi hui rectangle (dikha rahi hai — "yahan hai pothole!"). Ek box = ek pothole ka location.

---

### 🅲 C

**CC BY 4.0 (Creative Commons Attribution 4.0)**
> Matlab: Ek open license. "Tum freely use karo — bas credit do original creator ko." Jaise Wikipedia content free hai but source batao.

**Checkpoint**
> Matlab: Training ke dauran saved model ka state. Jaise game mein save point. `best.pt` = sabse best checkpoint.

**Classification vs Detection**
> - **Classification**: "Yeh photo mein pothole hai ya nahi?" (Haan/Nahi)
> - **Detection**: "Photo mein pothole KAI hain, aur KAHAN hain?" (Location + Count)

**COCO Format**
> Matlab: Ek popular tarika annotations save karne ka (JSON file mein). Microsoft ne banaya. Humne ise YOLO format mein convert kiya.

**Comparative Analysis**
> Matlab: Do ya zyada cheezein compare karna — systematically. Jaise phone review mein feature-by-feature comparison.

**Confidence / Confidence Score**
> Matlab: Model kitna "sure" hai ki usne sahi detect kiya. 0.25 confidence threshold = "25% se zyada sure ho tabhi batao."

**CPU (Central Processing Unit)**
> Matlab: Normal computer ka processor. AI training ke liye slow hai. Hamara experiment CPU pe chala (GPU nahi tha).

---

### 🅳 D

**Dataset**
> Matlab: Data ka collection. Yahan: 735 sadak ki photos + har photo mein pothole ke locations marked.

**Deep Learning**
> Matlab: AI ka ek type jisme artificial "neurons" (brain jaisi structure) use hote hain. YOLO deep learning model hai.

**Deployment**
> Matlab: Model ko real world mein use karna — app ya server pe laga dena. Hamara model abhi deployment ready nahi.

**Detector / Object Detector**
> Matlab: Woh AI model jo images mein objects dhundh ke unke location batata hai.

**Deterministic**
> Matlab: Same input → same output hamesha. `seed=42` set kiya toh results reproducible hain.

---

### 🅴 E

**Epoch**
> Matlab: Ek baar poora training dataset dekha. 5 epochs = model ne 5 baar poori dataset dekhi.
> Analogy: Jaise ek student ne same book 5 baar padhi — zyada padhe toh zyada seekhega.

**Evaluation / Evaluate**
> Matlab: Model kitna accha hai yeh measure karna — test data pe.

---

### 🅵 F

**F1 Score**
> Matlab: Precision aur Recall ka balanced average (harmonic mean).
> Formula: `F1 = 2 × (Precision × Recall) / (Precision + Recall)`
> Matlab: Ek number jo batata hai overall kitna accha model hai.

**False Negative (FN) / Miss**
> Matlab: Actual pothole tha, but model ne detect nahi kiya. 
> Analogy: Doctor ne cancer miss kar diya — dangerous!
> YOLOv8n ne 115 potholes miss kiye.

**False Positive (FP) / False Alarm**
> Matlab: Pothole tha nahi, but model ne bola "yahan hai pothole."
> Analogy: Alarm baja but chor nahi tha — false alarm.
> YOLOv8n ke 50 FP, YOLOv8s ke 77 FP.

**Fine-tuning**
> Matlab: Pehle se trained (pretrained) model ko apne specific data pe aur train karna.
> Analogy: Ek experienced doctor ko specialty mein aur training dena.

---

### 🅶 G

**GPU (Graphics Processing Unit)**
> Matlab: Powerful processor jo AI training mein CPU se 10-100x fast hai. Gaming cards bhi GPU use karte hain. Hamara experiment GPU ke bina tha.

**Ground Truth**
> Matlab: "Sach" — actually correctly marked data. Jab hum compare karte hain model ke output ko, toh ground truth se compare karte hain.

---

### 🅷 H

**Harmonic Mean**
> Matlab: Ek khaas average jo unequal values handle karta hai. F1 score mein use hota hai.
> Simple: `2/(1/P + 1/R)` — do numbers ka harmonic mean.

**Held-out Test Set**
> Matlab: Woh data jo training ke dauran bilkul use nahi kiya — sirf akhir mein final test ke liye. "Exam ke questions pehle nahi dekhne diye" wala rule.

**Hugging Face**
> Matlab: Ek popular website (huggingface.co) jahan AI models aur datasets freely available hain. Hamara dataset yahan se aaya.

---

### 🅸 I

**Image Size (imgsz)**
> Matlab: Training se pehle photos ko is size mein resize kiya. Hamara `imgsz=416` = 416×416 pixels.

**IoU — Intersection over Union**
> Matlab: Do boxes kitna overlap karte hain, yeh measure karna.
> Formula: `IoU = Overlap Area / Total Combined Area`
> `IoU = 0.5` = 50% overlap ho tabhi sahi detection mana.
> 
> ```
> [Ground Truth Box]
>    [Predicted Box  ]
>  |---overlap---|
>  IoU = overlap/total
> ```

---

### 🅹 J

**JSON / JSONL**
> Matlab: Ek file format data save karne ka. COCO annotations JSON format mein the.

---

### 🅺 K

*(Koi specific term nahi is report mein)*

---

### 🅻 L

**Label / Class Label**
> Matlab: Annotation ka naam — "yeh object kya hai?" Hamari ek hi class thi: `pothole`.

**License**
> Matlab: Rules ki jo batate hain ki data/model ka use kaise kar sakte ho. CC BY 4.0 = free use with credit.

**Loss Function**
> Matlab: Training mein model kitna galat hai yeh measure karne ka formula. Training ka goal: loss kam karo.

---

### 🅼 M

**mAP — Mean Average Precision**
> Matlab: Object detection ka main accuracy metric.
> Simple: "Model overall kitna precise hai, saare confidence levels pe average karke."
> 
> - **mAP@0.5**: 50% IoU threshold pe — relatively easy standard
> - **mAP@0.5:0.95**: 50% se 95% tak ke strict thresholds ka average — harder standard

**Model**
> Matlab: Trained AI — ek file (`.pt` extension) jisme learning stored hai. Jaise ek trained brain.

**Model Selection**
> Matlab: Available models mein se best choose karna — ek rule ke basis pe.

---

### 🅽 N

**Nano (YOLOv8n)**
> Matlab: YOLOv8 ka sabse chhota/lightest version. `n` = nano. Fast hai, memory kam lagti hai.

**NeurIPS**
> Matlab: "Neural Information Processing Systems" — AI ka ek top conference jahan Faster R-CNN paper publish hua.

**Normalize / Normalization**
> Matlab: Values ko 0 se 1 ke beech scale karna. YOLO format mein box coordinates normalized hain (image size ke relative).

---

### 🅾 O

**Object Detection**
> Matlab: Image mein objects dhundh ke unka location batana.
> Output: "Yahan pothole hai [x, y, width, height] ke box mein, 85% confidence ke saath."

**One-class / Single-class**
> Matlab: Sirf ek type ka object detect karna. Hamara model sirf `pothole` detect karta hai.

**One-stage Detector**
> Matlab: Ek hi baar mein boxes aur classes predict karta hai. Fast hai.
> Examples: YOLO, SSD

**Output Artifact**
> Matlab: Experiment ke baad save hone wali files — CSV results, charts, model files.

---

### 🅿 P

**Patience**
> Matlab: Training mein kitne epochs tak wait karo improvement ke bina. `patience=0` = early stopping disabled.

**Pilot Results**
> Matlab: Preliminary/initial results — poora experiment nahi, sirf shuruat. Final results nahi hain.

**Pothole**
> Matlab: Sadak mein gadda 🕳️ — yahi detect karna tha.

**Precision**
> Matlab: Jab model bolta hai "pothole hai" — kitni baar sach mein hota hai?
> Formula: `Precision = TP / (TP + FP)`
> High precision = kam false alarms
> YOLOv8n precision: **0.638** (63.8% sahi)

**Pretrained Model**
> Matlab: Pehle se kisi bade dataset pe trained model. Hum ise apne data pe aur train karte hain (fine-tune).
> Analogy: Ek graduate doctor jo already books padh chuka — ab usse specialty training deni hai.

**Protocol**
> Matlab: Rules aur steps ka set jo experiment mein follow kiye. "Matched protocol" = dono models ke liye same rules.

---

### 🆀 Q

*(Koi specific term nahi)*

---

### 🆁 R

**R-CNN / Faster R-CNN**
> Matlab: "Region-based Convolutional Neural Network." Pehle regions propose karo (kahan object ho sakta hai), phir classify karo. 2-stage process — accurate but slow.

**Random Seed**
> Matlab: Ek number jo randomness ko fix karta hai. `seed=42` matlab har baar same random choices. Reproducibility ke liye zaroori.

**Recall**
> Matlab: Total actual potholes mein se kitne detect hue?
> Formula: `Recall = TP / (TP + FN)`
> High recall = kam misses
> YOLOv8n recall: **0.433** (43.3% potholes pakde gaye — 56.7% miss!)

**Reproducibility**
> Matlab: Koi bhi yahi experiment dobara kare toh same results mile. Scientific honesty ke liye zaroori.

**RDD2022**
> Matlab: "Road Damage Detection 2022" — ek multinational road damage dataset. India, Japan, Norway, USA ki roads include. Future work mein use karna hai.

**Region Proposal Network (RPN)**
> Matlab: Faster R-CNN ka ek part jo "yahan object ho sakta hai" locations suggest karta hai.

**Roboflow**
> Matlab: Ek platform jahan annotated computer vision datasets available hain. Hamara dataset originally yahan se tha.

---

### 🆂 S

**Seed (Random Seed)**
> Dekho "Random Seed" upar.

**SHA-256 Hashing**
> Matlab: Ek mathematical function jo har file ka unique "fingerprint" banata hai. 
> Use: Check karo ki same file do jagah toh nahi — exact duplicates pakdo.
> SHA-256 = "Secure Hash Algorithm 256-bit"

**Small (YOLOv8s)**
> Matlab: YOLOv8 ka thoda bada version. `s` = small. YOLOv8n se bada, zyada parameters.

**Split**
> Matlab: Dataset ko teen hisso mein baantna — Train, Validation, Test.

**SSD — Single Shot MultiBox Detector**
> Matlab: Ek one-stage detector. "Single shot" = ek hi pass mein boxes predict karta hai. Fast hai.

---

### 🆃 T

**Test Set / Test Split**
> Matlab: Woh data jo training mein bilkul use nahi kiya — sirf final evaluation ke liye. 
> YOLOv8n test pe: 79 images, 203 boxes.

**Threshold (Confidence Threshold)**
> Matlab: Minimum confidence level. 0.25 = model 25% se zyada sure ho tabhi detect batao.

**Training**
> Matlab: Model ko data dikhana baar baar taaki woh seekhe. Jaise bachcha baar baar practice karta hai.

**Training Set**
> Matlab: Woh data jisse model sikhta hai. Hamara: 515 images.

**True Positive (TP)**
> Matlab: Actual pothole tha, model ne detect kiya — sahi! 
> YOLOv8n: 88 TP.

**Two-stage Detector**
> Matlab: Pehle regions propose karo, phir classify karo. Accurate but slow.
> Example: Faster R-CNN

---

### 🆄 U

**Ultralytics**
> Matlab: Woh company/team jisne YOLOv8 banaya. Unka `ultralytics` Python package use kiya.

---

### 🆅 V

**Validation Set / Val Split**
> Matlab: Training ke dauran progress check karne ke liye data. Model yeh data training mein nahi dekhta but performance check hoti rahti hai.
> YOLOv8n val mAP@0.5: **0.627** ← is number se winner chuna.

---

### 🆆 W

**Weight / Model Weights**
> Matlab: Trained model ke internal numbers jo learning store karte hain. `.pt` file mein save hote hain.
> Pretrained weights: `yolov8n.pt`, `yolov8s.pt`

---

### 🆇 X

*(Koi specific term nahi)*

---

### 🆈 Y

**YOLO — You Only Look Once**
> Matlab: Ek popular object detection algorithm. "You Only Look Once" = ek hi baar image dekh ke sab kuch predict karo.
> 
> Versions:
> - v1 (2016): Original — J. Redmon ne banaya
> - v8 (2023): Latest stable — Ultralytics ne banaya
> 
> Variants in this report:
> - **YOLOv8n**: Nano (chhota, fast) — **Winner ✅**
> - **YOLOv8s**: Small (thoda bada, slow)

**YOLO Format**
> Matlab: Annotation save karne ka YOLO-specific tarika.
> Format: `class_id center_x center_y width height` (sab normalized 0-1 ke beech)

---

### 🆉 Z

*(Koi specific term nahi)*

---

## 📊 Metric Quick Reference Card

| Metric | Full Name | Simple Matlab | Good = High/Low? |
|--------|-----------|---------------|------------------|
| **Precision** | Precision | "Jab bola pothole, kitna sach tha?" | High ↑ |
| **Recall** | Recall / Sensitivity | "Total potholes mein se kitne pakde?" | High ↑ |
| **F1** | F1 Score | Precision + Recall ka balance | High ↑ |
| **mAP@0.5** | Mean Average Precision at IoU 0.5 | Overall detection quality | High ↑ |
| **mAP@0.5:0.95** | Stricter mAP | Harder standard | High ↑ |
| **FP** | False Positive | Galat alarm (pothole nahi tha) | Low ↓ |
| **FN** | False Negative | Miss hua pothole | Low ↓ |
| **TP** | True Positive | Sahi detect hua pothole | High ↑ |
| **IoU** | Intersection over Union | Box overlap percentage | — |

---

## 🔄 Experiment Flow — Step by Step

```
1. Dataset Download (Hugging Face se)
         ↓
2. COCO → YOLO Format Convert
         ↓
3. SHA-256 Duplicate Check & Remove
         ↓
4. Train/Val/Test Split (515/141/79 images)
         ↓
5. YOLOv8n Train karo (5 epochs, CPU)
         ↓
6. YOLOv8s Train karo (5 epochs, CPU)
         ↓
7. Validation Results Compare karo
         ↓
8. Best Model Select karo (Validation mAP se) → YOLOv8n ✅
         ↓
9. Test Set pe Final Evaluation
         ↓
10. Graphs + Tables + Report Generate
```

---

*Dictionary complete hai! Koi bhi term samajh na aaye toh yahan dekho. 📚*

