"""RoadWatch | Road inspection and detector performance.

Place this file in the existing project root beside runs/, reports/ and data/.
Run: python -m streamlit run app.py
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageOps

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

ROOT = Path(__file__).resolve().parent
MODELS = {
    "YOLOv8n": ROOT / "runs/roadwatch-yolov8n-20261114-004822/weights/best.pt",
    "YOLOv8s": ROOT / "runs/roadwatch-yolov8s-20261114-114035/weights/best.pt",
}
REPORTS = ROOT / "reports"
SAMPLE_DIR = ROOT / "data/processed/test/images"
NMS_IOU = 0.70  # Same as the documented fixed-threshold test error analysis.
MAX_PIXELS = 30_000_000

# Documented benchmark values, used only when the local JSON is unavailable.
# Never used for live detections; the performance page labels this provenance.
SNAPSHOT = {
    "YOLOv8n": {"validation": {"map50": 0.6268, "map50_95": 0.3507},
                "map50": 0.5367, "map50_95": 0.2814, "precision": 0.6377,
                "recall": 88/203, "f1": 176/341,
                "test_inference_images_per_second": 10.41,
                "test_error_counts": {"tp": 88, "fp": 50, "fn": 115}},
    "YOLOv8s": {"validation": {"map50": 0.5837, "map50_95": 0.2946},
                "map50": 0.4945, "map50_95": 0.2492, "precision": 0.5444,
                "recall": 92/203, "f1": 184/372,
                "test_inference_images_per_second": 5.75,
                "test_error_counts": {"tp": 92, "fp": 77, "fn": 111}},
}

st.set_page_config(page_title="RoadWatch | Road inspection", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
    .block-container {max-width: 1420px; padding-top: 2.0rem; padding-bottom: 2rem;}
    h1, h2, h3 {letter-spacing: -0.025em;}
    h1 {font-size: 2rem !important; font-weight: 650 !important;}
    h2 {font-size: 1.34rem !important; font-weight: 650 !important;}
    h3 {font-size: 1.08rem !important; font-weight: 600 !important;}
    [data-testid="stSidebar"] {border-right: 1px solid #dce3e6;}
    [data-testid="stMetric"] {border: 1px solid #dce3e6; border-radius: 8px; padding: 13px 15px;}
    [data-testid="stMetricValue"] {font-size: 1.65rem; font-variant-numeric: tabular-nums;}
    [data-testid="stMetricLabel"] {font-size: .84rem;}
    [data-testid="stFileUploaderDropzone"] {border: 1px dashed #a5b8bb; border-radius: 8px;}
    .stButton button, .stDownloadButton button {border-radius: 6px; font-weight: 550;}
    .rw-brand {font-size: 1.65rem; font-weight: 750; letter-spacing: -.06em; line-height: 1.15;}
    .rw-muted {font-size: .84rem; color: #64777c; line-height: 1.5;}
    .rw-empty {border: 1px dashed #a5b8bb; border-radius: 8px; padding: 3rem 1.5rem; text-align: center; color: #64777c; min-height: 170px;}
    .rw-empty strong {display: block; font-size: 1.05rem; margin-bottom: .5rem;}
    .rw-rule {height: 1px; background: #dce3e6; margin: 1.1rem 0 1.3rem;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_model(path: str, modified_ns: int):
    """Cache the exact trained checkpoint; do not download pretrained weights."""
    if YOLO is None:
        raise RuntimeError("Ultralytics is not installed. Install your project requirements first.")
    return YOLO(path)


@st.cache_data(show_spinner=False)
def read_metrics(path: str, modified_ns: int) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Expected a JSON object")
    return data


def benchmark_metrics():
    records, sources = {}, {}
    for name, filename in [("YOLOv8n", "metrics.json"), ("YOLOv8s", "metrics_yolov8s.json")]:
        path = REPORTS / filename
        try:
            records[name] = read_metrics(str(path), path.stat().st_mtime_ns)
            sources[name] = f"Local file: {filename}"
        except (OSError, ValueError, TypeError):
            records[name] = SNAPSHOT[name]
            sources[name] = "Saved benchmark summary (local JSON unavailable)"
    return records, sources


def number(data, *keys):
    for key in keys:
        if not isinstance(data, dict):
            return None
        data = data.get(key)
    try:
        val = float(data)
        return val if np.isfinite(val) else None
    except (TypeError, ValueError):
        return None


def fmt(value, digits=3):
    return "—" if value is None else f"{value:.{digits}f}"


def decode_image(raw: bytes):
    with Image.open(io.BytesIO(raw)) as image:
        if image.width * image.height > MAX_PIXELS:
            raise ValueError("Choose a photo under 30 megapixels.")
        return ImageOps.exif_transpose(image).convert("RGB")


def run_detection(model, image, conf, imgsz):
    rgb = np.asarray(image.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    start = time.perf_counter()
    result = model.predict(source=bgr, conf=conf, iou=NMS_IOU, imgsz=imgsz,
                           device="cpu", verbose=False)[0]
    elapsed = time.perf_counter() - start
    canvas = bgr.copy()
    h, w = canvas.shape[:2]
    boxes = []
    if result.boxes is not None:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1, x2 = [int(np.clip(round(x), 0, w)) for x in (x1, x2)]
            y1, y2 = [int(np.clip(round(y), 0, h)) for y in (y1, y2)]
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append(dict(x1=x1, y1=y1, x2=x2, y2=y2, conf=float(box.conf[0])))
    boxes.sort(key=lambda b: b["conf"], reverse=True)
    # One annotation colour: confidence is not a physical damage severity rating.
    colour = (105, 111, 10)  # BGR teal
    scale = max(.42, min(1.5, min(w, h) / 650))
    stroke = max(2, round(scale * 3))
    for i, b in enumerate(boxes, 1):
        x1, y1, x2, y2 = (b[k] for k in ("x1", "y1", "x2", "y2"))
        cv2.rectangle(canvas, (x1, y1), (min(x2,w-1), min(y2,h-1)), colour, stroke)
        label = f"Pothole {i}  {b['conf']:.0%}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, stroke-1)
        label_h = th + baseline + 10
        left = max(0, min(x1, w-tw-12))
        top = max(0, min(y1-label_h, h-label_h))
        cv2.rectangle(canvas, (left, top), (min(w-1,left+tw+10), min(h-1,top+label_h)), colour, -1)
        cv2.putText(canvas, label, (left+5, top+th+4), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, (255,255,255), stroke-1, cv2.LINE_AA)
    return Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)), boxes, elapsed


def png_bytes(image):
    out = io.BytesIO()
    image.save(out, format="PNG")
    return out.getvalue()


def detection_rows(boxes, image_size):
    w, h = image_size
    return [{"ID": i, "Confidence (%)": round(b["conf"]*100, 1),
             "Width (px)": b["x2"]-b["x1"], "Height (px)": b["y2"]-b["y1"],
             "Image area (%)": round(100*(b["x2"]-b["x1"])*(b["y2"]-b["y1"])/(w*h), 2),
             "x1": b["x1"], "y1": b["y1"], "x2": b["x2"], "y2": b["y2"]}
            for i, b in enumerate(boxes, 1)]


def csv_bytes(rows, result):
    out = io.StringIO()
    fields = ["model", "confidence_threshold", "input_size", "nms_iou", "ID", "Confidence (%)",
              "Width (px)", "Height (px)", "Image area (%)", "x1", "y1", "x2", "y2"]
    writer = csv.DictWriter(out, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow({"model": result["model"], "confidence_threshold": result["conf"],
                         "input_size": result["imgsz"], "nms_iou": NMS_IOU, **row})
    return out.getvalue().encode("utf-8")


def empty(title, description):
    # Only called with fixed application strings, never with uploaded content.
    st.markdown(f'<div class="rw-empty"><strong>{title}</strong>{description}</div>', unsafe_allow_html=True)


def page_heading(title, subtitle):
    st.title(title)
    st.caption(subtitle)
    st.markdown('<div class="rw-rule"></div>', unsafe_allow_html=True)


with st.sidebar:
    st.markdown('<div class="rw-brand">RoadWatch<span style="color:#176b69">.</span></div>', unsafe_allow_html=True)
    st.caption("Mumbai · Road inspection")
    st.markdown("---")
    page = st.radio("Navigation", ["Inspection", "Model performance"], label_visibility="collapsed")
    if page == "Inspection":
        st.markdown("---")
        st.markdown("**Detection settings**")
        model_name = st.selectbox("Detector", list(MODELS), help="Select a trained model. Nano is the default baseline.")
        conf = st.slider("Confidence threshold", .10, .90, .25, .05,
                         help="Minimum accepted prediction score. Lower values may increase false alarms.")
        imgsz = st.select_slider("Input size (px)", options=[320, 416, 640], value=416,
                                help="Larger inputs preserve more detail but may take longer to process.")
        st.caption("Processing device: CPU")
        if not MODELS[model_name].is_file():
            st.warning("Model checkpoint unavailable.")
        elif YOLO is None:
            st.warning("Ultralytics is not installed.")
        else:
            st.caption("Model checkpoint available")


if page == "Inspection":
    page_heading("Road inspection", "Detect potholes in a road photograph and review the predicted regions.")
    raw, source_name, source_img = None, None, None
    source_col, action_col = st.columns([3, 1], gap="large")
    with source_col:
        mode = st.radio("Image source", ["Upload image", "Test sample"], horizontal=True, label_visibility="collapsed")
        try:
            if mode == "Upload image":
                uploaded = st.file_uploader("Road photograph", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed")
                if uploaded is not None:
                    raw, source_name = uploaded.getvalue(), uploaded.name
            else:
                samples = sorted(p for p in SAMPLE_DIR.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}) if SAMPLE_DIR.is_dir() else []
                if samples:
                    chosen = st.selectbox("Test image", samples, format_func=lambda p: f"{samples.index(p)+1:02d} · {p.name}")
                    raw, source_name = chosen.read_bytes(), chosen.name
                    st.caption("Public test dataset; location unverified.")
                else:
                    st.info("No local test samples found. Upload an image instead.")
            if raw is not None:
                source_img = decode_image(raw)
        except (OSError, ValueError, Image.DecompressionBombError) as exc:
            st.error(f"Could not open image: {exc}")
    path = MODELS[model_name]
    ready = source_img is not None and path.is_file() and YOLO is not None
    with action_col:
        st.markdown("**Inspection**")
        st.caption(f"{model_name} · {imgsz} px · {conf:.2f} threshold")
        run = st.button("Run detection", type="primary", use_container_width=True, disabled=not ready)
        if source_img is not None and not ready:
            st.caption("Check model availability in the sidebar.")
        if st.button("Clear result", use_container_width=True, disabled="rw_result" not in st.session_state):
            st.session_state.pop("rw_result", None)

    signature = (hashlib.sha256(raw).hexdigest(), model_name, path.stat().st_mtime_ns if path.is_file() else None, conf, imgsz) if source_img is not None else None
    if run and ready:
        st.session_state.pop("rw_result", None)
        try:
            with st.spinner("Processing image…"):
                model = load_model(str(path), path.stat().st_mtime_ns)
                annotated, boxes, elapsed = run_detection(model, source_img, conf, imgsz)
                st.session_state["rw_result"] = {"signature": signature, "image": annotated, "boxes": boxes,
                    "elapsed": elapsed, "model": model_name, "conf": conf, "imgsz": imgsz}
        except Exception as exc:
            st.error(f"Detection failed: {exc}")
    stored = st.session_state.get("rw_result")
    result = stored if stored and signature is not None and stored["signature"] == signature else None

    st.markdown("---")
    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("Original image")
        if source_img is not None:
            st.image(source_img, use_container_width=True)
            st.caption(f"{source_name} · {source_img.width:,} × {source_img.height:,} px")
        else:
            empty("No image selected", "Upload a photograph or select a test sample.")
    with right:
        st.subheader("Detection output")
        if result:
            st.image(result["image"], use_container_width=True)
            st.caption(f"{result['model']} · confidence ≥ {result['conf']:.2f} · {result['imgsz']} px input")
        elif stored and source_img is not None:
            empty("New inspection required", "The image or settings have changed. Run detection again.")
        else:
            empty("Awaiting detection", "The annotated image will appear here.")

    if result:
        st.markdown("---")
        st.subheader("Results")
        boxes = result["boxes"]
        average = sum(b["conf"] for b in boxes) / len(boxes) if boxes else 0
        cols = st.columns(3)
        cols[0].metric("Pothole predictions", len(boxes))
        cols[1].metric("Mean confidence", f"{average:.1%}" if boxes else "—")
        cols[2].metric("Prediction time", f"{result['elapsed']*1000:,.0f} ms")
        rows = detection_rows(boxes, result["image"].size)
        if rows:
            st.dataframe([{k:v for k,v in row.items() if k in {"ID", "Confidence (%)", "Width (px)", "Height (px)"}} for row in rows], use_container_width=True, hide_index=True)
            st.caption("Box dimensions are in image pixels; confidence is not physical damage severity.")
        else:
            st.info("No predictions at this threshold. This does not confirm the absence of potholes.")
        a, b = st.columns(2)
        a.download_button("Download annotated image", png_bytes(result["image"]), "roadwatch_inspection.png", "image/png", use_container_width=True)
        b.download_button("Export detections (CSV)", csv_bytes(rows, result), "roadwatch_detections.csv", "text/csv", use_container_width=True)
    st.caption("Detection prototype · Mumbai field performance has not yet been validated.")

else:
    page_heading("Model performance", "Recorded results from the public-dataset benchmark, not live image predictions.")
    records, sources = benchmark_metrics()
    n, s = records["YOLOv8n"], records["YOLOv8s"]
    st.markdown("**Baseline selection: YOLOv8n** — higher validation mAP@0.5 in the documented pilot.")
    specs = [("Validation mAP@0.5", ("validation", "map50")), ("Test mAP@0.5", ("map50",)),
             ("Test mAP@0.5:0.95", ("map50_95",)), ("Test precision", ("precision",)),
             ("Test recall", ("recall",)), ("Test F1", ("f1",)), ("CPU images/sec", ("test_inference_images_per_second",))]
    table = []
    for label, keys in specs:
        vn, vs = number(n, *keys), number(s, *keys)
        digits = 2 if label == "CPU images/sec" else 3
        table.append({"Metric": label, "YOLOv8n": fmt(vn, digits), "YOLOv8s": fmt(vs, digits)})
    st.subheader("Detector comparison")
    st.dataframe(table, hide_index=True, use_container_width=True)
    st.caption("Test precision, recall and F1 at confidence 0.25 and match IoU ≥ 0.50. mAP summarises the precision–recall curve.")
    st.subheader("Detection errors")
    errors = [{"Model": name, "Matched (TP)": fmt(number(data,"test_error_counts","tp"),0),
               "False alarms (FP)": fmt(number(data,"test_error_counts","fp"),0),
               "Missed (FN)": fmt(number(data,"test_error_counts","fn"),0)} for name, data in records.items()]
    st.dataframe(errors, hide_index=True, use_container_width=True)
    st.caption("The documented pilot used 79 test images with 203 labelled potholes. Recall remains a limitation.")
    charts = [("comparison_metrics.png", "Test metrics"), ("comparison_error_counts.png", "Detection errors")]
    available = [(REPORTS / filename, title) for filename, title in charts if (REPORTS / filename).is_file()]
    if available:
        cols = st.columns(len(available))
        for col, (chart, title) in zip(cols, available):
            with col:
                st.image(str(chart), use_container_width=True)
                st.caption(f"{title} · saved benchmark figure")
    with st.expander("Benchmark details"):
        st.write("5 epochs · 416 px input · batch size 8 · CPU · seed 42. Public dataset: 735 images (507 train, 149 validation, 79 test).")
        for name, source in sources.items():
            st.caption(f"{name}: {source}")
