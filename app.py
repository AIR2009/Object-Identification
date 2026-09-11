import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import colorsys
from PIL import Image

st.set_page_config(page_title="Object Detector", page_icon="🔎", layout="centered")


def class_color(name):
    h = 0
    for ch in name:
        h = (h * 31 + ord(ch)) % 360
    r, g, b = colorsys.hls_to_rgb(h / 360, 0.62, 0.85)
    return (int(r * 255), int(g * 255), int(b * 255))


def draw_corner(img, px, py, dx, dy, color, thickness):
    cv2.line(img, (px, py), (px + dx, py), color, thickness)
    cv2.line(img, (px, py), (px, py + dy), color, thickness)


def detect(detector, image_bgr, min_confidence=0.3):
    h, w = image_bgr.shape[:2]
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    # This model expects float32 pixels in [0, 1]
    tensor = tf.image.convert_image_dtype(rgb, tf.float32)[tf.newaxis, ...]
    detector_fn = detector.signatures["default"]
    result = {k: v.numpy() for k, v in detector_fn(tensor).items()}

    boxes    = result["detection_boxes"]
    entities = result["detection_class_entities"]
    scores   = result["detection_scores"]

    detections = []
    for box, entity, score in zip(boxes, entities, scores):
        if score < min_confidence:
            continue
        ymin, xmin, ymax, xmax = box
        detections.append({
            "class": entity.decode("ascii"),
            "score": float(score),
            "bbox":  (int(xmin*w), int(ymin*h),
                      int((xmax-xmin)*w), int((ymax-ymin)*h)),
        })
    return sorted(detections, key=lambda d: d["score"], reverse=True)


def render(image_bgr, detections):
    out = image_bgr.copy()
    h, w = out.shape[:2]
    base       = max(2, w // 320)
    font_scale = max(0.4, w / 1200)
    thick_thin  = base
    thick_thick = max(2, int(base * 1.8))

    for det in detections:
        x, y, bw, bh = det["bbox"]
        color = class_color(det["class"])
        label = f"{det['class']}  {det['score']*100:.0f}%"

        overlay = out.copy()
        cv2.rectangle(overlay, (x, y), (x+bw, y+bh), color, thick_thin)
        cv2.addWeighted(overlay, 0.45, out, 0.55, 0, out)

        c = int(min(bw, bh) * 0.22)
        draw_corner(out, x,      y,      +c, +c, color, thick_thick)
        draw_corner(out, x+bw,   y,      -c, +c, color, thick_thick)
        draw_corner(out, x,      y+bh,   +c, -c, color, thick_thick)
        draw_corner(out, x+bw,   y+bh,   -c, -c, color, thick_thick)

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, font_scale, 1)
        pad = max(4, int(th * 0.4))
        label_y = y - th - pad*2 if y - th - pad*2 >= 0 else y
        cv2.rectangle(out, (x, label_y), (x + tw + pad*2, label_y + th + pad), color, cv2.FILLED)
        cv2.putText(out, label, (x + pad, label_y + th),
                    cv2.FONT_HERSHEY_DUPLEX, font_scale, (13,15,13), 1, cv2.LINE_AA)
    return out


@st.cache_resource(show_spinner=False)
def load_detector():
    return hub.load("https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1")


st.title("🔎 Multi-Object Detector")
st.caption(
    "Upload a photo and detect 600+ everyday objects using a lightweight "
    "SSD MobileNet v2 model (TensorFlow Hub, Open Images V4)."
)

min_confidence = st.slider(
    "Minimum confidence", min_value=0.05, max_value=0.90, value=0.30, step=0.05,
    help="Lower catches more objects but with more false positives.",
)

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    with st.spinner("Loading model — first run can take a minute…"):
        detector = load_detector()

    with st.spinner("Detecting objects…"):
        dets = detect(detector, img, min_confidence=min_confidence)

    annotated = render(img, dets)
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    st.image(annotated_rgb, width='stretch')

    if not dets:
        st.info("No objects found. Try lowering the confidence slider.")
    else:
        counts = {}
        for d in dets:
            counts[d["class"]] = counts.get(d["class"], 0) + 1
        st.subheader("Detected")
        for cls, n in sorted(counts.items(), key=lambda x: -x[1]):
            st.write(f"**{cls}** × {n}")
else:
    st.info("👆 Upload a photo to get started.")
