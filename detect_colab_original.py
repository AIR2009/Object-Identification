!pip install -q tensorflow tensorflow-hub opencv-python-headless

import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import colorsys
from google.colab import files
from IPython.display import display, Image as IPImage
from PIL import Image
import io

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

def show(image_bgr, max_width=900):
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    if pil.width > max_width:
        ratio = max_width / pil.width
        pil = pil.resize((max_width, int(pil.height * ratio)), Image.LANCZOS)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=92)
    display(IPImage(data=buf.getvalue()))

print("Loading model (Faster R-CNN + Open Images V4, 600 classes)…")
detector = hub.load("https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1")

# This model tends to run lower/noisier confidence scores than COCO SSD models,
# so the default threshold is lower. Raise it if you're seeing false positives.
MIN_CONFIDENCE = 0.30

print("Upload an image:")
uploaded = files.upload()

for filename, data in uploaded.items():
    arr  = np.frombuffer(data, np.uint8)
    img  = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    print(f"\n Detecting in {filename}…")
    dets = detect(detector, img, min_confidence=MIN_CONFIDENCE)

    if not dets:
        print("No objects found. Try lowering MIN_CONFIDENCE.")
    else:
        counts = {}
        for d in dets:
            counts[d["class"]] = counts.get(d["class"], 0) + 1
        print("\n── Detected ──────────────────────")
        for cls, n in sorted(counts.items(), key=lambda x: -x[1]):
            print(f"  {cls:<22} ×{n}")
        print("──────────────────────────────────")

    annotated = render(img, dets)
    show(annotated)
