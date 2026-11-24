
from flask import Flask, request, jsonify, send_file
import os
import io
import logging
import threading

import numpy as np
import onnxruntime as ort
from PIL import Image

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# =====================================
# 1. MODEL CONFIG
# =====================================

MODEL_PATH = "best.onnx"

CLASS_NAMES = [
    "Vô cơ",
    "Hữu cơ",
    "Rác khác"
]

CLASS_COMMANDS = ["A", "B", "C"]

AI_SIZE = 320
CONFIDENCE_THRESHOLD = 0.40
NMS_THRESHOLD = 0.45

# Giữ nguyên tiền xử lý hiện tại
ZOOM_FACTOR = 1.6

# Ảnh mới nhất lưu trong bộ nhớ
latest_image = None
image_lock = threading.Lock()

# =====================================
# 2. LOAD MODEL
# =====================================

session = None
input_name = None
output_name = None

try:
    session = ort.InferenceSession(
        MODEL_PATH,
        providers=["CPUExecutionProvider"]
    )

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    logging.info("AI MODEL LOADED")
    logging.info("INPUT: %s", session.get_inputs()[0].shape)
    logging.info("OUTPUT: %s", session.get_outputs()[0].shape)

except Exception:
    logging.exception("AI MODEL ERROR")

# =====================================
# 3. PREPROCESS IMAGE
# =====================================

def preprocess_image(image):

    image = image.convert("RGB")

    width, height = image.size

    crop_size = min(width, height) / ZOOM_FACTOR

    left = (width - crop_size) / 2
    top = (height - crop_size) / 2

    right = left + crop_size
    bottom = top + crop_size

    image = image.crop(
        (left, top, right, bottom)
    )

    image = image.resize(
        (AI_SIZE, AI_SIZE),
        Image.Resampling.BILINEAR
    )

    image = np.asarray(
        image,
        dtype=np.float32
    ) / 255.0

    image = np.transpose(image, (2, 0, 1))

    image = np.expand_dims(image, axis=0)

    return image

# =====================================
# 4. NMS
# =====================================

def calculate_iou(box1, box2):

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])

    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])

    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def non_max_suppression(detections):

    detections = sorted(
        detections,
        key=lambda x: x["score"],
        reverse=True
    )

    kept = []

    for detection in detections:

        duplicate = False

        for selected in kept:

            if detection["class_id"] != selected["class_id"]:
                continue

            if calculate_iou(
                detection["box"],
                selected["box"]
            ) > NMS_THRESHOLD:

                duplicate = True
                break

        if not duplicate:
            kept.append(detection)

    return kept

# =====================================
# 5. DECODE YOLO OUTPUT
# =====================================

def process_result(output):

    predictions = np.asarray(output)

    logging.info("RAW OUTPUT SHAPE: %s", predictions.shape)

    if predictions.ndim != 3 or predictions.shape[0] != 1:
        raise ValueError(
            f"Unexpected output shape: {predictions.shape}"
        )

    predictions = predictions[0]

    # Hỗ trợ cả [7, 2100] và [2100, 7]
    if predictions.shape[0] == 7:
        predictions = predictions.T

    elif predictions.shape[1] != 7:
        raise ValueError(
            f"Cannot decode YOLO output: {predictions.shape}"
        )

    detections = []

    for row in predictions:

        cx, cy, w, h = map(float, row[:4])

        scores = row[4:7]

        if not np.all(np.isfinite(row)):
            continue

        class_id = int(np.argmax(scores))
        confidence = float(scores[class_id])

        if confidence < CONFIDENCE_THRESHOLD:
            continue

        if w <= 0 or h <= 0:
            continue

        x1 = max(0.0, cx - w / 2)
        y1 = max(0.0, cy - h / 2)

        x2 = min(float(AI_SIZE), cx + w / 2)
        y2 = min(float(AI_SIZE), cy + h / 2)

        if x2 <= x1 or y2 <= y1:
            continue

        detections.append({
            "class_id": class_id,
            "score": confidence,
            "box": [x1, y1, x2, y2]
        })

    logging.info(
        "DETECTIONS BEFORE NMS: %d",
        len(detections)
    )

    detections = non_max_suppression(detections)

    logging.info(
        "DETECTIONS AFTER NMS: %d",
        len(detections)
    )

    if not detections:
        return None

    best = detections[0]

    class_id = best["class_id"]

    return {
        "class_id": class_id,
        "label": CLASS_NAMES[class_id],
        "command": CLASS_COMMANDS[class_id],
        "confidence": round(best["score"], 4)
    }

# =====================================
# 6. HOME
# =====================================

@app.route("/", methods=["GET"])
def home():

    return jsonify({
        "status": "online",
        "service": "SmartBin Cloud AI"
    })

# =====================================
# 7. HEALTH
# =====================================

@app.route("/health", methods=["GET"])
def health():

    return jsonify({
        "status": "ok",
        "ai_loaded": session is not None
    })

# =====================================
# 8. VIEW LATEST IMAGE
# =====================================

@app.route("/latest-image", methods=["GET"])
def get_latest_image():

    with image_lock:
        if latest_image is None:
            return jsonify({
                "error": "No image received yet"
            }), 404

        image_data = latest_image

    return send_file(
        io.BytesIO(image_data),
        mimetype="image/jpeg",
        download_name="latest.jpg"
    )

# =====================================
# 9. PREDICT
# =====================================

@app.route("/predict", methods=["POST"])
def predict():

    global latest_image

    if session is None:
        return jsonify({
            "status": "error",
            "error": "AI model not loaded"
        }), 500

    if "image" not in request.files:
        return jsonify({
            "status": "error",
            "error": "Missing image"
        }), 400

    try:

        file = request.files["image"]

        image_bytes = file.read()

        if not image_bytes:
            return jsonify({
                "status": "error",
                "error": "Empty image"
            }), 400

        # Lưu ảnh mới nhất trong RAM
        with image_lock:
            latest_image = image_bytes

        image = Image.open(
            io.BytesIO(image_bytes)
        )

        image.load()

        logging.info(
            "IMAGE RECEIVED: %s",
            image.size
        )

        tensor = preprocess_image(image)

        outputs = session.run(
            [output_name],
            {input_name: tensor}
        )

        result = process_result(outputs[0])

        if result is None:

            return jsonify({
                "status": "no_detection",
                "message": "Không nhận diện được rác"
            })

        return jsonify({
            "status": "success",
            **result
        })

    except Exception as error:

        logging.exception("PREDICT ERROR")

        return jsonify({
            "status": "error",
            "error": str(error)
        }), 500

# =====================================
# 10. RUN SERVER
# =====================================

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 10000))

    app.run(
        host="0.0.0.0",
        port=port
    )
