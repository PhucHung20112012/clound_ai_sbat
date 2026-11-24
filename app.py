from flask import Flask, request, jsonify
import os
import io
import numpy as np
import onnxruntime as ort
from PIL import Image

app = Flask(__name__)

# =========================
# CẤU HÌNH MODEL
# =========================

MODEL_PATH = "best.onnx"

CLASS_NAMES = [
    "Vô cơ",
    "Hữu cơ",
    "Rác khác"
]

CLASS_COMMANDS = ["A", "B", "C"]

CONFIDENCE_THRESHOLD = 0.4
AI_SIZE = 320
ZOOM_FACTOR = 1.6

# =========================
# KHỞI TẠO AI
# =========================

session = None

try:
    session = ort.InferenceSession(
        MODEL_PATH,
        providers=["CPUExecutionProvider"]
    )

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    print("AI MODEL LOADED")
    print("Input:", session.get_inputs()[0].shape)
    print("Output:", session.get_outputs()[0].shape)

except Exception as error:
    print("AI MODEL ERROR:", error)


# =========================
# TIỀN XỬ LÝ ẢNH
# =========================

def preprocess_image(image):

    image = image.convert("RGB")

    width, height = image.size

    # Cắt vùng trung tâm, tương tự zoom 1.6
    crop_size = min(width, height) / ZOOM_FACTOR

    left = (width - crop_size) / 2
    top = (height - crop_size) / 2
    right = left + crop_size
    bottom = top + crop_size

    image = image.crop((left, top, right, bottom))

    image = image.resize(
        (AI_SIZE, AI_SIZE),
        Image.Resampling.BILINEAR
    )

    image = np.asarray(image).astype(np.float32) / 255.0

    # HWC -> CHW
    image = np.transpose(image, (2, 0, 1))

    # Thêm batch dimension
    image = np.expand_dims(image, axis=0)

    return image


# =========================
# XỬ LÝ KẾT QUẢ YOLO
# =========================

def process_result(output):

    # Output có dạng [1, 7, 2100]
    predictions = output[0]

    if predictions.shape[0] != 7:
        raise ValueError(
            f"Output không đúng định dạng: {predictions.shape}"
        )

    num_boxes = predictions.shape[1]

    best_score = 0
    best_class = -1

    for i in range(num_boxes):

        cx = predictions[0, i]
        cy = predictions[1, i]
        w = predictions[2, i]
        h = predictions[3, i]

        class_scores = predictions[4:7, i]

        class_id = int(np.argmax(class_scores))
        score = float(class_scores[class_id])

        if w <= 20 or h <= 20:
            continue

        if cx - w / 2 < 0 or cy - h / 2 < 0:
            continue

        if score > best_score:
            best_score = score
            best_class = class_id

    if best_class == -1 or best_score < CONFIDENCE_THRESHOLD:
        return None

    return {
        "class_id": best_class,
        "label": CLASS_NAMES[best_class],
        "command": CLASS_COMMANDS[best_class],
        "confidence": round(best_score, 4)
    }


# =========================
# API
# =========================

@app.route("/", methods=["GET"])
def home():

    return jsonify({
        "status": "online",
        "service": "SmartBin Cloud AI"
    })


@app.route("/health", methods=["GET"])
def health():

    return jsonify({
        "status": "ok",
        "ai_loaded": session is not None
    })


@app.route("/predict", methods=["POST"])
def predict():

    if session is None:
        return jsonify({
            "error": "AI model not loaded"
        }), 500

    if "image" not in request.files:
        return jsonify({
            "error": "Missing image"
        }), 400

    try:

        file = request.files["image"]

        image = Image.open(io.BytesIO(file.read()))

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

        print("PREDICT ERROR:", error)

        return jsonify({
            "error": str(error)
        }), 500


if __name__ == "__main__":

    port = int(os.environ.get("PORT", 10000))

    app.run(
        host="0.0.0.0",
        port=port
    )
