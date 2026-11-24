
from flask import Flask, request, jsonify
import os

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "online",
        "service": "SmartBin Cloud AI"
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "Missing image"}), 400

    image = request.files["image"]

    if image.filename == "":
        return jsonify({"error": "Empty image"}), 400

    # AI model inference will be added in the next step.
    return jsonify({
        "status": "received",
        "message": "Image received successfully"
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
