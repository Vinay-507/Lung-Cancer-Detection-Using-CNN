import os
# MUST set these before importing TensorFlow / Keras to reduce noisy logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import numpy as np
import traceback
import cv2

from tensorflow.keras.models import load_model

# ---------------------------
# Config
# ---------------------------
UPLOAD_FOLDER = "static/uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
MODEL_PATH = os.path.join("model", "lung_cancer_cnn.h5")
IMG_SIZE = 224

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.secret_key = "replace_this_with_random_key"

# ---------------------------
# Helpers
# ---------------------------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image_from_path(image_path, target_size=(IMG_SIZE, IMG_SIZE)):
    """
    Normalizes uploaded CT image to match training dataset style:
    - convert to grayscale
    - histogram equalization
    - resize to 224x224
    - scale 0-255 → 0-1
    - convert grayscale → 3-channel
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Error reading input image.")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = cv2.resize(gray, target_size)
    gray = gray.astype("float32") / 255.0

    img_3ch = np.stack([gray, gray, gray], axis=-1)
    img_3ch = np.expand_dims(img_3ch, axis=0)

    return img_3ch


# ---------------------------
# Load Model
# ---------------------------
model = None
try:
    model = load_model(MODEL_PATH, compile=False)
    print(f"[INFO] Model loaded: {MODEL_PATH}")
except Exception as e:
    print(f"[ERROR] Could not load model: {e}")


# ---------------------------
# Routes
# ---------------------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    global model
    try:
        if model is None:
            return jsonify({"error": "Model not loaded on server"}), 500

        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]

        if file.filename == "":
            return jsonify({"error": "Empty filename"}), 400

        if not allowed_file(file.filename):
            return jsonify({"error": "Invalid file type. Allowed: png, jpg, jpeg"}), 400

        filename = secure_filename(file.filename)
        saved_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(saved_path)

        # Preprocess
        img_arr = preprocess_image_from_path(saved_path)

        # Predict
        raw_preds = model.predict(img_arr)
        preds = np.array(raw_preds).flatten()  # ensure (4,)

        # ---------------------------
        # Correct class indices
        # 0 = adenocarcinoma
        # 1 = large cell carcinoma
        # 2 = normal
        # 3 = squamous cell carcinoma
        # ---------------------------
        normal_prob = float(preds[2])
        cancer_prob = float(preds[0] + preds[1] + preds[3])

        if normal_prob >= cancer_prob:
            final_label = "No Cancer"
            confidence = round(normal_prob * 100, 2)
        else:
            final_label = "Cancer"
            confidence = round(cancer_prob * 100, 2)

        try:
            os.remove(saved_path)
        except:
            pass

        return jsonify({
            "prediction": final_label,
            "confidence": confidence
        }), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "error": "Server error during prediction",
            "details": str(e)
        }), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=True, use_reloader=False)