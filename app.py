from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import os

app = Flask(__name__)
CORS(app)

print("🌱 Loading AI Plant Doctor model...")

MODEL_PATH = "plant_disease_model_fixed.keras"
model = tf.keras.models.load_model(MODEL_PATH)

print("✅ AI model loaded successfully!")

# Warm up TensorFlow so the first real prediction is not slow
dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
model.predict(dummy, verbose=0)

print("✅ AI model warmed up successfully!")

class_names = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]


def get_severity(disease):
    if "healthy" in disease.lower():
        return "Healthy ✅"
    return "Needs Attention ⚠️"


def get_recommendation(disease):
    if "healthy" in disease.lower():
        return (
            "The plant appears healthy. Continue regular watering, "
            "proper nutrition, and monitor the plant regularly."
        )

    return (
        "The plant may be affected by this disease. Remove severely "
        "affected leaves, maintain good air circulation, avoid excessive "
        "moisture on leaves, and consider an appropriate plant disease treatment."
    )


def clean_name(name):
    return (
        name.replace("___", " — ")
            .replace("_", " ")
            .strip()
    )


@app.route("/")
def home():
    return """
<!DOCTYPE html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AI Plant Doctor 🌿</title>

<style>
* {
    box-sizing: border-box;
    font-family: Arial, sans-serif;
}

body {
    margin: 0;
    min-height: 100vh;
    background: linear-gradient(135deg, #0b3d2e, #145a32, #1e8449);
    color: white;
    display: flex;
    justify-content: center;
    align-items: center;
    padding: 25px;
}

.container {
    width: 100%;
    max-width: 800px;
    background: rgba(255,255,255,0.10);
    backdrop-filter: blur(15px);
    border-radius: 25px;
    padding: 35px;
    box-shadow: 0 20px 60px rgba(0,0,0,0.35);
}

h1 {
    text-align: center;
    font-size: 38px;
    margin-bottom: 8px;
}

.subtitle {
    text-align: center;
    color: #d5f5e3;
    margin-bottom: 30px;
}

.upload {
    border: 2px dashed rgba(255,255,255,0.5);
    border-radius: 20px;
    padding: 30px;
    text-align: center;
    cursor: pointer;
}

.upload:hover {
    background: rgba(255,255,255,0.08);
}

input {
    display: none;
}

#preview {
    display: none;
    text-align: center;
    margin: 20px;
}

#preview img {
    max-width: 300px;
    max-height: 300px;
    border-radius: 18px;
}

button {
    width: 100%;
    padding: 16px;
    border: none;
    border-radius: 14px;
    background: white;
    color: #145a32;
    font-size: 18px;
    font-weight: bold;
    cursor: pointer;
    margin-top: 15px;
}

button:disabled {
    opacity: 0.6;
}

#loading {
    display: none;
    text-align: center;
    margin: 25px;
}

#result {
    display: none;
    margin-top: 25px;
}

.card {
    background: rgba(0,0,0,0.18);
    padding: 18px;
    border-radius: 15px;
    margin: 12px 0;
}

.label {
    font-size: 13px;
    color: #b8e6c9;
    text-transform: uppercase;
    margin-bottom: 6px;
}

.value {
    font-size: 20px;
    font-weight: bold;
    line-height: 1.5;
}

#error {
    display: none;
    background: rgba(192,57,43,0.4);
    padding: 15px;
    border-radius: 12px;
    margin-top: 20px;
    text-align: center;
}
</style>
</head>

<body>

<div class="container">

<h1>🌿🩺 AI Plant Doctor</h1>

<p class="subtitle">
AI-powered plant disease detection using deep learning
</p>

<label class="upload" for="image">
    <h2>📷 Upload a Plant Leaf</h2>
    <p>Click here to choose an image</p>
    <p>JPG, JPEG and PNG supported</p>
</label>

<input type="file" id="image" accept="image/*">

<div id="preview">
    <img id="previewImage">
</div>

<button id="analyze" disabled>
    🔍 Analyze Plant
</button>

<div id="loading">
    🌱 AI is analyzing your plant...
</div>

<div id="error"></div>

<div id="result">

    <div class="card">
        <div class="label">Detected Disease</div>
        <div class="value" id="disease">-</div>
    </div>

    <div class="card">
        <div class="label">Confidence</div>
        <div class="value" id="confidence">-</div>
    </div>

    <div class="card">
        <div class="label">Severity</div>
        <div class="value" id="severity">-</div>
    </div>

    <div class="card">
        <div class="label">Recommendation</div>
        <div class="value" id="recommendation">-</div>
    </div>

</div>

</div>

<script>

let selectedFile = null;

const imageInput = document.getElementById("image");
const preview = document.getElementById("preview");
const previewImage = document.getElementById("previewImage");
const analyze = document.getElementById("analyze");
const loading = document.getElementById("loading");
const result = document.getElementById("result");
const errorBox = document.getElementById("error");

imageInput.addEventListener("change", function() {

    selectedFile = this.files[0];

    if (!selectedFile) {
        analyze.disabled = true;
        preview.style.display = "none";
        return;
    }

    previewImage.src = URL.createObjectURL(selectedFile);
    preview.style.display = "block";
    analyze.disabled = false;
    result.style.display = "none";
    errorBox.style.display = "none";
});


analyze.addEventListener("click", async function() {

    if (!selectedFile) return;

    analyze.disabled = true;
    loading.style.display = "block";
    result.style.display = "none";
    errorBox.style.display = "none";

    const formData = new FormData();
    formData.append("image", selectedFile);

    try {

        const response = await fetch("/predict", {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || "Prediction failed");
        }

        document.getElementById("disease").textContent =
            data.disease;

        document.getElementById("confidence").textContent =
            data.confidence;

        document.getElementById("severity").textContent =
            data.severity;

        document.getElementById("recommendation").textContent =
            data.recommendation;

        result.style.display = "block";

    } catch (error) {

        console.error(error);

        errorBox.textContent =
            "❌ " + error.message;

        errorBox.style.display = "block";

    } finally {

        loading.style.display = "none";
        analyze.disabled = false;

    }

});

</script>

</body>
</html>
"""


@app.route("/predict", methods=["POST"])
def predict():

    print("📷 Prediction request received")

    try:

        if "image" not in request.files:
            return jsonify({
                "error": "No image uploaded"
            }), 400

        file = request.files["image"]

        image = Image.open(file).convert("RGB")
        image = image.resize((224, 224))

        image_array = np.asarray(
            image,
            dtype=np.float32
        )

        image_array = np.expand_dims(
            image_array,
            axis=0
        )

        print("🧠 Running prediction...")

        predictions = model.predict(
            image_array,
            verbose=0
        )

        index = int(
            np.argmax(predictions[0])
        )

        confidence = (
            float(predictions[0][index]) * 100
        )

        disease = class_names[index]

        print(
            f"✅ Prediction: {disease} "
            f"({confidence:.2f}%)"
        )

        return jsonify({
            "disease": clean_name(disease),
            "confidence": f"{confidence:.2f}%",
            "severity": get_severity(disease),
            "recommendation": get_recommendation(disease)
        })

    except Exception as e:

        print("❌ PREDICTION ERROR:", repr(e))

        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":

    port = int(
        os.environ.get("PORT", 5000)
    )

    app.run(
        host="0.0.0.0",
        port=port
    )
