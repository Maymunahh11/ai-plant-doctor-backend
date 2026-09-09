from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

# Keep CORS enabled, although the website will now be served
# from this same Flask application.
CORS(app, resources={r"/*": {"origins": "*"}})

MODEL_PATH = "plant_disease_model_fixed.keras"

print("======================================")
print("🌱 AI PLANT DOCTOR")
print("======================================")
print("Loading AI Plant Doctor model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("AI model loaded successfully!")

# Warm up TensorFlow so the first real user prediction
# does not pay the model/tracing startup cost.
print("Warming up AI model...")

dummy_image = np.zeros((1, 224, 224, 3), dtype=np.float32)
model.predict(dummy_image, verbose=0)

print("AI model warmed up successfully!")

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


def get_recommendation(disease):
    if "healthy" in disease.lower():
        return (
            "The plant appears healthy. "
            "Continue regular watering, proper nutrition, "
            "and monitor the plant regularly."
        )

    return (
        "The plant may be affected by this disease. "
        "Remove severely affected leaves, maintain good air circulation, "
        "avoid excessive moisture on leaves, and consider an appropriate "
        "plant disease treatment."
    )


def get_severity(disease):
    if "healthy" in disease.lower():
        return "Healthy ✅"

    return "Needs Attention ⚠️"


def format_disease(name):
    if not name:
        return "Unknown"

    return (
        name.replace("___", " — ")
            .replace("_", " ")
            .replace("(", " (")
            .replace("  ", " ")
            .strip()
    )


@app.route("/", methods=["GET"])
def home():
    return HTML_PAGE


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "online",
        "model": "loaded",
        "classes": len(class_names)
    })


@app.route("/predict", methods=["POST"])
def predict():

    print("Received prediction request")

    try:

        if "image" not in request.files:
            print("ERROR: No image uploaded")
            return jsonify({
                "error": "No image uploaded"
            }), 400

        file = request.files["image"]

        image = Image.open(file).convert("RGB")
        image = image.resize((224, 224))

        image_array = np.array(image).astype("float32")
        image_array = np.expand_dims(image_array, axis=0)

        print("Running AI prediction...")

        predictions = model.predict(
            image_array,
            verbose=0
        )

        predicted_index = int(
            np.argmax(predictions[0])
        )

        confidence = (
            float(predictions[0][predicted_index]) * 100
        )

        disease = class_names[predicted_index]

        print(
            "Prediction:",
            disease,
            f"{confidence:.2f}%"
        )

        return jsonify({
            "disease": disease,
            "confidence": f"{confidence:.2f}%",
            "severity": get_severity(disease),
            "recommendation": get_recommendation(disease)
        })

    except Exception as e:

        print("PREDICTION ERROR:", str(e))

        return jsonify({
            "error": str(e)
        }), 500


HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">

<head>

<meta charset="UTF-8">

<meta name="viewport"
      content="width=device-width, initial-scale=1.0">

<title>AI Plant Doctor 🌿</title>

<style>

* {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
    font-family: Arial, sans-serif;
}

body {
    min-height: 100vh;
    background: linear-gradient(
        135deg,
        #0b3d2e,
        #145a32,
        #1e8449
    );
    color: white;
    display: flex;
    justify-content: center;
    align-items: center;
    padding: 25px;
}

.container {
    width: 100%;
    max-width: 850px;
    background: rgba(255,255,255,0.10);
    backdrop-filter: blur(18px);
    border: 1px solid rgba(255,255,255,0.2);
    border-radius: 28px;
    padding: 35px;
    box-shadow: 0 20px 60px rgba(0,0,0,0.35);
}

.header {
    text-align: center;
    margin-bottom: 30px;
}

.logo {
    font-size: 55px;
    margin-bottom: 8px;
}

h1 {
    font-size: 38px;
    margin-bottom: 8px;
}

.subtitle {
    color: #d5f5e3;
    font-size: 16px;
}

.upload-box {
    border: 2px dashed rgba(255,255,255,0.5);
    border-radius: 20px;
    padding: 30px;
    text-align: center;
    cursor: pointer;
    transition: 0.3s;
    margin-bottom: 20px;
}

.upload-box:hover {
    background: rgba(255,255,255,0.08);
    border-color: white;
}

.upload-icon {
    font-size: 45px;
    margin-bottom: 10px;
}

.upload-box p {
    margin: 8px;
    color: #e8f8f5;
}

input[type="file"] {
    display: none;
}

.preview {
    display: none;
    text-align: center;
    margin: 20px 0;
}

.preview img {
    max-width: 300px;
    max-height: 300px;
    border-radius: 18px;
    border: 3px solid rgba(255,255,255,0.5);
    box-shadow: 0 10px 30px rgba(0,0,0,0.3);
}

.button {
    width: 100%;
    border: none;
    padding: 16px;
    border-radius: 14px;
    background: white;
    color: #145a32;
    font-size: 18px;
    font-weight: bold;
    cursor: pointer;
}

.button:disabled {
    opacity: 0.6;
    cursor: not-allowed;
}

.loading {
    display: none;
    text-align: center;
    margin: 25px 0;
}

.spinner {
    width: 40px;
    height: 40px;
    border: 4px solid rgba(255,255,255,0.3);
    border-top: 4px solid white;
    border-radius: 50%;
    animation: spin 1s linear infinite;
    margin: auto auto 12px;
}

@keyframes spin {
    100% {
        transform: rotate(360deg);
    }
}

.result {
    display: none;
    margin-top: 28px;
    background: rgba(255,255,255,0.12);
    border-radius: 20px;
    padding: 25px;
}

.result-title {
    text-align: center;
    font-size: 24px;
    margin-bottom: 20px;
}

.result-card {
    background: rgba(0,0,0,0.15);
    border-radius: 15px;
    padding: 18px;
    margin-bottom: 12px;
}

.label {
    font-size: 13px;
    color: #b8e6c9;
    margin-bottom: 6px;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.value {
    font-size: 21px;
    font-weight: bold;
}

.recommendation {
    line-height: 1.6;
    font-size: 16px;
}

.error {
    display: none;
    margin-top: 20px;
    padding: 15px;
    border-radius: 12px;
    background: rgba(192,57,43,0.35);
    text-align: center;
}

.footer {
    text-align: center;
    margin-top: 25px;
    color: #c8e6d5;
    font-size: 13px;
}

</style>

</head>

<body>

<div class="container">

<div class="header">

<div class="logo">🌿🩺</div>

<h1>AI Plant Doctor</h1>

<p class="subtitle">
AI-powered plant disease detection using deep learning
</p>

</div>

<label class="upload-box" for="imageInput">

<div class="upload-icon">📷</div>

<h3>Upload a Plant Leaf</h3>

<p>Click here to choose an image</p>

<p>Supports JPG, JPEG and PNG</p>

</label>

<input
    type="file"
    id="imageInput"
    accept="image/*"
>

<div class="preview" id="preview">

<img
    id="previewImage"
    alt="Plant preview"
>

</div>

<button
    class="button"
    id="predictButton"
    disabled
>

🔍 Analyze Plant

</button>

<div class="loading" id="loading">

<div class="spinner"></div>

<p>AI is analyzing your plant...</p>

</div>

<div class="error" id="error"></div>

<div class="result" id="result">

<div class="result-title">
🌱 Diagnosis Result
</div>

<div class="result-card">

<div class="label">
Detected Disease
</div>

<div class="value" id="disease">
-
</div>

</div>

<div class="result-card">

<div class="label">
Confidence
</div>

<div class="value" id="confidence">
-
</div>

</div>

<div class="result-card">

<div class="label">
Severity
</div>

<div class="value" id="severity">
-
</div>

</div>

<div class="result-card">

<div class="label">
Recommendation
</div>

<div class="value recommendation"
     id="recommendation">
-
</div>

</div>

</div>

<div class="footer">

Powered by MobileNetV2 • TensorFlow • Flask • AI/ML

<br><br>

🌿 AI Plant Doctor — 38 Plant Disease Classes

</div>

</div>

<script>

const imageInput =
document.getElementById("imageInput");

const preview =
document.getElementById("preview");

const previewImage =
document.getElementById("previewImage");

const predictButton =
document.getElementById("predictButton");

const loading =
document.getElementById("loading");

const result =
document.getElementById("result");

const errorBox =
document.getElementById("error");

let selectedFile = null;


imageInput.addEventListener(
"change",
function () {

selectedFile = this.files[0];

if (!selectedFile) {

predictButton.disabled = true;
preview.style.display = "none";

return;

}

const reader = new FileReader();

reader.onload = function(event) {

previewImage.src =
event.target.result;

preview.style.display =
"block";

predictButton.disabled =
false;

result.style.display =
"none";

errorBox.style.display =
"none";

};

reader.readAsDataURL(
selectedFile
);

});


predictButton.addEventListener(
"click",
async function () {

if (!selectedFile) {
return;
}

predictButton.disabled =
true;

loading.style.display =
"block";

result.style.display =
"none";

errorBox.style.display =
"none";


const formData =
new FormData();

formData.append(
"image",
selectedFile
);


try {

const response =
await fetch(
"/predict",
{
method: "POST",
body: formData
}
);


const data =
await response.json()
.catch(() => ({}));


if (!response.ok) {

throw new Error(
data.error ||
"Server error: " +
response.status
);

}


document.getElementById(
"disease"
).textContent =
formatDisease(
data.disease
);


document.getElementById(
"confidence"
).textContent =
data.confidence;


document.getElementById(
"severity"
).textContent =
data.severity;


document.getElementById(
"recommendation"
).textContent =
data.recommendation;


result.style.display =
"block";

}

catch(error) {

console.error(error);

errorBox.textContent =
"❌ " + error.message;

errorBox.style.display =
"block";

}

finally {

loading.style.display =
"none";

predictButton.disabled =
false;

}

});


function formatDisease(name) {

if (!name) {
return "Unknown";
}

return name
.replaceAll("___", " — ")
.replaceAll("_", " ")
.replaceAll("(", " (")
.replaceAll("  ", " ")
.trim();

}

</script>

</body>

</html>
"""


if __name__ == "__main__":

    port = int(
        os.environ.get("PORT", 5000)
    )

    app.run(
        host="0.0.0.0",
        port=port
    )
