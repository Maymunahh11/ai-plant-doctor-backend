from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import os

app = Flask(__name__)
CORS(app)

# ==============================
# LOAD AI MODEL
# ==============================

MODEL_PATH = "plant_disease_model_fixed.keras"

print("Loading AI Plant Doctor model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("AI model loaded successfully!")


# ==============================
# 38 PLANT DISEASE CLASSES
# ==============================

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


# ==============================
# RECOMMENDATIONS
# ==============================

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


# ==============================
# HOME ROUTE
# ==============================

@app.route("/", methods=["GET"])
def home():

    return jsonify({
        "status": "online",
        "message": "🌱 AI Plant Doctor is running",
        "classes": len(class_names)
    })


# ==============================
# PREDICTION ROUTE
# ==============================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        if "image" not in request.files:

            return jsonify({
                "error": "No image uploaded"
            }), 400

        file = request.files["image"]

        image = Image.open(file).convert("RGB")

        image = image.resize((224, 224))

        image_array = np.array(image).astype("float32")

        image_array = np.expand_dims(image_array, axis=0)

        # IMPORTANT:
        # The model already contains the Rescaling layer.
        # Therefore we do NOT use MobileNetV2 preprocess_input here.

        predictions = model.predict(image_array, verbose=0)

        predicted_index = int(np.argmax(predictions[0]))

        confidence = float(predictions[0][predicted_index]) * 100

        disease = class_names[predicted_index]

        recommendation = get_recommendation(disease)

        severity = get_severity(disease)

        return jsonify({

            "disease": disease,

            "confidence": f"{confidence:.2f}%",

            "severity": severity,

            "recommendation": recommendation

        })

    except Exception as e:

        print("Prediction error:", str(e))

        return jsonify({
            "error": str(e)
        }), 500


# ==============================
# START SERVER
# ==============================

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))

    print("======================================")
    print("🌱 AI PLANT DOCTOR BACKEND")
    print("======================================")
    print("Classes:", len(class_names))
    print("Starting Flask server...")
    print("Port:", port)
    print("======================================")

    app.run(
        host="0.0.0.0",
        port=port
    )
