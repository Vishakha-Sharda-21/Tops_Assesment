"""
M16-A1 - Section B - Task 2
Flask REST API for Order Predictions

Loads delivery_model.joblib ONCE at startup (not per-request) and exposes
POST /predict for a mobile app / frontend to get live delivery estimates.

Run:
    python task2_flask_api.py

Test with curl (success case):
    curl -X POST http://127.0.0.1:5000/predict \
         -H "Content-Type: application/json" \
         -d '{"distance_km": 4.2, "num_items": 3, "rain_flag": 1}'

Test with curl (missing field -> should return 400, not crash):
    curl -X POST http://127.0.0.1:5000/predict \
         -H "Content-Type: application/json" \
         -d '{"distance_km": 4.2, "num_items": 3}'
"""

from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

MODEL_PATH = "delivery_model.joblib"
REQUIRED_FIELDS = ["distance_km", "num_items", "rain_flag"]

# Load the model once at startup, keep it in memory for the process lifetime.
# NOTE: run task1_save_load_model.py first so this file exists.
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"'{MODEL_PATH}' not found. Run task1_save_load_model.py first to generate it."
    )

model = joblib.load(MODEL_PATH)
print(f"Loaded model from '{MODEL_PATH}' at startup.")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)

    if data is None:
        return jsonify({"error": "Request body must be valid JSON"}), 400

    missing = [field for field in REQUIRED_FIELDS if field not in data]
    if missing:
        return jsonify({
            "error": "Missing required field(s)",
            "missing_fields": missing
        }), 400

    try:
        X_new = pd.DataFrame([{
            "distance_km": float(data["distance_km"]),
            "num_items": int(data["num_items"]),
            "rain_flag": int(data["rain_flag"]),
        }])
    except (ValueError, TypeError):
        return jsonify({"error": "Fields must be numeric"}), 400

    prediction = model.predict(X_new)[0]

    return jsonify({
        "predicted_delivery_time_min": round(float(prediction), 1)
    }), 200


@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "QuickBite delivery-time API is running"}), 200


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
