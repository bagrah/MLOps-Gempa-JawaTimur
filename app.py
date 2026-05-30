from flask import Flask, request, jsonify
import mlflow.pyfunc
import mlflow
import pandas as pd
import glob
import os
import time
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

app = Flask(__name__)

mlflow.set_tracking_uri("sqlite:///mlflow.db")

# Prometheus metrics
REQUEST_COUNT = Counter(
    "prediction_requests_total",
    "Total jumlah request prediksi",
    ["result"]
)
REQUEST_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Latensi prediksi dalam detik"
)
PREDICTION_SCORE = Histogram(
    "prediction_probability",
    "Distribusi probabilitas prediksi",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

def load_best_model():
    model_files = glob.glob("models/rf_n*.pkl")
    if not model_files:
        raise Exception("Tidak ada model ditemukan!")
    import pickle
    latest = max(model_files, key=os.path.getmtime)
    with open(latest, "rb") as f:
        return pickle.load(f)

model = load_best_model()

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": "GempaJawaTimur-RandomForest"})

@app.route("/predict", methods=["POST"])
def predict():
    start_time = time.time()
    data = request.json
    input_df = pd.DataFrame([{
        "magnitude": data["magnitude"],
        "kedalaman_km": data["kedalaman_km"],
        "lintang": data["lintang"],
        "bujur": data["bujur"],
        "jam": data["jam"]
    }])
    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0]
    label = "BERPOTENSI TSUNAMI" if pred == 1 else "TIDAK BERPOTENSI TSUNAMI"

    # Catat ke Prometheus
    latency = time.time() - start_time
    REQUEST_COUNT.labels(result=label).inc()
    REQUEST_LATENCY.observe(latency)
    PREDICTION_SCORE.observe(float(max(prob)))

    return jsonify({"prediksi": label, "nilai": int(pred)})

@app.route("/metrics")
def metrics():
    return generate_latest(), 200, {"Content-Type": CONTENT_TYPE_LATEST}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)