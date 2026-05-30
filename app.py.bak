from flask import Flask, request, jsonify
import mlflow
import mlflow.sklearn
import pandas as pd
import os
import glob

app = Flask(__name__)

mlflow.set_tracking_uri("sqlite:///mlflow.db")

def load_best_model():
    # Cari MLmodel file terbaru
    mlmodel_files = glob.glob("mlruns/1/models/*/artifacts/MLmodel")
    if not mlmodel_files:
        raise Exception("Tidak ada model artifact ditemukan!")
    
    # Ambil yang terbaru berdasarkan waktu modifikasi
    latest = max(mlmodel_files, key=os.path.getmtime)
    model_dir = os.path.dirname(latest)
    print(f"Loading model dari: {model_dir}")
    return mlflow.sklearn.load_model(model_dir)

model = load_best_model()

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": "GempaJawaTimur-RandomForest"})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    input_df = pd.DataFrame([{
        "magnitude": data["magnitude"],
        "kedalaman_km": data["kedalaman_km"],
        "lintang": data["lintang"],
        "bujur": data["bujur"],
        "jam": data["jam"]
    }])
    pred = model.predict(input_df)[0]
    label = "DIRASAKAN" if pred == 1 else "TIDAK DIRASAKAN"
    return jsonify({"prediksi": label, "nilai": int(pred)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
