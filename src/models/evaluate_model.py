import pandas as pd
import mlflow
import pickle
import glob
import os
from sklearn.metrics import confusion_matrix, classification_report

TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
mlflow.set_tracking_uri(TRACKING_URI)

FEATURES = ["magnitude", "kedalaman_km", "lintang", "bujur", "jam"]
TARGET = "potensi_tsunami"

def evaluate_model():
    print("Evaluating best model...")

    model_files = glob.glob("models/rf_n*.pkl")
    if not model_files:
        print("Tidak ada model ditemukan!")
        return

    latest_model = max(model_files, key=os.path.getmtime)
    print(f"Loading model: {latest_model}")

    with open(latest_model, "rb") as f:
        model = pickle.load(f)

    df = pd.read_csv("data/processed/gempa_indonesia.csv")
    X = df[FEATURES]
    y = df[TARGET]
    y_pred = model.predict(X)

    print("\n=== Classification Report ===")
    print(classification_report(y, y_pred, target_names=["Tidak Berpotensi", "Berpotensi Tsunami"]))

    cm = confusion_matrix(y, y_pred)
    print("=== Confusion Matrix ===")
    print(f"  TN={cm[0][0]} | FP={cm[0][1]}")
    print(f"  FN={cm[1][0]} | TP={cm[1][1]}")

if __name__ == "__main__":
    evaluate_model()