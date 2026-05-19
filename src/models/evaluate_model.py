import pandas as pd
import mlflow
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

mlflow.set_tracking_uri("file:./mlruns")

FEATURES = ["magnitude", "kedalaman_km", "lintang", "bujur", "jam"]
TARGET = "dirasakan"

def evaluate_model():
    print("Evaluating best model from MLflow...")

    # Ambil run terbaik berdasarkan f1_score
    df_runs = mlflow.search_runs(
        experiment_names=["gempa-jatim-experiment"],
        order_by=["metrics.f1_score DESC"]
    )

    if df_runs.empty:
        print("Tidak ada run ditemukan!")
        return

    best_run = df_runs.iloc[0]
    run_id = best_run["run_id"]
    print(f"Best run_id: {run_id}")
    print(f"  F1 Score : {best_run['metrics.f1_score']:.4f}")
    print(f"  Accuracy : {best_run['metrics.accuracy']:.4f}")
    print(f"  Precision: {best_run['metrics.precision']:.4f}")
    print(f"  Recall   : {best_run['metrics.recall']:.4f}")

    # Load model dari run terbaik
    model_uri = f"runs:/{run_id}/model"
    model = mlflow.sklearn.load_model(model_uri)

    # Load data processed
    df = pd.read_csv("data/processed/gempa_jatim.csv")
    X = df[FEATURES]
    y = df[TARGET]

    y_pred = model.predict(X)

    print("\n=== Classification Report ===")
    print(classification_report(y, y_pred, target_names=["Tidak Dirasakan", "Dirasakan"]))

    print("=== Confusion Matrix ===")
    cm = confusion_matrix(y, y_pred)
    print(f"  TN={cm[0][0]} | FP={cm[0][1]}")
    print(f"  FN={cm[1][0]} | TP={cm[1][1]}")

if __name__ == "__main__":
    evaluate_model()
