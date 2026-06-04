import mlflow
import os
import sys

TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
mlflow.set_tracking_uri(TRACKING_URI)

EXPERIMENT_NAME = "tsunami-indonesia-experiment"
THRESHOLD_ACCURACY = 0.85
THRESHOLD_F1 = 0.60

def check_performance():
    print("Mengecek performa model terkini...")

    df_runs = mlflow.search_runs(
        experiment_names=[EXPERIMENT_NAME],
        order_by=["start_time DESC"]
    )

    if df_runs.empty:
        print("Tidak ada run ditemukan! Trigger retrain.")
        sys.exit(1)

    latest = df_runs.iloc[0]
    acc = latest.get("metrics.accuracy", 0)
    f1  = latest.get("metrics.f1_score", 0)

    print(f"Model terkini:")
    print(f"  Accuracy : {acc:.4f} (threshold: {THRESHOLD_ACCURACY})")
    print(f"  F1 Score : {f1:.4f} (threshold: {THRESHOLD_F1})")

    if acc < THRESHOLD_ACCURACY or f1 < THRESHOLD_F1:
        print("PERFORMA DI BAWAH THRESHOLD! Trigger retrain...")
        sys.exit(1)
    else:
        print("Performa OK. Tidak perlu retrain.")
        sys.exit(0)

if __name__ == "__main__":
    check_performance()
