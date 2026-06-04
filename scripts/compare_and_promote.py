import mlflow
from mlflow import MlflowClient
import os
import pickle
import glob

TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
mlflow.set_tracking_uri(TRACKING_URI)

EXPERIMENT_NAME = "tsunami-indonesia-experiment"
MODEL_NAME = "TsunamiRisk-RandomForest"

def compare_and_promote():
    client = MlflowClient()

    df_runs = mlflow.search_runs(
        experiment_names=[EXPERIMENT_NAME],
        order_by=["metrics.f1_score DESC"]
    )

    if df_runs.empty:
        print("Tidak ada run ditemukan!")
        return

    best_run = df_runs.iloc[0]
    best_run_id = best_run["run_id"]
    new_f1 = best_run.get("metrics.f1_score", 0)
    new_acc = best_run.get("metrics.accuracy", 0)

    print(f"Model baru  - Accuracy: {new_acc:.4f} | F1: {new_f1:.4f}")

    # Cek model Production yang ada
    try:
        versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])
        if versions:
            prod_run_id = versions[0].run_id
            prod_run = mlflow.get_run(prod_run_id)
            old_f1 = prod_run.data.metrics.get("f1_score", 0)
            old_acc = prod_run.data.metrics.get("accuracy", 0)
            print(f"Model lama  - Accuracy: {old_acc:.4f} | F1: {old_f1:.4f}")

            if new_f1 <= old_f1:
                print("Model baru tidak lebih baik. Tetap pakai model lama.")
                return
            print("Model baru lebih baik! Promosi ke Production...")
        else:
            print("Belum ada model Production. Langsung promosi...")
    except Exception:
        print("Belum ada registry. Membuat baru...")

    # Load model pickle dan register
    model_files = glob.glob("models/rf_n*.pkl")
    if not model_files:
        print("Tidak ada model pickle!")
        return

    latest_model = max(model_files, key=os.path.getmtime)
    with open(latest_model, "rb") as f:
        model = pickle.load(f)

    with mlflow.start_run(run_id=best_run_id):
        mlflow.sklearn.log_model(model, artifact_path="model")

    mv = mlflow.register_model(f"runs:/{best_run_id}/model", MODEL_NAME)
    print(f"Model registered: {MODEL_NAME} v{mv.version}")

    client.transition_model_version_stage(
        name=MODEL_NAME, version=mv.version, stage="Production"
    )
    print(f"Model v{mv.version} → Production ✅")

if __name__ == "__main__":
    compare_and_promote()
