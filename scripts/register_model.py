import mlflow
from mlflow import MlflowClient

mlflow.set_tracking_uri("sqlite:///mlflow.db")

MODEL_NAME = "GempaJawaTimur-RandomForest"

def register_model():
    client = MlflowClient()

    # Ambil best run berdasarkan f1_score
    df_runs = mlflow.search_runs(
        experiment_names=["gempa-jatim-experiment"],
        order_by=["metrics.f1_score DESC"]
    )

    best_run_id = df_runs.iloc[0]["run_id"]
    print(f"Best run_id: {best_run_id}")

    # Register model
    model_uri = f"runs:/{best_run_id}/model"
    mv = mlflow.register_model(model_uri, MODEL_NAME)
    print(f"Model registered: {MODEL_NAME} v{mv.version}")

    # Transisi ke Staging
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=mv.version,
        stage="Staging"
    )
    print(f"Model v{mv.version} → Staging")

    # Transisi ke Production
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=mv.version,
        stage="Production"
    )
    print(f"Model v{mv.version} → Production")

    # Verifikasi load model Production
    production_model = mlflow.pyfunc.load_model(
        model_uri=f"models:/{MODEL_NAME}/Production"
    )
    print(f"\nModel Production berhasil di-load: {type(production_model)}")
    return mv.version

if __name__ == "__main__":
    register_model()
