import mlflow
import pandas as pd

mlflow.set_tracking_uri("file:./mlruns")

FEATURES = ["magnitude", "kedalaman_km", "lintang", "bujur", "jam"]

def test_mlflow_runs_exist():
    df_runs = mlflow.search_runs(experiment_names=["gempa-jatim-experiment"])
    assert len(df_runs) > 0

def test_model_predict():
    df_runs = mlflow.search_runs(
        experiment_names=["gempa-jatim-experiment"],
        order_by=["metrics.f1_score DESC"]
    )
    model = mlflow.sklearn.load_model(f"runs:/{df_runs.iloc[0]['run_id']}/model")
    input_data = pd.DataFrame([{
        "magnitude": 4.5, "kedalaman_km": 15,
        "lintang": -7.98, "bujur": 112.63, "jam": 14
    }])
    pred = model.predict(input_data)
    assert pred[0] in [0, 1]

def test_model_accuracy_threshold():
    df_runs = mlflow.search_runs(experiment_names=["gempa-jatim-experiment"])
    best_acc = df_runs["metrics.accuracy"].max()
    assert best_acc >= 0.60
