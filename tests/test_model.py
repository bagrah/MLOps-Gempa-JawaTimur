import pandas as pd
import mlflow
import pickle
import glob
import os

TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
mlflow.set_tracking_uri(TRACKING_URI)

FEATURES = ["magnitude", "kedalaman_km", "lintang", "bujur", "jam"]

def test_mlflow_runs_exist():
    df_runs = mlflow.search_runs(experiment_names=["gempa-jatim-experiment"])
    assert len(df_runs) > 0

def test_model_predict():
    model_files = glob.glob("models/rf_n*.pkl")
    assert len(model_files) > 0
    with open(max(model_files, key=os.path.getmtime), "rb") as f:
        model = pickle.load(f)
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
