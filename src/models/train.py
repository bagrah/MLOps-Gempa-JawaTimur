import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import os
import pickle

TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment("gempa-jatim-experiment")

FEATURES = ["magnitude", "kedalaman_km", "lintang", "bujur", "jam"]
TARGET = "dirasakan"

def load_data():
    df = pd.read_csv("data/processed/gempa_jatim.csv")
    return df

def train(n_estimators=100):
    df = load_data()
    X = df[FEATURES]
    y = df[TARGET]

    print(f"Total data: {len(df)}")
    print(f"Distribusi label:\n{y.value_counts()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weight = dict(zip(classes, weights))

    with mlflow.start_run() as run:
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            class_weight=class_weight,
            random_state=42
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)


        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec  = recall_score(y_test, y_pred, zero_division=0)
        f1   = f1_score(y_test, y_pred, zero_division=0)

        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("features", FEATURES)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)

        # Simpan model pickle lokal saja, tanpa log artifact ke MLflow
        os.makedirs("models", exist_ok=True)
        model_path = f"models/rf_n{n_estimators}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        print(f"\nn_estimators={n_estimators}")
        print(f"  Accuracy : {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall   : {rec:.4f}")
        print(f"  F1 Score : {f1:.4f}")

        return acc, f1

if __name__ == "__main__":
    results = []
    for n in [50, 100, 200, 300]:
        acc, f1 = train(n_estimators=n)
        results.append((n, acc, f1))

    print("\n=== Hasil Semua Eksperimen ===")
    best = max(results, key=lambda x: x[2])
    for n, acc, f1 in results:
        print(f"  n={n:3d} | acc={acc:.4f} | f1={f1:.4f}")
    print(f"\nBest: n_estimators={best[0]} | F1={best[2]:.4f}")

    if best[2] < 0.40:
        raise Exception(f"F1 Score {best[2]:.4f} di bawah threshold 0.40!")
