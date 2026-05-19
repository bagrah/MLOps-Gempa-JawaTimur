import pandas as pd
import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri("file:./mlruns")

FEATURES = ["magnitude", "kedalaman_km", "lintang", "bujur", "jam"]

def load_best_model():
    df_runs = mlflow.search_runs(
        experiment_names=["gempa-jatim-experiment"],
        order_by=["metrics.f1_score DESC"]
    )
    if df_runs.empty:
        raise Exception("Tidak ada model ditemukan di MLflow!")
    
    best_run_id = df_runs.iloc[0]["run_id"]
    model_uri = f"runs:/{best_run_id}/model"
    model = mlflow.sklearn.load_model(model_uri)
    print(f"Model loaded dari run_id: {best_run_id}")
    return model

def predict(magnitude, kedalaman_km, lintang, bujur, jam):
    model = load_best_model()

    input_data = pd.DataFrame([{
        "magnitude": magnitude,
        "kedalaman_km": kedalaman_km,
        "lintang": lintang,
        "bujur": bujur,
        "jam": jam
    }])

    pred = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0]

    label = "DIRASAKAN" if pred == 1 else "TIDAK DIRASAKAN"
    print(f"\n=== Hasil Prediksi ===")
    print(f"  Input : M{magnitude}, kedalaman {kedalaman_km}km, ({lintang}, {bujur}), jam {jam}")
    print(f"  Prediksi  : {label}")
    print(f"  Confidence: {max(prob)*100:.1f}%")
    return pred, prob

if __name__ == "__main__":
    # Contoh: gempa M4.5, kedalaman 15km, di sekitar Malang, jam 14
    predict(
        magnitude=4.5,
        kedalaman_km=15,
        lintang=-7.98,
        bujur=112.63,
        jam=14
    )

    # Contoh: gempa dalam M6.0, kedalaman 180km
    predict(
        magnitude=6.0,
        kedalaman_km=180,
        lintang=-8.50,
        bujur=111.50,
        jam=3
    )
