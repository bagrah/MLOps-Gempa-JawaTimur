import pandas as pd
import pickle
import glob
import os

FEATURES = ["magnitude", "kedalaman_km", "lintang", "bujur", "jam"]

def load_best_model():
    model_files = glob.glob("models/rf_n*.pkl")
    if not model_files:
        raise Exception("Tidak ada model ditemukan!")
    latest = max(model_files, key=os.path.getmtime)
    with open(latest, "rb") as f:
        return pickle.load(f)

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
    label = "BERPOTENSI TSUNAMI" if pred == 1 else "TIDAK BERPOTENSI TSUNAMI"

    print(f"\n=== Hasil Prediksi ===")
    print(f"  Input     : M{magnitude}, kedalaman {kedalaman_km}km, ({lintang}, {bujur}), jam {jam}")
    print(f"  Prediksi  : {label}")
    print(f"  Confidence: {max(prob)*100:.1f}%")
    return pred, prob

if __name__ == "__main__":
    # Gempa dangkal besar di laut - berpotensi tsunami
    predict(magnitude=7.2, kedalaman_km=10, lintang=-8.50, bujur=115.50, jam=14)

    # Gempa dalam - tidak berpotensi
    predict(magnitude=6.0, kedalaman_km=180, lintang=-3.50, bujur=120.50, jam=3)