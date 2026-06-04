import pandas as pd
import numpy as np
import os

np.random.seed(99)
n = 100

# Data "shifted" - distribusi berbeda dari training data
# Simulasi: tiba-tiba banyak gempa dangkal besar (pola tidak biasa)
df = pd.DataFrame({
    "datetime": pd.date_range("2026-06-01", periods=n, freq="2h").astype(str),
    "tanggal": pd.date_range("2026-06-01", periods=n, freq="2h").strftime("%d %b %Y"),
    "jam": np.random.randint(0, 24, n),
    "magnitude": np.round(np.random.uniform(6.0, 8.0, n), 1),  # shifted: semua gempa besar
    "kedalaman_km": np.random.randint(5, 30, n),               # shifted: semua dangkal
    "lintang": np.round(np.random.uniform(-11.0, 6.0, n), 2),
    "bujur": np.round(np.random.uniform(95.0, 141.0, n), 2),
    "wilayah": "Simulasi Drift Data",
    "potensi_tsunami": 1,  # semua berpotensi tsunami (extreme shift)
})

os.makedirs("data/raw/batch", exist_ok=True)
output = "data/raw/batch/gempa_drift_simulation.csv"
df.to_csv(output, index=False)

print(f"Drift data dibuat: {df.shape}")
print(f"Distribusi magnitude: min={df['magnitude'].min()}, max={df['magnitude'].max()}")
print(f"Distribusi kedalaman: min={df['kedalaman_km'].min()}, max={df['kedalaman_km'].max()}")
print(f"Saved ke: {output}")
