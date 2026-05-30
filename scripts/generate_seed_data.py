import pandas as pd
import numpy as np
import os

np.random.seed(42)
n = 500

df = pd.DataFrame({
    "datetime": pd.date_range("2023-01-01", periods=n, freq="12h").astype(str),
    "tanggal": pd.date_range("2023-01-01", periods=n, freq="12h").strftime("%d %b %Y"),
    "jam": np.random.randint(0, 24, n),
    "magnitude": np.round(np.random.uniform(2.0, 7.5, n), 1),
    "kedalaman_km": np.random.randint(5, 200, n),
    "lintang": np.round(np.random.uniform(-11.0, 6.0, n), 2),
    "bujur": np.round(np.random.uniform(95.0, 141.0, n), 2),
    "wilayah": "Indonesia (seed)",
})

# Tsunami berpotensi jika gempa di laut, magnitude >= 6.5, kedalaman < 70km
df["potensi_tsunami"] = (
    (df["magnitude"] >= 6.5) & (df["kedalaman_km"] < 70)
).astype(int)

# Tambah noise
noise_idx = np.random.choice(df.index, size=20, replace=False)
df.loc[noise_idx, "potensi_tsunami"] = 1 - df.loc[noise_idx, "potensi_tsunami"]

os.makedirs("data/raw/batch", exist_ok=True)
df.to_csv("data/raw/batch/gempa_seed_2023.csv", index=False)

print(f"Seed data dibuat: {df.shape}")
print(df["potensi_tsunami"].value_counts())