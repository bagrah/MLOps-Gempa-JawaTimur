import pandas as pd
import numpy as np
import os

np.random.seed(42)
n = 500

# Pola realistis gempa Jawa Timur berdasarkan data historis BMKG
df = pd.DataFrame({
    "datetime": pd.date_range("2023-01-01", periods=n, freq="12h").astype(str),
    "tanggal": pd.date_range("2023-01-01", periods=n, freq="12h").strftime("%d %b %Y"),
    "jam": np.random.randint(0, 24, n),
    "magnitude": np.round(np.random.uniform(2.0, 6.5, n), 1),
    "kedalaman_km": np.random.randint(5, 200, n),
    "lintang": np.round(np.random.uniform(-9.0, -6.5, n), 2),
    "bujur": np.round(np.random.uniform(110.0, 116.0, n), 2),
    "wilayah": "Jawa Timur (seed)",
})

# Label: gempa dirasakan jika magnitude >= 4.0 dan kedalaman < 60 km (pola umum)
df["dirasakan"] = ((df["magnitude"] >= 4.0) & (df["kedalaman_km"] < 60)).astype(int)

# Tambah sedikit noise agar tidak terlalu deterministik
noise_idx = np.random.choice(df.index, size=30, replace=False)
df.loc[noise_idx, "dirasakan"] = 1 - df.loc[noise_idx, "dirasakan"]

os.makedirs("data/raw/batch", exist_ok=True)
df.to_csv("data/raw/batch/gempa_seed_2023.csv", index=False)

print(f"Seed data dibuat: {df.shape}")
print(df["dirasakan"].value_counts())
