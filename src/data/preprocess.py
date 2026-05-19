import pandas as pd
import os
import glob

def preprocess_data():
    print("Preprocessing data gempa Jawa Timur...")

    # Load semua file batch yang ada
    batch_files = glob.glob("data/raw/batch/gempa_*.csv")
    
    if not batch_files:
        print("Tidak ada data batch ditemukan di data/raw/batch/")
        return

    print(f"Ditemukan {len(batch_files)} file batch: {batch_files}")

    # Gabungkan semua batch
    df = pd.concat([pd.read_csv(f) for f in batch_files], ignore_index=True)
    print(f"Total data sebelum cleaning: {df.shape}")

    # Hapus duplikat
    df = df.drop_duplicates(subset=["datetime"])

    # Hapus baris dengan nilai kosong di kolom penting
    features = ["magnitude", "kedalaman_km", "lintang", "bujur", "jam", "dirasakan"]
    df = df.dropna(subset=features)

    # Pastikan tipe data benar
    df["magnitude"] = df["magnitude"].astype(float)
    df["kedalaman_km"] = df["kedalaman_km"].astype(int)
    df["lintang"] = df["lintang"].astype(float)
    df["bujur"] = df["bujur"].astype(float)
    df["jam"] = df["jam"].astype(int)
    df["dirasakan"] = df["dirasakan"].astype(int)

    print(f"Total data setelah cleaning: {df.shape}")
    print(f"\nDistribusi label:")
    print(df["dirasakan"].value_counts())
    print(f"\nStatistik fitur:")
    print(df[features].describe())

    # Simpan hasil
    os.makedirs("data/processed", exist_ok=True)
    output_path = "data/processed/gempa_jatim.csv"
    df.to_csv(output_path, index=False)
    print(f"\nData processed disimpan ke: {output_path}")

if __name__ == "__main__":
    preprocess_data()
