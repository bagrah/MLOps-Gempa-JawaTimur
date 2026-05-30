import requests
import pandas as pd
import os
from datetime import datetime

ENDPOINTS = {
    "terkini": "https://data.bmkg.go.id/DataMKG/TEWS/gempaterkini.json",
    "dirasakan": "https://data.bmkg.go.id/DataMKG/TEWS/gempadirasakan.json",
}

def parse_koordinat(lintang_str, bujur_str):
    try:
        lat = float(lintang_str.split()[0])
        if "LS" in lintang_str:
            lat = -lat
        lon = float(bujur_str.split()[0])
        return lat, lon
    except:
        return None, None

def fetch_gempa(endpoint_url):
    try:
        response = requests.get(endpoint_url, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data["Infogempa"]["gempa"]
    except Exception as e:
        print(f"Error fetching {endpoint_url}: {e}")
        return []

def parse_gempa(gempa_list):
    records = []
    for g in gempa_list:
        lat, lon = parse_koordinat(g.get("Lintang", ""), g.get("Bujur", ""))
        if lat is None:
            continue
        try:
            jam = int(g.get("Jam", "00:00:00 WIB").split(":")[0])
            kedalaman = int(g.get("Kedalaman", "0 km").split()[0])
            magnitude = float(g.get("Magnitude", 0))
        except:
            continue

        # Label dari field Potensi
        potensi = g.get("Potensi", "")
        tsunami = 1 if "berpotensi tsunami" in potensi.lower() and "tidak" not in potensi.lower() else 0

        record = {
            "datetime": g.get("DateTime", ""),
            "tanggal": g.get("Tanggal", ""),
            "jam": jam,
            "magnitude": magnitude,
            "kedalaman_km": kedalaman,
            "lintang": lat,
            "bujur": lon,
            "wilayah": g.get("Wilayah", ""),
            "potensi_tsunami": tsunami,
        }
        records.append(record)
    return pd.DataFrame(records)

def ingest_data():
    print("Fetching data dari BMKG (seluruh Indonesia)...")

    gempa_terkini = fetch_gempa(ENDPOINTS["terkini"])
    gempa_dirasakan = fetch_gempa(ENDPOINTS["dirasakan"])

    print(f"  gempaterkini  : {len(gempa_terkini)} gempa")
    print(f"  gempadirasakan: {len(gempa_dirasakan)} gempa")

    df_terkini = parse_gempa(gempa_terkini)
    df_dirasakan = parse_gempa(gempa_dirasakan)

    df_all = pd.concat([df_terkini, df_dirasakan], ignore_index=True)

    if df_all.empty:
        print("Tidak ada data gempa. Skip.")
        return

    df_all = df_all.drop_duplicates(subset=["datetime"])

    print(f"\nTotal gempa hari ini: {len(df_all)}")
    print(df_all[["tanggal", "magnitude", "kedalaman_km", "wilayah", "potensi_tsunami"]])

    os.makedirs("data/raw/batch", exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")
    output_path = f"data/raw/batch/gempa_{today}.csv"

    if os.path.exists(output_path):
        existing = pd.read_csv(output_path)
        df_all = pd.concat([existing, df_all], ignore_index=True)
        df_all = df_all.drop_duplicates(subset=["datetime"])

    df_all.to_csv(output_path, index=False)
    print(f"\nData disimpan ke: {output_path}")

if __name__ == "__main__":
    ingest_data()