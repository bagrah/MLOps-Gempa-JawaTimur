import requests
import pandas as pd
import os
from datetime import datetime

# Koordinat bounding box Jawa Timur
LAT_MIN, LAT_MAX = -9.0, -6.5
LON_MIN, LON_MAX = 110.0, 116.0

ENDPOINTS = {
    "terkini": "https://data.bmkg.go.id/DataMKG/TEWS/gempaterkini.json",
    "dirasakan": "https://data.bmkg.go.id/DataMKG/TEWS/gempadirasakan.json",
}

def parse_koordinat(lintang_str, bujur_str):
    """Parse string Lintang/Bujur BMKG ke float."""
    try:
        lat = float(lintang_str.split()[0])
        if "LS" in lintang_str:
            lat = -lat
        lon = float(bujur_str.split()[0])
        return lat, lon
    except:
        return None, None

def fetch_gempa(endpoint_url):
    """Fetch data gempa dari endpoint BMKG."""
    try:
        response = requests.get(endpoint_url, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data["Infogempa"]["gempa"]
    except Exception as e:
        print(f"Error fetching {endpoint_url}: {e}")
        return []

def parse_gempa(gempa_list, label_dirasakan=None):
    """Parse list gempa BMKG ke DataFrame."""
    records = []
    for g in gempa_list:
        lat, lon = parse_koordinat(g.get("Lintang", ""), g.get("Bujur", ""))
        if lat is None:
            continue

        # Filter Jawa Timur
        if not (LAT_MIN <= lat <= LAT_MAX and LON_MIN <= lon <= LON_MAX):
            continue

        try:
            jam = int(g.get("Jam", "00:00:00 WIB").split(":")[0])
            kedalaman = int(g.get("Kedalaman", "0 km").split()[0])
            magnitude = float(g.get("Magnitude", 0))
        except:
            continue

        record = {
            "datetime": g.get("DateTime", ""),
            "tanggal": g.get("Tanggal", ""),
            "jam": jam,
            "magnitude": magnitude,
            "kedalaman_km": kedalaman,
            "lintang": lat,
            "bujur": lon,
            "wilayah": g.get("Wilayah", ""),
            "dirasakan": label_dirasakan if label_dirasakan is not None else (1 if g.get("Dirasakan", "") else 0),
        }
        records.append(record)
    return pd.DataFrame(records)

def ingest_data():
    print("Fetching data dari BMKG...")

    # Fetch kedua endpoint
    gempa_terkini = fetch_gempa(ENDPOINTS["terkini"])
    gempa_dirasakan = fetch_gempa(ENDPOINTS["dirasakan"])

    print(f"  gempaterkini  : {len(gempa_terkini)} gempa")
    print(f"  gempadirasakan: {len(gempa_dirasakan)} gempa")

    # Parse masing-masing
    df_terkini = parse_gempa(gempa_terkini, label_dirasakan=0)
    df_dirasakan = parse_gempa(gempa_dirasakan)  # label dari field Dirasakan

    # Gabungkan, hapus duplikat berdasarkan datetime
    df_all = pd.concat([df_terkini, df_dirasakan], ignore_index=True)
    df_all = df_all.drop_duplicates(subset=["datetime"])

    # Gempa yang ada di dirasakan → override label jadi 1
    datetime_dirasakan = set(df_dirasakan["datetime"].tolist())
    df_all["dirasakan"] = df_all["datetime"].apply(
        lambda x: 1 if x in datetime_dirasakan else 0
    )

    print(f"\nTotal gempa Jawa Timur hari ini: {len(df_all)}")
    if df_all.empty:
        print("Tidak ada gempa baru di Jawa Timur. Skip.")
        return

    print(df_all[["tanggal", "magnitude", "kedalaman_km", "lintang", "bujur", "dirasakan"]])

    # Simpan dengan nama file tanggal hari ini
    os.makedirs("data/raw/batch", exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")
    output_path = f"data/raw/batch/gempa_{today}.csv"

    # Append ke file historis jika sudah ada, buat baru jika belum
    if os.path.exists(output_path):
        existing = pd.read_csv(output_path)
        df_all = pd.concat([existing, df_all], ignore_index=True)
        df_all = df_all.drop_duplicates(subset=["datetime"])

    df_all.to_csv(output_path, index=False)
    print(f"\nData disimpan ke: {output_path}")

if __name__ == "__main__":
    ingest_data()
