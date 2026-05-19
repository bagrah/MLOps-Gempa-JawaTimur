# 🌋 MLOps-Gempa-JawaTimur

> **End-to-end MLOps pipeline** untuk prediksi gempa bumi yang dirasakan di wilayah Jawa Timur menggunakan data real-time dari BMKG (Badan Meteorologi, Klimatologi, dan Geofisika).

---

## 📋 Deskripsi Proyek

Proyek ini membangun sistem Machine Learning berbasis **MLOps production-ready** yang secara otomatis mengambil data gempa bumi terbaru dari API resmi BMKG, melatih ulang model klasifikasi secara berkala, dan menghasilkan prediksi apakah suatu gempa akan **dirasakan** atau **tidak dirasakan** oleh masyarakat di Jawa Timur.

| Atribut | Detail |
|---|---|
| **Domain** | Seismologi / Kebencanaan |
| **ML Task** | Binary Classification |
| **Target** | Gempa Dirasakan (1) / Tidak Dirasakan (0) |
| **Wilayah** | Jawa Timur (Lat: -9.0 s/d -6.5, Lon: 110.0 s/d 116.0) |
| **Sumber Data** | BMKG Open Data API (Real-time) |
| **Model** | Random Forest Classifier |
| **Tracking** | MLflow (SQLite backend) |
| **Automation** | GitHub Actions (trigger harian 08.00 WIB) |

---

## 🏗️ Arsitektur Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                     SUMBER DATA DINAMIS                         │
│         BMKG Open Data API (update real-time)                   │
│  gempaterkini.json │ gempadirasakan.json │ autogempa.json        │
└──────────────────────────┬──────────────────────────────────────┘
                           │ Pull harian (GitHub Actions cron)
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                     DATA INGESTION                              │
│  src/data/ingest_data.py                                        │
│  • Fetch 3 endpoint BMKG                                        │
│  • Filter koordinat Jawa Timur                                  │
│  • Parse & labeling otomatis (dirasakan: 0/1)                   │
│  • Simpan ke data/raw/batch/gempa_YYYY-MM-DD.csv                │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                     PREPROCESSING                               │
│  src/data/preprocess.py                                         │
│  • Merge semua batch CSV                                        │
│  • Drop duplicates berdasarkan datetime                         │
│  • Validasi tipe data & missing values                          │
│  • Output: data/processed/gempa_jatim.csv                       │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                     MODEL TRAINING                              │
│  src/models/train.py                                            │
│  • Random Forest Classifier                                     │
│  • Stratified train/test split (80/20)                          │
│  • Handle imbalanced class (class_weight=balanced)              │
│  • Log params & metrics ke MLflow                               │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                     EVALUATION & REGISTRY                       │
│  src/models/evaluate_model.py                                   │
│  • Ambil best run dari MLflow berdasarkan F1 Score              │
│  • Classification report & confusion matrix                     │
│  • Threshold F1 >= 0.40                                         │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                     INFERENCE                                   │
│  src/inference/predict.py                                       │
│  • Load best model dari MLflow                                  │
│  • Prediksi: DIRASAKAN / TIDAK DIRASAKAN                        │
│  • Output confidence score (%)                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📁 Struktur Direktori

```
MLOps-Gempa-JawaTimur/
│
├── .github/
│   └── workflows/
│       └── mlops-automation.yaml     # CI/CD pipeline otomatis
│
├── configs/
│   └── config.yaml                   # Konfigurasi project
│
├── data/
│   ├── raw/
│   │   └── batch/
│   │       ├── gempa_seed_2023.csv   # Data historis awal (500 records)
│   │       └── gempa_YYYY-MM-DD.csv  # Data harian dari BMKG
│   └── processed/
│       └── gempa_jatim.csv           # Data gabungan siap training
│
├── notebooks/
│   └── eda.ipynb                     # Exploratory Data Analysis
│
├── scripts/
│   ├── run_pipeline.py               # Jalankan full pipeline
│   └── generate_seed_data.py         # Generate seed data historis
│
├── src/
│   ├── data/
│   │   ├── ingest_data.py            # Fetch data dari BMKG API
│   │   └── preprocess.py             # Cleaning & transformasi data
│   ├── features/
│   │   └── build_features.py         # Feature engineering
│   ├── models/
│   │   ├── train.py                  # Training & MLflow logging
│   │   └── evaluate_model.py         # Evaluasi model terbaik
│   └── inference/
│       └── predict.py                # Inference / prediksi
│
├── tests/
│   ├── test_basic.py                 # Test data pipeline
│   └── test_model.py                 # Test model & MLflow
│
├── Dockerfile                        # Container image
├── docker-compose.yaml               # Orkestrasi services
├── requirements.txt                  # Python dependencies
├── mlflow.db                         # MLflow SQLite tracking
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/bagrah/MLOps-Gempa-JawaTimur.git
cd MLOps-Gempa-JawaTimur
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Generate Seed Data (pertama kali)

```bash
python scripts/generate_seed_data.py
```

### 4. Jalankan Full Pipeline

```bash
# Jalankan semua tahap sekaligus
export PYTHONPATH=$PYTHONPATH:$(pwd)
python scripts/run_pipeline.py
```

Atau jalankan per tahap:

```bash
# Step 1: Ingest data dari BMKG
python -m src.data.ingest_data

# Step 2: Preprocessing
python -m src.data.preprocess

# Step 3: Training
python -m src.models.train

# Step 4: Evaluasi
python -m src.models.evaluate_model

# Step 5: Prediksi
python -m src.inference.predict
```

### 5. Jalankan Tests

```bash
pytest tests/ -v
```

---

## 📊 Data & Fitur

### Sumber Data

Data diambil secara real-time dari **3 endpoint BMKG**:

| Endpoint | Deskripsi | Update |
|---|---|---|
| `gempaterkini.json` | 15 gempa M5.0+ terkini | Real-time |
| `gempadirasakan.json` | 15 gempa yang dirasakan | Real-time |
| `autogempa.json` | Gempa terbaru | Real-time |

### Filter Wilayah Jawa Timur

```
Lintang : -9.0 s/d -6.5 (LS)
Bujur   : 110.0 s/d 116.0 (BT)
```

### Fitur Model

| Fitur | Tipe | Deskripsi |
|---|---|---|
| `magnitude` | float | Kekuatan gempa (skala Richter) |
| `kedalaman_km` | int | Kedalaman pusat gempa (km) |
| `lintang` | float | Koordinat lintang (negatif = LS) |
| `bujur` | float | Koordinat bujur |
| `jam` | int | Jam kejadian gempa (0-23 WIB) |

### Label (Target)

| Label | Nilai | Keterangan |
|---|---|---|
| Tidak Dirasakan | 0 | Gempa tidak terasa di permukaan |
| Dirasakan | 1 | Gempa dirasakan oleh masyarakat |

---

## 🤖 Model & Performa

### Algoritma

**Random Forest Classifier** dengan konfigurasi:
- `class_weight = balanced` (menangani imbalanced data)
- `stratify = y` pada train/test split
- Eksperimen dengan `n_estimators`: 50, 100, 200, 300

### Hasil Eksperimen (Data Saat Ini)

| n_estimators | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| 50 | 0.9208 | 0.8824 | 0.7143 | 0.7895 |
| 100 | 0.9208 | 0.8824 | 0.7143 | 0.7895 |
| 200 | 0.9208 | 0.8824 | 0.7143 | 0.7895 |
| 300 | 0.9208 | 0.8824 | 0.7143 | 0.7895 |

### Confusion Matrix (Best Model)

```
              Prediksi
              0      1
Aktual  0  [ 397 |   2 ]
        1  [   6 |  96 ]

TN=397 | FP=2 | FN=6 | TP=96
```

### Threshold Keberhasilan

| Metrik | Threshold | Status |
|---|---|---|
| F1 Score | ≥ 0.40 | ✅ 0.7895 |
| Accuracy | ≥ 0.60 | ✅ 0.9208 |

---

## ⚙️ Continual Learning Strategy

Proyek ini dirancang untuk **Continuous Training (CT)** — model dilatih ulang secara otomatis setiap ada data baru:

### Trigger Otomatis

```yaml
schedule:
  - cron: '0 1 * * *'  # Setiap hari 08.00 WIB
```

### Alur Continual Learning

```
Hari ke-N:
  1. GitHub Actions trigger (cron harian)
  2. ingest_data.py → fetch gempa baru dari BMKG
  3. Data baru di-append ke batch CSV (tidak menimpa data lama)
  4. preprocess.py → merge semua batch
  5. train.py → retrain model dengan data terbaru
  6. MLflow log run baru → bandingkan dengan run sebelumnya
  7. Model terbaik tersedia untuk inference
```

### Potensi Data Drift

| Jenis Drift | Contoh Skenario |
|---|---|
| **Feature Drift** | Perubahan pola kedalaman gempa musiman |
| **Label Drift** | Perubahan threshold BMKG untuk gempa "dirasakan" |
| **Concept Drift** | Pergeseran lempeng tektonik mengubah pola seismik |

---

## 🔄 GitHub Actions CI/CD

Pipeline otomatis berjalan pada setiap:
- **Push** ke branch `main`
- **Pull Request** ke branch `main`
- **Jadwal harian** jam 08.00 WIB

### Tahapan Pipeline

```
1. Checkout repository
2. Setup Python 3.10
3. Install dependencies
4. Ingest data dari BMKG API
5. Preprocessing data
6. Training model (+ MLflow tracking)
7. Run pytest (7 test cases)
8. Evaluate model terbaik
```

---

## 🧪 Testing

```bash
pytest tests/ -v
```

| Test | File | Deskripsi |
|---|---|---|
| `test_processed_data_exists` | test_basic.py | File processed ada |
| `test_processed_data_not_empty` | test_basic.py | Data tidak kosong |
| `test_processed_data_columns` | test_basic.py | Kolom lengkap |
| `test_label_binary` | test_basic.py | Label hanya 0 atau 1 |
| `test_mlflow_runs_exist` | test_model.py | Ada run di MLflow |
| `test_model_predict` | test_model.py | Model bisa prediksi |
| `test_model_accuracy_threshold` | test_model.py | Accuracy ≥ 60% |

---

## 🐳 Docker

### Build & Run

```bash
# Build image
docker build -t mlops-gempa-jatim .

# Run dengan docker-compose
docker-compose up
```

### Services

| Service | Port | Deskripsi |
|---|---|---|
| `mlflow-server` | 5001 | MLflow UI untuk tracking eksperimen |
| `api-service` | 5000 | Inference service |

---

## 📈 MLflow Tracking

Setelah training, buka MLflow UI:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
```

Buka browser: `http://localhost:5001`

Experiment: `gempa-jatim-experiment`

---

## 🔧 Konfigurasi

Edit `configs/config.yaml` untuk menyesuaikan:

```yaml
data:
  filter:
    lat_min: -9.0   # Ubah untuk memperluas/mempersempit wilayah
    lat_max: -6.5
    lon_min: 110.0
    lon_max: 116.0

model:
  threshold_accuracy: 0.60  # Threshold minimum accuracy

mlflow:
  tracking_uri: sqlite:///mlflow.db
```

---

## 📚 Referensi

- [BMKG Open Data](https://data.bmkg.go.id/) — Sumber data gempa bumi resmi Indonesia
- [MLflow Documentation](https://mlflow.org/docs/latest/) — ML experiment tracking
- [Scikit-learn RandomForest](https://scikit-learn.org/stable/modules/ensemble.html#forest) — Dokumentasi model
- [GitHub Actions](https://docs.github.com/en/actions) — CI/CD automation

---

## 👤 Author

**Muhammad Bagas Anugrah**

Mata Kuliah: Machine Learning Operations