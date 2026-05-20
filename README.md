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
│                   DATA VERSIONING (DVC)                         │
│  • dvc add data/raw/batch/                                      │
│  • Track perubahan data setiap batch baru                       │
│  • Hash-based versioning tanpa membebani Git                    │
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
│  • Log params & metrics ke MLflow (SQLite)                      │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                  MODEL REGISTRY (MLflow)                        │
│  scripts/register_model.py                                      │
│  • Register model ke MLflow Model Registry                      │
│  • Transisi stage: None → Staging → Production                  │
│  • Model aktif: GempaJawaTimur-RandomForest v1                  │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                 INFERENCE & SERVING (Docker)                    │
│  app.py + docker-compose.yaml                                   │
│  • Flask REST API (port 5000)                                   │
│  • MLflow UI (port 5001)                                        │
│  • 3 replicas api-service untuk horizontal scaling              │
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
├── .dvc/                             # Konfigurasi DVC
│   └── config
│
├── configs/
│   └── config.yaml                   # Konfigurasi project
│
├── data/
│   ├── raw/
│   │   └── batch/
│   │       ├── gempa_seed_2023.csv      # Data historis awal (500 records)
│   │       ├── gempa_seed_2023.csv.dvc  # DVC tracking file
│   │       └── gempa_YYYY-MM-DD.csv     # Data harian dari BMKG
│   └── processed/
│       └── gempa_jatim.csv           # Data gabungan siap training
│
├── notebooks/
│   └── eda.ipynb                     # Exploratory Data Analysis
│
├── scripts/
│   ├── run_pipeline.py               # Jalankan full pipeline
│   ├── generate_seed_data.py         # Generate seed data historis
│   └── register_model.py            # Register model ke MLflow Registry
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
│   ├── test_basic.py                 # Test data pipeline (4 tests)
│   └── test_model.py                 # Test model & MLflow (3 tests)
│
├── app.py                            # Flask REST API server
├── Dockerfile                        # Container image
├── docker-compose.yaml               # Orkestrasi services + scaling
├── requirements.txt                  # Python dependencies
├── mlflow.db                         # MLflow SQLite tracking
├── .dvcignore                        # DVC ignore rules
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

# Step 5: Register model ke Registry
python scripts/register_model.py

# Step 6: Prediksi
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

### Hasil Eksperimen

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

## 📦 DVC Data Versioning

DVC digunakan untuk melacak perubahan dataset tanpa membebani Git.

### Track Dataset Baru

```bash
dvc add data/raw/batch/gempa_YYYY-MM-DD.csv
git add data/raw/batch/gempa_YYYY-MM-DD.csv.dvc
git commit -m "data: tambah batch gempa terbaru"
```

### Cek Perubahan Data

```bash
dvc diff
dvc status
```

### Alur Versioning

```
Batch baru masuk → dvc add → hash baru tersimpan di .dvc file
                           → git commit .dvc file
                           → riwayat data terlacak tanpa simpan file besar di Git
```

---

## 🧪 MLflow Experiment Tracking

### Jalankan MLflow UI

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
```

Buka browser: `http://localhost:5001` — Experiment: `gempa-jatim-experiment`

### Model Registry

Model terbaik terdaftar sebagai:
- **Nama:** `GempaJawaTimur-RandomForest`
- **Versi:** v1
- **Stage:** Production

Register ulang model terbaru:

```bash
python scripts/register_model.py
```

---

## ⚙️ Continual Learning Strategy

### Trigger Otomatis

```yaml
schedule:
  - cron: '0 1 * * *'  # Setiap hari 08.00 WIB
```

### Alur Harian

```
GitHub Actions trigger (cron)
  → ingest_data.py    : fetch gempa baru dari BMKG
  → preprocess.py     : merge & cleaning data
  → train.py          : retrain model + log ke MLflow
  → pytest            : validasi integritas sistem
  → evaluate_model.py : evaluasi model terbaru
```

### Potensi Data Drift

| Jenis Drift | Skenario |
|---|---|
| Feature Drift | Perubahan pola kedalaman gempa musiman |
| Label Drift | Perubahan threshold BMKG untuk "dirasakan" |
| Concept Drift | Pergeseran lempeng tektonik mengubah pola seismik |

---

## 🔄 GitHub Actions CI/CD

Pipeline otomatis pada setiap push, pull request, dan jadwal harian:

```
1. Checkout repository
2. Setup Python 3.10
3. Install dependencies
4. Ingest data dari BMKG API
5. Preprocessing data
6. Training model + MLflow tracking
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

## 🐳 Docker & Scaling

### Jalankan Semua Services

```bash
docker compose up -d
```

### Cek Status Services

```bash
docker compose ps
```

Output yang diharapkan:

```
NAME                                  SERVICE         STATUS
mlflow-server                         mlflow-server   Up
mlops-gempa-jawatimur-api-service-1   api-service     Up
mlops-gempa-jawatimur-api-service-2   api-service     Up
mlops-gempa-jawatimur-api-service-3   api-service     Up
```

### Services

| Service | Port | Deskripsi |
|---|---|---|
| `api-service` | 5000 | Flask REST API untuk inferensi |
| `mlflow-server` | 5001 | MLflow UI untuk tracking eksperimen |

### Scaling Replika Dinamis

Untuk mengubah jumlah replika secara dinamis tanpa restart penuh:

```bash
docker compose up -d --scale api-service=5  # Scale up ke 5 replika
docker compose up -d --scale api-service=1  # Scale down ke 1 replika
```

Atau edit `docker-compose.yaml` bagian `deploy`:

```yaml
api-service:
  deploy:
    replicas: 3  # Ubah angka ini sesuai kebutuhan
```

### Matikan Semua Services

```bash
docker compose down
```

---

## 🌐 API Endpoints

Base URL: `http://localhost:5000`

### GET /health

```bash
curl http://localhost:5000/health
```

Response:

```json
{
  "model": "GempaJawaTimur-RandomForest",
  "status": "ok"
}
```

### POST /predict

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "magnitude": 4.5,
    "kedalaman_km": 15,
    "lintang": -7.98,
    "bujur": 112.63,
    "jam": 14
  }'
```

Response:

```json
{
  "nilai": 1,
  "prediksi": "DIRASAKAN"
}
```

### Parameter Input

| Parameter | Tipe | Contoh | Keterangan |
|---|---|---|---|
| `magnitude` | float | 4.5 | Kekuatan gempa |
| `kedalaman_km` | int | 15 | Kedalaman dalam km |
| `lintang` | float | -7.98 | Koordinat lintang (negatif = LS) |
| `bujur` | float | 112.63 | Koordinat bujur |
| `jam` | int | 14 | Jam kejadian (0-23) |

---

## 🔧 Konfigurasi

Edit `configs/config.yaml`:

```yaml
data:
  filter:
    lat_min: -9.0
    lat_max: -6.5
    lon_min: 110.0
    lon_max: 116.0
model:
  threshold_accuracy: 0.60
mlflow:
  tracking_uri: sqlite:///mlflow.db
```

---

## 📚 Referensi

- [BMKG Open Data](https://data.bmkg.go.id/)
- [MLflow Documentation](https://mlflow.org/docs/latest/)
- [DVC Documentation](https://dvc.org/doc)
- [Scikit-learn RandomForest](https://scikit-learn.org/stable/modules/ensemble.html)
- [GitHub Actions](https://docs.github.com/en/actions)

---

## 👤 Author

**Muhammad Bagas Anugrah**
Mata Kuliah: Machine Learning Operations