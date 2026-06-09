# 🌊 MLOps-Gempa-JawaTimur
### Sistem Klasifikasi Potensi Tsunami Otomatis Berbasis Data Seismik Real-time BMKG

> Pipeline MLOps production-ready yang secara otomatis mengambil data gempa bumi terbaru dari API resmi BMKG, melatih ulang model klasifikasi secara berkala, dan menghasilkan prediksi apakah suatu gempa berpotensi menyebabkan tsunami — dalam milidetik.

[![GitHub Actions](https://github.com/bagrah/MLOps-Gempa-JawaTimur/actions/workflows/mlops-automation.yaml/badge.svg)](https://github.com/bagrah/MLOps-Gempa-JawaTimur/actions)

---

## 📋 Deskripsi Proyek

Indonesia adalah negara dengan risiko tsunami tertinggi di dunia. Sistem InaTEWS milik BMKG saat ini masih mengandalkan **pre-calculated database** dan simulasi numerik konvensional yang bersifat statis — tidak belajar otomatis dari data baru.

Projek ini hadir sebagai **komplemen** bagi InaTEWS dengan menghadirkan pipeline ML yang:
- Retrain otomatis setiap hari dari data gempa terbaru
- Adaptif terhadap perubahan pola seismik (data drift)
- Dapat diakses terbuka via REST API
- Termonitor real-time via Prometheus + Grafana

| Atribut | Detail |
|---|---|
| **Domain** | Kebencanaan / Seismologi |
| **ML Task** | Binary Classification |
| **Target** | Berpotensi Tsunami (1) / Tidak Berpotensi (0) |
| **Wilayah** | Seluruh Indonesia |
| **Sumber Data** | BMKG Open Data API (Real-time) |
| **Model** | Random Forest Classifier |
| **Tracking** | MLflow (SQLite backend) |
| **Automation** | GitHub Actions (trigger harian 08.00 WIB) |

---

## 🏗️ Arsitektur Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                   SUMBER DATA DINAMIS                       │
│         BMKG Open Data API (update real-time)               │
│    gempaterkini.json  │  gempadirasakan.json                 │
└──────────────────────┬──────────────────────────────────────┘
                       │ Pull harian (GitHub Actions cron)
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                   DATA INGESTION                            │
│  src/data/ingest_data.py                                    │
│  • Fetch gempa terbaru se-Indonesia                         │
│  • Parse label dari field "Potensi" BMKG                    │
│  • Simpan ke data/raw/batch/gempa_YYYY-MM-DD.csv            │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                DATA VERSIONING (DVC)                        │
│  • dvc add setiap batch baru                                │
│  • Hash-based versioning tanpa membebani Git                │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                   PREPROCESSING                             │
│  src/data/preprocess.py                                     │
│  • Merge semua batch CSV                                    │
│  • Drop duplikat, validasi tipe data                        │
│  • Output: data/processed/gempa_indonesia.csv               │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                  MODEL TRAINING                             │
│  src/models/train.py                                        │
│  • Random Forest Classifier (4 variasi n_estimators)        │
│  • class_weight=balanced (handle imbalanced data)           │
│  • Log params & metrics ke MLflow                           │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│             CONTINUOUS TRAINING (CT)                        │
│  scripts/check_model_performance.py                         │
│  • Cek accuracy ≥ 0.85 dan F1 ≥ 0.60                       │
│  scripts/compare_and_promote.py                             │
│  • Bandingkan model baru vs Production                      │
│  • Promosi otomatis jika model baru lebih baik              │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│            INFERENCE & SERVING (Docker)                     │
│  Flask API → Nginx → 3 replika api-service                  │
│  Prometheus → Grafana (monitoring real-time)                │
└─────────────────────────────────────────────────────────────┘
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
│
├── configs/
│   └── config.yaml                   # Konfigurasi project
│
├── data/
│   ├── raw/batch/
│   │   ├── gempa_seed_2023.csv       # Seed data historis (500 records)
│   │   ├── gempa_drift_simulation.csv # Data simulasi drift
│   │   └── gempa_YYYY-MM-DD.csv      # Data harian dari BMKG
│   └── processed/
│       └── gempa_indonesia.csv       # Data gabungan siap training
│
├── models/
│   ├── rf_n50.pkl                    # Model n_estimators=50
│   ├── rf_n100.pkl                   # Model n_estimators=100
│   ├── rf_n200.pkl                   # Model n_estimators=200
│   └── rf_n300.pkl                   # Model n_estimators=300
│
├── scripts/
│   ├── run_pipeline.py               # Jalankan full pipeline
│   ├── generate_seed_data.py         # Generate seed data historis
│   ├── register_model.py             # Register model ke MLflow Registry
│   ├── check_model_performance.py    # Cek threshold performa model
│   ├── compare_and_promote.py        # Bandingkan & promosi model
│   └── simulate_drift.py             # Simulasi data drift
│
├── src/
│   ├── data/
│   │   ├── ingest_data.py            # Fetch data dari BMKG API
│   │   └── preprocess.py             # Cleaning & transformasi data
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
├── docker-compose.yaml               # Orkestrasi 5 services
├── nginx.conf                        # Load balancer config
├── prometheus.yml                    # Prometheus scraping config
├── requirements.txt                  # Python dependencies
└── mlflow.db                         # MLflow SQLite tracking
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

Atau per tahap:

```bash
python -m src.data.ingest_data        # Fetch data BMKG
python -m src.data.preprocess         # Preprocessing
python -m src.models.train            # Training
python scripts/check_model_performance.py  # Cek performa
python scripts/compare_and_promote.py     # Promosi model
python -m src.models.evaluate_model   # Evaluasi
```

### 5. Jalankan Tests

```bash
pytest tests/ -v
```

### 6. Jalankan Semua Services (Docker)

```bash
docker compose up -d
docker compose ps
```

---

## 📊 Data & Fitur

### Sumber Data

| Endpoint | Deskripsi | Update |
|---|---|---|
| `gempaterkini.json` | 15 gempa M5.0+ terkini | Real-time |
| `gempadirasakan.json` | 15 gempa yang dirasakan | Real-time |

### Pembuatan Label

Label `potensi_tsunami` dibuat otomatis dari field `Potensi` di data BMKG:

```
"Berpotensi tsunami"       → label = 1
"Tidak berpotensi tsunami" → label = 0
```

### Fitur Model

| Fitur | Tipe | Penjelasan |
|---|---|---|
| `magnitude` | float | Kekuatan gempa (skala Richter) |
| `kedalaman_km` | int | Kedalaman pusat gempa — gempa dangkal lebih berisiko |
| `lintang` | float | Koordinat lintang |
| `bujur` | float | Koordinat bujur |
| `jam` | int | Jam kejadian gempa (0-23 WIB) |

---

## 🤖 Model & Performa

### Algoritma

**Random Forest Classifier** dengan:
- `class_weight = balanced` — menangani imbalanced data
- `stratify = y` pada train/test split
- 4 variasi `n_estimators`: 50, 100, 200, 300

### Hasil Eksperimen

| n_estimators | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| 50 | 0.9524 | 1.0000 | 0.4444 | 0.6154 |
| 100 | 0.9524 | 1.0000 | 0.4444 | 0.6154 |
| 200 | 0.9524 | 1.0000 | 0.8889 | 0.9412 |
| 300 | 0.9524 | 1.0000 | 0.8929 | 0.8929 |

### Confusion Matrix (Best Model)

```
              Prediksi
              0      1
Aktual  0  [ 484 |   0 ]
        1  [   5 |  38 ]

TN=484 | FP=0 | FN=5 | TP=38
Precision=100% — tidak ada false alarm!
```

### Threshold Keberhasilan

| Metrik | Threshold | Hasil | Status |
|---|---|---|---|
| Accuracy | ≥ 0.85 | 0.9524 | ✅ |
| F1 Score | ≥ 0.60 | 0.8929 | ✅ |

---

## 📦 DVC Data Versioning

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

---

## 🧪 MLflow Experiment Tracking

### Jalankan MLflow UI

```bash
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
```

Buka: `http://localhost:5001` — Experiment: `tsunami-indonesia-experiment`

### Model Registry

| Atribut | Detail |
|---|---|
| Nama | `TsunamiRisk-RandomForest` |
| Versi aktif | v2 |
| Stage | Production |

```bash
python scripts/register_model.py
```

---

## ⚙️ Continual Learning & Continuous Training

### Trigger Otomatis

**Trigger 1 — Schedule-based (LK-12 Skenario C)**
```yaml
schedule:
  - cron: '0 1 * * *'  # Setiap hari 08.00 WIB
```

**Trigger 2 — Performance-based (LK-12 Skenario A)**

Jika accuracy < 0.85 atau F1 < 0.60 → pipeline retrain otomatis:
```bash
python scripts/check_model_performance.py
```

### Evaluasi Komparatif Otomatis

Model baru hanya dipromosi ke Production jika F1 Score lebih tinggi:
```bash
python scripts/compare_and_promote.py
```

Contoh hasil simulasi drift:

| | Model Lama | Model Baru |
|---|---|---|
| Accuracy | 0.9528 | 0.9524 |
| F1 Score | 0.6154 | 0.8929 |
| Status | — | ✅ Dipromosi ke Production v2 |

### Simulasi Data Drift

```bash
python scripts/simulate_drift.py
```

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
7. Check model performance (threshold)
8. Compare & promote model
9. Run pytest (7 test cases)
10. Evaluate model terbaik
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

### Cek Status

```bash
docker compose ps
```

Output yang diharapkan:

```
NAME                                  SERVICE         STATUS
grafana                               grafana         Up
mlflow-server                         mlflow-server   Up
mlops-gempa-jawatimur-api-service-1   api-service     Up
mlops-gempa-jawatimur-api-service-2   api-service     Up
mlops-gempa-jawatimur-api-service-3   api-service     Up
nginx                                 nginx           Up
prometheus                            prometheus      Up
```

### Services

| Service | Port | Deskripsi |
|---|---|---|
| `api-service` (via Nginx) | 5000 | Flask REST API inferensi |
| `mlflow-server` | 5001 | MLflow UI tracking |
| `prometheus` | 9090 | Metrics collector |
| `grafana` | 3000 | Monitoring dashboard |

### Scaling Dinamis

```bash
docker compose up -d --scale api-service=5  # Scale up
docker compose up -d --scale api-service=1  # Scale down
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
  "model": "TsunamiRisk-RandomForest",
  "status": "ok"
}
```

### POST /predict

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "magnitude": 7.2,
    "kedalaman_km": 10,
    "lintang": -8.50,
    "bujur": 115.50,
    "jam": 14
  }'
```

Response:
```json
{
  "nilai": 1,
  "prediksi": "BERPOTENSI TSUNAMI"
}
```

### GET /metrics

```bash
curl http://localhost:5000/metrics
```

Mengembalikan metrik Prometheus: `prediction_requests_total`, `prediction_latency_seconds`, `prediction_probability`.

### Parameter Input /predict

| Parameter | Tipe | Contoh | Keterangan |
|---|---|---|---|
| `magnitude` | float | 7.2 | Kekuatan gempa |
| `kedalaman_km` | int | 10 | Kedalaman dalam km |
| `lintang` | float | -8.50 | Koordinat lintang |
| `bujur` | float | 115.50 | Koordinat bujur |
| `jam` | int | 14 | Jam kejadian (0-23) |

---

## 📈 Monitoring (Prometheus + Grafana)

### Akses Dashboard

- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000` (admin/admin)

### Setup Grafana

1. Connections → Data Sources → Add → Prometheus
2. URL: `http://prometheus:9090` → Save & Test
3. Dashboards → New → Add visualization
4. Query: `prediction_requests_total`

### Metrik yang Dipantau

| Metrik | Kegunaan |
|---|---|
| `prediction_requests_total` | Jumlah total request prediksi |
| `prediction_latency_seconds` | Latensi inferensi |
| `prediction_probability` | Distribusi confidence score — indikator model decay |

### Deteksi Model Decay

Kalau `prediction_probability` rata-rata turun drastis → model mulai tidak yakin → sinyal perlu retrain.

---

## 🔧 Konfigurasi

Edit `configs/config.yaml`:

```yaml
model:
  threshold_accuracy: 0.85  # Minimum accuracy
  threshold_f1: 0.60        # Minimum F1 Score

mlflow:
  experiment_name: tsunami-indonesia-experiment
  tracking_uri: sqlite:///mlflow.db
```

---

## 🗺️ Pemetaan LK

| LK | Topik | Implementasi |
|---|---|---|
| LK-01 | Inisiasi Proyek | Definisi domain, data dinamis, strategi CT, metrik sukses |
| LK-02 | Setup Infrastruktur | Repo GitHub, Codespaces, struktur direktori |
| LK-03 | Arsitektur Pipeline | Diagram ETL di README, rencana DVC |
| LK-04 | Data Ingestion | `ingest_data.py` + `preprocess.py` |
| LK-05 | DVC Versioning | `dvc init`, `dvc add`, versioning v1→v2 |
| LK-06 | MLflow Tracking | `train.py` + 4 eksperimen + MLflow UI |
| LK-07 | Model Registry | `register_model.py` + stage Production |
| LK-08 | GitHub Actions | `mlops-automation.yaml` + cron harian |
| LK-09 | Docker Compose | 5 services + network + volumes |
| LK-10 | Model Serving | Flask API + Nginx + 3 replika |
| LK-11 | Observability | Prometheus + Grafana dashboard |
| LK-12 | Continuous Training | check_performance + compare_promote + simulate_drift |

---

## 📚 Referensi

- [BMKG Open Data](https://data.bmkg.go.id/)
- [InaTEWS BMKG](https://inatews.bmkg.go.id/)
- [MLflow Documentation](https://mlflow.org/docs/latest/)
- [DVC Documentation](https://dvc.org/doc)
- [Prometheus](https://prometheus.io/docs/)
- [Grafana](https://grafana.com/docs/)

---

## 👤 Author

**Muhammad Bagas Anugrah**
Mata Kuliah: Machine Learning Operations