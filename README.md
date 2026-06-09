# 🌊 MLOps-Gempa-JawaTimur
### Sistem Klasifikasi Potensi Tsunami Otomatis Berbasis Data Seismik Real-time BMKG

> Pipeline MLOps production-ready yang secara otomatis mengambil data gempa bumi terbaru dari API resmi BMKG, melatih ulang model klasifikasi secara berkala, dan menghasilkan prediksi apakah suatu gempa berpotensi menyebabkan tsunami — dalam milidetik.

[![GitHub Actions](https://github.com/bagrah/MLOps-Gempa-JawaTimur/actions/workflows/mlops-automation.yaml/badge.svg)](https://github.com/bagrah/MLOps-Gempa-JawaTimur/actions)

---

## 🧭 Tentang Projek Ini

Indonesia adalah negara dengan risiko tsunami tertinggi di dunia. Berada di pertemuan tiga lempeng tektonik besar — Eurasia, Indo-Australia, dan Pasifik — Indonesia mengalami ratusan gempa setiap harinya. Tidak semua gempa berpotensi tsunami, namun keterlambatan dalam mengklasifikasikan potensi tersebut bisa berakibat fatal.

Sistem InaTEWS (Indonesia Tsunami Early Warning System) milik BMKG sudah ada dan sangat canggih dalam hal infrastruktur sensor fisik. Namun InaTEWS masih mengandalkan **pre-calculated database** — ribuan skenario yang dibuat manual oleh tim ahli dan bersifat statis. Sistem ini tidak belajar otomatis dari data baru.

Projek ini hadir sebagai **komplemen InaTEWS**, menghadirkan pipeline Machine Learning yang:

- Belajar otomatis dari data gempa terbaru setiap hari
- Adaptif terhadap perubahan pola seismik (data drift)
- Dapat diakses terbuka via REST API oleh siapapun
- Termonitor real-time via dashboard Grafana

**InaTEWS unggul di infrastruktur sensor fisik. Projek ini unggul di kecerdasan adaptif dan otomatisasi pipeline.**

---

## 🎯 Apa yang Diprediksi?

Sistem ini menjawab satu pertanyaan:

> **"Dari data gempa yang baru saja terdeteksi sensor, apakah gempa ini berpotensi menyebabkan tsunami?"**

- **Input:** 5 karakteristik gempa (magnitude, kedalaman, koordinat, jam)
- **Output:** `BERPOTENSI TSUNAMI` atau `TIDAK BERPOTENSI TSUNAMI`

Prediksi terjadi **sesaat setelah sensor mendeteksi gempa** — sebelum getaran sampai ke permukaan dan sebelum analisis manual BMKG selesai. Di jeda waktu itulah sistem ini bekerja.

---

## 📊 Fitur & Label

### Fitur Input Model

| Fitur | Tipe | Penjelasan | Kenapa Penting |
|---|---|---|---|
| `magnitude` | float | Kekuatan gempa (skala Richter) | Gempa M6.5+ lebih berisiko tsunami |
| `kedalaman_km` | int | Kedalaman pusat gempa | Gempa dangkal (<70km) langsung menggeser dasar laut |
| `lintang` | float | Koordinat lintang | Posisi — laut vs darat, zona subduksi |
| `bujur` | float | Koordinat bujur | Posisi geografis spesifik |
| `jam` | int | Jam kejadian (0-23 WIB) | Pola temporal kejadian gempa |

### Label (Target)

Label dibuat **otomatis** dari field `Potensi` di data BMKG:

```
"Berpotensi tsunami"       → potensi_tsunami = 1
"Tidak berpotensi tsunami" → potensi_tsunami = 0
```

Tidak perlu labeling manual — BMKG sudah menyertakan informasi ini di setiap data gempa.

---

## 🏗️ Arsitektur Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                   SUMBER DATA DINAMIS                       │
│         BMKG Open Data API (update real-time)               │
│    gempaterkini.json  |  gempadirasakan.json                 │
└──────────────────────┬──────────────────────────────────────┘
                       │ Pull harian (GitHub Actions cron)
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                   DATA INGESTION                            │
│  src/data/ingest_data.py                                    │
│  - Fetch gempa terbaru se-Indonesia                         │
│  - Parse label dari field "Potensi" BMKG                    │
│  - Simpan ke data/raw/batch/gempa_YYYY-MM-DD.csv            │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                DATA VERSIONING (DVC)                        │
│  - dvc add setiap batch baru                                │
│  - Hash-based versioning tanpa membebani Git                │
│  - Riwayat perubahan data terlacak                          │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                   PREPROCESSING                             │
│  src/data/preprocess.py                                     │
│  - Merge semua batch CSV menjadi satu                       │
│  - Drop duplikat berdasarkan datetime                       │
│  - Validasi tipe data                                       │
│  - Output: data/processed/gempa_indonesia.csv               │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                  MODEL TRAINING                             │
│  src/models/train.py                                        │
│  - Random Forest Classifier (4 variasi n_estimators)        │
│  - class_weight=balanced (handle imbalanced data)           │
│  - Semua eksperimen dicatat ke MLflow                       │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│             CONTINUOUS TRAINING (CT)                        │
│  scripts/check_model_performance.py                         │
│  - Cek accuracy >= 0.85 dan F1 >= 0.60                      │
│  scripts/compare_and_promote.py                             │
│  - Bandingkan model baru vs Production lama                 │
│  - Promosi otomatis jika model baru lebih baik              │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│            INFERENCE & SERVING (Docker)                     │
│  Flask API -> Nginx -> 3 replika api-service                │
│  Prometheus scraping /metrics setiap 15 detik               │
│  Grafana dashboard monitoring real-time                     │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 Struktur Direktori

```
MLOps-Gempa-JawaTimur/
├── .github/workflows/
│   └── mlops-automation.yaml     # CI/CD pipeline (LK-08)
├── .dvc/                         # Konfigurasi DVC (LK-05)
├── configs/
│   └── config.yaml               # Konfigurasi project (LK-01)
├── data/
│   ├── raw/batch/                # Data harian dari BMKG (LK-04)
│   │   ├── gempa_seed_2023.csv   # Seed data historis 500 records
│   │   ├── gempa_drift_simulation.csv  # Simulasi drift (LK-12)
│   │   └── gempa_YYYY-MM-DD.csv  # Data harian
│   └── processed/
│       └── gempa_indonesia.csv   # Data siap training (LK-04)
├── models/                       # Model pickle (LK-06)
│   ├── rf_n50.pkl
│   ├── rf_n100.pkl
│   ├── rf_n200.pkl
│   └── rf_n300.pkl
├── scripts/
│   ├── run_pipeline.py           # Full pipeline
│   ├── generate_seed_data.py     # Seed data historis
│   ├── register_model.py         # Model Registry (LK-07)
│   ├── check_model_performance.py # Cek threshold (LK-12)
│   ├── compare_and_promote.py    # Promosi model (LK-12)
│   └── simulate_drift.py         # Simulasi drift (LK-12)
├── src/
│   ├── data/
│   │   ├── ingest_data.py        # Fetch BMKG API (LK-04)
│   │   └── preprocess.py         # Cleaning data (LK-04)
│   ├── models/
│   │   ├── train.py              # Training + MLflow (LK-06)
│   │   └── evaluate_model.py     # Evaluasi model (LK-06)
│   └── inference/
│       └── predict.py            # Inference (LK-07)
├── tests/
│   ├── test_basic.py             # Test data 4 cases (LK-08)
│   └── test_model.py             # Test model 3 cases (LK-08)
├── app.py                        # Flask REST API (LK-10)
├── Dockerfile                    # Container image (LK-09)
├── docker-compose.yaml           # 5 services (LK-09)
├── nginx.conf                    # Load balancer (LK-10)
├── prometheus.yml                # Monitoring config (LK-11)
├── requirements.txt              # Python dependencies
└── mlflow.db                     # MLflow tracking (LK-06)
```

---

## 🗺️ Perjalanan Projek — LK-01 sampai LK-12

### [LK-01] Inisiasi Proyek

**Narasi:** Mendefinisikan seluruh fondasi projek — masalah apa yang diselesaikan, data apa yang dipakai, bagaimana sistem akan belajar terus, dan metrik apa yang menentukan keberhasilan.

**Domain & Masalah:** Indonesia mengalami ratusan gempa setiap hari. Tidak semua berpotensi tsunami, tapi keterlambatan klasifikasi bisa fatal. Projek ini membangun sistem klasifikasi otomatis yang memprediksi potensi tsunami dari karakteristik gempa.

**ML Task:** Binary Classification — output hanya dua: `1 (BERPOTENSI)` atau `0 (TIDAK BERPOTENSI)`

**Kenapa data ini dinamis:** Gempa terjadi setiap hari — data selalu bertambah secara alami. BMKG update API real-time setiap ada gempa baru. Tidak ada hari tanpa gempa di Indonesia.

**Strategi Continual Learning:** Model dilatih ulang otomatis setiap hari jam 08.00 WIB via GitHub Actions. Ada dua trigger: schedule harian dan performance-based (kalau akurasi turun di bawah threshold).

**Kriteria Keberhasilan:**

| Metrik | Threshold | Alasan |
|---|---|---|
| Accuracy | >= 0.85 | Overall correctness |
| F1 Score | >= 0.60 | Penting karena data imbalanced |
| Recall | >= 0.80 | Miss lebih berbahaya dari false alarm |
| Latensi API | < 200ms | Sistem peringatan harus cepat |

**File teknis:** `configs/config.yaml` dan `.github/workflows/mlops-automation.yaml`

---

### [LK-02] Setup Infrastruktur

**Narasi:** Membangun fondasi teknis — repositori GitHub, lingkungan pengembangan Codespaces, dan struktur direktori standar industri agar projek reproducible di manapun.

**Yang dihasilkan:**
- Repo GitHub publik: `github.com/bagrah/MLOps-Gempa-JawaTimur`
- GitHub Codespaces dikonfigurasi — environment konsisten
- Struktur direktori Cookiecutter Data Science: `src/`, `data/`, `configs/`, `scripts/`, `tests/`
- README awal sebagai dokumentasi projek

---

### [LK-03] Perancangan Arsitektur Pipeline

**Narasi:** Merancang secara teknis bagaimana data akan mengalir dalam sistem dari sumber eksternal BMKG sampai siap training.

**Rancangan ETL:**
- **Extract:** `ingest_data.py` hit API BMKG, ambil JSON gempa terbaru
- **Transform:** parse string ke numerik, buat label dari field `Potensi`, filter duplikat
- **Load:** simpan ke `data/raw/batch/gempa_YYYY-MM-DD.csv`

**Rencana versioning:** DVC akan melacak setiap perubahan dataset tanpa menyimpan file besar di Git.

---

### [LK-04] Implementasi Data Ingestion

**Narasi:** Mengimplementasikan dua script utama pengolahan data yang berjalan otomatis setiap hari.

**`src/data/ingest_data.py`:**
```python
# Fetch dari API BMKG
response = requests.get("https://data.bmkg.go.id/DataMKG/TEWS/gempaterkini.json")

# Parse dan buat label otomatis
potensi = g.get("Potensi", "")
tsunami = 1 if "berpotensi tsunami" in potensi.lower()
              and "tidak" not in potensi.lower() else 0

# Simpan per tanggal — tidak menimpa data lama
output_path = f"data/raw/batch/gempa_{today}.csv"
```

**`src/data/preprocess.py`:**
```python
# Gabungkan semua batch
batch_files = glob.glob("data/raw/batch/gempa_*.csv")
df = pd.concat([pd.read_csv(f) for f in batch_files])

# Bersihkan
df = df.drop_duplicates(subset=["datetime"])
df = df.dropna(subset=features)
df.to_csv("data/processed/gempa_indonesia.csv")
```

**Poin penting:** Data lama tidak pernah dihapus — setiap hari diakumulasi. Inilah yang membuat data benar-benar bergerak dan bertambah.

---

### [LK-05] DVC Data Versioning

**Narasi:** Mengintegrasikan DVC untuk melacak perubahan dataset tanpa menyimpan file besar di Git.

**Masalah yang diselesaikan:** Git dirancang untuk kode, bukan data. File CSV besar tidak boleh di Git. DVC menyimpan hanya hash pointer ke file datanya.

```bash
# Track dataset
dvc add data/raw/batch/gempa_seed_2023.csv
# Membuat gempa_seed_2023.csv.dvc (berisi hash, bukan data)

# Lihat perubahan antar versi
dvc diff
```

**Simulasi versioning:**
- v1: gempa_seed_2023.csv → hash: abc123
- (tambah data baru)
- v2: gempa_seed_2023.csv → hash: def456 — `dvc diff` menunjukkan perubahan

---

### [LK-06] MLflow Experiment Tracking

**Narasi:** Mengintegrasikan MLflow untuk mencatat semua eksperimen training secara otomatis — parameter, metrik, dan model tersimpan rapi dan bisa dibandingkan kapanpun.

**Kenapa MLflow penting:** Tanpa MLflow, kalau training dijalankan berkali-kali dengan parameter berbeda, tidak ada catatan mana yang terbaik. MLflow otomatis menyimpan semua eksperimen.

**Implementasi di `src/models/train.py`:**
```python
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("tsunami-indonesia-experiment")

with mlflow.start_run():
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
```

**4 Eksperimen:**

| Run | n_estimators | Accuracy | F1 Score |
|---|---|---|---|
| 1 | 50 | 0.9524 | 0.6154 |
| 2 | 100 | 0.9524 | 0.6154 |
| 3 | 200 | 0.9524 | 0.9412 |
| 4 | 300 | 0.9524 | 0.8929 |

**Akses MLflow UI:**
```bash
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
# Buka: http://localhost:5001
# Experiment: tsunami-indonesia-experiment
```

---

### [LK-07] Model Registry

**Narasi:** Mengelola siklus hidup model dari eksperimen ke production menggunakan MLflow Model Registry.

**Alur stage:** `None → Staging → Production`

**`scripts/register_model.py`:**
```python
# Register model terbaik
mv = mlflow.register_model(model_uri, "TsunamiRisk-RandomForest")

# Transisi ke Production
client.transition_model_version_stage(
    name="TsunamiRisk-RandomForest",
    version=mv.version,
    stage="Production"
)
```

**Model aktif:** `TsunamiRisk-RandomForest` v2 — Production

---

### [LK-08] GitHub Actions Automation

**Narasi:** Membangun CI/CD yang otomatis menjalankan seluruh pipeline setiap ada perubahan kode atau sesuai jadwal harian. Prinsip: "Code as a Trigger".

**`.github/workflows/mlops-automation.yaml`:**
```yaml
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
  schedule:
    - cron: '0 1 * * *'  # Setiap hari 08.00 WIB
```

**Tahapan pipeline:**
1. Checkout repository
2. Setup Python 3.10
3. Install dependencies
4. Ingest data dari BMKG
5. Preprocessing
6. Training (4 variasi)
7. Check model performance
8. Compare & promote
9. Pytest (7 test cases)
10. Evaluate model

**7 Test Cases:**

| Test | File | Yang Diuji |
|---|---|---|
| `test_processed_data_exists` | test_basic.py | File processed ada |
| `test_processed_data_not_empty` | test_basic.py | Data tidak kosong |
| `test_processed_data_columns` | test_basic.py | Kolom lengkap |
| `test_label_binary` | test_basic.py | Label hanya 0 atau 1 |
| `test_mlflow_runs_exist` | test_model.py | Ada run di MLflow |
| `test_model_predict` | test_model.py | Model bisa prediksi |
| `test_model_accuracy_threshold` | test_model.py | Accuracy >= 60% |

---

### [LK-09] Docker Compose

**Narasi:** Membungkus seluruh sistem ke container Docker yang bisa dijalankan dengan satu perintah di manapun — portable dan reproducible.

**5 Services:**

| Service | Image | Port | Fungsi |
|---|---|---|---|
| `api-service` (3x) | custom build | — | Flask REST API inferensi |
| `nginx` | nginx:alpine | 5000 | Load balancer |
| `mlflow-server` | python:3.10-slim | 5001 | MLflow UI |
| `prometheus` | prom/prometheus | 9090 | Metrics collector |
| `grafana` | grafana/grafana | 3000 | Monitoring dashboard |

Semua dalam satu jaringan `mlops-network` — container saling mengenal via nama service.

```bash
docker compose up -d
docker compose ps
```

---

### [LK-10] Model Serving & Scaling

**Narasi:** Menyajikan model sebagai REST API yang bisa diakses publik, dengan simulasi horizontal scaling menggunakan 3 replika untuk handle beban tinggi.

**Flask API (`app.py`):**
```python
@app.route("/predict", methods=["POST"])
def predict():
    pred = model.predict(input_df)[0]
    label = "BERPOTENSI TSUNAMI" if pred == 1 else "TIDAK BERPOTENSI TSUNAMI"
    return jsonify({"prediksi": label, "nilai": int(pred)})
```

**3 Replika + Nginx:**
```yaml
api-service:
  deploy:
    replicas: 3  # 3 instance berjalan bersamaan
```

Nginx mendistribusikan request ke 3 replika secara otomatis (round-robin).

**Contoh penggunaan:**
```bash
# Health check
curl http://localhost:5000/health
# {"model": "TsunamiRisk-RandomForest", "status": "ok"}

# Prediksi
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"magnitude": 7.2, "kedalaman_km": 10,
       "lintang": -8.50, "bujur": 115.50, "jam": 14}'
# {"nilai": 1, "prediksi": "BERPOTENSI TSUNAMI"}
```

---

### [LK-11] Observability & Dashboard

**Narasi:** Membangun sistem pemantauan real-time yang bisa mendeteksi kalau model mulai bermasalah tanpa harus cek manual — "CCTV untuk model ML".

**Arsitektur:** `Flask /metrics → Prometheus (tiap 15 detik) → Grafana`

**3 Metrik yang dipantau:**

| Metrik | Tipe | Kegunaan |
|---|---|---|
| `prediction_requests_total` | Counter | Distribusi BERPOTENSI vs TIDAK |
| `prediction_latency_seconds` | Histogram | Kecepatan respons model |
| `prediction_probability` | Histogram | Confidence — indikator model decay |

**Implementasi di `app.py`:**
```python
from prometheus_client import Counter, Histogram

REQUEST_COUNT = Counter("prediction_requests_total", "Total request", ["result"])
REQUEST_LATENCY = Histogram("prediction_latency_seconds", "Latensi")

# Di setiap predict:
REQUEST_COUNT.labels(result=label).inc()
REQUEST_LATENCY.observe(latency)
```

**Cara deteksi model decay:** Kalau `prediction_probability` rata-rata turun dari 90% ke 55% → model tidak yakin → perlu retrain.

**Setup Grafana:**
1. Buka `http://localhost:3000` (admin/admin)
2. Connections → Data Sources → Prometheus → URL: `http://prometheus:9090` → Save & Test
3. Dashboards → New → Add visualization → query `prediction_requests_total`

---

### [LK-12] Continuous Training

**Narasi:** Menutup siklus MLOps — sistem tidak hanya berjalan otomatis tapi juga memperbarui dirinya sendiri ketika performa model menurun. Inilah yang membedakan MLOps dari sekedar ML biasa.

**Skenario A — Performance-based (`check_model_performance.py`):**
```python
THRESHOLD_ACCURACY = 0.85
THRESHOLD_F1 = 0.60

if acc < THRESHOLD_ACCURACY or f1 < THRESHOLD_F1:
    print("PERFORMA DI BAWAH THRESHOLD! Trigger retrain...")
    sys.exit(1)  # GitHub Actions otomatis jalankan ulang pipeline
```

**Skenario C — Schedule-based:** Cron harian jam 08.00 WIB (sudah ada sejak LK-08).

**Evaluasi Komparatif Otomatis (`compare_and_promote.py`):**
```python
if new_f1 <= old_f1:
    print("Model baru tidak lebih baik. Tetap pakai model lama.")
    return

# Kalau lebih baik, promosi otomatis
client.transition_model_version_stage(
    name=MODEL_NAME, version=mv.version, stage="Production"
)
```

**Simulasi Data Drift — Hasil Nyata:**

| Kondisi | Model Lama | Model Baru (post-drift) |
|---|---|---|
| F1 Score | 0.6154 | 0.8929 |
| Status | Production v1 | Dipromosi ke Production v2 |

Sistem berhasil mendeteksi peningkatan dan melakukan promosi otomatis tanpa intervensi manusia.

---

## 🤖 Performa Model Terbaik

| Metrik | Nilai | Interpretasi |
|---|---|---|
| Accuracy | 95.24% | 95 dari 100 prediksi benar |
| Precision | 100% | Tidak ada false alarm tsunami sama sekali |
| Recall | 88.89% | 8 dari 9 gempa berpotensi berhasil dideteksi |
| F1 Score | 0.8929 | Keseimbangan sangat baik |

**Confusion Matrix:**
```
              Prediksi
              0      1
Aktual  0  [ 484 |   0 ]   <- Tidak ada false alarm!
        1  [   5 |  38 ]   <- 5 miss dari 43 berpotensi

TN=484 | FP=0 | FN=5 | TP=38
```

**Kenapa Recall lebih penting dari Precision di sini:** Dalam konteks tsunami, miss lebih berbahaya dari false alarm. Lebih baik orang evakuasi sia-sia daripada tidak evakuasi saat tsunami benar terjadi. Precision 100% artinya tidak ada kepanikan sia-sia.

---

## ⚙️ Bagaimana Sistem Terus Belajar

```
Hari ke-1 :  500 data seed  → model v1 → F1: 0.6154
Hari ke-2 :  +15 data baru  → retrain  → F1 sama → tidak promosi
...
Hari ke-N :  +100 data drift → retrain → F1: 0.8929 → PROMOSI ke v2!
```

Setiap hari model mendapat data baru dari BMKG. Semakin banyak data, semakin akurat prediksi.

---

## 🚀 Quick Start

```bash
# Clone
git clone https://github.com/bagrah/MLOps-Gempa-JawaTimur.git
cd MLOps-Gempa-JawaTimur

# Install
pip install -r requirements.txt

# Generate seed data (pertama kali)
python scripts/generate_seed_data.py

# Jalankan pipeline lengkap
export PYTHONPATH=$PYTHONPATH:$(pwd)
python scripts/run_pipeline.py

# Jalankan semua services
docker compose up -d
docker compose ps

# Test
pytest tests/ -v
```

**Akses services:**
- API: `http://localhost:5000`
- MLflow UI: `http://localhost:5001`
- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000`

---

## 🐳 Docker Services Detail

### Jalankan & Cek

```bash
docker compose up -d
docker compose ps
```

### Scaling Dinamis

```bash
docker compose up -d --scale api-service=5  # Scale up ke 5 replika
docker compose up -d --scale api-service=1  # Scale down ke 1 replika
```

### Matikan

```bash
docker compose down
```

---

## 🌐 API Endpoints Lengkap

### GET /health
```bash
curl http://localhost:5000/health
```
```json
{"model": "TsunamiRisk-RandomForest", "status": "ok"}
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
```json
{"nilai": 1, "prediksi": "BERPOTENSI TSUNAMI"}
```

### GET /metrics
```bash
curl http://localhost:5000/metrics
# Mengembalikan metrik Prometheus
```

### Parameter /predict

| Parameter | Tipe | Contoh | Keterangan |
|---|---|---|---|
| `magnitude` | float | 7.2 | Kekuatan gempa |
| `kedalaman_km` | int | 10 | Kedalaman dalam km |
| `lintang` | float | -8.50 | Koordinat lintang |
| `bujur` | float | 115.50 | Koordinat bujur |
| `jam` | int | 14 | Jam kejadian (0-23) |

---

## 📦 DVC Commands

```bash
# Track dataset baru
dvc add data/raw/batch/gempa_YYYY-MM-DD.csv
git add data/raw/batch/gempa_YYYY-MM-DD.csv.dvc
git commit -m "data: tambah batch gempa terbaru"

# Cek status dan perubahan
dvc status
dvc diff
```

---

## 🔧 Konfigurasi (`configs/config.yaml`)

```yaml
project:
  name: MLOps-Gempa-JawaTimur
  domain: Seismologi / Kebencanaan
  task: Klasifikasi Potensi Tsunami

model:
  threshold_accuracy: 0.85  # Minimum accuracy sebelum trigger retrain
  threshold_f1: 0.60        # Minimum F1 Score

mlflow:
  experiment_name: tsunami-indonesia-experiment
  tracking_uri: sqlite:///mlflow.db
```

---

## 🔁 Continuous Training Commands

```bash
# Cek performa model saat ini
python scripts/check_model_performance.py

# Bandingkan dan promosi model
python scripts/compare_and_promote.py

# Simulasi data drift
python scripts/simulate_drift.py
```

---

## 📚 Referensi

- [BMKG Open Data](https://data.bmkg.go.id/) — Sumber data gempa resmi Indonesia
- [InaTEWS BMKG](https://inatews.bmkg.go.id/) — Sistem peringatan dini tsunami Indonesia
- [MLflow Documentation](https://mlflow.org/docs/latest/)
- [DVC Documentation](https://dvc.org/doc)
- [Prometheus](https://prometheus.io/docs/)
- [Grafana](https://grafana.com/docs/)
- [Scikit-learn RandomForest](https://scikit-learn.org/stable/modules/ensemble.html)
- [GitHub Actions](https://docs.github.com/en/actions)

---

## 👤 Author

**Muhammad Bagas Anugrah**
Mata Kuliah: Machine Learning Operations

---

*Projek ini adalah demonstrasi pipeline MLOps production-ready menggunakan domain kebencanaan Indonesia. Model terus belajar dari data seismik real-time BMKG setiap harinya — adaptif, otomatis, dan termonitor.*