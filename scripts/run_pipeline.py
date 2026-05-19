from src.data.ingest_data import ingest_data
from src.data.preprocess import preprocess_data
from src.models.train import train
from src.models.evaluate_model import evaluate_model

def run_pipeline():
    print("=" * 50)
    print("🚀 STARTING MLOPS PIPELINE - GEMPA JAWA TIMUR")
    print("=" * 50)

    print("\n[1/4] Ingesting data dari BMKG...")
    ingest_data()

    print("\n[2/4] Preprocessing data...")
    preprocess_data()

    print("\n[3/4] Training model...")
    train(n_estimators=100)

    print("\n[4/4] Evaluating model...")
    evaluate_model()

    print("\n" + "=" * 50)
    print("✅ PIPELINE SELESAI!")
    print("=" * 50)

if __name__ == "__main__":
    run_pipeline()
