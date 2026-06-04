import pandas as pd
import os

def test_processed_data_exists():
    assert os.path.exists("data/processed/gempa_indonesia.csv")

def test_processed_data_not_empty():
    df = pd.read_csv("data/processed/gempa_indonesia.csv")
    assert len(df) > 0

def test_processed_data_columns():
    df = pd.read_csv("data/processed/gempa_indonesia.csv")
    required = ["magnitude", "kedalaman_km", "lintang", "bujur", "jam", "potensi_tsunami"]
    for col in required:
        assert col in df.columns

def test_label_binary():
    df = pd.read_csv("data/processed/gempa_indonesia.csv")
    assert df["potensi_tsunami"].isin([0, 1]).all()
