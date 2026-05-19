import pandas as pd
import os

def test_processed_data_exists():
    assert os.path.exists("data/processed/gempa_jatim.csv")

def test_processed_data_not_empty():
    df = pd.read_csv("data/processed/gempa_jatim.csv")
    assert len(df) > 0

def test_processed_data_columns():
    df = pd.read_csv("data/processed/gempa_jatim.csv")
    required = ["magnitude", "kedalaman_km", "lintang", "bujur", "jam", "dirasakan"]
    for col in required:
        assert col in df.columns

def test_label_binary():
    df = pd.read_csv("data/processed/gempa_jatim.csv")
    assert df["dirasakan"].isin([0, 1]).all()
