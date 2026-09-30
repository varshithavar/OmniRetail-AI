from pathlib import Path
import pandas as pd


DATA_PATH = Path("data/raw/aiml_training_data.csv")


def load_training_data() -> pd.DataFrame:
    """Load the raw retail training dataset."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    return pd.read_csv(DATA_PATH)


if __name__ == "__main__":
    df = load_training_data()

    print("Dataset loaded successfully.")
    print("Shape:", df.shape)
    print(df.head())