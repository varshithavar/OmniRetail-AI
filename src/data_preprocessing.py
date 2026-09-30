import pandas as pd
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_FILE = PROJECT_ROOT / "data" / "raw" / "aiml_training_data.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_FILE = PROCESSED_DIR / "cleaned_training_data.csv"


def load_data():
    """Load the raw training dataset."""
    return pd.read_csv(RAW_FILE)


def preprocess_data(df):
    """Clean and prepare the dataset for machine learning."""

    df = df.copy()

    # Convert booking date to datetime
    df["booking_date"] = pd.to_datetime(df["booking_date"])

    # Sort chronologically
    df = df.sort_values("booking_date").reset_index(drop=True)

    # Create date features
    df["year"] = df["booking_date"].dt.year
    df["month"] = df["booking_date"].dt.month
    df["day"] = df["booking_date"].dt.day
    df["weekofyear"] = df["booking_date"].dt.isocalendar().week.astype(int)

    return df


def save_data(df):
    """Save the processed dataset."""

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df.to_csv(PROCESSED_FILE, index=False)


def main():

    print("Loading raw dataset...")

    df = load_data()

    print(f"Raw dataset shape: {df.shape}")

    print("\nPreprocessing data...")

    df_processed = preprocess_data(df)

    print(f"Processed dataset shape: {df_processed.shape}")

    print("\nColumns:")
    print(df_processed.columns.tolist())

    print("\nMissing values:")
    print(df_processed.isnull().sum().sum())

    save_data(df_processed)

    print("\nProcessed dataset saved successfully:")
    print(PROCESSED_FILE)


if __name__ == "__main__":
    main()