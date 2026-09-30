import pandas as pd
from pathlib import Path


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_FILE = PROJECT_ROOT / "data" / "raw" / "aiml_training_data.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_FILE = PROCESSED_DIR / "cleaned_training_data.csv"


def load_data():
    """Load the raw training dataset."""
    df = pd.read_csv(RAW_FILE)
    return df


def preprocess_data(df):
    """Clean and prepare the dataset for ML."""

    df = df.copy()

    # Convert booking date to datetime
    df["booking_date"] = pd.to_datetime(df["booking_date"])

    # Sort chronologically
    df = df.sort_values("booking_date").reset_index(drop=True)

    # Extract useful date features
    df["year"] = df["booking_date"].dt.year
    df["month"] = df["booking_date"].dt.month
    df["day"] = df["booking_date"].dt.day
    df["weekofyear"] = df["booking_date"].dt.isocalendar().week.astype(int)

    # Convert categorical columns to category type
    categorical_columns = [
        "city",
        "service_type",
        "day_of_week",
        "weather",
        "status",
    ]

    for column in categorical_columns:
        df[column] = df[column].astype("category")

    return df


def save_data(df):
    """Save processed dataset."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_FILE, index=False)


def main():
    print("Loading raw dataset...")

    df = load_data()

    print(f"Raw dataset shape: {df.shape}")

    print("\nPreprocessing data...")

    df_processed = preprocess_data(df)

    print(f"Processed dataset shape: {df_processed.shape}")

    print("\nColumns after preprocessing:")
    print(df_processed.columns.tolist())

    print("\nMissing values:")
    print(df_processed.isnull().sum().sum())

    save_data(df_processed)

    print(f"\nProcessed dataset saved to:")
    print(PROCESSED_FILE)


if __name__ == "__main__":
    main()