import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cleaned_training_data.csv"
)


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

def load_data():
    """Load the processed dataset."""

    df = pd.read_csv(DATA_FILE)

    df["booking_date"] = pd.to_datetime(df["booking_date"])

    return df


# ---------------------------------------------------------
# Create daily demand dataset
# ---------------------------------------------------------

def prepare_daily_demand(df):
    """
    Aggregate booking-level data into daily demand.

    The provided dataset does not contain SKU-level sales.
    Therefore, demand_index is used as the demand target.
    """

    daily = (
        df.groupby("booking_date")["demand_index"]
        .mean()
        .reset_index()
    )

    daily = daily.sort_values("booking_date").reset_index(drop=True)

    return daily


# ---------------------------------------------------------
# Create baseline prediction
# ---------------------------------------------------------

def create_baseline_predictions(daily):
    """
    Baseline forecasting method:
    predict today's demand using the previous day's demand.
    """

    daily["prediction"] = daily["demand_index"].shift(1)

    daily = daily.dropna().reset_index(drop=True)

    return daily


# ---------------------------------------------------------
# Metrics
# ---------------------------------------------------------

def calculate_mape(y_true, y_pred):
    """Calculate Mean Absolute Percentage Error."""

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mask = y_true != 0

    if not np.any(mask):
        return np.nan

    return (
        np.mean(
            np.abs(
                (y_true[mask] - y_pred[mask])
                / y_true[mask]
            )
        )
        * 100
    )


def calculate_metrics(y_true, y_pred):

    mae = mean_absolute_error(y_true, y_pred)

    rmse = np.sqrt(
        mean_squared_error(y_true, y_pred)
    )

    mape = calculate_mape(y_true, y_pred)

    return mae, rmse, mape


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    print("=" * 60)
    print("OMNIRETAIL AI - BASELINE DEMAND FORECAST")
    print("=" * 60)

    print("\nLoading processed dataset...")

    df = load_data()

    print(f"Dataset shape: {df.shape}")

    print("\nPreparing daily demand data...")

    daily = prepare_daily_demand(df)

    print(f"Daily demand records: {len(daily)}")

    print("\nDate range:")

    print(
        f"{daily['booking_date'].min().date()} "
        f"to "
        f"{daily['booking_date'].max().date()}"
    )

    print("\nCreating baseline predictions...")

    predictions = create_baseline_predictions(daily)

    # -----------------------------------------------------
    # Time-based train/test split
    # -----------------------------------------------------

    split_index = int(len(predictions) * 0.8)

    train = predictions.iloc[:split_index]

    test = predictions.iloc[split_index:]

    print(f"\nTraining records: {len(train)}")
    print(f"Testing records: {len(test)}")

    # -----------------------------------------------------
    # Evaluate baseline on test data
    # -----------------------------------------------------

    y_true = test["demand_index"]

    y_pred = test["prediction"]

    mae, rmse, mape = calculate_metrics(
        y_true,
        y_pred
    )

    # -----------------------------------------------------
    # Results
    # -----------------------------------------------------

    print("\n" + "=" * 60)
    print("BASELINE MODEL RESULTS")
    print("=" * 60)

    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"MAPE : {mape:.2f}%")

    print("=" * 60)

    print("\nBaseline forecasting completed successfully.")


if __name__ == "__main__":
    main()