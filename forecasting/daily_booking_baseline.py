
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = PROJECT_ROOT / "data" / "raw" / "aiml_training_data.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def load_daily_bookings():
    """Create a complete daily booking-count time series."""

    df = pd.read_csv(DATA_FILE)

    df["booking_date"] = pd.to_datetime(df["booking_date"])

    daily = (
        df.groupby("booking_date")
        .size()
        .sort_index()
    )

    # Include dates with zero bookings.
    date_range = pd.date_range(
        start=daily.index.min(),
        end=daily.index.max(),
        freq="D"
    )

    daily = daily.reindex(date_range, fill_value=0)
    daily.index.name = "booking_date"
    daily.name = "actual_bookings"

    return daily


def calculate_metrics(actual, predicted):
    """Calculate forecasting evaluation metrics."""

    mae = mean_absolute_error(actual, predicted)

    rmse = np.sqrt(
        mean_squared_error(actual, predicted)
    )

    total_actual = np.sum(actual)

    wape = (
        np.sum(np.abs(actual - predicted)) / total_actual * 100
        if total_actual > 0 else np.nan
    )

    nonzero = actual != 0

    mape = (
        np.mean(
            np.abs(
                (actual[nonzero] - predicted[nonzero])
                / actual[nonzero]
            )
        ) * 100
        if np.any(nonzero) else np.nan
    )

    return {
        "MAE": mae,
        "RMSE": rmse,
        "WAPE_percent": wape,
        "MAPE_percent_nonzero": mape
    }


def main():

    print("=" * 60)
    print("OMNIRETAIL AI - DAILY BOOKING FORECAST BASELINES")
    print("=" * 60)

    daily = load_daily_bookings()

    print("\nDataset records:", len(daily))
    print("Total bookings:", int(daily.sum()))

    # Chronological 80/20 split.
    split_index = int(len(daily) * 0.8)

    train = daily.iloc[:split_index]
    test = daily.iloc[split_index:]

    print("\nTraining days:", len(train))
    print("Testing days:", len(test))
    print("Test start:", test.index.min().date())
    print("Test end:", test.index.max().date())

    # Rolling one-day-ahead predictions.
    predictions = pd.DataFrame(index=daily.index)

    predictions["actual"] = daily

    predictions["previous_day"] = daily.shift(1)

    predictions["previous_week"] = daily.shift(7)

    predictions["moving_average_7"] = (
        daily.shift(1)
        .rolling(window=7)
        .mean()
    )

    test_predictions = predictions.loc[test.index].dropna()

    results = []

    for model_name in [
        "previous_day",
        "previous_week",
        "moving_average_7"
    ]:

        actual = test_predictions["actual"].to_numpy()
        predicted = test_predictions[model_name].to_numpy()

        metrics = calculate_metrics(actual, predicted)

        results.append({
            "model": model_name,
            **metrics
        })

    results_df = pd.DataFrame(results)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(
        OUTPUT_DIR / "daily_booking_baseline_metrics.csv",
        index=False
    )

    test_predictions.to_csv(
        OUTPUT_DIR / "daily_booking_baseline_predictions.csv"
    )

    print("\n" + "=" * 60)
    print("BASELINE COMPARISON")
    print("=" * 60)

    print(results_df.to_string(index=False))

    print("\nFiles saved:")
    print(OUTPUT_DIR / "daily_booking_baseline_metrics.csv")
    print(OUTPUT_DIR / "daily_booking_baseline_predictions.csv")

    print("\nForecasting baseline completed.")


if __name__ == "__main__":
    main()
