
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "raw" / "aiml_training_data.csv"
OUTPUT_DIR = ROOT / "outputs"


def main():
    print("=" * 60)
    print("OMNIRETAIL AI - ADAPTIVE WEEKDAY FORECAST")
    print("=" * 60)

    df = pd.read_csv(DATA_FILE)
    df["booking_date"] = pd.to_datetime(df["booking_date"])

    daily = df.groupby("booking_date").size().sort_index()

    dates = pd.date_range(
        daily.index.min(),
        daily.index.max(),
        freq="D"
    )

    daily = daily.reindex(dates, fill_value=0)
    daily.index.name = "booking_date"

    split_index = int(len(daily) * 0.8)

    train = daily.iloc[:split_index]
    test = daily.iloc[split_index:]

    # Initialize weekday averages using training data.
    weekday_history = {
        day: list(train[train.index.dayofweek == day])
        for day in range(7)
    }

    predictions = []

    # Rolling one-day-ahead evaluation.
    for date, actual in test.items():

        weekday = date.dayofweek

        # Predict using previously observed values for this weekday.
        prediction = np.mean(weekday_history[weekday])

        predictions.append({
            "booking_date": date,
            "actual": actual,
            "adaptive_prediction": prediction
        })

        # Update after actual bookings become available.
        weekday_history[weekday].append(actual)

    result = pd.DataFrame(predictions)

    actual = result["actual"].to_numpy()
    predicted = result["adaptive_prediction"].to_numpy()

    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))

    wape = (
        np.sum(np.abs(actual - predicted))
        / np.sum(actual) * 100
    )

    nonzero = actual != 0

    mape = np.mean(
        np.abs(
            (actual[nonzero] - predicted[nonzero])
            / actual[nonzero]
        )
    ) * 100

    print("\nTraining days:", len(train))
    print("Testing days:", len(test))

    print("\nADAPTIVE WEEKDAY RESULTS")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"WAPE: {wape:.2f}%")
    print(f"MAPE: {mape:.2f}%")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    result.to_csv(
        OUTPUT_DIR / "daily_booking_adaptive_predictions.csv",
        index=False
    )

    pd.DataFrame([{
        "MAE": mae,
        "RMSE": rmse,
        "WAPE_percent": wape,
        "MAPE_percent_nonzero": mape
    }]).to_csv(
        OUTPUT_DIR / "daily_booking_adaptive_metrics.csv",
        index=False
    )

    print("\nAdaptive forecasting completed.")
    print("Prediction and metric files saved in outputs/")


if __name__ == "__main__":
    main()
