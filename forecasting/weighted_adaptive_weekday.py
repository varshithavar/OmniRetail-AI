
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "raw" / "aiml_training_data.csv"
OUTPUT_DIR = ROOT / "outputs"


def calculate_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))

    total_actual = np.sum(actual)

    wape = (
        np.sum(np.abs(actual - predicted)) / total_actual * 100
        if total_actual > 0 else 0.0
    )

    nonzero = actual != 0

    mape = (
        np.mean(
            np.abs(
                (actual[nonzero] - predicted[nonzero])
                / actual[nonzero]
            )
        ) * 100
        if nonzero.any() else 0.0
    )

    return {
        "MAE": mae,
        "RMSE": rmse,
        "WAPE_percent": wape,
        "MAPE_percent": mape
    }


def predict_weekday(history, strategy):
    if not history:
        return 0.0

    if strategy == "All History":
        return float(np.mean(history))

    if strategy == "Last 4":
        recent = history[-4:]
        return float(np.mean(recent))

    if strategy == "Last 8":
        recent = history[-8:]
        return float(np.mean(recent))

    if strategy == "Weighted Last 8":
        recent = np.asarray(history[-8:], dtype=float)
        weights = np.arange(1, len(recent) + 1)

        return float(np.average(recent, weights=weights))

    raise ValueError(f"Unknown strategy: {strategy}")


def main():
    print("=" * 65)
    print("OMNIRETAIL AI - WEIGHTED ADAPTIVE WEEKDAY FORECAST")
    print("=" * 65)

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

    strategies = [
        "All History",
        "Last 4",
        "Last 8",
        "Weighted Last 8"
    ]

    histories = {
        strategy: {
            day: list(train[train.index.dayofweek == day])
            for day in range(7)
        }
        for strategy in strategies
    }

    predictions = []

    # Rolling one-day-ahead evaluation.
    for date, actual in test.items():

        weekday = date.dayofweek

        row = {
            "booking_date": date,
            "actual": actual
        }

        for strategy in strategies:

            prediction = predict_weekday(
                histories[strategy][weekday],
                strategy
            )

            # Booking demand cannot be negative.
            prediction = max(0.0, prediction)

            row[strategy] = prediction

        predictions.append(row)

        # Update histories only after all predictions for this day.
        for strategy in strategies:
            histories[strategy][weekday].append(actual)

    result = pd.DataFrame(predictions)

    metric_rows = []

    for strategy in strategies:

        metrics = calculate_metrics(
            result["actual"],
            result[strategy]
        )

        metric_rows.append({
            "Model": strategy,
            **metrics
        })

    comparison = pd.DataFrame(metric_rows)

    print("\nTRAINING DAYS:", len(train))
    print("TESTING DAYS:", len(test))
    print("TEST START:", test.index.min().date())
    print("TEST END:", test.index.max().date())

    print("\nWEIGHTED ADAPTIVE MODEL COMPARISON")
    print(
        comparison.round(3).to_string(index=False)
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    result.to_csv(
        OUTPUT_DIR / "weighted_adaptive_predictions.csv",
        index=False
    )

    comparison.to_csv(
        OUTPUT_DIR / "weighted_adaptive_comparison.csv",
        index=False
    )

    print("\nSaved:")
    print("outputs/weighted_adaptive_predictions.csv")
    print("outputs/weighted_adaptive_comparison.csv")


if __name__ == "__main__":
    main()
