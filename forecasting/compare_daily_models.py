
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "outputs"

files = {
    "7-Day Moving Average": (
        "daily_booking_baseline_predictions.csv",
        "moving_average_7"
    ),
    "Random Forest": (
        "daily_booking_random_forest_predictions.csv",
        "random_forest_prediction"
    ),
    "Weekday Average": (
        "daily_booking_weekday_predictions.csv",
        "weekday_prediction"
    ),
    "Gradient Boosting": (
        "daily_booking_gradient_boosting_predictions.csv",
        "gradient_boosting_prediction"
    ),
    "Adaptive Weekday": (
        "daily_booking_adaptive_predictions.csv",
        "adaptive_prediction"
    )
}

comparison = None

for model_name, (filename, prediction_column) in files.items():

    df = pd.read_csv(OUTPUT / filename)

    df["booking_date"] = pd.to_datetime(df["booking_date"])

    # Keep only the date, actual bookings and model prediction.
    df = df[["booking_date", "actual", prediction_column]]

    if comparison is None:
        comparison = df.rename(columns={
            prediction_column: model_name
        })
    else:
        # Check actual booking counts before merging.
        check = comparison[["booking_date", "actual"]].merge(
            df[["booking_date", "actual"]],
            on="booking_date",
            suffixes=("_first", "_new")
        )

        if not np.allclose(
            check["actual_first"],
            check["actual_new"]
        ):
            raise ValueError(
                f"Actual booking counts do not match for {model_name}"
            )

        # Merge only predictions, avoiding duplicate actual columns.
        comparison = comparison.merge(
            df[["booking_date", prediction_column]],
            on="booking_date"
        )

        comparison = comparison.rename(columns={
            prediction_column: model_name
        })

# Use common test dates for all five models.
comparison = comparison.dropna()

actual = comparison["actual"].to_numpy()
nonzero = actual != 0

results = []

for model_name in files:
    predicted = comparison[model_name].to_numpy()

    results.append({
        "Model": model_name,
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "WAPE (%)": (
            np.sum(np.abs(actual - predicted))
            / np.sum(actual) * 100
        ),
        "MAPE (%)": (
            np.mean(
                np.abs(
                    (actual[nonzero] - predicted[nonzero])
                    / actual[nonzero]
                )
            ) * 100
        )
    })

results_df = pd.DataFrame(results)

results_df.to_csv(
    OUTPUT / "daily_booking_model_comparison.csv",
    index=False
)

print("COMMON TEST DAYS:", len(comparison))
print("\nFAIR FIVE-MODEL COMPARISON")
print(results_df.round(2).to_string(index=False))

print("\nSaved: outputs/daily_booking_model_comparison.csv")


if __name__ == "__main__":
    pass
