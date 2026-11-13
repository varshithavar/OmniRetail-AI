
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "outputs" / "daily_booking_weekday_predictions.csv"
OUTPUT = ROOT / "outputs"

df = pd.read_csv(INPUT)
df["booking_date"] = pd.to_datetime(df["booking_date"])

weekday_names = [
    "Monday", "Tuesday", "Wednesday",
    "Thursday", "Friday", "Saturday", "Sunday"
]

df["weekday"] = df["booking_date"].dt.dayofweek.map(
    lambda x: weekday_names[x]
)

df["error"] = (
    df["actual"] - df["weekday_prediction"]
)

df["absolute_error"] = df["error"].abs()

df["error_type"] = np.where(
    df["error"] > 0,
    "Underprediction",
    np.where(df["error"] < 0, "Overprediction", "Exact")
)

print("=" * 60)
print("WEEKDAY MODEL - ERROR ANALYSIS")
print("=" * 60)

print("\nAverage error by weekday:")

weekday_errors = (
    df.groupby("weekday")
    .agg(
        actual_average=("actual", "mean"),
        predicted_average=("weekday_prediction", "mean"),
        MAE=("absolute_error", "mean")
    )
    .reindex(weekday_names)
)

print(weekday_errors.round(2).to_string())

print("\nPrediction error summary:")
print(df["error_type"].value_counts().to_string())

print("\nTop 10 largest prediction errors:")

top_errors = df.nlargest(10, "absolute_error")[
    ["booking_date", "weekday", "actual",
     "weekday_prediction", "error", "absolute_error"]
]

print(top_errors.round(2).to_string(index=False))

df.to_csv(
    OUTPUT / "weekday_error_analysis.csv",
    index=False
)

weekday_errors.to_csv(
    OUTPUT / "weekday_error_by_day.csv"
)

print("\nError analysis files saved successfully.")


if __name__ == "__main__":
    pass
