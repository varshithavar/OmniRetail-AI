
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = PROJECT_ROOT / "data" / "raw" / "aiml_training_data.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def main():
    print("=" * 60)
    print("OMNIRETAIL AI - WEEKDAY BOOKING FORECAST")
    print("=" * 60)

    df = pd.read_csv(DATA_FILE)
    df["booking_date"] = pd.to_datetime(df["booking_date"])

    daily = df.groupby("booking_date").size().sort_index()

    # Include dates with zero bookings.
    date_range = pd.date_range(
        daily.index.min(),
        daily.index.max(),
        freq="D"
    )

    daily = daily.reindex(date_range, fill_value=0)
    daily.index.name = "booking_date"

    # Chronological 80/20 split.
    split_index = int(len(daily) * 0.8)

    train = daily.iloc[:split_index]
    test = daily.iloc[split_index:]

    # Learn average bookings for each weekday from training only.
    weekday_means = train.groupby(train.index.dayofweek).mean()

    # Predict test days using their weekday averages.
    predictions = pd.DataFrame({
        "booking_date": test.index,
        "actual": test.to_numpy(),
        "weekday_prediction": [
            weekday_means.loc[date.dayofweek]
            for date in test.index
        ]
    })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    predictions.to_csv(
        OUTPUT_DIR / "daily_booking_weekday_predictions.csv",
        index=False
    )

    print("\nTraining days:", len(train))
    print("Testing days:", len(test))
    print("Test start:", test.index.min().date())
    print("Test end:", test.index.max().date())

    print("\nAverage bookings by weekday:")

    weekday_names = [
        "Monday", "Tuesday", "Wednesday",
        "Thursday", "Friday", "Saturday", "Sunday"
    ]

    for day, average in weekday_means.items():
        print(f"{weekday_names[day]}: {average:.2f}")

    print("\nWeekday baseline predictions saved successfully.")
    print("outputs/daily_booking_weekday_predictions.csv")


if __name__ == "__main__":
    main()
