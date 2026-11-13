
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = PROJECT_ROOT / "data" / "raw" / "aiml_training_data.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def load_daily_data():
    df = pd.read_csv(DATA_FILE)
    df["booking_date"] = pd.to_datetime(df["booking_date"])

    daily = df.groupby("booking_date").size().sort_index()

    date_range = pd.date_range(
        daily.index.min(),
        daily.index.max(),
        freq="D"
    )

    daily = daily.reindex(date_range, fill_value=0)
    daily.index.name = "booking_date"

    data = pd.DataFrame({"actual_bookings": daily})

    # Calendar features
    data["day_of_week"] = data.index.dayofweek
    data["month"] = data.index.month
    data["day_of_year"] = data.index.dayofyear
    data["is_weekend"] = (data.index.dayofweek >= 5).astype(int)

    # Historical booking features
    data["lag_1"] = data["actual_bookings"].shift(1)
    data["lag_7"] = data["actual_bookings"].shift(7)

    data["rolling_mean_7"] = (
        data["actual_bookings"].shift(1).rolling(7).mean()
    )

    data["rolling_std_7"] = (
        data["actual_bookings"].shift(1).rolling(7).std()
    )

    return data.dropna()


def calculate_metrics(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))

    wape = (
        np.sum(np.abs(actual - predicted))
        / np.sum(actual) * 100
    )

    nonzero = actual != 0

    mape = (
        np.mean(
            np.abs(
                (actual[nonzero] - predicted[nonzero])
                / actual[nonzero]
            )
        ) * 100
    )

    return {
        "MAE": mae,
        "RMSE": rmse,
        "WAPE_percent": wape,
        "MAPE_percent_nonzero": mape
    }


def main():
    print("=" * 60)
    print("OMNIRETAIL AI - RANDOM FOREST DAILY BOOKING FORECAST")
    print("=" * 60)

    data = load_daily_data()

    features = [
        "day_of_week",
        "month",
        "day_of_year",
        "is_weekend",
        "lag_1",
        "lag_7",
        "rolling_mean_7",
        "rolling_std_7"
    ]

    split_index = int(len(data) * 0.8)

    train = data.iloc[:split_index]
    test = data.iloc[split_index:]

    X_train = train[features]
    y_train = train["actual_bookings"]

    X_test = test[features]
    y_test = test["actual_bookings"]

    print("\nTraining records:", len(train))
    print("Testing records:", len(test))
    print("Test start:", test.index.min().date())
    print("Test end:", test.index.max().date())

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=5,
        min_samples_leaf=3,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    # Booking counts cannot be negative.
    predictions = np.maximum(predictions, 0)

    results = calculate_metrics(
        y_test.to_numpy(),
        predictions
    )

    output = pd.DataFrame({
        "booking_date": test.index,
        "actual": y_test.to_numpy(),
        "random_forest_prediction": predictions
    })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output.to_csv(
        OUTPUT_DIR / "daily_booking_random_forest_predictions.csv",
        index=False
    )

    pd.DataFrame([results]).to_csv(
        OUTPUT_DIR / "daily_booking_random_forest_metrics.csv",
        index=False
    )

    print("\n" + "=" * 60)
    print("RANDOM FOREST RESULTS")
    print("=" * 60)

    for metric, value in results.items():
        print(f"{metric}: {value:.2f}")

    print("\nSaved prediction and metric files in outputs/")
    print("Random Forest forecasting completed.")


if __name__ == "__main__":
    main()
