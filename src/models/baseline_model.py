
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "processed" / "cleaned_training_data.csv"
OUTPUT_DIR = ROOT / "outputs"
REPORT_DIR = ROOT / "reports"
MODEL_DIR = ROOT / "models"
TARGET = "demand_index"


def evaluate(actual, predicted):
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)

    nonzero = actual != 0
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "mape_percent": (
            float(np.mean(np.abs(
                (actual[nonzero] - predicted[nonzero]) / actual[nonzero]
            )) * 100)
            if nonzero.any() else None
        ),
    }


def main():
    for folder in (OUTPUT_DIR, REPORT_DIR, MODEL_DIR):
        folder.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    df["booking_date"] = pd.to_datetime(df["booking_date"], errors="coerce")
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=["booking_date", TARGET])
    df = df.sort_values("booking_date").reset_index(drop=True)

    # Exclude identifiers, target, raw date, and fields that may leak
    # information unavailable at prediction time.
    excluded = {
        "booking_id", "booking_date", TARGET, "status", "rating"
    }
    features = [c for c in df.columns if c not in excluded]

    split = int(len(df) * 0.8)
    train, test = df.iloc[:split], df.iloc[split:]
    X_train, X_test = train[features], test[features]
    y_train, y_test = train[TARGET], test[TARGET]

    numeric = X_train.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = [c for c in features if c not in numeric]

    def make_pipeline(model):
        preprocessing = ColumnTransformer([
            ("numeric", Pipeline([
                ("imputer", SimpleImputer(strategy="median"))
            ]), numeric),
            ("categorical", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore"))
            ]), categorical),
        ])
        return Pipeline([
            ("preprocessing", preprocessing),
            ("model", model),
        ])

    models = {
        "dummy_mean": make_pipeline(DummyRegressor(strategy="mean")),
        "random_forest": make_pipeline(RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            min_samples_leaf=2,
            n_jobs=-1,
        )),
    }

    results = {}
    predictions = pd.DataFrame({
        "booking_date": test["booking_date"].dt.strftime("%Y-%m-%d"),
        "actual_demand_index": y_test.to_numpy(),
    })

    for name, model in models.items():
        model.fit(X_train, y_train)
        predicted = model.predict(X_test)
        results[name] = evaluate(y_test, predicted)
        predictions[f"{name}_prediction"] = predicted

        if name == "random_forest":
            joblib.dump(model, MODEL_DIR / "baseline_random_forest.joblib")

    results["dataset"] = {
        "valid_rows": int(len(df)),
        "training_rows": int(len(train)),
        "testing_rows": int(len(test)),
        "feature_count": len(features),
        "first_test_date": str(test["booking_date"].min().date()),
        "last_test_date": str(test["booking_date"].max().date()),
        "target": TARGET,
        "split_method": "chronological 80/20",
    }

    (OUTPUT_DIR / "baseline_metrics.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    predictions.to_csv(OUTPUT_DIR / "baseline_predictions.csv", index=False)

    report = [
        "# OmniRetail-AI Baseline Model Report",
        "",
        "## Objective",
        "Establish a benchmark for predicting demand_index.",
        "",
        "## Data split",
        f"- Valid rows: {len(df)}",
        f"- Training rows: {len(train)}",
        f"- Test rows: {len(test)}",
        f"- Features: {len(features)}",
        f"- First test date: {test['booking_date'].min().date()}",
        f"- Last test date: {test['booking_date'].max().date()}",
        "",
        "## Test-set metrics",
    ]

    for name, metrics in results.items():
        if name == "dataset":
            continue
        report.extend([f"### {name.replace('_', ' ').title()}"])
        report.extend([
            f"- {key}: {value:.5f}" if value is not None
            else f"- {key}: Not defined"
            for key, value in metrics.items()
        ])
        report.append("")

    report.extend([
        "## Interpretation",
        "Lower MAE and RMSE indicate smaller prediction errors. Compare the",
        "Random Forest with the Dummy Regressor on the same held-out dates.",
        "",
        "## Limitations",
        "This dataset predicts a service demand index, not SKU-level retail",
        "sales. Results do not establish inventory optimization or pricing impact.",
        "MAPE may be misleading for targets close to zero.",
    ])

    (REPORT_DIR / "baseline_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )

    print("Baseline evaluation complete.")
    print(json.dumps(results, indent=2))
    print("Model: models/baseline_random_forest.joblib")
    print("Metrics: outputs/baseline_metrics.json")
    print("Predictions: outputs/baseline_predictions.csv")
    print("Report: reports/baseline_report.md")


if __name__ == "__main__":
    main()
