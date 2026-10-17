
import pandas as pd
import pytest

from src.data_preprocessing import preprocess_data


@pytest.fixture
def sample_data():
    return pd.DataFrame({
        "booking_id": ["B002", "B001"],
        "city": ["Mysuru", "Bengaluru"],
        "service_type": ["Cleaning", "Repair"],
        "booking_date": ["2026-02-02", "2026-01-01"],
        "booking_hour": [10, 14],
        "day_of_week": ["Monday", "Thursday"],
        "weather": ["Sunny", "Rainy"],
        "price_inr": [500, 800],
        "duration_minutes": [60, 90],
        "rating": [4.5, 4.0],
        "status": ["Completed", "Completed"],
        "demand_index": [0.7, 0.5],
    })


def test_preprocess_adds_date_features(sample_data):
    result = preprocess_data(sample_data)

    assert {"year", "month", "day", "weekofyear"}.issubset(
        result.columns
    )


def test_preprocess_sorts_by_date(sample_data):
    result = preprocess_data(sample_data)

    dates = pd.to_datetime(result["booking_date"])
    assert dates.is_monotonic_increasing


def test_preprocess_preserves_row_count(sample_data):
    result = preprocess_data(sample_data)

    assert len(result) == len(sample_data)


def test_preprocess_does_not_modify_original(sample_data):
    original = sample_data.copy(deep=True)

    preprocess_data(sample_data)

    pd.testing.assert_frame_equal(sample_data, original)


def test_preprocess_rejects_missing_booking_date(sample_data):
    sample_data = sample_data.drop(columns=["booking_date"])

    with pytest.raises(KeyError):
        preprocess_data(sample_data)
