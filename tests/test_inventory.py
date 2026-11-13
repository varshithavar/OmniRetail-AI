
import pytest

from inventory.replenishment import (
    calculate_safety_stock,
    check_inventory,
    get_service_daily_demand,
    get_service_demand_stats,
)


def test_inventory_healthy():
    result = check_inventory(
        "Laptop",
        current_stock=100,
        average_daily_demand=5,
        lead_time_days=4,
        safety_stock=10
    )

    assert result["status"] == "HEALTHY"
    assert result["alert"] is False
    assert result["recommended_order"] == 0


def test_inventory_low_stock():
    result = check_inventory(
        "Laptop",
        current_stock=25,
        average_daily_demand=5,
        lead_time_days=4,
        safety_stock=10
    )

    assert result["status"] == "LOW STOCK"
    assert result["alert"] is True
    assert result["recommended_order"] == 5


def test_inventory_critical():
    result = check_inventory(
        "Laptop",
        current_stock=5,
        average_daily_demand=5,
        lead_time_days=4,
        safety_stock=10
    )

    assert result["status"] == "CRITICAL"
    assert result["alert"] is True
    assert result["recommended_order"] == 25


def test_empty_product_name():
    with pytest.raises(ValueError):
        check_inventory(
            "",
            current_stock=10,
            average_daily_demand=2,
            lead_time_days=3
        )


@pytest.mark.parametrize(
    "stock,demand,lead,safety",
    [
        (-1, 2, 4, 5),
        (10, -1, 4, 5),
        (10, 2, -1, 5),
        (10, 2, 4, -1),
    ],
)
def test_negative_inventory_values(stock, demand, lead, safety):
    with pytest.raises(ValueError):
        check_inventory(
            "Laptop",
            current_stock=stock,
            average_daily_demand=demand,
            lead_time_days=lead,
            safety_stock=safety
        )


def test_service_daily_demand():
    demand = get_service_daily_demand("Plumbing")

    assert isinstance(demand, float)
    assert demand > 0


def test_service_daily_demand_case_insensitive():
    lower = get_service_daily_demand("plumbing")
    upper = get_service_daily_demand("PLUMBING")

    assert lower == upper


def test_invalid_service_name():
    with pytest.raises(ValueError):
        get_service_daily_demand("Unknown Service")


def test_service_name_none():
    with pytest.raises(ValueError):
        get_service_daily_demand(None)


def test_missing_dataset(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "inventory.replenishment.DATA_FILE",
        tmp_path / "missing.csv"
    )

    with pytest.raises(FileNotFoundError):
        get_service_daily_demand("Plumbing")


def test_service_not_found():
    with pytest.raises(ValueError):
        get_service_daily_demand("Nonexistent Service")


def test_inventory_main_execution():
    import runpy

    runpy.run_module(
        "inventory.replenishment",
        run_name="__main__"
    )


def test_service_demand_stats():
    stats = get_service_demand_stats("Plumbing")

    assert stats["service_type"] == "Plumbing"
    assert stats["total_bookings"] > 0
    assert stats["calendar_days"] > 0
    assert stats["active_days"] > 0
    assert stats["average_daily_demand"] > 0
    assert stats["demand_std_dev"] >= 0
    assert stats["max_daily_demand"] > 0


def test_calculate_safety_stock():
    result = calculate_safety_stock(
        demand_std_dev=2,
        lead_time_days=4,
        service_level_z=1.645
    )

    assert result == 7


def test_safety_stock_zero_lead_time():
    result = calculate_safety_stock(
        demand_std_dev=2,
        lead_time_days=0
    )

    assert result == 0


@pytest.mark.parametrize(
    "std_dev,lead_time,z_score",
    [
        (-1, 4, 1.645),
        (2, -1, 1.645),
        (2, 4, 0),
        (2, 4, -1),
        (float("nan"), 4, 1.645),
        (float("inf"), 4, 1.645),
        (True, 4, 1.645),
        (2, True, 1.645),
        (2, 4, True),
    ],
)
def test_invalid_safety_stock_inputs(std_dev, lead_time, z_score):
    with pytest.raises(ValueError):
        calculate_safety_stock(std_dev, lead_time, z_score)


def test_inventory_uses_dynamic_safety_stock():
    result = check_inventory(
        product_name="Plumbing Kit",
        current_stock=8,
        average_daily_demand=2,
        lead_time_days=4,
        demand_std_dev=2,
        service_level_z=1.645
    )

    assert result["safety_stock"] == 7
    assert result["lead_time_demand"] == 8
    assert result["reorder_level"] == 15
    assert result["recommended_order"] == 7
    assert result["status"] == "LOW STOCK"
