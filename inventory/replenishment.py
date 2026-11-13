
import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "raw" / "aiml_training_data.csv"


def get_service_demand_stats(service_type):
    """Calculate calendar-based daily demand statistics."""

    if not isinstance(service_type, str) or not service_type.strip():
        raise ValueError("Enter a valid service type.")

    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

    df = pd.read_csv(DATA_FILE)

    required = {"booking_date", "service_type"}
    if not required.issubset(df.columns):
        raise ValueError("Dataset is missing required columns.")

    df["booking_date"] = pd.to_datetime(
        df["booking_date"], errors="coerce"
    ).dt.normalize()

    df = df.dropna(subset=["booking_date", "service_type"])

    service_data = df[
        df["service_type"].astype(str).str.strip().str.lower()
        == service_type.strip().lower()
    ]

    if service_data.empty:
        raise ValueError(f"No data found for {service_type}.")

    all_dates = pd.date_range(
        df["booking_date"].min(),
        df["booking_date"].max(),
        freq="D"
    )

    daily = (
        service_data.groupby("booking_date")
        .size()
        .reindex(all_dates, fill_value=0)
    )

    return {
        "service_type": service_type.strip(),
        "total_bookings": int(daily.sum()),
        "calendar_days": len(daily),
        "active_days": int((daily > 0).sum()),
        "average_daily_demand": float(daily.mean()),
        "demand_std_dev": float(daily.std(ddof=1)),
        "max_daily_demand": int(daily.max())
    }


def get_service_daily_demand(service_type):
    """Maintain backward compatibility with existing code."""

    stats = get_service_demand_stats(service_type)
    return round(stats["average_daily_demand"], 3)


def calculate_safety_stock(
    demand_std_dev,
    lead_time_days,
    service_level_z=1.645
):
    """Calculate safety stock using demand variability."""

    values = [demand_std_dev, lead_time_days, service_level_z]

    # Validate inputs, including rejecting Boolean values.
    if not all(
        isinstance(v, (int, float))
        and not isinstance(v, bool)
        and math.isfinite(v)
        for v in values
    ):
        raise ValueError("Safety stock inputs must be finite numbers.")

    if demand_std_dev < 0 or lead_time_days < 0:
        raise ValueError("Demand variation and lead time cannot be negative.")

    if service_level_z <= 0:
        raise ValueError("Service level factor must be positive.")

    return math.ceil(
        service_level_z * demand_std_dev * math.sqrt(lead_time_days)
    )


def check_inventory(
    product_name,
    current_stock,
    average_daily_demand,
    lead_time_days,
    safety_stock=5,
    demand_std_dev=None,
    service_level_z=1.645
):
    """Generate inventory alerts and reorder recommendations."""

    if not isinstance(product_name, str) or not product_name.strip():
        raise ValueError("Product name cannot be empty.")

    values = [current_stock, average_daily_demand, lead_time_days]

    if not all(
        isinstance(v, (int, float))
        and not isinstance(v, bool)
        and math.isfinite(v)
        for v in values
    ):
        raise ValueError("Inventory inputs must be finite numbers.")

    if current_stock < 0 or average_daily_demand < 0:
        raise ValueError("Stock and demand cannot be negative.")

    if lead_time_days < 0:
        raise ValueError("Lead time cannot be negative.")

    if demand_std_dev is not None:
        safety_stock = calculate_safety_stock(
            demand_std_dev,
            lead_time_days,
            service_level_z
        )

    if (
        not isinstance(safety_stock, (int, float))
        or isinstance(safety_stock, bool)
        or not math.isfinite(safety_stock)
        or safety_stock < 0
    ):
        raise ValueError("Safety stock must be a valid non-negative number.")

    lead_time_demand = average_daily_demand * lead_time_days
    reorder_level = lead_time_demand + safety_stock

    recommended_order = max(
        0, math.ceil(reorder_level - current_stock)
    )

    if current_stock <= safety_stock:
        status = "CRITICAL"
    elif current_stock <= reorder_level:
        status = "LOW STOCK"
    else:
        status = "HEALTHY"

    return {
        "product_name": product_name.strip(),
        "current_stock": current_stock,
        "average_daily_demand": round(average_daily_demand, 3),
        "lead_time_days": lead_time_days,
        "safety_stock": safety_stock,
        "lead_time_demand": round(lead_time_demand, 2),
        "reorder_level": round(reorder_level, 2),
        "status": status,
        "recommended_order": recommended_order,
        "alert": status != "HEALTHY"
    }


if __name__ == "__main__":

    service = "Plumbing"

    stats = get_service_demand_stats(service)

    print("\nSERVICE DEMAND ANALYSIS")
    print("-" * 40)

    for key, value in stats.items():
        print(f"{key}: {value}")

    result = check_inventory(
        product_name="Plumbing Service Kit",
        current_stock=8,
        average_daily_demand=stats["average_daily_demand"],
        lead_time_days=4,
        demand_std_dev=stats["demand_std_dev"],
        service_level_z=1.645
    )

    print("\nINVENTORY MANAGEMENT REPORT")
    print("-" * 40)

    for key, value in result.items():
        print(f"{key}: {value}")
