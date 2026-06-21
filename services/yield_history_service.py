"""Historical cotton yield analytics helpers."""

from __future__ import annotations

from statistics import mean


HISTORICAL_YIELD_DATA = [
    {"year": 2021, "region": "maharashtra", "crop": "cotton", "yield_q_per_acre": 17.8, "rainfall_mm": 812},
    {"year": 2022, "region": "maharashtra", "crop": "cotton", "yield_q_per_acre": 18.9, "rainfall_mm": 774},
    {"year": 2023, "region": "maharashtra", "crop": "cotton", "yield_q_per_acre": 20.2, "rainfall_mm": 802},
    {"year": 2024, "region": "maharashtra", "crop": "cotton", "yield_q_per_acre": 19.6, "rainfall_mm": 735},
    {"year": 2021, "region": "gujarat", "crop": "cotton", "yield_q_per_acre": 18.4, "rainfall_mm": 690},
    {"year": 2022, "region": "gujarat", "crop": "cotton", "yield_q_per_acre": 19.1, "rainfall_mm": 711},
    {"year": 2023, "region": "gujarat", "crop": "cotton", "yield_q_per_acre": 20.8, "rainfall_mm": 748},
    {"year": 2024, "region": "gujarat", "crop": "cotton", "yield_q_per_acre": 21.3, "rainfall_mm": 726},
]


def build_yield_history_report(crop="cotton", region=None):
    """Return filtered historical yield trend and forecast baseline."""
    crop_key = (crop or "cotton").strip().lower()
    region_key = region.strip().lower() if region else None

    records = [record for record in HISTORICAL_YIELD_DATA if record["crop"] == crop_key]
    if region_key:
        records = [record for record in records if record["region"] == region_key]

    if not records:
        return {
            "count": 0,
            "records": [],
            "summary": None,
            "recommendation": "No historical yield data is available for the selected filters.",
        }

    records = sorted(records, key=lambda record: record["year"])
    yields = [record["yield_q_per_acre"] for record in records]
    first_yield = yields[0]
    latest_yield = yields[-1]
    trend_delta = round(latest_yield - first_yield, 2)
    trend_pct = round((trend_delta / first_yield) * 100, 2) if first_yield else 0

    forecast_baseline = round(mean(yields[-3:]), 2)
    trend_label = "improving" if trend_delta > 0 else "declining" if trend_delta < 0 else "stable"

    return {
        "count": len(records),
        "records": records,
        "summary": {
            "average_yield_q_per_acre": round(mean(yields), 2),
            "latest_yield_q_per_acre": latest_yield,
            "trend_delta_q_per_acre": trend_delta,
            "trend_percent": trend_pct,
            "trend_label": trend_label,
            "forecast_baseline_q_per_acre": forecast_baseline,
        },
        "recommendation": (
            f"Historical trend is {trend_label}. Use {forecast_baseline} q/acre "
            "as the baseline before applying live crop health and weather adjustments."
        ),
    }
