"""Market price analytics helpers for crop procurement decisions."""

MARKET_PRICE_DATA = [
    {
        "crop": "cotton",
        "state": "maharashtra",
        "market": "Nagpur",
        "price_per_quintal": 6900,
        "trend": "up",
    },
    {
        "crop": "cotton",
        "state": "gujarat",
        "market": "Rajkot",
        "price_per_quintal": 7050,
        "trend": "stable",
    },
    {
        "crop": "wheat",
        "state": "punjab",
        "market": "Ludhiana",
        "price_per_quintal": 2425,
        "trend": "up",
    },
    {
        "crop": "rice",
        "state": "west bengal",
        "market": "Burdwan",
        "price_per_quintal": 2180,
        "trend": "down",
    },
]


def build_market_price_report(crop=None, state=None):
    """Return filtered market prices with summary procurement guidance."""
    records = MARKET_PRICE_DATA

    if crop:
        crop_key = crop.strip().lower()
        records = [record for record in records if record["crop"] == crop_key]

    if state:
        state_key = state.strip().lower()
        records = [record for record in records if record["state"] == state_key]

    prices = [record["price_per_quintal"] for record in records]
    if not prices:
        return {
            "count": 0,
            "records": [],
            "summary": None,
            "recommendation": "No matching market data is available yet.",
        }

    best_market = max(records, key=lambda record: record["price_per_quintal"])
    average_price = sum(prices) / len(prices)

    return {
        "count": len(records),
        "records": records,
        "summary": {
            "average_price_per_quintal": round(average_price, 2),
            "min_price_per_quintal": min(prices),
            "max_price_per_quintal": max(prices),
            "best_market": best_market["market"],
        },
        "recommendation": (
            f"Best listed market is {best_market['market']} at "
            f"{best_market['price_per_quintal']} per quintal."
        ),
    }
