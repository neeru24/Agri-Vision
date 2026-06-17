"""Pest early-warning and treatment recommendation helpers."""

PEST_PROFILES = {
    "aphids": {
        "risk_factors": {"humidity": 65, "temperature_min": 20, "temperature_max": 32},
        "organic": "Use neem oil spray and encourage ladybird beetles.",
        "chemical": "Use locally approved systemic insecticide only above threshold.",
    },
    "bollworm": {
        "risk_factors": {"humidity": 55, "temperature_min": 24, "temperature_max": 34},
        "organic": "Use pheromone traps and Bacillus thuringiensis at early stage.",
        "chemical": "Escalate to recommended bollworm control after extension advice.",
    },
    "whitefly": {
        "risk_factors": {"humidity": 50, "temperature_min": 25, "temperature_max": 38},
        "organic": "Install yellow sticky traps and remove heavily infested leaves.",
        "chemical": "Use selective whitefly control to protect beneficial insects.",
    },
}


def assess_pest_risk(pest="aphids", humidity=None, temperature_c=None, visible_damage=None):
    """Estimate pest risk and return farmer-facing next actions."""
    pest_key = (pest or "aphids").lower()
    profile = PEST_PROFILES.get(pest_key, PEST_PROFILES["aphids"])
    humidity_value = _to_float(humidity, 55.0)
    temperature = _to_float(temperature_c, 28.0)
    damage = _to_float(visible_damage, 0.0)
    factors = profile["risk_factors"]

    score = 20
    if humidity_value >= factors["humidity"]:
        score += 25
    if factors["temperature_min"] <= temperature <= factors["temperature_max"]:
        score += 25
    if damage >= 20:
        score += 25
    elif damage >= 5:
        score += 15

    score = min(score, 100)
    if score >= 70:
        level = "high"
        inspection_window = "Inspect affected blocks within 24 hours."
    elif score >= 45:
        level = "medium"
        inspection_window = "Re-scout the field within 48 hours."
    else:
        level = "low"
        inspection_window = "Continue routine scouting every 3 to 5 days."

    return {
        "pest": pest_key,
        "risk_score": score,
        "risk_level": level,
        "inspection_window": inspection_window,
        "recommendations": {
            "organic": profile["organic"],
            "chemical": profile["chemical"],
        },
        "inputs": {
            "humidity": humidity_value,
            "temperature_c": temperature,
            "visible_damage_percent": damage,
        },
    }


def _to_float(value, default):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
