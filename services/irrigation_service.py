"""Rule-based irrigation advisory foundation for soil moisture inputs."""


def build_irrigation_advice(
    soil_moisture=None,
    rainfall_mm=0,
    temperature_c=None,
    crop_stage="vegetative",
):
    """Return irrigation timing and water-use guidance from field signals."""
    moisture = _to_float(soil_moisture, default=45.0)
    rainfall = _to_float(rainfall_mm, default=0.0)
    temperature = _to_float(temperature_c, default=28.0)
    stage = (crop_stage or "vegetative").lower()

    if rainfall >= 12:
        action = "skip"
        urgency = "low"
        water_mm = 0
        reason = "Recent rainfall is enough to delay irrigation."
    elif moisture < 25:
        action = "irrigate_now"
        urgency = "high"
        water_mm = 22 if stage in {"flowering", "boll"} else 18
        reason = "Soil moisture is critically low."
    elif moisture < 40:
        action = "irrigate_soon"
        urgency = "medium"
        water_mm = 14 if temperature < 34 else 18
        reason = "Soil moisture is below the preferred range."
    else:
        action = "monitor"
        urgency = "low"
        water_mm = 0
        reason = "Soil moisture is currently within a safe range."

    return {
        "action": action,
        "urgency": urgency,
        "recommended_water_mm": water_mm,
        "reason": reason,
        "inputs": {
            "soil_moisture": moisture,
            "rainfall_mm": rainfall,
            "temperature_c": temperature,
            "crop_stage": stage,
        },
        "sustainability_tips": [
            "Water early morning to reduce evaporation.",
            "Avoid overhead irrigation when fungal disease pressure is high.",
            "Recheck soil moisture before repeating irrigation.",
        ],
    }


def _to_float(value, default):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
