"""
Agri-Vision Yield Estimation Service
=====================================
Rule-based cotton yield estimator using agronomic constants from ICAR
(Indian Council of Agricultural Research) baseline for Bt cotton in India.

Base yield reference: 15–25 quintals/acre (ICAR, Bt cotton average ~20 q/acre)
Sources:
  - ICAR-CICR Cotton Production Guide (2022)
  - NCIPM Integrated Pest Management for Cotton
  - IMD agro-advisory bulletins for heat/humidity stress factors
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

BASE_YIELD_PER_ACRE = 20.0       # quintals/acre, ICAR Bt cotton average
QUINTALS_TO_KG_PER_HECTARE = 247.1  # conversion factor (1 q/acre = 247.1 kg/ha)

# Confidence labels mapped to combined multiplier ranges
CONFIDENCE_LABELS = [
    (0.85, "High",   "#28a745"),
    (0.65, "Medium", "#ffc107"),
    (0.00, "Low",    "#dc3545"),
]


# ── Stage Multiplier ──────────────────────────────────────────────────────────

STAGE_MULTIPLIERS = {
    "Cotton Bud":           0.30,   # Pre-flowering; boll set not confirmed
    "Cotton Blossom":       0.40,   # Flowering; highly variable outcome
    "Early Boll":           0.65,   # Bolls forming; some may abort
    "Green Cotton Boll":    0.75,   # Bolls developing; moderate confidence
    "Matured Cotton Boll":  0.95,   # Near-full potential
    "Split Cotton Boll":    1.00,   # Harvest-ready; maximum yield realised
}

STAGE_NOTES = {
    "Cotton Bud":           "Crop is pre-flowering. Yield estimate has high uncertainty — bolls have not yet set.",
    "Cotton Blossom":       "Crop is flowering. Final boll count depends on pollination success and pest pressure.",
    "Early Boll":           "Bolls are forming. Protect against boll weevil and maintain irrigation for best fill.",
    "Green Cotton Boll":    "Bolls are developing. Ensure adequate nutrition; avoid water stress at this stage.",
    "Matured Cotton Boll":  "Bolls are mature. Plan harvest logistics; reduce irrigation to harden bolls.",
    "Split Cotton Boll":    "Crop is harvest-ready. Harvest promptly to avoid fibre degradation and boll rot.",
}


def get_stage_multiplier(growth_stage: str) -> tuple:
    """
    Map YOLOv8 detected growth stage to a yield multiplier.
    Returns (multiplier, note).
    """
    mult = STAGE_MULTIPLIERS.get(growth_stage, 0.50)
    note = STAGE_NOTES.get(growth_stage, "Growth stage not recognised. Using conservative estimate.")
    return mult, note


# ── Health Multiplier ─────────────────────────────────────────────────────────

def get_health_multiplier(health_score: float) -> tuple:
    """
    Map ResNet50 health score (0–100) to a yield condition multiplier.
    Returns (multiplier, note).
    """
    if health_score is None:
        return 0.70, "Health score unavailable. Using moderate condition estimate."

    if health_score >= 80:
        return 1.00, "Crop is in excellent health. Full yield potential expected."
    elif health_score >= 60:
        return 0.85, "Crop health is good. Minor disease pressure may reduce yield slightly."
    elif health_score >= 40:
        return 0.70, "Moderate disease/stress detected. Yield likely reduced — treat promptly."
    elif health_score >= 20:
        return 0.55, "Significant crop stress detected. Yield substantially impacted."
    else:
        return 0.40, "Severe crop stress or disease. Urgent intervention required to salvage yield."


# ── Weather Multiplier ────────────────────────────────────────────────────────

def get_weather_multiplier(weather: Optional[dict]) -> tuple:
    """
    Map current weather conditions to a yield stress multiplier.
    Returns (multiplier, list of weather stress notes).
    """
    if not weather:
        return 1.00, []

    mult = 1.00
    notes = []

    temp = weather.get("temperature")
    humidity = weather.get("humidity")
    precipitation = weather.get("precipitation", 0)

    if temp is not None and temp > 38:
        mult *= 0.85
        notes.append(f"Heat stress ({temp}°C) — reduces boll fill and fibre quality.")
    elif temp is not None and temp < 15:
        mult *= 0.90
        notes.append(f"Cold stress ({temp}°C) — slows boll development.")

    if humidity is not None and humidity > 85:
        mult *= 0.88
        notes.append(f"High humidity ({humidity}%) — elevated disease pressure on bolls.")

    if precipitation and precipitation > 5:
        mult *= 0.90
        notes.append(f"Recent heavy rain ({precipitation}mm) — risk of boll rot and fibre staining.")

    if not notes:
        notes.append("Weather conditions are favourable for cotton.")

    return round(mult, 3), notes


# ── Main Estimator ────────────────────────────────────────────────────────────

def estimate_yield(
    disease_result: dict,
    growth_result: dict,
    weather: Optional[dict] = None,
    field_acres: float = 1.0,
) -> dict:
    """
    Main yield estimation function.

    Args:
        disease_result: dict from ResNet50 disease classifier (must have 'health_score')
        growth_result:  dict from YOLOv8 growth stage detector (must have 'main_class')
        weather:        optional dict from weather_service.get_weather()
        field_acres:    field size in acres (default 1.0)

    Returns:
        Structured dict with yield range, confidence, multiplier breakdown,
        harvest advice, and unit conversions.
    """
    if field_acres is None or field_acres <= 0:
        field_acres = 1.0

    # ── Extract inputs ──
    growth_stage = growth_result.get("main_class") if growth_result else None
    health_score = disease_result.get("health_score") if disease_result else None

    # ── Get multipliers ──
    stage_mult, stage_note   = get_stage_multiplier(growth_stage or "Unknown")
    health_mult, health_note = get_health_multiplier(health_score)
    weather_mult, weather_notes = get_weather_multiplier(weather)

    # ── Combined multiplier ──
    combined = round(stage_mult * health_mult * weather_mult, 3)

    # ── Yield range per acre ──
    base = BASE_YIELD_PER_ACRE * combined
    yield_min_acre = round(base * 0.85, 2)
    yield_max_acre = round(base * 1.15, 2)

    # ── Scale to field size ──
    yield_min_total = round(yield_min_acre * field_acres, 2)
    yield_max_total = round(yield_max_acre * field_acres, 2)

    # ── kg/hectare conversion ──
    yield_min_kg_ha = round(yield_min_acre * QUINTALS_TO_KG_PER_HECTARE, 0)
    yield_max_kg_ha = round(yield_max_acre * QUINTALS_TO_KG_PER_HECTARE, 0)

    # ── Confidence label ──
    confidence_label, confidence_color = "Low", "#dc3545"
    for threshold, label, color in CONFIDENCE_LABELS:
        if combined >= threshold:
            confidence_label, confidence_color = label, color
            break

    # ── Harvest timing advice ──
    harvest_advice = _get_harvest_advice(growth_stage, health_score)

    return {
        "growth_stage":       growth_stage or "Unknown",
        "health_score":       health_score,
        "field_acres":        field_acres,

        # Per-acre estimates
        "yield_min_acre":     yield_min_acre,
        "yield_max_acre":     yield_max_acre,

        # Total field estimates
        "yield_min_total":    yield_min_total,
        "yield_max_total":    yield_max_total,

        # kg/hectare
        "yield_min_kg_ha":    int(yield_min_kg_ha),
        "yield_max_kg_ha":    int(yield_max_kg_ha),

        # Multiplier breakdown (for transparency in UI)
        "stage_multiplier":   stage_mult,
        "health_multiplier":  health_mult,
        "weather_multiplier": weather_mult,
        "combined_multiplier": combined,

        # Confidence
        "confidence_label":   confidence_label,
        "confidence_color":   confidence_color,
        "confidence_pct":     round(combined * 100, 1),

        # Explanatory notes
        "stage_note":         stage_note,
        "health_note":        health_note,
        "weather_notes":      weather_notes,
        "harvest_advice":     harvest_advice,
    }


def _get_harvest_advice(growth_stage: Optional[str], health_score: Optional[float]) -> str:
    """Generate a harvest timing recommendation string."""
    if growth_stage == "Split Cotton Boll":
        return "🟢 Harvest NOW — bolls are open. Delay risks fibre degradation and boll rot."
    elif growth_stage == "Matured Cotton Boll":
        if health_score and health_score < 50:
            return "🟡 Consider early harvest — bolls are mature but crop health is poor. Delay may worsen losses."
        return "🟡 Harvest within 1–2 weeks — bolls are mature. Monitor daily for splitting."
    elif growth_stage == "Green Cotton Boll":
        return "🔵 Harvest in 3–5 weeks — bolls are still filling. Maintain irrigation and nutrition."
    elif growth_stage == "Early Boll":
        return "🔵 Harvest in 6–8 weeks — bolls are forming. Focus on pest management and boll protection."
    elif growth_stage == "Cotton Blossom":
        return "⚪ Harvest in 10–12 weeks — crop is still flowering. Boll set will determine final yield."
    elif growth_stage == "Cotton Bud":
        return "⚪ Harvest in 12–14 weeks — crop is pre-flowering. Too early for reliable yield estimate."
    else:
        return "⚪ Growth stage not detected. Upload a clearer image for a more accurate harvest timeline."