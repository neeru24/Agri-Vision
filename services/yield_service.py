"""
Agri-Vision Yield Estimation Service
=====================================
XGBoost-based yield estimator with graceful fallback to the original
rule-based system when model files are not present.

Inference flow:
  1. Try to load crop-specific XGBoost model from ai_models/<crop>_yield_model.pkl
  2. If found  → run ML inference (captures feature interaction effects)
  3. If missing → fall back to original ICAR multiplier logic (rule-based)

The API response format is backward-compatible PLUS two new fields:
  - predicted_yield_kg_acre  (point estimate, more useful for UI)
  - model_used               ("xgboost" | "legacy" — helps debug/audit)

Sources (rule-based fallback):
  - ICAR-CICR Cotton Production Guide (2022)
  - NCIPM Integrated Pest Management for Cotton
  - IMD agro-advisory bulletins for heat/humidity stress factors
"""

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────

_ROOT = Path(__file__).resolve().parent.parent
_MODELS_DIR = _ROOT / "ai_models"

# ── Model cache (loaded once per process) ────────────────────────────────────

_MODEL_CACHE: dict = {}


def _load_model(crop: str) -> Optional[dict]:
    """
    Load and cache a crop-specific yield model bundle.
    Returns None if model file doesn't exist (triggers fallback).
    """
    if crop in _MODEL_CACHE:
        return _MODEL_CACHE[crop]

    model_path = _MODELS_DIR / f"{crop}_yield_model.pkl"
    if not model_path.exists():
        logger.warning(
            f"Yield model not found for '{crop}' at {model_path}. "
            "Run training/train_yield_model.py to generate it. "
            "Falling back to rule-based estimation."
        )
        _MODEL_CACHE[crop] = None
        return None

    try:
        with open(model_path, "rb") as f:
            bundle = pickle.load(f)
        _MODEL_CACHE[crop] = bundle
        logger.info(
            f"Loaded {crop} yield model | "
            f"CV MAE: {bundle.get('cv_mae_mean', 'N/A'):.1f} kg/acre | "
            f"CV R²: {bundle.get('cv_r2_mean', 'N/A'):.4f}"
        )
        return bundle
    except Exception as e:
        logger.error(f"Failed to load yield model for '{crop}': {e}")
        _MODEL_CACHE[crop] = None
        return None


# ══════════════════════════════════════════════════════════════════════════════
#  ML INFERENCE PATH
# ══════════════════════════════════════════════════════════════════════════════

def _build_feature_vector(
    bundle: dict,
    growth_stage: str,
    disease_class: str,
    disease_confidence: float,
    temperature: float,
    humidity: float,
    precipitation: float,
    field_acres: float,
) -> np.ndarray:
    """
    Reproduce the same feature engineering used during training.
    Must stay in sync with engineer_features() in train_yield_model.py.
    """
    stage_enc = bundle["stage_enc"]
    disease_enc = bundle["disease_enc"]

    # Handle unseen labels gracefully
    known_stages = set(stage_enc.classes_)
    known_diseases = set(disease_enc.classes_)

    safe_stage = growth_stage if growth_stage in known_stages else stage_enc.classes_[0]
    safe_disease = disease_class if disease_class in known_diseases else disease_enc.classes_[0]

    if safe_stage != growth_stage:
        logger.warning(f"Unknown growth stage '{growth_stage}' → using '{safe_stage}'")
    if safe_disease != disease_class:
        logger.warning(f"Unknown disease class '{disease_class}' → using '{safe_disease}'")

    stage_encoded = stage_enc.transform([safe_stage])[0]
    disease_encoded = disease_enc.transform([safe_disease])[0]

    # Interaction features (same as training)
    disease_impact = disease_confidence * disease_encoded
    temp_humidity_stress = abs(temperature - 30) * (humidity / 100)
    stage_health_interact = stage_encoded * (1 - disease_confidence)

    feature_vector = np.array([[
        stage_encoded,
        disease_encoded,
        disease_confidence,
        temperature,
        humidity,
        precipitation,
        field_acres,
        disease_impact,
        temp_humidity_stress,
        stage_health_interact,
    ]])

    return feature_vector


def _run_ml_inference(
    bundle: dict,
    growth_stage: str,
    disease_class: str,
    disease_confidence: float,
    weather: dict,
    field_acres: float,
) -> float:
    """
    Run XGBoost inference. Returns predicted yield in kg/acre.
    """
    temp = weather.get("temperature", 28.0)
    humidity = weather.get("humidity", 60.0)
    precipitation = weather.get("precipitation", 0.0)

    X = _build_feature_vector(
        bundle,
        growth_stage,
        disease_class,
        disease_confidence,
        float(temp),
        float(humidity),
        float(precipitation),
        float(field_acres),
    )

    predicted = float(bundle["model"].predict(X)[0])
    return max(1.0, round(predicted, 2))


# ══════════════════════════════════════════════════════════════════════════════
#  RULE-BASED FALLBACK (original ICAR multiplier system — unchanged)
# ══════════════════════════════════════════════════════════════════════════════

BASE_YIELD_PER_ACRE = 20.0
QUINTALS_TO_KG_PER_HECTARE = 247.1

CONFIDENCE_LABELS = [
    (0.85, "High",   "#28a745"),
    (0.65, "Medium", "#ffc107"),
    (0.00, "Low",    "#dc3545"),
]

STAGE_MULTIPLIERS = {
    "Cotton Bud":           0.30,
    "Cotton Blossom":       0.40,
    "Early Boll":           0.65,
    "Green Cotton Boll":    0.75,
    "Matured Cotton Boll":  0.95,
    "Split Cotton Boll":    1.00,
    # Tomato stages
    "Early Vegetative":     0.35,
    "Flowering Initiation": 0.65,
    # Potato stages
    "Vegetative":           0.25,
    "Tuber Initiation":     0.55,
    "Tuber Bulking":        0.85,
    "Maturation":           1.00,
}

STAGE_NOTES = {
    "Cotton Bud":           "Crop is pre-flowering. Yield estimate has high uncertainty.",
    "Cotton Blossom":       "Crop is flowering. Final boll count depends on pollination success.",
    "Early Boll":           "Bolls are forming. Protect against boll weevil and maintain irrigation.",
    "Green Cotton Boll":    "Bolls are developing. Ensure adequate nutrition.",
    "Matured Cotton Boll":  "Bolls are mature. Plan harvest logistics.",
    "Split Cotton Boll":    "Crop is harvest-ready. Harvest promptly.",
    "Early Vegetative":     "Tomato is in early vegetative stage. Focus on plant establishment.",
    "Flowering Initiation": "Tomato is initiating flowers. Critical period for fruit set.",
    "Vegetative":           "Potato is in vegetative growth. Focus on canopy establishment.",
    "Tuber Initiation":     "Tubers are initiating. Maintain consistent soil moisture.",
    "Tuber Bulking":        "Tubers are bulking rapidly. Highest nutrient demand period.",
    "Maturation":           "Potato crop is maturing. Reduce irrigation to harden skin.",
}


def _get_stage_multiplier(growth_stage: str) -> tuple:
    mult = STAGE_MULTIPLIERS.get(growth_stage, 0.50)
    note = STAGE_NOTES.get(growth_stage, "Growth stage not recognised. Using conservative estimate.")
    return mult, note


def _get_health_multiplier(health_score: float) -> tuple:
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
        return 0.40, "Severe crop stress or disease. Urgent intervention required."


def _get_weather_multiplier(weather: Optional[dict]) -> tuple:
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
        notes.append(f"Cold stress ({temp}°C) — slows crop development.")

    if humidity is not None and humidity > 85:
        mult *= 0.88
        notes.append(f"High humidity ({humidity}%) — elevated disease pressure.")

    if precipitation and precipitation > 5:
        mult *= 0.90
        notes.append(f"Recent heavy rain ({precipitation}mm) — risk of rot and stress.")

    if not notes:
        notes.append("Weather conditions are favourable for the crop.")

    return round(mult, 3), notes


def _legacy_estimate(
    disease_result: dict,
    growth_result: dict,
    weather: Optional[dict],
    field_acres: float,
    crop_type: str,
) -> dict:
    """
    Original rule-based yield estimation (ICAR multiplier system).
    Used as fallback when XGBoost model is unavailable.
    """
    growth_stage = growth_result.get("main_class") if growth_result else None
    health_score = disease_result.get("health_score") if disease_result else None

    stage_mult, stage_note = _get_stage_multiplier(growth_stage or "Unknown")
    health_mult, health_note = _get_health_multiplier(health_score)
    weather_mult, weather_notes = _get_weather_multiplier(weather)

    combined = round(stage_mult * health_mult * weather_mult, 3)

    # Crop-specific base yields (quintals/acre for cotton; kg/acre for others)
    base_yields = {
        "cotton": 20.0 * 100,   # 20 q/acre → 2000 kg/acre seed cotton
        "tomato": 13000.0,
        "potato": 7000.0,
    }
    base = base_yields.get(crop_type, 20.0 * 100) * combined

    yield_min_acre = round(base * 0.85, 2)
    yield_max_acre = round(base * 1.15, 2)
    predicted_yield_kg_acre = round(base, 2)

    yield_min_total = round(yield_min_acre * field_acres, 2)
    yield_max_total = round(yield_max_acre * field_acres, 2)

    yield_min_kg_ha = round(yield_min_acre * QUINTALS_TO_KG_PER_HECTARE / 100, 0)
    yield_max_kg_ha = round(yield_max_acre * QUINTALS_TO_KG_PER_HECTARE / 100, 0)

    confidence_label, confidence_color = "Low", "#dc3545"
    for threshold, label, color in CONFIDENCE_LABELS:
        if combined >= threshold:
            confidence_label, confidence_color = label, color
            break

    harvest_advice = _get_harvest_advice(growth_stage, health_score)

    return {
        "growth_stage":             growth_stage or "Unknown",
        "health_score":             health_score,
        "field_acres":              field_acres,
        "crop_type":                crop_type,

        # Point estimate (NEW — useful for UI display)
        "predicted_yield_kg_acre":  predicted_yield_kg_acre,

        # Range estimates
        "yield_min_acre":           yield_min_acre,
        "yield_max_acre":           yield_max_acre,
        "yield_min_total":          yield_min_total,
        "yield_max_total":          yield_max_total,
        "yield_min_kg_ha":          int(yield_min_kg_ha),
        "yield_max_kg_ha":          int(yield_max_kg_ha),

        # Multiplier breakdown (kept for transparency / backward compat)
        "stage_multiplier":         stage_mult,
        "health_multiplier":        health_mult,
        "weather_multiplier":       weather_mult,
        "combined_multiplier":      combined,

        # Confidence
        "confidence_label":         confidence_label,
        "confidence_color":         confidence_color,
        "confidence_pct":           round(combined * 100, 1),

        # Explanatory notes
        "stage_note":               stage_note,
        "health_note":              health_note,
        "weather_notes":            weather_notes,
        "harvest_advice":           harvest_advice,

        # Audit field (NEW)
        "model_used":               "legacy",
    }


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN PUBLIC FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def estimate_yield(
    disease_result: dict,
    growth_result: dict,
    weather: Optional[dict] = None,
    field_acres: float = 1.0,
    crop_type: str = "cotton",
) -> dict:
    """
    Main yield estimation entry point.

    Tries XGBoost model first; falls back to ICAR rule-based system
    automatically if the model file is not found.

    Args:
        disease_result : dict from ResNet50 — must have 'predicted_class',
                         'confidence', and 'health_score'
        growth_result  : dict from YOLOv8 — must have 'main_class'
        weather        : optional dict from weather_service.get_weather()
        field_acres    : field size in acres (default 1.0)
        crop_type      : "cotton" | "tomato" | "potato" (default "cotton")

    Returns:
        Structured dict with yield range, point estimate, confidence,
        multiplier breakdown, harvest advice, and model_used flag.
    """
    if field_acres is None or field_acres <= 0:
        field_acres = 1.0
    crop_type = (crop_type or "cotton").lower().strip()

    # ── Extract disease info ──────────────────────────────────────────────────
    growth_stage = growth_result.get("main_class") if growth_result else None
    disease_class = disease_result.get("predicted_class") if disease_result else None
    disease_confidence = disease_result.get("confidence", 0.5) if disease_result else 0.5
    health_score = disease_result.get("health_score") if disease_result else None

    # ── Try ML path ───────────────────────────────────────────────────────────
    bundle = _load_model(crop_type)

    if bundle is not None and growth_stage and disease_class:
        try:
            safe_weather = weather or {"temperature": 28.0, "humidity": 60.0, "precipitation": 0.0}
            predicted_yield_kg_acre = _run_ml_inference(
                bundle,
                growth_stage,
                disease_class,
                float(disease_confidence),
                safe_weather,
                float(field_acres),
            )

            # Build weather notes + harvest advice (still useful even with ML)
            _, weather_notes = _get_weather_multiplier(weather)
            _, stage_note = _get_stage_multiplier(growth_stage)
            harvest_advice = _get_harvest_advice(growth_stage, health_score)

            # Yield range: ±15% around point estimate
            yield_min_acre = round(predicted_yield_kg_acre * 0.85, 2)
            yield_max_acre = round(predicted_yield_kg_acre * 1.15, 2)
            yield_min_total = round(yield_min_acre * field_acres, 2)
            yield_max_total = round(yield_max_acre * field_acres, 2)

            # Confidence label based on disease_confidence score
            confidence_label, confidence_color = "Low", "#dc3545"
            for threshold, label, color in CONFIDENCE_LABELS:
                if disease_confidence >= threshold:
                    confidence_label, confidence_color = label, color
                    break

            logger.info(
                f"XGBoost yield estimate | crop={crop_type} | "
                f"stage={growth_stage} | disease={disease_class} "
                f"({disease_confidence:.0%}) | "
                f"yield={predicted_yield_kg_acre:.0f} kg/acre"
            )

            return {
                "growth_stage":             growth_stage,
                "health_score":             health_score,
                "field_acres":              field_acres,
                "crop_type":                crop_type,

                # Point estimate
                "predicted_yield_kg_acre":  predicted_yield_kg_acre,

                # Range
                "yield_min_acre":           yield_min_acre,
                "yield_max_acre":           yield_max_acre,
                "yield_min_total":          yield_min_total,
                "yield_max_total":          yield_max_total,

                # Multipliers set to None in ML path (not applicable)
                # Kept in response for backward compat, clearly marked
                "stage_multiplier":         None,
                "health_multiplier":        None,
                "weather_multiplier":       None,
                "combined_multiplier":      None,

                # Confidence
                "confidence_label":         confidence_label,
                "confidence_color":         confidence_color,
                "confidence_pct":           round(disease_confidence * 100, 1),

                # Notes
                "stage_note":               stage_note,
                "health_note":              f"Disease detected: {disease_class} "
                                            f"(confidence: {disease_confidence:.0%})",
                "weather_notes":            weather_notes,
                "harvest_advice":           harvest_advice,

                # Audit
                "model_used":               "xgboost",
                "model_cv_mae":             bundle.get("cv_mae_mean"),
            }

        except Exception as e:
            logger.error(
                f"XGBoost inference failed for {crop_type}: {e}. "
                "Falling back to rule-based estimation.",
                exc_info=True,
            )
            # Fall through to legacy path

    # ── Fallback: rule-based ──────────────────────────────────────────────────
    logger.info(f"Using legacy rule-based yield estimation for crop='{crop_type}'")
    return _legacy_estimate(disease_result, growth_result, weather, field_acres, crop_type)


# ── Harvest Timing Advice ─────────────────────────────────────────────────────

def _get_harvest_advice(growth_stage: Optional[str], health_score: Optional[float]) -> str:
    """Generate a harvest timing recommendation string."""
    if growth_stage == "Split Cotton Boll":
        return "🟢 Harvest NOW — bolls are open. Delay risks fibre degradation and boll rot."
    elif growth_stage == "Matured Cotton Boll":
        if health_score and health_score < 50:
            return "🟡 Consider early harvest — bolls are mature but crop health is poor."
        return "🟡 Harvest within 1–2 weeks — bolls are mature. Monitor daily for splitting."
    elif growth_stage == "Green Cotton Boll":
        return "🔵 Harvest in 3–5 weeks — bolls are still filling. Maintain irrigation."
    elif growth_stage == "Early Boll":
        return "🔵 Harvest in 6–8 weeks — bolls are forming. Focus on pest management."
    elif growth_stage == "Cotton Blossom":
        return "⚪ Harvest in 10–12 weeks — crop is still flowering."
    elif growth_stage == "Cotton Bud":
        return "⚪ Harvest in 12–14 weeks — crop is pre-flowering."
    elif growth_stage == "Flowering Initiation":
        return "🔵 Tomato fruit set in 3–4 weeks. Maintain pollinator access and nutrition."
    elif growth_stage == "Early Vegetative":
        return "⚪ Tomato harvest in 10–12 weeks. Focus on root establishment."
    elif growth_stage == "Tuber Bulking":
        return "🔵 Potato harvest in 4–6 weeks. Do not water-stress during bulking."
    elif growth_stage == "Maturation":
        return "🟡 Potato harvest ready in 1–2 weeks. Reduce irrigation to harden skin."
    else:
        return "⚪ Growth stage not detected. Upload a clearer image for harvest timeline."