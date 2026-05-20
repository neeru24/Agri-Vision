"""
tests/test_yield.py
Unit tests for Agri-Vision yield estimation service.
Run with: python -m pytest tests/test_yield.py -v
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from services.yield_service import (
    estimate_yield,
    get_stage_multiplier,
    get_health_multiplier,
    get_weather_multiplier,
    BASE_YIELD_PER_ACRE,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_disease(health_score=75.0, predicted_class="Healthy"):
    return {"predicted_class": predicted_class, "health_score": health_score}

def make_growth(main_class="Matured Cotton Boll"):
    return {"main_class": main_class}

def make_weather(temp=28, humidity=55, precipitation=0):
    return {"temperature": temp, "humidity": humidity, "precipitation": precipitation}


# ── Stage multiplier tests ────────────────────────────────────────────────────

class TestStageMultiplier:

    def test_split_boll_is_max(self):
        mult, _ = get_stage_multiplier("Split Cotton Boll")
        assert mult == 1.00

    def test_matured_boll_is_high(self):
        mult, _ = get_stage_multiplier("Matured Cotton Boll")
        assert mult >= 0.90

    def test_bud_is_lowest(self):
        mult, _ = get_stage_multiplier("Cotton Bud")
        assert mult <= 0.35

    def test_unknown_stage_returns_fallback(self):
        mult, note = get_stage_multiplier("Unknown")
        assert 0 < mult <= 0.55
        assert isinstance(note, str)

    def test_all_known_stages_return_valid_mult(self):
        stages = [
            "Cotton Bud", "Cotton Blossom", "Early Boll",
            "Green Cotton Boll", "Matured Cotton Boll", "Split Cotton Boll"
        ]
        for stage in stages:
            mult, note = get_stage_multiplier(stage)
            assert 0 < mult <= 1.0
            assert len(note) > 0


# ── Health multiplier tests ───────────────────────────────────────────────────

class TestHealthMultiplier:

    def test_high_health_returns_full_multiplier(self):
        mult, _ = get_health_multiplier(90)
        assert mult == 1.00

    def test_zero_health_returns_lowest_multiplier(self):
        mult, _ = get_health_multiplier(5)
        assert mult == 0.40

    def test_moderate_health_returns_intermediate(self):
        mult, _ = get_health_multiplier(50)
        assert 0.65 <= mult <= 0.75

    def test_none_health_returns_default(self):
        mult, note = get_health_multiplier(None)
        assert mult == 0.70
        assert isinstance(note, str)

    def test_boundaries(self):
        assert get_health_multiplier(80)[0] == 1.00
        assert get_health_multiplier(79)[0] == 0.85
        assert get_health_multiplier(60)[0] == 0.85
        assert get_health_multiplier(59)[0] == 0.70
        assert get_health_multiplier(40)[0] == 0.70
        assert get_health_multiplier(39)[0] == 0.55
        assert get_health_multiplier(20)[0] == 0.55
        assert get_health_multiplier(19)[0] == 0.40


# ── Weather multiplier tests ──────────────────────────────────────────────────

class TestWeatherMultiplier:

    def test_none_weather_returns_1(self):
        mult, notes = get_weather_multiplier(None)
        assert mult == 1.00
        assert notes == []

    def test_extreme_heat_reduces_multiplier(self):
        mult, notes = get_weather_multiplier(make_weather(temp=42))
        assert mult < 1.00
        assert any("heat" in n.lower() or "°C" in n for n in notes)

    def test_high_humidity_reduces_multiplier(self):
        mult, notes = get_weather_multiplier(make_weather(humidity=90))
        assert mult < 1.00
        assert any("humidity" in n.lower() for n in notes)

    def test_heavy_rain_reduces_multiplier(self):
        mult, notes = get_weather_multiplier(make_weather(precipitation=10))
        assert mult < 1.00
        assert any("rain" in n.lower() for n in notes)

    def test_normal_conditions_return_1(self):
        mult, notes = get_weather_multiplier(make_weather(temp=28, humidity=55, precipitation=0))
        assert mult == 1.00
        assert any("favourable" in n.lower() for n in notes)

    def test_multiple_stressors_compound(self):
        bad_weather = make_weather(temp=40, humidity=90, precipitation=8)
        mult, _ = get_weather_multiplier(bad_weather)
        assert mult < 0.80  # compounded reduction


# ── Main estimator tests ──────────────────────────────────────────────────────

class TestEstimateYield:

    def test_returns_complete_dict(self):
        result = estimate_yield(make_disease(), make_growth())
        required_keys = [
            "yield_min_acre", "yield_max_acre", "yield_min_total", "yield_max_total",
            "yield_min_kg_ha", "yield_max_kg_ha", "confidence_label", "confidence_pct",
            "stage_multiplier", "health_multiplier", "weather_multiplier",
            "combined_multiplier", "stage_note", "health_note", "weather_notes",
            "harvest_advice", "field_acres"
        ]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_yield_min_less_than_max(self):
        result = estimate_yield(make_disease(), make_growth())
        assert result["yield_min_acre"] < result["yield_max_acre"]
        assert result["yield_min_total"] < result["yield_max_total"]

    def test_field_size_scales_total(self):
        r1 = estimate_yield(make_disease(), make_growth(), field_acres=1.0)
        r5 = estimate_yield(make_disease(), make_growth(), field_acres=5.0)
        assert abs(r5["yield_min_total"] - r1["yield_min_total"] * 5) < 0.01

    def test_none_weather_still_works(self):
        result = estimate_yield(make_disease(), make_growth(), weather=None)
        assert result["weather_multiplier"] == 1.00
        assert result["yield_min_acre"] > 0

    def test_negative_field_acres_defaults_to_1(self):
        result = estimate_yield(make_disease(), make_growth(), field_acres=-3)
        assert result["field_acres"] == 1.0

    def test_zero_field_acres_defaults_to_1(self):
        result = estimate_yield(make_disease(), make_growth(), field_acres=0)
        assert result["field_acres"] == 1.0

    def test_high_confidence_for_healthy_split_boll(self):
        result = estimate_yield(
            make_disease(health_score=90),
            make_growth("Split Cotton Boll"),
            make_weather()
        )
        assert result["confidence_label"] == "High"
        assert result["combined_multiplier"] >= 0.85

    def test_low_confidence_for_sick_bud(self):
        result = estimate_yield(
            make_disease(health_score=10),
            make_growth("Cotton Bud"),
            make_weather(temp=41, humidity=90)
        )
        assert result["confidence_label"] == "Low"
        assert result["combined_multiplier"] < 0.65

    def test_kg_ha_conversion_reasonable(self):
        result = estimate_yield(make_disease(health_score=80), make_growth("Split Cotton Boll"))
        # For healthy split boll, expect roughly 4000–6000 kg/ha
        assert 1000 < result["yield_max_kg_ha"] < 10000

    def test_harvest_advice_is_string(self):
        result = estimate_yield(make_disease(), make_growth())
        assert isinstance(result["harvest_advice"], str)
        assert len(result["harvest_advice"]) > 10

    def test_split_boll_harvest_advice_urgent(self):
        result = estimate_yield(make_disease(), make_growth("Split Cotton Boll"))
        assert "now" in result["harvest_advice"].lower() or "harvest" in result["harvest_advice"].lower()