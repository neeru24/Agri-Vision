"""
tests/test_weather.py
Unit tests for Agri-Vision weather service integration.
Run with: python -m pytest tests/test_weather.py -v
"""

import pytest
from unittest.mock import patch, MagicMock
import sys
import os

# Allow importing from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from services.weather_service import (
    get_weather_open_meteo,
    geocode_city,
    generate_weather_recommendations,
    _wmo_description,
    _wmo_icon,
)


# ── Fixtures ────────────────────────────────────────────────────────

MOCK_OPEN_METEO_RESPONSE = {
    "current": {
        "temperature_2m": 34.0,
        "apparent_temperature": 38.0,
        "relative_humidity_2m": 82,
        "precipitation": 2.5,
        "wind_speed_10m": 35.0,
        "uv_index": 9.0,
        "weather_code": 61,
    }
}

MOCK_GEOCODING_RESPONSE = {
    "results": [
        {
            "latitude": 21.14,
            "longitude": 79.08,
            "name": "Nagpur",
            "country": "India",
        }
    ]
}


# ── Weather fetch tests ──────────────────────────────────────────────

class TestGetWeatherOpenMeteo:

    @patch("services.weather_service.requests.get")
    def test_returns_structured_dict(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = MOCK_OPEN_METEO_RESPONSE
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        result = get_weather_open_meteo(21.14, 79.08)

        assert result is not None
        assert result["temperature"] == 34.0
        assert result["humidity"] == 82
        assert result["precipitation"] == 2.5
        assert result["wind_speed"] == 35.0
        assert result["uv_index"] == 9.0
        assert result["source"] == "open-meteo"
        assert result["lat"] == 21.14
        assert result["lon"] == 79.08

    @patch("services.weather_service.requests.get")
    def test_returns_none_on_network_error(self, mock_get):
        mock_get.side_effect = Exception("Network error")
        result = get_weather_open_meteo(0.0, 0.0)
        assert result is None

    @patch("services.weather_service.requests.get")
    def test_description_and_icon_set(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = MOCK_OPEN_METEO_RESPONSE
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        result = get_weather_open_meteo(0.0, 0.0)
        assert result["description"] == "Slight rain"
        assert result["icon"] == "🌧️"


# ── Geocoding tests ──────────────────────────────────────────────────

class TestGeocodeCity:

    @patch("services.weather_service.requests.get")
    def test_returns_lat_lon_for_valid_city(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = MOCK_GEOCODING_RESPONSE
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        result = geocode_city("Nagpur")

        assert result is not None
        assert result["lat"] == 21.14
        assert result["lon"] == 79.08
        assert result["name"] == "Nagpur"

    @patch("services.weather_service.requests.get")
    def test_returns_none_for_unknown_city(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"results": []}
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        result = geocode_city("Xyz_Nonexistent_City_999")
        assert result is None

    @patch("services.weather_service.requests.get")
    def test_returns_none_on_error(self, mock_get):
        mock_get.side_effect = Exception("Timeout")
        result = geocode_city("Delhi")
        assert result is None


# ── Weather recommendations tests ────────────────────────────────────

class TestGenerateWeatherRecommendations:

    def test_high_humidity_triggers_fungal_warning(self):
        weather = {"temperature": 28, "humidity": 90, "precipitation": 0, "wind_speed": 10, "uv_index": 5}
        recs = generate_weather_recommendations(weather)
        assert any("humidity" in r.lower() or "fungal" in r.lower() for r in recs)

    def test_extreme_heat_triggers_warning(self):
        weather = {"temperature": 42, "humidity": 30, "precipitation": 0, "wind_speed": 5, "uv_index": 7}
        recs = generate_weather_recommendations(weather)
        assert any("heat" in r.lower() or "38" in r for r in recs)

    def test_rainfall_triggers_harvest_warning(self):
        weather = {"temperature": 25, "humidity": 75, "precipitation": 5.0, "wind_speed": 10, "uv_index": 4}
        recs = generate_weather_recommendations(weather)
        assert any("rain" in r.lower() or "harvest" in r.lower() for r in recs)

    def test_strong_wind_triggers_spray_warning(self):
        weather = {"temperature": 28, "humidity": 55, "precipitation": 0, "wind_speed": 40, "uv_index": 6}
        recs = generate_weather_recommendations(weather)
        assert any("wind" in r.lower() or "spray" in r.lower() for r in recs)

    def test_high_uv_triggers_warning(self):
        weather = {"temperature": 30, "humidity": 50, "precipitation": 0, "wind_speed": 12, "uv_index": 10}
        recs = generate_weather_recommendations(weather)
        assert any("uv" in r.lower() or "UV" in r for r in recs)

    def test_empty_weather_returns_empty_list(self):
        recs = generate_weather_recommendations(None)
        assert recs == []

    def test_returns_at_most_3_recommendations(self):
        weather = {"temperature": 42, "humidity": 92, "precipitation": 8.0, "wind_speed": 45, "uv_index": 11}
        recs = generate_weather_recommendations(weather)
        assert len(recs) <= 3

    def test_normal_conditions_return_no_warnings(self):
        weather = {"temperature": 25, "humidity": 55, "precipitation": 0, "wind_speed": 10, "uv_index": 5}
        recs = generate_weather_recommendations(weather)
        # Normal conditions shouldn't fire any of the threshold alerts
        assert len(recs) == 0


# ── WMO helper tests ─────────────────────────────────────────────────

class TestWMOHelpers:

    def test_known_code_returns_description(self):
        assert _wmo_description(0) == "Clear sky"
        assert _wmo_description(61) == "Slight rain"
        assert _wmo_description(95) == "Thunderstorm"

    def test_unknown_code_returns_fallback(self):
        assert "42" in _wmo_description(42)

    def test_none_code_returns_unknown(self):
        assert _wmo_description(None) == "Unknown"

    def test_known_code_returns_icon(self):
        assert _wmo_icon(0) == "☀️"
        assert _wmo_icon(61) == "🌧️"

    def test_none_code_returns_default_icon(self):
        assert _wmo_icon(None) == "🌡️"