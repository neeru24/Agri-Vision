"""
Agri-Vision Weather Service
Fetches real-time weather data using Open-Meteo API (free, no API key required).
Optionally supports OpenWeatherMap if OPENWEATHER_API_KEY is set in .env
"""

import requests
from requests.exceptions import (
    Timeout,
    ConnectionError,
    HTTPError,
    RequestException,
    JSONDecodeError,
)
import logging
from typing import Optional

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

# Retry strategy for temporary API/network failures
retry_strategy = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"],
)

adapter = HTTPAdapter(max_retries=retry_strategy)

session = requests.Session()
session.mount("https://", adapter)
session.mount("http://", adapter)

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
OPENWEATHER_URL = "https://api.openweathermap.org/data/2.5/weather"
GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"


def geocode_city(city: str) -> Optional[dict]:
    """Convert a city name to lat/lon using Open-Meteo's free geocoding API."""
    try:
        resp = session.get(
            GEOCODING_URL,
            params={
                "name": city,
                "count": 1,
                "language": "en",
                "format": "json",
            },
            timeout=5,
        )

        resp.raise_for_status()

        data = resp.json()

        if not isinstance(data, dict):
            logger.error("Invalid geocoding API response format")
            return None

        results = data.get("results", [])

        if results:
            r = results[0]

            return {
                "lat": r["latitude"],
                "lon": r["longitude"],
                "name": r.get("name", city),
                "country": r.get("country", ""),
            }

        logger.warning(f"No geocoding results found for '{city}'")

    except Timeout:
        logger.error(f"Geocoding request timed out for '{city}'")

    except ConnectionError:
        logger.error(f"Connection error during geocoding for '{city}'")

    except HTTPError as e:
        logger.error(f"HTTP error during geocoding for '{city}': {e}")

    except JSONDecodeError:
        logger.error(f"Invalid JSON received from geocoding API for '{city}'")

    except RequestException as e:
        logger.error(f"Request failure during geocoding for '{city}': {e}")

    return None


def get_weather_open_meteo(lat: float, lon: float) -> Optional[dict]:
    """
    Fetch current weather from Open-Meteo (free, no API key).
    Returns a structured dict with all weather fields Agri-Vision needs.
    """
    try:
        params = {
            "latitude": lat,
            "longitude": lon,
            "current": [
                "temperature_2m",
                "relative_humidity_2m",
                "apparent_temperature",
                "precipitation",
                "weather_code",
                "wind_speed_10m",
                "uv_index",
            ],
            "timezone": "auto",
        }
        resp = session.get(OPEN_METEO_URL, params=params, timeout=7)
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, dict):
            logger.error("Malformed Open-Meteo API response")
            return None
        current = data.get("current")
        if not isinstance(current, dict):
            logger.error("Missing 'current' weather data in Open-Meteo response")
            return None

        return {
            "source": "open-meteo",
            "lat": lat,
            "lon": lon,
            "temperature": current.get("temperature_2m"),        # °C
            "feels_like": current.get("apparent_temperature"),   # °C
            "humidity": current.get("relative_humidity_2m"),     # %
            "precipitation": current.get("precipitation", 0),   # mm
            "wind_speed": current.get("wind_speed_10m"),         # km/h
            "uv_index": current.get("uv_index"),
            "weather_code": current.get("weather_code"),
            "description": _wmo_description(current.get("weather_code")),
            "icon": _wmo_icon(current.get("weather_code")),
        }
    except Timeout:
        logger.error("Open-Meteo request timed out")

    except ConnectionError:
        logger.error("Connection error while fetching Open-Meteo weather data")

    except HTTPError as e:
        logger.error(f"Open-Meteo HTTP error: {e}")

    except JSONDecodeError:
        logger.error("Invalid JSON received from Open-Meteo API")

    except RequestException as e:
        logger.error(f"Open-Meteo request failed: {e}")

    return None


def get_weather_openweathermap(lat: float, lon: float, api_key: str) -> Optional[dict]:
    """
    Fetch current weather from OpenWeatherMap (requires API key).
    Falls back to this only if OPENWEATHER_API_KEY is set in .env
    """
    try:
        params = {
            "lat": lat,
            "lon": lon,
            "appid": api_key,
            "units": "metric",
        }
        # Reuse shared session so OpenWeatherMap calls also benefit from retry policy.
        resp = session.get(OPENWEATHER_URL, params=params, timeout=7)
        resp.raise_for_status()
        data = resp.json()
        main = data.get("main", {})
        wind = data.get("wind", {})
        weather = data.get("weather", [{}])[0]
        return {
            "source": "openweathermap",
            "lat": lat,
            "lon": lon,
            "temperature": main.get("temp"),
            "feels_like": main.get("feels_like"),
            "humidity": main.get("humidity"),
            "precipitation": data.get("rain", {}).get("1h", 0),
            "wind_speed": wind.get("speed", 0) * 3.6,  # m/s → km/h
            "uv_index": None,  # needs separate UV call in OWM
            "weather_code": None,
            "description": weather.get("description", "").title(),
            "icon": weather.get("icon"),
        }
    except Timeout:
        logger.error("OpenWeatherMap request timed out")
        
    except ConnectionError:
        logger.error("Connection error while fetching OpenWeatherMap data")

    except HTTPError as e:
        logger.error(f"OpenWeatherMap HTTP error: {e}")

    except JSONDecodeError:
        logger.error("Invalid JSON received from OpenWeatherMap API")

    except RequestException as e:
        logger.error(f"OpenWeatherMap request failed: {e}")

    return None


def get_weather(lat: float, lon: float, owm_api_key: str = None) -> Optional[dict]:
    """
    Main entry point. Tries OpenWeatherMap first if API key is available,
    otherwise falls back to Open-Meteo (always free).
    Returns None gracefully if all sources fail.
    """
    if owm_api_key:
        result = get_weather_openweathermap(lat, lon, owm_api_key)
        if result:
            return result
    return get_weather_open_meteo(lat, lon)


def generate_weather_recommendations(weather: dict) -> list:
    """
    Generate crop-specific recommendations based on current weather conditions.
    Returns a list of recommendation strings.
    """
    if not weather:
        return []

    recs = []
    temp = weather.get("temperature")
    humidity = weather.get("humidity")
    precipitation = weather.get("precipitation", 0)
    wind_speed = weather.get("wind_speed", 0)
    uv_index = weather.get("uv_index")

    # --- Humidity-based ---
    if humidity is not None:
        if humidity > 85:
            recs.append(
                "⚠️ Very high humidity detected — conditions favour fungal diseases "
                "(Powdery Mildew, Target Spot). Apply preventive fungicide and improve canopy airflow."
            )
        elif humidity > 70:
            recs.append(
                "🌫️ Elevated humidity — monitor closely for early signs of fungal infection. "
                "Avoid overhead irrigation."
            )

    # --- Temperature-based ---
    if temp is not None:
        if temp > 38:
            recs.append(
                "🌡️ Extreme heat (>38°C) — high risk of heat stress. Ensure adequate irrigation "
                "and avoid pesticide spraying during peak afternoon hours."
            )
        elif temp > 32:
            recs.append(
                "☀️ High temperatures — increase irrigation frequency and scout for spider mites "
                "which thrive in hot, dry conditions."
            )
        elif temp < 15:
            recs.append(
                "🧊 Cool temperatures — crop growth will slow. Delay any fertiliser applications "
                "until temperatures recover."
            )

    # --- Precipitation-based ---
    if precipitation and precipitation > 0:
        recs.append(
            f"🌧️ Recent rainfall ({precipitation:.1f} mm) — delay harvest of Split Cotton Bolls "
            "to prevent moisture damage and fibre staining. Check drainage channels."
        )
    elif precipitation == 0 and humidity is not None and humidity < 40:
        recs.append(
            "💧 Dry conditions — increase irrigation, especially during boll development phase. "
            "Watch for water stress symptoms."
        )

    # --- Wind-based ---
    if wind_speed and wind_speed > 30:
        recs.append(
            f"💨 Strong winds ({wind_speed:.0f} km/h) — avoid pesticide or fertiliser spraying "
            "to prevent drift and uneven coverage."
        )

    # --- UV-based ---
    if uv_index is not None and uv_index > 8:
        recs.append(
            f"☀️ Very high UV index ({uv_index:.0f}) — seedlings and young plants may experience "
            "photoinhibition. Consider shading and early-morning field operations."
        )

    return recs[:3]  # Cap at 3 weather tips to avoid overwhelming the recommendations list


# --- WMO Weather Code helpers ---

WMO_DESCRIPTIONS = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Depositing rime fog",
    51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
    61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
    71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow",
    80: "Slight showers", 81: "Moderate showers", 82: "Violent showers",
    95: "Thunderstorm", 96: "Thunderstorm with hail", 99: "Thunderstorm with heavy hail",
}

WMO_ICONS = {
    0: "☀️", 1: "🌤️", 2: "⛅", 3: "☁️",
    45: "🌫️", 48: "🌫️",
    51: "🌦️", 53: "🌦️", 55: "🌧️",
    61: "🌧️", 63: "🌧️", 65: "⛈️",
    71: "🌨️", 73: "🌨️", 75: "❄️",
    80: "🌦️", 81: "🌧️", 82: "⛈️",
    95: "⛈️", 96: "⛈️", 99: "⛈️",
}


def _wmo_description(code: int) -> str:
    if code is None:
        return "Unknown"
    return WMO_DESCRIPTIONS.get(code, f"Weather code {code}")


def _wmo_icon(code: int) -> str:
    if code is None:
        return "🌡️"
    return WMO_ICONS.get(code, "🌡️")
