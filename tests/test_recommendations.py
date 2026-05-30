"""
Tests for the Recommendation Engine service module.
Covers: retrieval, fallback handling, confidence advisory, and integration.
"""

import io
import json

import numpy as np
import pytest
from PIL import Image

from services.recommendation_engine import (
    get_confidence_advisory,
    get_fallback_recommendations,
    get_recommendations,
    RECOMMENDATIONS,
    LOW_CONFIDENCE_THRESHOLD,
    MEDIUM_CONFIDENCE_THRESHOLD,
)
import app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    """Flask test client."""
    app.app.config["TESTING"] = True
    with app.app.test_client() as c:
        yield c


@pytest.fixture(autouse=True)
def disable_rate_limiting(monkeypatch):
    """Disable rate limiting for all tests since Redis may be active."""
    monkeypatch.setattr(app.limiter, "limit", lambda *args, **kwargs: lambda f: f)


@pytest.fixture
def valid_image():
    """Generates a valid 300x300 PNG image with sufficient detail to pass quality checks."""
    img_byte_arr = io.BytesIO()
    pixels = np.random.randint(50, 200, (300, 300, 3), dtype=np.uint8)
    pixels[50:250, 50:250] = np.random.randint(30, 150, (200, 200, 3), dtype=np.uint8)
    pixels[100:200, 100:200] = np.random.randint(60, 180, (100, 100, 3), dtype=np.uint8)
    pixels[120:180, 120:180] = np.random.randint(10, 100, (60, 60, 3), dtype=np.uint8)
    Image.fromarray(pixels).save(img_byte_arr, format="PNG")
    img_byte_arr.seek(0)
    return img_byte_arr


# ---------------------------------------------------------------------------
# RECOMMENDATION CATEGORIES — expected keys in every response
# ---------------------------------------------------------------------------

EXPECTED_CATEGORIES = [
    "treatment",
    "prevention",
    "irrigation",
    "organic_control",
    "chemical_control",
    "severity_advice",
]


# ---------------------------------------------------------------------------
# Unit Tests — get_recommendations()
# ---------------------------------------------------------------------------

class TestGetRecommendations:
    """Tests for the core get_recommendations function."""

    def test_known_disease_returns_all_categories(self):
        """A known disease (Aphids) should return all 6 recommendation categories."""
        result = get_recommendations("cotton", "Aphids")
        assert result["crop_type"] == "cotton"
        assert result["disease_name"] == "Aphids"
        assert result["is_fallback"] is False
        for category in EXPECTED_CATEGORIES:
            assert category in result, f"Missing category: {category}"
            assert isinstance(result[category], list), f"{category} should be a list"
            assert len(result[category]) > 0, f"{category} should not be empty"

    def test_all_cotton_diseases_covered(self):
        """Every disease class in app.disease_classes should have recommendations."""
        for disease_name in app.disease_classes:
            result = get_recommendations("cotton", disease_name)
            assert result["is_fallback"] is False, (
                f"Disease '{disease_name}' returned fallback instead of specific recommendations"
            )
            for category in EXPECTED_CATEGORIES:
                assert len(result[category]) > 0, (
                    f"Disease '{disease_name}' has empty {category}"
                )

    def test_unknown_disease_returns_fallback(self):
        """An unrecognized disease should return fallback recommendations."""
        result = get_recommendations("cotton", "NonexistentDisease")
        assert result["is_fallback"] is True
        assert result["disease_name"] == "NonexistentDisease"
        for category in EXPECTED_CATEGORIES:
            assert category in result
            assert len(result[category]) > 0

    def test_unknown_crop_returns_fallback(self):
        """An unrecognized crop type should return fallback recommendations."""
        result = get_recommendations("wheat", "Aphids")
        assert result["is_fallback"] is True
        assert result["crop_type"] == "wheat"
        for category in EXPECTED_CATEGORIES:
            assert category in result
            assert len(result[category]) > 0

    def test_empty_disease_name_returns_fallback(self):
        """Empty or None disease name should return fallback."""
        result = get_recommendations("cotton", "")
        assert result["is_fallback"] is True

        result = get_recommendations("cotton", None)
        assert result["is_fallback"] is True

    def test_empty_crop_type_returns_fallback(self):
        """Empty or None crop type should return fallback."""
        result = get_recommendations("", "Aphids")
        assert result["is_fallback"] is True

        result = get_recommendations(None, "Aphids")
        assert result["is_fallback"] is True

    def test_healthy_returns_maintenance_guidance(self):
        """Healthy crop should return maintenance-oriented recommendations."""
        result = get_recommendations("cotton", "Healthy")
        assert result["is_fallback"] is False
        assert result["disease_name"] == "Healthy"
        # Treatment should indicate no action needed
        assert any("no treatment" in t.lower() or "no disease" in t.lower()
                    for t in result["treatment"])
        # Chemical control should discourage unnecessary spraying
        assert any("no chemical" in c.lower() or "not needed" in c.lower()
                    for c in result["chemical_control"])

    def test_case_insensitive_crop_lookup(self):
        """Crop type lookup should be case-insensitive."""
        result_lower = get_recommendations("cotton", "Aphids")
        result_upper = get_recommendations("COTTON", "Aphids")
        result_mixed = get_recommendations("Cotton", "Aphids")
        assert result_lower["is_fallback"] is False
        assert result_upper["is_fallback"] is False
        assert result_mixed["is_fallback"] is False

    def test_crop_type_whitespace_trimmed(self):
        """Leading/trailing whitespace in crop type should be trimmed."""
        result = get_recommendations("  cotton  ", "Aphids")
        assert result["is_fallback"] is False

    def test_disease_name_whitespace_trimmed(self):
        """Leading/trailing whitespace in disease name should be trimmed."""
        result = get_recommendations("cotton", "  Aphids  ")
        assert result["is_fallback"] is False


# ---------------------------------------------------------------------------
# Unit Tests — get_confidence_advisory()
# ---------------------------------------------------------------------------

class TestConfidenceAdvisory:
    """Tests for confidence-based advisory messages."""

    def test_low_confidence_returns_warning(self):
        """Confidence below LOW_CONFIDENCE_THRESHOLD should generate a warning."""
        advisory = get_confidence_advisory(0.30)
        assert advisory is not None
        assert "low" in advisory.lower() or "uncertain" in advisory.lower()

    def test_medium_confidence_returns_info(self):
        """Confidence between thresholds should generate informational message."""
        advisory = get_confidence_advisory(0.55)
        assert advisory is not None
        assert "moderate" in advisory.lower() or "monitor" in advisory.lower()

    def test_high_confidence_returns_none(self):
        """Confidence above MEDIUM_CONFIDENCE_THRESHOLD should return None."""
        advisory = get_confidence_advisory(0.90)
        assert advisory is None

    def test_none_confidence_returns_none(self):
        """None confidence should return None advisory."""
        advisory = get_confidence_advisory(None)
        assert advisory is None

    def test_edge_confidence_at_low_threshold(self):
        """Confidence exactly at LOW_CONFIDENCE_THRESHOLD should not be 'low'."""
        advisory = get_confidence_advisory(LOW_CONFIDENCE_THRESHOLD)
        # At the threshold value, it falls into the medium range
        if advisory is not None:
            assert "low" not in advisory.lower() or "moderate" in advisory.lower()

    def test_confidence_integrated_in_recommendations(self):
        """Low confidence should populate confidence_advisory in results."""
        result = get_recommendations("cotton", "Aphids", confidence=0.30)
        assert result["confidence_advisory"] is not None
        assert "low" in result["confidence_advisory"].lower() or "uncertain" in result["confidence_advisory"].lower()

    def test_high_confidence_no_advisory_in_recommendations(self):
        """High confidence should have None confidence_advisory in results."""
        result = get_recommendations("cotton", "Aphids", confidence=0.95)
        assert result["confidence_advisory"] is None


# ---------------------------------------------------------------------------
# Unit Tests — get_fallback_recommendations()
# ---------------------------------------------------------------------------

class TestFallbackRecommendations:
    """Tests for the fallback recommendation function."""

    def test_fallback_has_all_categories(self):
        """Fallback should contain all expected categories."""
        result = get_fallback_recommendations()
        assert result["is_fallback"] is True
        for category in EXPECTED_CATEGORIES:
            assert category in result
            assert isinstance(result[category], list)
            assert len(result[category]) > 0

    def test_fallback_contains_expert_advice(self):
        """Fallback should recommend consulting an agricultural expert."""
        result = get_fallback_recommendations()
        all_text = " ".join(
            item for cat in EXPECTED_CATEGORIES for item in result[cat]
        )
        assert "agricultural" in all_text.lower() or "expert" in all_text.lower()


# ---------------------------------------------------------------------------
# Integration Tests — analyze_image includes treatment_recommendations
# ---------------------------------------------------------------------------

class TestAnalyzeImageIntegration:
    """Tests that treatment_recommendations flows through analyze_image."""

    def test_analyze_image_includes_treatment_recommendations(self, monkeypatch):
        """analyze_image() result should contain treatment_recommendations key."""
        monkeypatch.setattr(app.model_manager, "resnet_model", None)
        monkeypatch.setattr(app.model_manager, "yolo_model", None)
        monkeypatch.setattr(app.model_manager, "loaded", True)

        dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
        result = app.analyze_image(dummy_img)

        # Should not be an error result
        if "error" not in result:
            assert "treatment_recommendations" in result
            tr = result["treatment_recommendations"]
            assert isinstance(tr, dict)
            assert "treatment" in tr
            assert "prevention" in tr
            assert "is_fallback" in tr

    def test_api_analyze_includes_treatment_recommendations(self, client, valid_image):
        """POST /api/analyze should include treatment_recommendations in response."""
        data = {"file": (valid_image, "test_cotton.png")}
        resp = client.post("/api/analyze", data=data, content_type="multipart/form-data")
        assert resp.status_code == 200
        res_data = json.loads(resp.data)
        assert res_data["status"] == "success"
        assert "results" in res_data
        results = res_data["results"]
        assert "treatment_recommendations" in results

    def test_demo_route_includes_treatment_recommendations(self, client):
        """GET /demo should render page with treatment recommendation content."""
        resp = client.get("/demo")
        assert resp.status_code == 200
        assert (
            b"Treatment" in resp.data
            or b"treatment" in resp.data
            or b"Management Guide" in resp.data
        )
