"""Crop rotation planning helpers for soil health recommendations."""

CROP_FAMILIES = {
    "cotton": {
        "family": "malvaceae",
        "nutrient_load": "heavy",
        "recommended_next": ["chickpea", "green gram", "wheat"],
    },
    "rice": {
        "family": "poaceae",
        "nutrient_load": "heavy",
        "recommended_next": ["mustard", "lentil", "vegetables"],
    },
    "wheat": {
        "family": "poaceae",
        "nutrient_load": "medium",
        "recommended_next": ["mung bean", "cotton", "soybean"],
    },
    "maize": {
        "family": "poaceae",
        "nutrient_load": "heavy",
        "recommended_next": ["cowpea", "potato", "mustard"],
    },
}

SOIL_COVER_CROPS = {
    "sandy": ["cowpea", "sesbania"],
    "clay": ["mustard", "chickpea"],
    "loamy": ["green gram", "clover"],
    "default": ["green manure", "legume cover crop"],
}


def build_rotation_plan(previous_crop=None, soil_type=None, goal=None):
    """Return next-crop and cover-crop recommendations."""
    crop = (previous_crop or "cotton").strip().lower()
    soil = (soil_type or "default").strip().lower()
    rotation_goal = (goal or "soil_health").strip().lower()

    profile = CROP_FAMILIES.get(crop, CROP_FAMILIES["cotton"])
    cover_crops = SOIL_COVER_CROPS.get(soil, SOIL_COVER_CROPS["default"])
    has_legume = any("gram" in crop_name or "bean" in crop_name for crop_name in profile["recommended_next"])
    sustainability_score = 84 if has_legume else 76

    return {
        "previous_crop": crop,
        "crop_family": profile["family"],
        "nutrient_load": profile["nutrient_load"],
        "recommended_next_crops": profile["recommended_next"],
        "cover_crops": cover_crops,
        "soil_type": soil,
        "goal": rotation_goal,
        "sustainability_score": sustainability_score,
        "guidance": [
            "Avoid repeating the same crop family in consecutive seasons.",
            "Use legumes or green manure to rebuild nitrogen.",
            "Track yield and pest pressure after each season.",
        ],
    }
