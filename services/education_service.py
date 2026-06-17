"""Curated farmer education resources for in-app learning."""

EDUCATION_RESOURCES = [
    {
        "id": "cotton-disease-scouting",
        "title": "Cotton Disease Scouting Basics",
        "crop": "cotton",
        "topic": "disease",
        "language": "en",
        "duration_minutes": 8,
        "difficulty": "beginner",
        "summary": "Identify common leaf spots, mildew, boll rot, and when to request expert help.",
        "steps": [
            "Inspect upper and lower leaves in morning light.",
            "Check five random plants from each field corner and the center.",
            "Photograph symptoms before spraying.",
            "Compare spread after 48 hours.",
        ],
    },
    {
        "id": "cotton-pest-ipm",
        "title": "Integrated Pest Management for Cotton",
        "crop": "cotton",
        "topic": "pest",
        "language": "en",
        "duration_minutes": 10,
        "difficulty": "intermediate",
        "summary": "Use field scouting, threshold-based treatment, and lower-risk controls first.",
        "steps": [
            "Use sticky traps near field borders.",
            "Record pest count and affected plant percentage.",
            "Prefer biological or neem-based controls for early infestations.",
            "Escalate to local expert guidance if damage crosses threshold.",
        ],
    },
    {
        "id": "safe-irrigation",
        "title": "Safe Irrigation and Leaf Wetness Control",
        "crop": "all",
        "topic": "irrigation",
        "language": "en",
        "duration_minutes": 6,
        "difficulty": "beginner",
        "summary": "Avoid overwatering and reduce disease risk with simple moisture checks.",
        "steps": [
            "Check soil moisture 5 cm below surface before watering.",
            "Avoid overhead irrigation when fungal pressure is high.",
            "Water early in the day to reduce overnight leaf wetness.",
            "Track rainfall before repeating irrigation.",
        ],
    },
]


def list_education_resources(crop=None, topic=None, language=None):
    """Return filtered education resources with stable API-ready fields."""
    resources = EDUCATION_RESOURCES

    if crop:
        crop = crop.lower()
        resources = [item for item in resources if item["crop"] in {crop, "all"}]

    if topic:
        topic = topic.lower()
        resources = [item for item in resources if item["topic"] == topic]

    if language:
        language = language.lower()
        resources = [item for item in resources if item["language"] == language]

    return {
        "count": len(resources),
        "resources": resources,
        "topics": sorted({item["topic"] for item in EDUCATION_RESOURCES}),
    }
