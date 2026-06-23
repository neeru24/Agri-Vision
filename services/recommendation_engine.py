"""
Agri-Vision Recommendation Engine
==================================
Context-aware, rule-based crop treatment recommendation service.
Provides structured agricultural guidance based on detected disease,
crop type, and prediction confidence.

Supports 6 recommendation categories per disease:
  - Treatment suggestions
  - Prevention tips
  - Irrigation guidance
  - Organic control methods
  - Chemical/pesticide category suggestions (generic, no brand names)
  - Severity-level advice

Designed to be modular and easily extendable to new crops and diseases.
"""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RECOMMENDATION DATA — Cotton Crop
# ---------------------------------------------------------------------------
# Each disease maps to 6 categories of farmer-friendly guidance.
# All recommendations are generic (no branded pesticide names).
# ---------------------------------------------------------------------------

RECOMMENDATIONS: Dict[str, Dict[str, Dict[str, Any]]] = {
    "cotton": {
        "Aphids": {
            "treatment": [
                "Spray a strong jet of water on affected leaves to dislodge aphid colonies.",
                "Remove and destroy heavily infested leaves and shoots.",
                "Apply neem-based insecticide spray at recommended dosage.",
                "Introduce or conserve natural predators such as ladybugs and lacewings.",
            ],
            "prevention": [
                "Avoid excessive nitrogen fertilization which promotes soft, sap-rich growth.",
                "Use reflective mulches to deter aphids from landing on crops.",
                "Plant trap crops like mustard on field borders to attract aphids away.",
                "Scout fields weekly for early aphid colonies on leaf undersides.",
            ],
            "irrigation": [
                "Maintain consistent soil moisture — water-stressed plants attract more aphids.",
                "Use drip irrigation to keep foliage dry and reduce honeydew-related issues.",
                "Avoid overhead sprinklers during active aphid infestation.",
            ],
            "organic_control": [
                "Apply neem oil (1–2%) foliar spray at weekly intervals.",
                "Use insecticidal soap solution targeting soft-bodied insects.",
                "Release Chrysoperla (lacewing) larvae as biological control agents.",
            ],
            "chemical_control": [
                "Systemic insecticides (imidacloprid group) may be used for severe infestations.",
                "Contact insecticides (pyrethroid group) can provide quick knockdown.",
                "Rotate between insecticide groups to prevent resistance buildup.",
            ],
            "severity_advice": [
                "Mild: Monitor and use water jets. No chemical intervention needed.",
                "Moderate: Apply neem oil or insecticidal soap. Scout every 3 days.",
                "Severe: Use recommended systemic insecticide. Consult local extension officer.",
            ],
        },
        "Army worm": {
            "treatment": [
                "Hand-pick and destroy visible larvae during early morning or late evening.",
                "Apply Bacillus thuringiensis (Bt) spray for biological larval control.",
                "Use light traps to monitor and reduce adult moth populations.",
                "Remove crop residues and weeds that serve as alternate hosts.",
            ],
            "prevention": [
                "Deep plough fields after harvest to expose pupae to predators and sun.",
                "Maintain clean field borders free of weeds that harbor larvae.",
                "Install pheromone traps to monitor moth activity and time interventions.",
                "Encourage bird populations by placing perches near fields.",
            ],
            "irrigation": [
                "Flood irrigation can help expose and drown soil-dwelling pupae.",
                "Maintain normal irrigation schedule — army worms are not moisture-driven.",
                "Avoid waterlogging which can weaken plant root systems during attack.",
            ],
            "organic_control": [
                "Spray Bacillus thuringiensis (Bt) at the early larval stage for best results.",
                "Apply neem kernel extract (5%) as a feeding deterrent.",
                "Release Trichogramma egg parasitoids to target moth eggs.",
            ],
            "chemical_control": [
                "Emamectin benzoate-based insecticides are effective against caterpillars.",
                "Chlorantraniliprole group offers targeted caterpillar control with low toxicity.",
                "Apply insecticide during evening hours when larvae are actively feeding.",
            ],
            "severity_advice": [
                "Mild: Hand-pick larvae and monitor. Bt spray if count exceeds threshold.",
                "Moderate: Apply Bt or neem extract. Increase scouting to every 2 days.",
                "Severe: Use recommended insecticide immediately. Consult agricultural expert.",
            ],
        },
        "Bacterial blight": {
            "treatment": [
                "Remove and destroy infected plant parts immediately to limit spread.",
                "Apply copper-based bactericide spray at recommended intervals.",
                "Avoid working in fields when foliage is wet to prevent bacterial spread.",
                "Ensure proper plant spacing to improve air circulation.",
            ],
            "prevention": [
                "Use certified disease-free seeds from reliable sources.",
                "Treat seeds with recommended bactericide before sowing.",
                "Practice crop rotation with non-host crops for at least 2 seasons.",
                "Remove and burn crop residues after harvest to eliminate bacterial inoculum.",
            ],
            "irrigation": [
                "Avoid overhead or sprinkler irrigation which splashes bacteria between plants.",
                "Use drip or furrow irrigation to keep leaves dry.",
                "Irrigate during morning hours so foliage dries quickly.",
            ],
            "organic_control": [
                "Apply copper hydroxide spray as a preventive bactericide.",
                "Use Pseudomonas fluorescens-based biocontrol agents as seed treatment.",
                "Maintain soil health with compost to boost plant disease resistance.",
            ],
            "chemical_control": [
                "Copper oxychloride sprays are the primary chemical control for bacterial blight.",
                "Streptocycline (antibiotic) at recommended dose can reduce bacterial load.",
                "Combine copper spray with mancozeb for broader protection.",
            ],
            "severity_advice": [
                "Mild: Remove affected leaves. Apply preventive copper spray.",
                "Moderate: Increase spray frequency. Avoid overhead irrigation immediately.",
                "Severe: Destroy heavily infected plants. Consult extension officer for area-wide management.",
            ],
        },
        "Cotton Boll Rot": {
            "treatment": [
                "Remove and destroy all rotten bolls promptly to prevent spore spread.",
                "Improve field drainage to reduce standing water around plant base.",
                "Apply recommended fungicide spray targeting boll rot pathogens.",
                "Manage insect pests (bollworms) that create entry wounds for rot organisms.",
            ],
            "prevention": [
                "Ensure proper plant-to-plant and row-to-row spacing for air circulation.",
                "Avoid excess nitrogen which promotes dense canopy and moisture retention.",
                "Control boll-feeding insects that create wounds for secondary infection.",
                "Time irrigation to avoid prolonged soil saturation during boll development.",
            ],
            "irrigation": [
                "Reduce irrigation frequency during boll maturation to prevent excess moisture.",
                "Ensure field has adequate drainage — waterlogging accelerates boll rot.",
                "Stop irrigation entirely once bolls begin to split open.",
            ],
            "organic_control": [
                "Apply Trichoderma-based bio-fungicide as a soil drench around plant base.",
                "Use neem cake in soil to suppress soil-borne rot organisms.",
                "Maintain balanced organic matter in soil for beneficial microbial activity.",
            ],
            "chemical_control": [
                "Carbendazim or mancozeb-based fungicides target common boll rot pathogens.",
                "Copper fungicide sprays can provide broad-spectrum protection.",
                "Avoid fungicide application during rain — apply in dry weather for best uptake.",
            ],
            "severity_advice": [
                "Mild: Remove affected bolls. Improve drainage and reduce watering.",
                "Moderate: Apply fungicide and adjust irrigation. Monitor healthy bolls closely.",
                "Severe: Widespread rot — consider early harvest of unaffected bolls. Seek expert advice.",
            ],
        },
        "Green Cotton Boll": {
            "treatment": [
                "No active disease treatment required — bolls are in normal development.",
                "Continue regular scouting for early signs of boll-feeding pests.",
                "Monitor boll surface for any discoloration or insect entry holes.",
                "Maintain balanced nutrient supply to support healthy boll filling.",
            ],
            "prevention": [
                "Scout for bollworm eggs and early-instar larvae on boll surfaces.",
                "Maintain clean field borders to reduce pest pressure on bolls.",
                "Ensure adequate potassium supply for proper boll fiber development.",
                "Monitor weather forecasts — heavy rain during boll fill increases rot risk.",
            ],
            "irrigation": [
                "Provide consistent and adequate irrigation during boll development phase.",
                "Avoid water stress which can cause premature boll shedding.",
                "Reduce irrigation gradually as bolls approach maturity.",
            ],
            "organic_control": [
                "Release Trichogramma wasps to control bollworm eggs.",
                "Apply neem oil as a general pest deterrent during boll development.",
                "Use pheromone traps to monitor bollworm moth activity.",
            ],
            "chemical_control": [
                "Apply targeted insecticide only if bollworm threshold is exceeded.",
                "Avoid broad-spectrum sprays that kill beneficial insects.",
                "Use integrated pest management (IPM) as the primary strategy.",
            ],
            "severity_advice": [
                "Healthy boll development detected. Continue normal crop management.",
                "Monitor for pest damage and environmental stress during this critical phase.",
                "Ensure timely irrigation and nutrition for optimal fiber quality.",
            ],
        },
        "Healthy": {
            "treatment": [
                "No treatment required — crop is healthy with no disease symptoms detected.",
                "Continue routine field monitoring at least once per week.",
                "Maintain current crop management practices.",
                "Document this health status for comparison in future scans.",
            ],
            "prevention": [
                "Continue preventive scouting for early signs of pest or disease.",
                "Maintain balanced fertilization to sustain plant vigor.",
                "Practice crop rotation in subsequent seasons to preserve soil health.",
                "Keep field borders clean and free of weed hosts.",
            ],
            "irrigation": [
                "Maintain current irrigation schedule based on crop growth stage.",
                "Monitor soil moisture levels and adjust watering as needed.",
                "Use mulching to conserve soil moisture and reduce irrigation frequency.",
            ],
            "organic_control": [
                "Apply compost or organic manure to maintain soil microbial health.",
                "Use bio-fertilizers to enhance nutrient uptake naturally.",
                "Maintain biodiversity in and around the field to support natural pest control.",
            ],
            "chemical_control": [
                "No chemical intervention needed for healthy crops.",
                "Avoid unnecessary pesticide applications to protect beneficial organisms.",
                "Reserve chemical controls for confirmed pest or disease outbreaks.",
            ],
            "severity_advice": [
                "Crop is healthy. No immediate action required.",
                "Continue monitoring and maintain preventive practices.",
                "Focus on optimizing yield through proper nutrition and irrigation management.",
            ],
        },
        "Powdery mildew": {
            "treatment": [
                "Remove and destroy heavily infected leaves to reduce spore load.",
                "Apply sulfur-based or systemic fungicide spray at first sign of infection.",
                "Improve plant spacing and prune lower canopy to increase airflow.",
                "Avoid late-evening irrigation that keeps foliage wet overnight.",
            ],
            "prevention": [
                "Select disease-resistant cotton varieties where available.",
                "Avoid excessive nitrogen which promotes dense, susceptible foliage.",
                "Maintain proper plant spacing for adequate air circulation.",
                "Scout fields regularly during humid weather for early white patches.",
            ],
            "irrigation": [
                "Irrigate in the morning so foliage dries during the day.",
                "Use drip irrigation to minimize leaf wetness.",
                "Avoid overhead sprinklers especially in humid conditions.",
            ],
            "organic_control": [
                "Apply potassium bicarbonate spray as an organic fungicide alternative.",
                "Use milk-water solution (1:9 ratio) spray — shown to suppress mildew.",
                "Neem oil spray can help as a preventive measure against fungal spread.",
            ],
            "chemical_control": [
                "Wettable sulfur is the primary contact fungicide for powdery mildew.",
                "Triazole-based systemic fungicides provide curative and preventive control.",
                "Rotate fungicide groups to prevent pathogen resistance.",
            ],
            "severity_advice": [
                "Mild: Remove affected leaves. Apply sulfur spray preventively.",
                "Moderate: Use systemic fungicide. Increase air circulation by pruning.",
                "Severe: Intensive fungicide program needed. Consult plant pathologist.",
            ],
        },
        "Target Spot": {
            "treatment": [
                "Remove and destroy lower canopy leaves showing circular lesions.",
                "Apply broad-spectrum fungicide at recommended intervals.",
                "Reduce leaf wetness duration by adjusting irrigation timing.",
                "Improve canopy airflow through proper spacing and pruning.",
            ],
            "prevention": [
                "Practice crop rotation — avoid planting cotton in the same field consecutively.",
                "Remove and destroy crop residues after harvest to eliminate fungal inoculum.",
                "Use resistant varieties where available in your region.",
                "Maintain proper nitrogen balance — excessive N promotes susceptible growth.",
            ],
            "irrigation": [
                "Water plants at the base, not overhead, to keep leaves dry.",
                "Irrigate in early morning to allow foliage to dry before nightfall.",
                "Ensure good field drainage to reduce ambient humidity around plants.",
            ],
            "organic_control": [
                "Apply Trichoderma-based bio-fungicide preventively to the soil and foliage.",
                "Use copper-based organic fungicide for early-stage infections.",
                "Maintain healthy soil biology with compost to suppress fungal pathogens.",
            ],
            "chemical_control": [
                "Strobilurin-based fungicides provide effective preventive control.",
                "Triazole fungicides offer curative action against target spot.",
                "Alternate between fungicide groups to manage resistance.",
            ],
            "severity_advice": [
                "Mild: Remove lower affected leaves. Monitor spread pattern.",
                "Moderate: Begin fungicide spray program. Adjust irrigation to reduce wetness.",
                "Severe: Intensive treatment required. Premature defoliation risk — consult expert.",
            ],
        },
    },
}


# ---------------------------------------------------------------------------
# FALLBACK RECOMMENDATIONS — used when disease/crop is not recognized
# ---------------------------------------------------------------------------

FALLBACK_RECOMMENDATIONS: Dict[str, Any] = {
    "treatment": [
        "Consult a local agricultural extension officer for specific treatment advice.",
        "Upload a clearer image for more accurate disease identification.",
        "Monitor the crop closely and document any changes in symptoms.",
    ],
    "prevention": [
        "Practice crop rotation and maintain field hygiene.",
        "Use certified disease-free seeds from reliable sources.",
        "Scout fields regularly for early signs of pests and diseases.",
    ],
    "irrigation": [
        "Maintain a regular irrigation schedule appropriate for the crop stage.",
        "Avoid overwatering and ensure adequate field drainage.",
    ],
    "organic_control": [
        "Use neem-based products as a general pest and disease deterrent.",
        "Maintain soil health with organic compost and bio-fertilizers.",
    ],
    "chemical_control": [
        "Avoid applying chemicals without confirmed disease identification.",
        "Consult a local agricultural expert before any pesticide application.",
    ],
    "severity_advice": [
        "Unable to determine severity without a confirmed diagnosis.",
        "Consult a local agricultural expert for further crop management guidance.",
    ],
}


# ---------------------------------------------------------------------------
# CONFIDENCE THRESHOLDS
# ---------------------------------------------------------------------------

LOW_CONFIDENCE_THRESHOLD = 0.45
MEDIUM_CONFIDENCE_THRESHOLD = 0.70
HIGH_CONFIDENCE_THRESHOLD = 0.85  # Below this, treatment recommendations are withheld


# ---------------------------------------------------------------------------
# PUBLIC API
# ---------------------------------------------------------------------------

def get_confidence_advisory(confidence: Optional[float]) -> Optional[str]:
    """
    Generate a confidence-based advisory message.

    Args:
        confidence: Model prediction confidence (0.0 to 1.0), or None.

    Returns:
        Advisory string if confidence is low/medium, None if high or absent.
    """
    if confidence is None:
        return None

    if confidence < LOW_CONFIDENCE_THRESHOLD:
        return (
            "⚠️ Low prediction confidence. The AI model is uncertain about this diagnosis. "
            "Please upload a clearer image with better lighting, or consult a local "
            "agricultural expert for manual verification before applying any treatment."
        )
    elif confidence < MEDIUM_CONFIDENCE_THRESHOLD:
        return (
            "ℹ️ Moderate prediction confidence. The recommendations below are based on the "
            "most likely diagnosis. Monitor the crop closely and consider a second opinion "
            "if symptoms do not match the predicted disease."
        )

    return None


def get_fallback_recommendations() -> Dict[str, Any]:
    """
    Return generic agricultural guidance when no specific recommendation
    mapping is found for the detected disease or crop.

    Returns:
        Dict with all recommendation categories populated with generic advice.
    """
    return {
        "crop_type": "unknown",
        "disease_name": "unknown",
        "treatment": FALLBACK_RECOMMENDATIONS["treatment"],
        "prevention": FALLBACK_RECOMMENDATIONS["prevention"],
        "irrigation": FALLBACK_RECOMMENDATIONS["irrigation"],
        "organic_control": FALLBACK_RECOMMENDATIONS["organic_control"],
        "chemical_control": FALLBACK_RECOMMENDATIONS["chemical_control"],
        "severity_advice": FALLBACK_RECOMMENDATIONS["severity_advice"],
        "confidence_advisory": None,
        "is_fallback": True,
    }


def get_recommendations(
    crop_type: str,
    disease_name: str,
    confidence: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Retrieve structured treatment recommendations for a detected disease.

    Args:
        crop_type:    The crop type (e.g., "cotton").
        disease_name: The predicted disease class name.
        confidence:   Optional model confidence score (0.0 to 1.0).

    Returns:
        Dict containing:
            - crop_type (str)
            - disease_name (str)
            - treatment (list[str])
            - prevention (list[str])
            - irrigation (list[str])
            - organic_control (list[str])
            - chemical_control (list[str])
            - severity_advice (list[str])
            - confidence_advisory (str or None)
            - is_fallback (bool)
    """
    # Block recommendations when model confidence is too low to be actionable.
    # A sub-threshold prediction should not drive farmer treatment decisions.
    if confidence is not None and confidence < HIGH_CONFIDENCE_THRESHOLD:
        logger.warning(
            "Prediction confidence %.2f below threshold %.2f for crop '%s' / disease '%s'. "
            "Withholding treatment recommendations to prevent false positives.",
            confidence, HIGH_CONFIDENCE_THRESHOLD, crop_type, disease_name,
        )
        return {
            "crop_type": crop_type or "unknown",
            "disease_name": disease_name or "unknown",
            "treatment": [],
            "prevention": [],
            "irrigation": [],
            "organic_control": [],
            "chemical_control": [],
            "severity_advice": [],
            "confidence_advisory": (
                "Prediction confidence is too low to make a reliable treatment recommendation. "
                "Please re-upload a clearer image or consult a local agricultural expert."
            ),
            "is_fallback": True,
            "no_treatment": True,
        }

    # Normalize inputs for safe lookup
    crop_key = crop_type.strip().lower() if crop_type else ""
    disease_key = disease_name.strip() if disease_name else ""

    # Look up crop-level recommendations
    crop_data = RECOMMENDATIONS.get(crop_key)
    if crop_data is None:
        logger.info(
            "No recommendations found for crop '%s'. Returning fallback.", crop_type
        )
        result = get_fallback_recommendations()
        result["crop_type"] = crop_type or "unknown"
        result["disease_name"] = disease_name or "unknown"
        result["confidence_advisory"] = get_confidence_advisory(confidence)
        return result

    # Look up disease-level recommendations
    disease_data = crop_data.get(disease_key)
    if disease_data is None:
        logger.info(
            "No recommendations found for disease '%s' in crop '%s'. Returning fallback.",
            disease_name,
            crop_type,
        )
        result = get_fallback_recommendations()
        result["crop_type"] = crop_type or "unknown"
        result["disease_name"] = disease_name or "unknown"
        result["confidence_advisory"] = get_confidence_advisory(confidence)
        return result

    # Build response from matched data
    return {
        "crop_type": crop_type,
        "disease_name": disease_name,
        "treatment": disease_data.get("treatment", []),
        "prevention": disease_data.get("prevention", []),
        "irrigation": disease_data.get("irrigation", []),
        "organic_control": disease_data.get("organic_control", []),
        "chemical_control": disease_data.get("chemical_control", []),
        "severity_advice": disease_data.get("severity_advice", []),
        "confidence_advisory": get_confidence_advisory(confidence),
        "is_fallback": False,
    }


def build_smart_farming_plan(
    recommendation_data: Dict[str, Any],
    *,
    health_score: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Build a short post-analysis action plan for farmers.

    The full recommendation object can be long. This helper extracts the most
    useful next steps into the four categories requested by the results UI.
    """
    if not recommendation_data:
        recommendation_data = get_fallback_recommendations()

    monitoring = list(recommendation_data.get("severity_advice") or [])
    if health_score is not None:
        if health_score < 50:
            monitoring.insert(0, "Scout the affected area every 2 days until the crop health score improves.")
        elif health_score < 70:
            monitoring.insert(0, "Recheck the crop within 4-5 days and compare symptoms with the current scan.")
        else:
            monitoring.insert(0, "Keep weekly scouting records so future scans can confirm the healthy trend.")

    return {
        "prevention_tips": list(recommendation_data.get("prevention") or [])[:3],
        "treatment_steps": list(recommendation_data.get("treatment") or [])[:3],
        "irrigation_suggestions": list(recommendation_data.get("irrigation") or [])[:3],
        "monitoring_recommendations": monitoring[:3],
        "confidence_advisory": recommendation_data.get("confidence_advisory"),
        "disease_name": recommendation_data.get("disease_name", "unknown"),
    }
