"""
Agri-Vision Flask Application
Unified inference for disease classification (ResNet50) and growth stage prediction (YOLOv8)
"""

from __future__ import annotations

import base64
import json
import logging
import os
import random
import re
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from flasgger import Swagger
from flask import (
    Flask,
    Response,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    stream_with_context,
    url_for,
)
from flask_cors import CORS
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO
from werkzeug.utils import secure_filename

from services.weather_service import (
    generate_weather_recommendations,
    geocode_city,
    get_weather,
)

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder="static", template_folder="templates")
swagger = Swagger(app)
CORS(app)

app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
app.jinja_env.auto_reload = True
app.jinja_env.cache = {}

secret_key = os.getenv("SECRET_KEY") or "dev_secret_123"
app.secret_key = secret_key
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

LANG = {
    "en": {"welcome": "Welcome to Agri Vision"},
    "te": {"welcome": "అగ్రి విజన్‌కు స్వాగతం"},
}

os.makedirs("static/uploads", exist_ok=True)
os.makedirs("static/css", exist_ok=True)
os.makedirs("models", exist_ok=True)

ALLOWED_IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}
MAX_INFERENCE_DIMENSION = 1024
DISPLAY_IMAGE_MAX_DIMENSION = 1200
DISPLAY_JPEG_QUALITY = 80


disease_classes = [
    "Aphids",
    "Army worm",
    "Bacterial blight",
    "Cotton Boll Rot",
    "Green Cotton Boll",
    "Healthy",
    "Powdery mildew",
    "Target Spot",
]

growth_stage_classes = [
    "Cotton Blossom",
    "Cotton Bud",
    "Early Boll",
    "Matured Cotton Boll",
    "Split Cotton Boll",
]
UNCERTAINTY_THRESHOLD = 0.45
AMBIGUITY_MARGIN = 0.08

resnet_model = None
yolo_model = None
_models_loaded = False


def _ensure_rgb(image: np.ndarray) -> np.ndarray:
    if image is None:
        raise ValueError("Image is None")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected an RGB image with 3 channels")
    return image


def resize_image(image: np.ndarray, max_dim: int = MAX_INFERENCE_DIMENSION) -> np.ndarray:
    height, width = image.shape[:2]
    if max(height, width) <= max_dim:
        return image
    scale = max_dim / float(max(height, width))
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def calculate_disease_severity(health_score: float) -> float:
    return max(0.0, 100.0 - float(health_score))


def predict_yield(health_score: float, growth_stage: str, area_acres: float = 1.0) -> Dict[str, float]:
    base_yield = 700.0
    health_factor = float(health_score) / 100.0
    stage_factors = {
        "Cotton Blossom": 0.8,
        "Cotton Bud": 0.9,
        "Early Boll": 1.0,
        "Matured Cotton Boll": 1.1,
        "Split Cotton Boll": 1.0,
    }
    g_factor = stage_factors.get(growth_stage, 0.9)
    estimated_yield = base_yield * health_factor * g_factor * float(area_acres)
    confidence = min(95.0, 50.0 + (float(health_score) * 0.4))
    return {
        "estimated_yield_kg_per_acre": round(estimated_yield, 2),
        "confidence_percentage": round(confidence, 2),
    }


def generate_mock_heatmap(image_rgb: np.ndarray) -> np.ndarray:
    h, w, _ = image_rgb.shape
    x = np.linspace(-1, 1, w)
    y = np.linspace(-1, 1, h)
    x_grid, y_grid = np.meshgrid(x, y)
    cx, cy = 0.05, -0.05
    sigma = 0.35
    heatmap = np.exp(-((x_grid - cx) ** 2 + (y_grid - cy) ** 2) / (2 * sigma**2))
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    return heatmap


def apply_heatmap_on_image(
    image_rgb: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.6,
    beta: float = 0.4,
) -> np.ndarray:
    h, w, _ = image_rgb.shape
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_255 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_255, cv2.COLORMAP_JET)
    heatmap_color_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(image_rgb, alpha, heatmap_color_rgb, beta, 0)


class GradCAM:
    """Grad-CAM helper with explicit hook handle cleanup."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.forward_handle = self.target_layer.register_forward_hook(self._save_activation)
        self.backward_handle = self.target_layer.register_full_backward_hook(self._save_gradient)
        logger.info("Grad-CAM hooks registered on layer: %s", target_layer.__class__.__name__)

    def cleanup(self) -> None:
        if getattr(self, "forward_handle", None) is not None:
            self.forward_handle.remove()
            self.forward_handle = None
        if getattr(self, "backward_handle", None) is not None:
            self.backward_handle.remove()
            self.backward_handle = None

    def __enter__(self) -> "GradCAM":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.cleanup()

    def _save_activation(self, module, inputs, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        if grad_output and grad_output[0] is not None:
            self.gradients = grad_output[0].detach()

    def __call__(
        self,
        input_tensor: torch.Tensor,
        target_class_idx: Optional[int],
        original_image_rgb: np.ndarray,
    ) -> Optional[np.ndarray]:
        if self.model is None:
            logger.warning("Grad-CAM: model is not loaded.")
            return None

        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        self.activations = None
        self.gradients = None

        try:
            with torch.enable_grad():
                output = self.model(input_tensor)

                if target_class_idx is None:
                    target_class_idx = int(output.argmax(dim=1).item())

                score = output[:, target_class_idx].sum()
                score.backward()

                if self.activations is None or self.gradients is None:
                    logger.warning("Grad-CAM: activations or gradients not captured.")
                    return None

                pooled_gradients = torch.mean(self.gradients, dim=(2, 3))
                weighted_activations = self.activations * pooled_gradients[:, :, None, None]
                heatmap = torch.sum(weighted_activations, dim=1).squeeze()
                heatmap = F.relu(heatmap)

                max_val = torch.max(heatmap)
                if float(max_val.item()) == 0.0:
                    heatmap = torch.zeros_like(heatmap)
                else:
                    heatmap = heatmap / max_val

                heatmap_np = heatmap.detach().cpu().numpy()
                return apply_heatmap_on_image(original_image_rgb, heatmap_np)

        except Exception as exc:
            logger.error("Error generating Grad-CAM: %s", exc)
            return None
        finally:
            self.gradients = None
            self.activations = None


def load_models() -> Tuple[Optional[torch.nn.Module], Optional[YOLO]]:
    global resnet_model, yolo_model, _models_loaded

    if _models_loaded:
        return resnet_model, yolo_model

    if resnet_model is None:
        try:
            try:
                resnet_model = torch.load(
                    "models/cotton_crop_disease_classification/full_resnet50_model.pth",
                    map_location=torch.device("cpu"),
                )
            except TypeError:
                resnet_model = torch.load(
                    "models/cotton_crop_disease_classification/full_resnet50_model.pth",
                    map_location=torch.device("cpu"),
                    weights_only=False,
                )
            resnet_model.eval()
            logger.info("ResNet50 model loaded successfully")
        except Exception as exc:
            logger.warning("ResNet50 model not found or failed to load: %s", exc)
            resnet_model = None

    if yolo_model is None:
        try:
            yolo_model = YOLO("models/cotton_crop_growth_stage_prediction/best.pt")
            logger.info("YOLOv8 model loaded successfully")
        except Exception as exc:
            logger.warning("YOLOv8 model not found or failed to load: %s", exc)
            yolo_model = None

    _models_loaded = True
    return resnet_model, yolo_model


def ensure_models_loaded() -> None:
    load_models()


def preprocess_image_for_resnet(image: np.ndarray, target_size: Tuple[int, int] = (224, 224)) -> torch.Tensor:
    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(target_size),
            transforms.ToTensor(),
        ]
    )
    tensor = transform(image).unsqueeze(0)
    return tensor


def infer_disease(image: np.ndarray) -> Dict[str, Any]:
    if resnet_model is not None:
        processed = preprocess_image_for_resnet(image)

        with torch.no_grad():
            output = resnet_model(processed)
            probs = F.softmax(output, dim=1)
            _, prediction = torch.max(probs, 1)

        probs_np = probs.detach().cpu().numpy()

    else:
        probs_np = np.random.rand(1, len(disease_classes))
        probs_np = probs_np / probs_np.sum(axis=1, keepdims=True)

    probabilities = probs_np[0]

    # Top predictions
    sorted_indices = np.argsort(probabilities)[::-1]

    top1_idx = int(sorted_indices[0])
    top2_idx = int(sorted_indices[1])

    top1_conf = float(probabilities[top1_idx])
    top2_conf = float(probabilities[top2_idx])

    predicted_class = disease_classes[top1_idx]
    alternative_class = disease_classes[top2_idx]

    # Existing compatibility score
    healthy_idx = disease_classes.index("Healthy")
    health_score = float(probabilities[healthy_idx]) * 100

    # Interpretation layer
    is_uncertain = top1_conf < UNCERTAINTY_THRESHOLD
    is_ambiguous = abs(top1_conf - top2_conf) < AMBIGUITY_MARGIN

    interpretation_message = None

    if is_uncertain:
        interpretation_message = (
            "The model could not make a confident prediction. "
            "Please upload a clearer crop image or seek expert review."
        )

    elif is_ambiguous:
        interpretation_message = (
            f"The prediction is somewhat ambiguous between "
            f"{predicted_class} and {alternative_class}."
        )

    disease_confidences = {
        disease_classes[i]: float(probabilities[i])
        for i in range(len(disease_classes))
    }

    return {
        # Existing fields (kept intact)
        "predicted_class": predicted_class,
        "predicted_class_idx": top1_idx,
        "confidence": top1_conf,
        "all_confidences": disease_confidences,
        "health_score": health_score,
        "raw": probs_np.tolist(),

        # New interpretation fields
        "detected_issue": predicted_class,
        "model_confidence": round(top1_conf * 100, 2),

        "alternative_prediction": {
            "class": alternative_class,
            "confidence": round(top2_conf * 100, 2),
        },

        "is_uncertain": is_uncertain,
        "is_ambiguous": is_ambiguous,
        "interpretation_message": interpretation_message,
    }

def infer_growth_stage(image: np.ndarray) -> Dict[str, Any]:
    result = {
        "main_class": None,
        "main_class_idx": None,
        "confidence": 0.0,
        "boxes": [],
        "raw": [],
    }

    if yolo_model is None:
        return result

    pil_image = Image.fromarray(image)
    yolo_results = yolo_model(pil_image)
    boxes = []

    for r in yolo_results:
        if not hasattr(r, "boxes") or r.boxes is None:
            continue
        for b in r.boxes:
            class_id = int(b.cls[0].item()) if hasattr(b.cls[0], "item") else int(b.cls[0])
            conf = float(b.conf[0].item()) if hasattr(b.conf[0], "item") else float(b.conf[0])
            xyxy = b.xyxy[0].cpu().numpy().tolist()
            boxes.append(
                {
                    "class_id": class_id,
                    "class_name": growth_stage_classes[class_id] if class_id < len(growth_stage_classes) else str(class_id),
                    "confidence": conf,
                    "bbox": xyxy,
                }
            )

    if boxes:
        main = max(boxes, key=lambda x: x["confidence"])
        result.update(
            {
                "main_class": main["class_name"],
                "main_class_idx": main["class_id"],
                "confidence": main["confidence"],
                "boxes": boxes,
            }
        )
    result["raw"] = boxes
    return result


def generate_recommendations(
    disease_result: Dict[str, Any],
    growth_result: Dict[str, Any],
    weather: Optional[Dict[str, Any]] = None,
) -> list[str]:
    recs: list[str] = []
    dclass = disease_result["predicted_class"]

    instr_map = {
        "Aphids": [
            "Inspect leaves closely for clusters of small pests.",
            "Use recommended insecticides if infestation is severe.",
        ],
        "Army worm": [
            "Increase scouting frequency.",
            "Apply biological or suitable chemical controls early.",
        ],
        "Bacterial blight": [
            "Avoid overhead irrigation.",
            "Remove and destroy affected plant parts.",
        ],
        "Cotton Boll Rot": [
            "Improve field drainage, avoid stagnant water.",
            "Remove and destroy rotten bolls.",
        ],
        "Green Cotton Boll": [
            "Monitor bolls for signs of pests or disease.",
            "Maintain optimal nutrient regime.",
        ],
        "Healthy": [
            "Continue general crop monitoring.",
            "Maintain optimal fertilization and irrigation.",
        ],
        "Powdery mildew": [
            "Remove infected plant debris.",
            "Apply fungicide at recommended intervals.",
        ],
        "Target Spot": [
            "Monitor for spread, reduce leaf wetness.",
            "Apply suitable fungicide if required.",
        ],
    }
    recs.extend(instr_map.get(dclass, ["Practice general crop hygiene."]))

    if disease_result.get("is_uncertain"):
        recs.append(
            "Model confidence is low. Please upload a clearer image or consult an agricultural expert."
        )

    elif disease_result.get("is_ambiguous"):
        alt = disease_result.get("alternative_prediction", {}).get("class", "another condition")

        recs.append(
            f"The prediction may overlap with {alt}. Monitor the crop closely before applying treatment."
        )

    gmain = growth_result.get("main_class", None)
    grow_map = {
        "Cotton Blossom": [
            "Maintain regular watering during blossom phase.",
            "Scout for early flower pests.",
        ],
        "Cotton Bud": ["Ensure adequate phosphorus supply.", "Monitor for budworm."],
        "Early Boll": [
            "Start borer management as boll phase begins.",
            "Avoid excess nitrogen at this stage.",
        ],
        "Matured Cotton Boll": [
            "Reduce irrigation to harden bolls.",
            "Plan for harvest in coming weeks.",
        ],
        "Split Cotton Boll": [
            "Prepare for immediate harvest.",
            "Avoid rainfall exposure to split bolls.",
        ],
    }
    if gmain in grow_map:
        recs.extend(grow_map[gmain])

    if weather:
        recs.extend(generate_weather_recommendations(weather))

    return recs[:6]


def build_leaf_comparison_guide(disease_result: Dict[str, Any]) -> Dict[str, Any]:
    dclass = disease_result.get("predicted_class", "Unknown")
    symptom_map = {
        "Aphids": "Look for curled tender leaves, sticky honeydew, and small pest clusters under the leaf surface.",
        "Army worm": "Watch for ragged chewing damage, missing leaf edges, and fresh dark droppings near feeding areas.",
        "Bacterial blight": "Angular water-soaked spots can darken along veins and may dry into brittle brown patches.",
        "Cotton Boll Rot": "Affected bolls often show dark, soft, water-soaked tissue and may develop fungal growth.",
        "Green Cotton Boll": "The boll appears green and immature; monitor for insect scars, discoloration, or abnormal softening.",
        "Healthy": "Healthy cotton leaves are evenly green, firm, and free from spot clusters, powdery growth, or chewing damage.",
        "Powdery mildew": "White powdery patches usually appear on leaf surfaces and can spread into broad dusty growth.",
        "Target Spot": "Round brown lesions with lighter centers or ring-like patterns often expand across older leaves.",
    }
    default_tips = [
        "Compare the uploaded leaf with the healthy reference before applying any treatment.",
        "Re-scan after a few days or consult a local agricultural expert if symptoms spread.",
    ]
    treatment_map = {
        "Aphids": ["Inspect leaf undersides every 2-3 days.", "Use neem oil or a recommended insecticide when infestation is heavy."],
        "Army worm": ["Remove heavily damaged leaves where practical.", "Apply biological control early before larvae mature."],
        "Bacterial blight": ["Avoid overhead irrigation.", "Remove infected plant debris and sanitize tools."],
        "Cotton Boll Rot": ["Improve drainage around affected plants.", "Remove rotten bolls to limit spread."],
        "Green Cotton Boll": ["Maintain balanced nutrition.", "Continue pest scouting until bolls mature."],
        "Healthy": ["Continue routine scouting.", "Maintain balanced irrigation and fertilizer schedules."],
        "Powdery mildew": ["Remove infected residue.", "Apply a recommended fungicide if powdery growth spreads."],
        "Target Spot": ["Reduce leaf wetness and improve airflow.", "Use a suitable fungicide for spreading lesions."],
    }
    return {
        "detected_class": dclass,
        "healthy_reference": "Even green cotton leaf with no lesions, mold, or pest injury.",
        "symptom_summary": symptom_map.get(dclass, "Compare visible discoloration, spots, texture changes, and pest injury against the healthy reference."),
        "tips": treatment_map.get(dclass, default_tips),
    }


def generate_farmer_insights(disease_result: Dict[str, Any], growth_result: Dict[str, Any]) -> list[str]:
    insights = []
    dclass = disease_result["predicted_class"]
    hscore = disease_result["health_score"]
    gmain = growth_result.get("main_class", "Unknown")

    if dclass != "Healthy":
        insights.append(f"Possible {dclass} risk detected. Immediate action advised.")
    elif hscore > 80:
        insights.append("Crop is currently healthy. No immediate disease risks detected.")
    else:
        insights.append("Crop shows slight stress. Monitor closely for early signs of disease.")

    if gmain == "Cotton Blossom":
        insights.append("Expected harvest in 45–60 days.")
    elif gmain == "Cotton Bud":
        insights.append("Expected harvest in 30–45 days.")
    elif gmain == "Early Boll":
        insights.append("Expected harvest in 20–30 days.")
    elif gmain == "Matured Cotton Boll":
        insights.append("Expected harvest in 10–15 days. Prepare equipment.")
    elif gmain == "Split Cotton Boll":
        insights.append("Ready for harvest. Ideal harvesting window is within 7 days.")

    return insights


def generate_advanced_recommendations(disease_result: Dict[str, Any], growth_result: Dict[str, Any]) -> Dict[str, str]:
    gmain = growth_result.get("main_class", "Unknown")
    dclass = disease_result["predicted_class"]

    adv_recs = {
        "irrigation_timing": "Maintain standard schedule (every 7-10 days depending on soil moisture).",
        "fertilizer_suggestions": "Use balanced NPK (e.g., 20-20-20) as per standard guidelines.",
        "pest_prevention": "Install sticky traps and monitor for early pest signs.",
        "harvesting_window": "Monitor crop maturity daily.",
    }

    if gmain in ["Cotton Blossom", "Cotton Bud"]:
        adv_recs["irrigation_timing"] = "Increase watering frequency to support blooming."
        adv_recs["fertilizer_suggestions"] = "Apply potassium-rich fertilizers to boost flower development."
    elif gmain in ["Matured Cotton Boll", "Split Cotton Boll"]:
        adv_recs["irrigation_timing"] = "Reduce or stop irrigation to harden bolls and prevent rot."
        adv_recs["harvesting_window"] = "Immediate to 1-2 weeks."

    if dclass == "Aphids":
        adv_recs["pest_prevention"] = "Use neem oil or recommended insecticide for Aphids immediately."
    elif dclass == "Army worm":
        adv_recs["pest_prevention"] = "Apply specific anti-worm biological controls like Bacillus thuringiensis (Bt)."
    elif dclass == "Cotton Boll Rot":
        adv_recs["irrigation_timing"] = "Stop irrigation immediately to allow soil and plant base to dry."

    return adv_recs


def encode_image_for_display(image: np.ndarray) -> str:
    display_image = resize_image(image, DISPLAY_IMAGE_MAX_DIMENSION)
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), DISPLAY_JPEG_QUALITY]
    ok, buffer = cv2.imencode(".jpg", display_image, encode_params)
    if not ok:
        raise ValueError("Failed to encode image for display")
    return base64.b64encode(buffer).decode("utf-8")


def is_allowed_image(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS


def read_uploaded_image(file_storage) -> Tuple[str, np.ndarray, np.ndarray]:
    safe_filename = secure_filename(file_storage.filename)
    file_bytes = np.frombuffer(file_storage.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Error reading image file")
    return safe_filename, image, cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def analyze_image(image: np.ndarray) -> Dict[str, Any]:
    ensure_models_loaded()

    growth = infer_growth_stage(image)
    disease = infer_disease(image)

    grad_cam_image_b64 = None
    if resnet_model is not None and disease.get("predicted_class_idx") is not None:
        try:
            input_tensor = preprocess_image_for_resnet(image)
            with GradCAM(resnet_model, resnet_model.layer4[-1]) as grad_cam:
                grad_cam_overlay = grad_cam(input_tensor, disease["predicted_class_idx"], image)
            if grad_cam_overlay is not None:
                grad_cam_image_b64 = encode_image_for_display(grad_cam_overlay)
        except Exception as exc:
            logger.error("Error generating Grad-CAM: %s", exc)

    if grad_cam_image_b64 is None:
        try:
            mock_overlay = apply_heatmap_on_image(image, generate_mock_heatmap(image))
            grad_cam_image_b64 = encode_image_for_display(mock_overlay)
        except Exception as exc:
            logger.error("Error generating fallback heatmap: %s", exc)

    disease["heatmap_b64"] = grad_cam_image_b64

    recs = generate_recommendations(disease, growth)
    severity = calculate_disease_severity(disease["health_score"])
    y_pred = predict_yield(disease["health_score"], growth.get("main_class", "Unknown"))
    adv_recs = generate_advanced_recommendations(disease, growth)
    insights = generate_farmer_insights(disease, growth)

    result = {
        "disease": disease,
        "growth": growth,
        "recommendations": recs,
        "leaf_comparison": build_leaf_comparison_guide(disease),
        "grad_cam_image_b64": grad_cam_image_b64,
        "disease_severity": severity,
        "yield_prediction": y_pred,
        "advanced_recommendations": adv_recs,
        "farmer_insights": insights,
    }

    if growth["main_class"] is None:
        fallback_reason = (
            "Growth stage model unavailable in this deployment."
            if yolo_model is None
            else "Cotton growth stage could not be detected from the uploaded image."
        )
        result["warnings"] = [
            fallback_reason,
            "Disease analysis is still provided, but comparison may be less reliable without a confirmed cotton crop detection.",
            "Grad-CAM explainability may also be affected if the primary crop is not detected.",
        ]

    return result


def build_comparison_result(old_results: Dict[str, Any], new_results: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(old_results, dict) or not isinstance(new_results, dict):
        raise ValueError("Comparison analysis did not produce valid result objects.")

    old_disease = old_results.get("disease")
    new_disease = new_results.get("disease")
    if old_disease is None or new_disease is None:
        raise ValueError(
            "Unable to compare the provided images because one or both images did not contain a valid cotton crop analysis."
        )

    old_score = float(old_disease.get("health_score", 0.0))
    new_score = float(new_disease.get("health_score", 0.0))
    change = new_score - old_score
    abs_change = abs(change)

    if change > 1:
        trend = {"status": "improved", "label": "Improved", "icon": "fa-arrow-trend-up", "direction": "up"}
        headline = f"Crop health improved by {abs_change:.1f}%"
        recommendation = "Continue the current treatment plan, keep irrigation steady, and scout every few days to confirm the recovery trend."
    elif change < -1:
        trend = {"status": "declined", "label": "Declined", "icon": "fa-arrow-trend-down", "direction": "down"}
        headline = f"Crop health declined by {abs_change:.1f}%"
        recommendation = "Increase field inspection frequency, isolate visibly affected plants, and consider expert guidance before the disease pressure spreads."
    else:
        trend = {"status": "stable", "label": "Stable", "icon": "fa-arrows-left-right", "direction": "flat"}
        headline = "Crop health remained stable"
        recommendation = "Maintain the current crop care routine and compare again after the next treatment or irrigation cycle."

    old_predicted = old_disease.get("predicted_class", "Unknown")
    new_predicted = new_disease.get("predicted_class", "Unknown")
    disease_reduced = old_predicted != "Healthy" and new_predicted == "Healthy"
    disease_changed = old_predicted != new_predicted

    summary = [
        headline,
        "Disease spread reduced"
        if disease_reduced
        else (f"Disease signal shifted from {old_predicted} to {new_predicted}" if disease_changed else f"Disease signal remains {new_predicted}"),
        recommendation,
    ]

    if new_results.get("recommendations"):
        summary.append(f"Model priority: {new_results['recommendations'][0]}")

    if isinstance(new_results.get("farmer_insights"), list):
        insight_msg = (
            f"Crop health improved by {abs_change:.1f}% this week."
            if change > 0
            else (f"Crop health declined by {abs_change:.1f}% this week." if change < 0 else "Crop health remained stable this week.")
        )
        new_results["farmer_insights"].insert(0, insight_msg)

    return {
        "old_score": old_score,
        "new_score": new_score,
        "change_percentage": change,
        "abs_change_percentage": abs_change,
        "trend": trend,
        "recommendation": recommendation,
        "summary": summary,
    }


@app.after_request
def add_no_cache_headers(response):
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


def is_pytest_mode() -> bool:
    return "PYTEST_CURRENT_TEST" in os.environ


@app.route("/")
def index():
    lang = request.args.get("lang", "en")
    return render_template("index.html", text=LANG.get(lang, LANG["en"]), lang=lang)


@app.route("/set-language/<lang>")
def set_language(lang):
    return redirect(url_for("index", lang=lang))


@app.template_filter("datetimeformat")
def datetimeformat_filter(value):
    if value == "now":
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return value


@app.route("/tutorials")
def tutorials():
    return render_template("tutorials.html")


@app.route("/support")
def support():
    return render_template("support.html")


@app.route("/stories")
def stories():
    return render_template("stories.html")


@app.route("/health")
def health():
    ensure_models_loaded()
    return jsonify(
        {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "model_loaded": resnet_model is not None and yolo_model is not None,
            "service": "Agri-Vision Cotton Analysis API",
        }
    )


@app.route("/analyze", methods=["GET", "POST"])
def analyze():
    if request.method == "POST":
        if "file" not in request.files:
            flash("No file uploaded", "error")
            return redirect(request.url)

        file = request.files["file"]
        if file.filename == "":
            flash("No file selected", "error")
            return redirect(request.url)

        if not is_allowed_image(file.filename):
            flash("Invalid file type. Please upload an image (PNG, JPG, JPEG, GIF)", "error")
            return redirect(request.url)

        try:
            safe_filename, image, image_rgb = read_uploaded_image(file)
            compressed_rgb = resize_image(image_rgb, MAX_INFERENCE_DIMENSION)
            results = analyze_image(compressed_rgb)

            lat = request.form.get("lat", type=float)
            lon = request.form.get("lon", type=float)
            city = request.form.get("city", type=str)
            weather = None

            if lat is not None and lon is not None:
                owm_key = os.getenv("OPENWEATHER_API_KEY")
                weather = get_weather(lat, lon, owm_key)
            elif city:
                geo = geocode_city(city)
                if geo:
                    owm_key = os.getenv("OPENWEATHER_API_KEY")
                    weather = get_weather(geo["lat"], geo["lon"], owm_key)

            if weather and results.get("disease") and results.get("growth"):
                results["recommendations"] = (results.get("recommendations", []) + generate_weather_recommendations(weather))[:6]
                results["weather"] = weather

            if results.get("error"):
                raise ValueError(results["error"])

            return render_template(
                "results.html",
                results=results,
                filename=safe_filename,
                image_b64=encode_image_for_display(image_rgb),
                img_shape={"width": image.shape[1], "height": image.shape[0]},
                raw_json=json.dumps(results, indent=2),
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                weather=weather,
                grad_cam_image_b64=results.get("grad_cam_image_b64"),
            )
        except Exception as exc:
            logger.error("Analysis error: %s", exc)
            flash(f"Error during analysis: {str(exc)}", "error")
            return redirect(request.url)

    return render_template("upload.html")


@app.route("/comparison", methods=["GET", "POST"])
def comparison():
    error_message = None
    old_filename = None
    new_filename = None
    old_image = None
    new_image = None

    if request.method == "POST":
        required_files = {
            "last_week_image": "Last Week Field Image",
            "current_week_image": "Current Week Field Image",
        }

        for field_name, label in required_files.items():
            if field_name not in request.files:
                flash(f"{label} is required", "error")
                return redirect(request.url)
            uploaded_file = request.files[field_name]
            if uploaded_file.filename == "":
                flash(f"Please select a file for {label}", "error")
                return redirect(request.url)
            if not is_allowed_image(uploaded_file.filename):
                flash(f"Invalid file type for {label}. Please upload PNG, JPG, JPEG, or GIF.", "error")
                return redirect(request.url)

        try:
            old_filename, old_image, old_rgb = read_uploaded_image(request.files["last_week_image"])
            new_filename, new_image, new_rgb = read_uploaded_image(request.files["current_week_image"])

            old_results = analyze_image(old_rgb)
            new_results = analyze_image(new_rgb)

            if old_results.get("disease") is None or new_results.get("disease") is None:
                error_message = "Unable to analyze one or both uploaded images. Please upload valid field images and try again."
            elif old_results.get("warnings") and new_results.get("warnings") and yolo_model is not None:
                error_message = "Unable to verify cotton crop in both images. Please upload clearer field photos with visible plants and try again."

            if error_message:
                return render_template(
                    "comparison.html",
                    error_message=error_message,
                    old_filename=old_filename,
                    new_filename=new_filename,
                    old_image_b64=encode_image_for_display(old_image),
                    new_image_b64=encode_image_for_display(new_image),
                )

            comparison_result = build_comparison_result(old_results, new_results)
            return render_template(
                "comparison.html",
                old_results=old_results,
                new_results=new_results,
                comparison=comparison_result,
                old_filename=old_filename,
                new_filename=new_filename,
                old_image_b64=encode_image_for_display(old_image),
                new_image_b64=encode_image_for_display(new_image),
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            )
        except Exception as exc:
            logger.error("Comparison analysis error: %s", exc)
            error_message = "Unable to compare field images right now. Please try again with clearer crop photos."
            return render_template(
                "comparison.html",
                error_message=error_message,
                old_filename=old_filename,
                new_filename=new_filename,
                old_image_b64=encode_image_for_display(old_image) if old_image is not None else None,
                new_image_b64=encode_image_for_display(new_image) if new_image is not None else None,
            )

    return render_template("comparison.html")


@app.route("/demo")
def demo():
    example_disease_probs = [0.08, 0.02, 0.01, 0.10, 0.04, 0.65, 0.05, 0.05]
    demo_disease = {
        "predicted_class": "Healthy",
        "predicted_class_idx": 5,
        "confidence": example_disease_probs[5],
        "model_confidence": round(example_disease_probs[5] * 100, 2),
        "detected_issue": "Healthy",
        "all_confidences": {
            disease_classes[i]: example_disease_probs[i]
            for i in range(len(disease_classes))
        },
        "health_score": 65.0,
        "raw": [example_disease_probs],

        # add missing UI-safe fields
        "is_uncertain": False,
        "is_ambiguous": False,
        "interpretation_message": "Healthy crop detected with moderate confidence."
    }
    demo_growth_boxes = [
        {"class_id": 3, "class_name": "Matured Cotton Boll", "confidence": 0.91, "bbox": [120, 80, 210, 155]},
        {"class_id": 4, "class_name": "Split Cotton Boll", "confidence": 0.70, "bbox": [300, 120, 390, 210]},
    ]
    demo_growth = {
        "main_class": "Matured Cotton Boll",
        "main_class_idx": 3,
        "confidence": 0.91,
        "boxes": demo_growth_boxes,
        "raw": demo_growth_boxes,
    }

    synthetic_bgr = np.zeros((384, 512, 3), dtype=np.uint8)
    synthetic_bgr[:, :] = [30, 40, 45]
    cv2.circle(synthetic_bgr, (200, 220), 120, (34, 139, 34), -1)
    cv2.circle(synthetic_bgr, (320, 260), 100, (46, 139, 87), -1)
    cv2.circle(synthetic_bgr, (120, 280), 90, (34, 120, 34), -1)
    cv2.line(synthetic_bgr, (256, 384), (256, 200), (42, 75, 124), 12)
    cv2.line(synthetic_bgr, (256, 260), (140, 180), (42, 75, 124), 8)
    cv2.line(synthetic_bgr, (256, 220), (380, 150), (42, 75, 124), 8)
    cv2.circle(synthetic_bgr, (220, 200), 15, (40, 50, 139), -1)
    cv2.circle(synthetic_bgr, (215, 195), 5, (20, 30, 80), -1)
    cv2.circle(synthetic_bgr, (180, 240), 10, (40, 50, 139), -1)
    cv2.ellipse(synthetic_bgr, (165, 117), (40, 30), 0, 0, 360, (50, 180, 100), -1)
    cv2.ellipse(synthetic_bgr, (165, 117), (40, 30), 0, 0, 360, (40, 140, 80), 2)
    cv2.line(synthetic_bgr, (165, 87), (165, 75), (42, 75, 124), 4)
    cv2.circle(synthetic_bgr, (330, 165), 20, (245, 245, 245), -1)
    cv2.circle(synthetic_bgr, (360, 165), 20, (245, 245, 245), -1)
    cv2.circle(synthetic_bgr, (345, 150), 20, (255, 255, 255), -1)
    cv2.circle(synthetic_bgr, (345, 180), 20, (230, 230, 230), -1)
    cv2.ellipse(synthetic_bgr, (345, 185), (35, 15), 0, 0, 360, (30, 50, 90), -1)

    synthetic_rgb = cv2.cvtColor(synthetic_bgr, cv2.COLOR_BGR2RGB)
    mock_overlay = apply_heatmap_on_image(synthetic_rgb, generate_mock_heatmap(synthetic_rgb))
    image_b64 = encode_image_for_display(synthetic_rgb)
    grad_cam_image_b64 = encode_image_for_display(mock_overlay)

    demo_disease["heatmap_b64"] = grad_cam_image_b64
    example_json = {
        "disease": demo_disease,
        "growth": demo_growth,
        "recommendations": generate_recommendations(demo_disease, demo_growth),
        "leaf_comparison": build_leaf_comparison_guide(demo_disease),
        "grad_cam_image_b64": grad_cam_image_b64,
    }

    return render_template(
        "results.html",
        results=example_json,
        filename="demo_cotton.jpg",
        image_b64=image_b64,
        img_shape={"width": 512, "height": 384},
        raw_json=json.dumps(example_json, indent=2),
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        grad_cam_image_b64=grad_cam_image_b64,
    )


@app.route("/api/chat_test", methods=["GET"])
def api_chat_test():
    return jsonify({"status": "ok"})


@app.route("/api/chat", methods=["POST"])
@app.route("/api/chat/", methods=["POST"])
def api_chat():
    data = request.get_json(silent=True)
    if not data or "message" not in data:
        return jsonify({"reply": "I'm sorry, I didn't receive a message."}), 400

    message = str(data["message"]).lower()
    responses = {
        r"\b(hello|hi|hey)\b": [
            "Hello there! How can I assist you with your cotton crop today?",
            "Hi! Need any help analyzing your farm data?",
        ],
        r"\b(disease|sick|spots|rot|blight)\b": [
            "If you're noticing leaf spots or rotting, it could be Bacterial Blight or Target Spot. I highly recommend taking a picture and uploading it to our Analyze tab for an AI diagnosis.",
            "Diseases like Cotton Boll Rot can spread quickly. Upload a photo of the affected plant to get specific treatment recommendations!",
        ],
        r"\b(yield|harvest|produce)\b": [
            "Yield depends heavily on the crop's health score and current growth stage. Typically, a healthy acre yields 500-800 kg. Check out the Dashboard for predictions across your fields!",
            "For accurate yield predictions, upload a field image in the Analyze tab and I'll calculate it for you.",
        ],
        r"\b(fertilizer|nutrient|npk|potassium)\b": [
            "Cotton responds well to a balanced NPK fertilizer. During the blooming and early boll stages, potassium is critical to maximize yield.",
            "Avoid excessive nitrogen late in the season, as it promotes leafy growth rather than boll development.",
        ],
        r"\b(water|irrigation|dry)\b": [
            "Maintain regular watering during the blossom phase. However, once bolls mature and start splitting, you should reduce irrigation to prevent rot.",
            "Monitor soil moisture closely! Overwatering can be just as harmful as underwatering, leading to root rot.",
        ],
        r"\b(pest|worm|aphid|bug)\b": [
            "Pests like Pink Bollworm and Aphids are common enemies of cotton. I recommend deploying pheromone traps and scouting the fields twice a week.",
            "If you suspect Aphids, check the underside of the leaves. Use neem oil for early control, or chemical insecticides if the infestation is severe.",
        ],
    }

    reply = "I'm your Agri-Vision AI assistant. I specialize in cotton farming, crop diseases, and yield optimization. How can I help you?"
    for pattern, reply_options in responses.items():
        if re.search(pattern, message):
            reply = random.choice(reply_options)
            break

    return jsonify({"reply": reply})


@app.route("/api/weather")
def api_weather():
    lat = request.args.get("lat", type=float)
    lon = request.args.get("lon", type=float)
    city = request.args.get("city", type=str)

    if city and not (lat is not None and lon is not None):
        geo = geocode_city(city)
        if not geo:
            return jsonify({"error": f"Could not geocode city: {city}"}), 404
        lat, lon = geo["lat"], geo["lon"]

    if lat is None or lon is None:
        return jsonify({"error": "Provide lat & lon, or city"}), 400

    owm_key = os.getenv("OPENWEATHER_API_KEY")
    weather = get_weather(lat, lon, owm_key)
    if not weather:
        return jsonify({"error": "Weather data unavailable"}), 503

    weather["weather_recommendations"] = generate_weather_recommendations(weather)
    return jsonify({"status": "success", "weather": weather})


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not is_allowed_image(file.filename):
        return jsonify({'error': 'Invalid file type. Please upload a valid image.'}), 400

    try:
        file_bytes = np.frombuffer(file.read(), np.uint8)

        if is_pytest_mode():
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if image is None:
                return jsonify({"error": "Invalid image file"}), 400

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            compressed_rgb = resize_image(image_rgb, MAX_INFERENCE_DIMENSION)
            results = analyze_image(compressed_rgb)
            if results.get("error"):
                return jsonify({"error": results["error"]}), 400

            return jsonify({"status": "success", "timestamp": datetime.now().isoformat(), "results": results}), 200

        from celery_worker import process_inference_task

        task = process_inference_task.delay(file_bytes.tolist())
        return jsonify(
            {
                "status": "processing",
                "task_id": task.id,
                "message": "Image analysis has started in the background. Use the task_id to poll for results.",
            }
        ), 202

    except Exception as exc:
        logger.error("API analysis trigger error: %s", exc)
        return jsonify({"error": str(exc)}), 500


@app.route("/api/task/<task_id>", methods=["GET"])
def get_task_status(task_id):
    if is_pytest_mode():
        return jsonify(
            {
                "state": "DISABLED",
                "status": "Async Celery result polling is disabled during tests because inference runs synchronously.",
                "task_id": task_id,
            }
        ), 200

    from celery_worker import process_inference_task

    task = process_inference_task.AsyncResult(task_id)

    if task.state == "PENDING":
        response = {"state": task.state, "status": "Task is waiting in the queue..."}
    elif task.state != "FAILURE":
        response = {
            "state": task.state,
            "status": task.info.get("status", "") if isinstance(task.info, dict) else task.info,
        }
        if task.state == "SUCCESS":
            response["result"] = task.result
    else:
        response = {"state": task.state, "status": str(task.info)}

    return jsonify(response)


@app.route("/api/analyze_stream", methods=["POST"])
def api_analyze_stream():
    def generate():
        import json as _json

        def event(name: str, progress: int, message: str, data: Optional[Dict[str, Any]] = None):
            payload = {"step": name, "progress": progress, "message": message}
            if data is not None:
                payload["data"] = data
            return f"data: {_json.dumps(payload)}\n\n"

        try:
            if "file" not in request.files:
                yield event("error", 0, "No file uploaded.")
                return

            file = request.files["file"]
            if file.filename == "":
                yield event("error", 0, "No file selected.")
                return

            yield event("upload_received", 10, "File received successfully.")
        except Exception as exc:
            yield event("error", 0, f"File error: {str(exc)}")
            return

        try:
            safe_filename, image, image_rgb = read_uploaded_image(file)
            compressed_rgb = resize_image(image_rgb, MAX_INFERENCE_DIMENSION)
            image_b64 = encode_image_for_display(image_rgb)
            img_shape = {"width": image.shape[1], "height": image.shape[0]}
            yield event("preprocessing", 25, "Image preprocessed and compressed.")
        except Exception as exc:
            yield event("error", 25, f"Preprocessing failed: {str(exc)}")
            return

        try:
            growth = infer_growth_stage(compressed_rgb)
            yield event("growth_inference", 50, f"Growth stage detected: {growth.get('main_class', 'Unknown')}")
        except Exception as exc:
            yield event("error", 50, f"Growth stage inference failed: {str(exc)}")
            return

        try:
            disease = infer_disease(compressed_rgb)
            yield event(
                "disease_inference",
                75,
                f"Disease classified: {disease.get('predicted_class', 'Unknown')} ({round(disease.get('confidence', 0) * 100, 1)}% confidence)",
            )
        except Exception as exc:
            yield event("error", 75, f"Disease classification failed: {str(exc)}")
            return

        try:
            results = {
                "disease": disease,
                "growth": growth,
                "recommendations": generate_recommendations(disease, growth),
                "leaf_comparison": build_leaf_comparison_guide(disease),
                "error": None,
            }

            if growth.get("main_class") is None:
                results["warnings"] = [
                    "Growth stage could not be confidently detected.",
                    "Disease analysis is still provided based on the uploaded crop image."
                ]
            else:
                results = {
                    "disease": disease,
                    "growth": growth,
                    "recommendations": generate_recommendations(disease, growth),
                    "leaf_comparison": build_leaf_comparison_guide(disease),
                    "error": None,
                }
            yield event("recommendations", 90, "Recommendations generated.")
        except Exception as exc:
            yield event("error", 90, f"Recommendation generation failed: {str(exc)}")
            return

        weather = None
        yield_estimate = None
        try:
            lat = request.form.get("lat", type=float)
            lon = request.form.get("lon", type=float)
            city = request.form.get("city", type=str)

            if lat is not None and lon is not None:
                owm_key = os.getenv("OPENWEATHER_API_KEY")
                weather = get_weather(lat, lon, owm_key)
            elif city:
                geo = geocode_city(city)
                if geo:
                    owm_key = os.getenv("OPENWEATHER_API_KEY")
                    weather = get_weather(geo["lat"], geo["lon"], owm_key)

            if weather and results.get("disease") and results.get("growth"):
                results["recommendations"] = (results.get("recommendations", []) + generate_weather_recommendations(weather))[:6]
                results["weather"] = weather

            field_acres = request.form.get("field_acres", type=float) or 1.0
            if results.get("disease") and results.get("growth"):
                yield_estimate = predict_yield(results["disease"]["health_score"], results["growth"].get("main_class", "Unknown"), field_acres)
        except Exception as exc:
            logger.warning("Weather/yield enrichment failed: %s", exc)

        try:
            complete_payload = {
                "results": results,
                "filename": safe_filename,
                "image_b64": image_b64,
                "img_shape": img_shape,
                "raw_json": _json.dumps(results, indent=2),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "weather": weather,
                "yield_estimate": yield_estimate,
            }
            yield event("complete", 100, "Analysis complete!", data=complete_payload)
        except Exception as exc:
            yield event("error", 95, f"Failed to finalise results: {str(exc)}")
            return

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/analyze_result", methods=["POST"])
def analyze_result():
    try:
        raw = request.form.get("payload", "")
        if not raw:
            flash("No analysis data received.", "error")
            return redirect(url_for("analyze"))

        payload = json.loads(raw)
        results = payload.get("results", {})
        filename = payload.get("filename", "unknown")
        image_b64 = payload.get("image_b64", "")
        img_shape = payload.get("img_shape", {})
        raw_json = payload.get("raw_json", "{}")
        timestamp = payload.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        weather = payload.get("weather")
        yield_estimate = payload.get("yield_estimate")

        if results.get("error"):
            flash(results["error"], "error")
            return redirect(url_for("analyze"))

        return render_template(
            "results.html",
            results=results,
            filename=filename,
            image_b64=image_b64,
            img_shape=img_shape,
            raw_json=raw_json,
            timestamp=timestamp,
            weather=weather,
            yield_estimate=yield_estimate,
        )
    except Exception as exc:
        logger.error("analyze_result error: %s", exc)
        flash(f"Failed to render results: {str(exc)}", "error")
        return redirect(url_for("analyze"))


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Agri-Vision Cotton Analysis System")
    logger.info("=" * 60)
    logger.info("Starting Flask application...")
    logger.info("Open http://localhost:5000 in your browser")
    logger.info("Endpoints:")
    logger.info("/              - Home page")
    logger.info("/analyze       - Upload and analyze image")
    logger.info("/comparison    - Compare two field images")
    logger.info("/demo          - View demo results")
    logger.info("/api/analyze   - API endpoint (POST)")
    logger.info("/health        - Health check")
    logger.info("=" * 60)

    ensure_models_loaded()
    is_debug = os.getenv("FLASK_DEBUG", "False").lower() in ("true", "1", "t")
    app.run(debug=is_debug, host="0.0.0.0", port=5000)
