"""
Agri-Vision Flask Application
Unified inference for disease classification (ResNet50) and growth stage prediction (YOLOv8)
"""
import hashlib
import logging
import os
import random
import re
import threading
import json
import base64
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Tuple
from collections import defaultdict
from io import BytesIO

# Load environment file if python-dotenv is available, but don't require it for tests
try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass
_flask_env = os.getenv("FLASK_ENV", "production").lower()
_secret_key = os.getenv("SECRET_KEY")
_GENERATED_EPHEMERAL_SECRET = False
if not _secret_key:
    if _flask_env in ("development", "dev", "testing") or os.getenv(
        "AGRI_VISION_ALLOW_DEV_SECRET", "false"
    ).lower() in ("1", "true", "t"):
        import secrets

        _secret_key = secrets.token_urlsafe(64)
        _GENERATED_EPHEMERAL_SECRET = True
    else:
        raise SystemExit("Missing required SECRET_KEY environment variable")

# Make validated values available for later configuration
_VALIDATED_SECRET_KEY = _secret_key
_VALIDATED_FLASK_ENV = _flask_env

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO
from werkzeug.utils import secure_filename


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
    Request,
    send_file,
    make_response,
)
from flask_cors import CORS
from flasgger import Swagger
from flask_sqlalchemy import SQLAlchemy
from flask_login import (
    LoginManager,
    login_user,
    logout_user,
    login_required,
    current_user,
)
from jinja2 import Environment, FileSystemLoader
from services.image_quality import safe_validate_image_quality

# redis and rate limiting imports
import redis
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from model_registry import registry
from services.weather_service import (
    generate_weather_recommendations,
    geocode_city,
    get_weather,
)
from services.gradcam import (
    GradCAM,
    apply_heatmap_on_image,
    generate_gradcam_explanation,
    generate_pure_heatmap,
)
from services.yield_service import estimate_yield
from services.recommendation_engine import (
    get_recommendations as get_treatment_recommendations,
)

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder="static", template_folder="templates")

@app.errorhandler(404)
def not_found(e):
    return render_template('404.html'), 404

# Try dynamic package loading to prevent crash on automated CI testing rigs
try:
    redis_client = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)
    redis_client.ping()
    logger.info("redis connected for caching and rate limiting")
    limiter = Limiter(
        get_remote_address,
        app=app,
        storage_uri="redis://localhost:6379",
        strategy="fixed-window",
    )
except (redis.ConnectionError, ModuleNotFoundError) as err:
    logger.warning(f"caching layer bypass active: {err}")
    redis_client = None

    class DummyLimiter:
        def limit(self, *args, **kwargs):
            return lambda f: f

    limiter = DummyLimiter()

# db config
app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv(
    "DATABASE_URL", "sqlite:///agri_vision.db"
)
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
from models import db

db.init_app(app)

# --- Login Manager Configuration ---
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"
login_manager.login_message = "Please log in to access this page."
login_manager.login_message_category = "info"


@login_manager.user_loader
def load_user(user_id):
    from models import User

    return User.query.get(user_id)


# --- Security Configuration ---
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


class CustomRequest(Request):
    max_form_memory_size = 25 * 1024 * 1024


app.request_class = CustomRequest

swagger = Swagger(app)
CORS(app)

app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
app.jinja_env.auto_reload = True
app.jinja_env.cache = {}

flask_env = _VALIDATED_FLASK_ENV
app.secret_key = _VALIDATED_SECRET_KEY
if _GENERATED_EPHEMERAL_SECRET:
    logger = logging.getLogger(__name__)
    logger.warning(
        "No SECRET_KEY set — generated ephemeral key for development/testing only. Do NOT use in production."
    )

cookie_secure_default = flask_env not in ("development", "dev", "testing")
app.config.setdefault("SESSION_COOKIE_SECURE", cookie_secure_default)
app.config.setdefault("SESSION_COOKIE_HTTPONLY", True)
app.config.setdefault("SESSION_COOKIE_SAMESITE", os.getenv("SESSION_COOKIE_SAMESITE", "Lax"))

LANG = {
    "en": {"welcome": "Welcome to Agri Vision"},
    "hi": {"welcome": "एग्री विज़न में आपका स्वागत है"},
    "ta": {"welcome": "அக்ரி விஷனுக்கு வரவேற்கிறோம்"},
    "te": {"welcome": "అగ్రి విజన్‌కు స్వాగతం"},
}

os.makedirs("static/uploads", exist_ok=True)
os.makedirs("static/css", exist_ok=True)
os.makedirs("static/generated/gradcam", exist_ok=True)
os.makedirs("models", exist_ok=True)

ALLOWED_IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}
MAX_INFERENCE_DIMENSION = 1024
DISPLAY_IMAGE_MAX_DIMENSION = 1200
DISPLAY_JPEG_QUALITY = 80

RESNET_MODEL_PATH = "models/cotton_crop_disease_classification/full_resnet50_model.pth"
YOLO_MODEL_PATH = "models/cotton_crop_growth_stage_prediction/best.pt"

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

disease_info_map = {
    "Aphids": {
        "healthy_image": "static/images/healthy_leaf.jpg",
        "description": "Aphids are small sap-sucking insects that weaken cotton plants by feeding on tender leaves and shoots.",
        "symptoms": "Curled leaves, sticky honeydew, yellowing, and clusters of tiny insects on the underside of leaves.",
        "treatment": "Remove heavily infested leaves, encourage natural predators, and use neem oil or recommended insecticide if infestation increases.",
    },
    "Army worm": {
        "healthy_image": "static/images/healthy_leaf.jpg",
        "description": "Army worms are leaf-feeding caterpillars that can quickly damage cotton foliage when populations build up.",
        "symptoms": "Chewed leaf edges, holes in leaves, skeletonized foliage, and visible larvae on plants.",
        "treatment": "Scout fields regularly, remove larvae where possible, and apply recommended biological or chemical control at early stages.",
    },
    "Bacterial blight": {
        "healthy_image": "static/images/healthy_leaf.jpg",
        "description": "Bacterial blight is a cotton disease that spreads through infected seed, crop residue, rain splash, and wind-driven moisture.",
        "symptoms": "Angular water-soaked leaf spots, dark lesions, yellowing, and drying of affected leaf tissue.",
        "treatment": "Avoid overhead irrigation, remove infected debris, use disease-free seed, and follow local copper-based spray recommendations if needed.",
    },
    "Cotton Boll Rot": {
        "healthy_image": "static/images/healthy_leaf.jpg",
        "description": "Cotton boll rot affects developing bolls, especially under humid conditions or poor field drainage.",
        "symptoms": "Soft or discolored bolls, fungal growth, rotting tissue, and premature boll drop.",
        "treatment": "Improve drainage and airflow, remove rotten bolls, avoid excess irrigation, and manage insects that create boll wounds.",
    },
    "Green Cotton Boll": {
        "healthy_image": "static/images/healthy_leaf.jpg",
        "description": "Green cotton boll indicates developing boll growth that should be monitored for nutrition, pests, and disease pressure.",
        "symptoms": "Green immature bolls with no clear disease symptoms unless stress, pest injury, or spotting appears.",
        "treatment": "Maintain balanced irrigation and nutrition, scout for pests, and continue regular field monitoring.",
    },
    "Healthy": {
        "healthy_image": "static/images/healthy_leaf.jpg",
        "description": "The leaf appears healthy with no major visible disease symptoms detected.",
        "symptoms": "Uniform green color, normal leaf shape, and no significant spots, mildew, curling, or pest damage.",
        "treatment": "Continue routine monitoring, balanced fertilization, proper irrigation, and preventive crop hygiene.",
    },
    "Powdery mildew": {
        "healthy_image": "static/images/healthy_leaf.jpg",
        "description": "Powdery mildew is a fungal disease that appears as white powdery growth on cotton leaves.",
        "symptoms": "White or gray powdery patches, yellowing leaves, reduced vigor, and premature leaf drying.",
        "treatment": "Improve airflow, remove infected debris, reduce leaf wetness, and apply recommended fungicide when needed.",
    },
    "Target Spot": {
        "healthy_image": "static/images/healthy_leaf.jpg",
        "description": "Target spot is a fungal leaf disease that produces circular lesions and can reduce cotton leaf area.",
        "symptoms": "Brown circular spots with ring-like patterns, yellow halos, leaf blight, and premature defoliation.",
        "treatment": "Reduce leaf wetness, improve spacing and airflow, remove infected residue, and use suitable fungicide if disease spreads.",
    },
}

UNCERTAINTY_THRESHOLD = 0.45
AMBIGUITY_MARGIN = 0.08  


class ModelManager:
    _instance = None
    _instance_lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return
        self._load_lock = threading.Lock()
        self.loaded = False
        self.errors = {"resnet": None, "yolo": None}
        self.resnet_model = None
        self.yolo_model = None
        self._initialized = True

    def load_models(self) -> Tuple[Optional[torch.nn.Module], Optional[YOLO]]:
        if self.loaded:
            return self.resnet_model, self.yolo_model

        with self._load_lock:
            if self.loaded:
                return self.resnet_model, self.yolo_model

            if self.resnet_model is None:
                try:
                    self.resnet_model = torch.load(
                        RESNET_MODEL_PATH,
                        map_location=torch.device("cpu"),
                        weights_only=True,
                    )
                    self.resnet_model.eval()
                    self.errors["resnet"] = None
                    logger.info("ResNet50 loaded")
                except Exception as exc:
                    self.errors["resnet"] = str(exc)
                    logger.warning(f"ResNet50 load failed: {exc}")
                    self.resnet_model = None

            if self.yolo_model is None:
                try:
                    self.yolo_model = YOLO(YOLO_MODEL_PATH)
                    self.errors["yolo"] = None
                    logger.info("YOLOv8 loaded")
                except Exception as exc:
                    self.errors["yolo"] = str(exc)
                    logger.warning(f"YOLOv8 load failed: {exc}")
                    self.yolo_model = None

            self.loaded = True
            return self.resnet_model, self.yolo_model

    def diagnostics(self) -> Dict[str, Any]:
        return {
            "resnet": {
                "loaded": self.resnet_model is not None,
                "path": RESNET_MODEL_PATH,
                "error": self.errors.get("resnet"),
            },
            "yolo": {
                "loaded": self.yolo_model is not None,
                "path": YOLO_MODEL_PATH,
                "error": self.errors.get("yolo"),
            },
        }


model_manager = ModelManager()

def ensure_models_loaded() -> None:
    model_manager.load_models()


def _ensure_rgb(image: np.ndarray) -> np.ndarray:
    if image is None:
        raise ValueError("Image is None")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected an RGB image with 3 channels")
    return image


def resize_image(
    image: np.ndarray, max_dim: int = MAX_INFERENCE_DIMENSION
) -> np.ndarray:
    height, width = image.shape[:2]
    if max(height, width) <= max_dim:
        return image
    scale = max_dim / float(max(height, width))
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def calculate_disease_severity(health_score: float) -> float:
    return max(0.0, 100.0 - float(health_score))


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


# -------------------------------------------------------------------
# INFERENCE PIPELINE
# -------------------------------------------------------------------


def preprocess_image_for_resnet(
    image: np.ndarray, target_size: Tuple[int, int] = (224, 224)
) -> torch.Tensor:
    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(target_size),
            transforms.ToTensor(),
        ]
    )
    tensor = transform(image).unsqueeze(0)
    return tensor


def infer_disease(image):
    if model_manager.resnet_model:
        processed = preprocess_image_for_resnet(image)
        with torch.no_grad():
            output = model_manager.resnet_model(processed)
            probs = F.softmax(output, dim=1)
            confidence, prediction = torch.max(probs, 1)
        probs_np = probs.numpy()
        class_idx = int(prediction.item())
        confidence_value = float(confidence.item())
        predicted_class = disease_classes[class_idx]
        healthy_idx = disease_classes.index("Healthy")
        health_score = float(probs_np[0][healthy_idx]) * 100
    else:
        probs_np = np.random.rand(1, len(disease_classes))
        probs_np = probs_np / probs_np.sum(axis=1, keepdims=True)
        class_idx = int(np.argmax(probs_np[0]))
        confidence_value = float(np.max(probs_np[0]))
        predicted_class = disease_classes[class_idx]
        health_score = float(np.max(probs_np[0])) * 100

    disease_confidences = {
        disease_classes[i]: float(probs_np[0][i]) for i in range(len(disease_classes))
    }

    return {
        "predicted_class": predicted_class,
        "predicted_class_idx": class_idx,
        "confidence": confidence_value,
        "all_confidences": disease_confidences,
        "health_score": health_score,
        "raw": probs_np.tolist(),
    }


def infer_growth_stage(image):
    result = {
        "main_class": None,
        "main_class_idx": None,
        "confidence": 0.0,
        "boxes": [],
        "raw": [],
    }
    if model_manager.yolo_model:
        pil_image = Image.fromarray(image)
        yolo_results = model_manager.yolo_model(pil_image)
        boxes = []
        for r in yolo_results:
            if hasattr(r, "boxes"):
                for b in r.boxes:
                    class_id = (
                        int(b.cls[0].item())
                        if hasattr(b.cls[0], "item")
                        else int(b.cls[0])
                    )
                    conf = (
                        float(b.conf[0].item())
                        if hasattr(b.conf[0], "item")
                        else float(b.conf[0])
                    )
                    xyxy = b.xyxy[0].cpu().numpy().tolist()
                    boxes.append(
                        {
                            "class_id": class_id,
                            "class_name": (
                                growth_stage_classes[class_id]
                                if class_id < len(growth_stage_classes)
                                else str(class_id)
                            ),
                            "confidence": conf,
                            "bbox": xyxy,
                        }
                    )
            else:
                continue

        if len(boxes):
            main = max(boxes, key=lambda x: x["confidence"])
            result.update(
                {
                    "main_class": main["class_name"],
                    "main_class_idx": main["class_id"],
                    "confidence": main["confidence"],
                }
            )
            result["boxes"] = boxes
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

    if disease_result["health_score"] < 50:
        recs.append("Consult an agricultural expert urgently for low health score.")
        recs.append("Consult an agricultural expert if symptoms persist.")
    elif disease_result["health_score"] < 70:
        recs.append("Increase frequency of crop monitoring based on moderate health.")

    if disease_result.get("is_uncertain"):
        recs.append(
            "Model confidence is low. Please upload a clearer image or consult an agricultural expert."
        )
    elif disease_result.get("is_ambiguous"):
        alt = disease_result.get("alternative_prediction", {}).get(
            "class", "another condition"
        )
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


def generate_farmer_insights(
    disease_result: Dict[str, Any], growth_result: Dict[str, Any]
) -> list[str]:
    insights = []
    dclass = disease_result["predicted_class"]
    hscore = disease_result["health_score"]
    gmain = growth_result.get("main_class", "Unknown")

    if dclass != "Healthy":
        insights.append(f"Possible {dclass} risk detected. Immediate action advised.")
    elif hscore > 80:
        insights.append(
            "Crop is currently healthy. No immediate disease risks detected."
        )
    else:
        insights.append(
            "Crop shows slight stress. Monitor closely for early signs of disease."
        )

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


def generate_advanced_recommendations(
    disease_result: Dict[str, Any], growth_result: Dict[str, Any]
) -> Dict[str, str]:
    gmain = growth_result.get("main_class", "Unknown")
    dclass = disease_result["predicted_class"]

    adv_recs = {
        "irrigation_timing": "Maintain standard schedule (every 7-10 days depending on soil moisture).",
        "fertilizer_suggestions": "Use balanced NPK (e.g., 20-20-20) as per standard guidelines.",
        "pest_prevention": "Install sticky traps and monitor for early pest signs.",
        "harvesting_window": "Monitor crop maturity daily.",
    }

    if gmain in ["Cotton Blossom", "Cotton Bud"]:
        adv_recs["irrigation_timing"] = (
            "Increase watering frequency to support blooming."
        )
        adv_recs["fertilizer_suggestions"] = (
            "Apply potassium-rich fertilizers to boost flower development."
        )
    elif gmain in ["Matured Cotton Boll", "Split Cotton Boll"]:
        adv_recs["irrigation_timing"] = (
            "Reduce or stop irrigation to harden bolls and prevent rot."
        )
        adv_recs["harvesting_window"] = "Immediate to 1-2 weeks."

    if dclass == "Aphids":
        adv_recs["pest_prevention"] = (
            "Use neem oil or recommended insecticide for Aphids immediately."
        )
    elif dclass == "Army worm":
        adv_recs["pest_prevention"] = (
            "Apply specific anti-worm biological controls like Bacillus thuringiensis (Bt)."
        )
    elif dclass == "Cotton Boll Rot":
        adv_recs["irrigation_timing"] = (
            "Stop irrigation immediately to allow soil and plant base to dry."
        )

    return adv_recs


def encode_image_for_display(image: np.ndarray) -> str:
    display_image = resize_image(image, DISPLAY_IMAGE_MAX_DIMENSION)
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), DISPLAY_JPEG_QUALITY]
    ok, buffer = cv2.imencode(".jpg", display_image, encode_params)
    if not ok:
        raise ValueError("Failed to encode image for display")
    return base64.b64encode(buffer).decode("utf-8")


def is_allowed_image(filename: str) -> bool:
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS
    )


def calculate_file_hash(file_storage) -> str:
    sha256_hash = hashlib.sha256()
    file_storage.seek(0)
    for byte_block in iter(lambda: file_storage.read(4096), b""):
        sha256_hash.update(byte_block)
    file_storage.seek(0)
    return sha256_hash.hexdigest()


def read_uploaded_image(file_storage) -> Tuple[str, np.ndarray, np.ndarray]:
    safe_filename = secure_filename(file_storage.filename)
    file_bytes = np.frombuffer(file_storage.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Error reading image file")
    return safe_filename, image, cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def should_continue_after_quality_warning(validation: Dict[str, Any]) -> bool:
    return (
        request.form.get("continue_analysis") == "1"
        or request.form.get("continue_anyway") == "1"
        or request.args.get("continue_analysis") == "1"
    )


def add_quality_warning_to_results(
    results: Dict[str, Any], validation: Dict[str, Any]
) -> Dict[str, Any]:
    results["image_quality"] = validation
    quality_warnings = validation.get("warnings") or []
    if quality_warnings:
        existing_warnings = list(results.get("warnings", []))
        for warning in quality_warnings:
            message = f"Image quality warning: {warning}"
            if message not in existing_warnings:
                existing_warnings.append(message)
        results["warnings"] = existing_warnings
    return results


GRAD_CAM_CACHE = {}
GRAD_CAM_CACHE_LOCK = threading.Lock()
MAX_CACHE_SIZE = 100


def get_cached_grad_cam(image_hash: str) -> Optional[Dict[str, Any]]:
    with GRAD_CAM_CACHE_LOCK:
        return GRAD_CAM_CACHE.get(image_hash)


def set_cached_grad_cam(
    image_hash: str,
    overlay_b64: Optional[str],
    heatmap_only_b64: Optional[str],
    overlay_path: Optional[str] = None,
    heatmap_path: Optional[str] = None,
    explainability: Optional[Dict[str, Any]] = None,
) -> None:
    with GRAD_CAM_CACHE_LOCK:
        if len(GRAD_CAM_CACHE) >= MAX_CACHE_SIZE:
            first_key = next(iter(GRAD_CAM_CACHE))
            GRAD_CAM_CACHE.pop(first_key, None)
        GRAD_CAM_CACHE[image_hash] = {
            "grad_cam_image_b64": overlay_b64,
            "heatmap_only_b64": heatmap_only_b64,
            "heatmap_image_path": overlay_path,
            "heatmap_only_path": heatmap_path,
            "explainability": explainability
            or {
                "available": bool(overlay_b64 and heatmap_only_b64),
                "status": (
                    "generated" if overlay_b64 and heatmap_only_b64 else "unavailable"
                ),
                "target_layer": "ResNet50 layer4[-1]",
            },
        }


def build_gradcam_payload(
    image: np.ndarray,
    disease: Dict[str, Any],
    model: Optional[torch.nn.Module],
) -> Dict[str, Any]:
    image_hash = hashlib.sha256(image.tobytes()).hexdigest()
    cached_result = get_cached_grad_cam(image_hash)
    if cached_result is not None:
        logger.info("Using cached Grad-CAM heatmap")
        return cached_result

    payload = {
        "grad_cam_image_b64": None,
        "heatmap_only_b64": None,
        "heatmap_image_path": None,
        "heatmap_only_path": None,
        "explainability": {
            "available": False,
            "status": "unavailable",
            "target_layer": "ResNet50 layer4[-1]",
        },
    }

    if model is None or disease.get("predicted_class_idx") is None:
        return payload

    try:
        input_tensor = preprocess_image_for_resnet(image)
        gradcam_result = generate_gradcam_explanation(
            model=model,
            input_tensor=input_tensor,
            image_rgb=image,
            target_class_idx=disease.get("predicted_class_idx"),
            filename_prefix=image_hash[:16],
        )
        payload["explainability"] = {
            "available": gradcam_result.available,
            "status": gradcam_result.status,
            "target_layer": gradcam_result.target_layer,
        }
        if gradcam_result.error:
            payload["explainability"]["error"] = gradcam_result.error

        if (
            gradcam_result.available
            and gradcam_result.overlay_image is not None
            and gradcam_result.heatmap_image is not None
        ):
            payload["grad_cam_image_b64"] = encode_image_for_display(
                gradcam_result.overlay_image
            )
            payload["heatmap_only_b64"] = encode_image_for_display(
                gradcam_result.heatmap_image
            )
            payload["heatmap_image_path"] = gradcam_result.overlay_path
            payload["heatmap_only_path"] = gradcam_result.heatmap_path
            set_cached_grad_cam(
                image_hash,
                payload["grad_cam_image_b64"],
                payload["heatmap_only_b64"],
                payload["heatmap_image_path"],
                payload["heatmap_only_path"],
                payload["explainability"],
            )
    except Exception as exc:
        logger.exception("Grad-CAM visualization failed: %s", exc)
        payload["explainability"] = {
            "available": False,
            "status": "failed",
            "target_layer": "ResNet50 layer4[-1]",
            "error": str(exc),
        }

    return payload


def analyze_image(image: np.ndarray) -> Dict[str, Any]:
    import time

    start_time = time.time()
    resnet_model, yolo_model = model_manager.load_models()
    try:
        try:
            growth = infer_growth_stage(image)
        except Exception as exc:
            logger.error("Error during growth stage inference: %s", exc)
            growth = {
                "main_class": None,
                "main_class_idx": None,
                "confidence": 0.0,
                "boxes": [],
                "raw": [],
            }

        disease = infer_disease(image)
        if (
            not isinstance(disease, dict)
            or "predicted_class" not in disease
            or "health_score" not in disease
        ):
            raise ValueError("Invalid disease model prediction output.")

        gradcam_payload = build_gradcam_payload(image, disease, resnet_model)
        grad_cam_image_b64 = gradcam_payload.get("grad_cam_image_b64")
        heatmap_only_b64 = gradcam_payload.get("heatmap_only_b64")

        disease["heatmap_b64"] = grad_cam_image_b64
        disease["heatmap_only_b64"] = heatmap_only_b64
        disease["heatmap_image_path"] = gradcam_payload.get("heatmap_image_path")
        disease["heatmap_only_path"] = gradcam_payload.get("heatmap_only_path")
        disease["explainability"] = gradcam_payload.get("explainability")

        # Track metrics in registry
        inference_time = time.time() - start_time
        try:
            if disease and disease.get("confidence"):
                registry.update_metrics(
                    model_type="resnet",
                    version="v1.0",
                    confidence=disease.get("confidence", 0.0),
                    inference_time=inference_time,
                    success=True,
                )

            if growth and growth.get("confidence"):
                registry.update_metrics(
                    model_type="yolo",
                    version="v1.0",
                    confidence=growth.get("confidence", 0.0),
                    inference_time=inference_time,
                    success=True,
                )
        except Exception as e:
            logger.error(f"Error tracking metrics: {e}")

        recs = generate_recommendations(disease, growth)
        severity = calculate_disease_severity(disease["health_score"])
        yield_est = estimate_yield(disease, growth, weather=None, field_acres=1.0)
        adv_recs = generate_advanced_recommendations(disease, growth)
        insights = generate_farmer_insights(disease, growth)

        # Context-aware treatment recommendations from the recommendation engine
        treatment_recs = get_treatment_recommendations(
            crop_type="cotton",
            disease_name=disease.get("predicted_class", ""),
            confidence=disease.get("confidence"),
        )

        result = {
            "disease": disease,
            "growth": growth,
            "recommendations": recs,
            "grad_cam_image_b64": grad_cam_image_b64,
            "heatmap_only_b64": heatmap_only_b64,
            "heatmap_image_path": gradcam_payload.get("heatmap_image_path"),
            "heatmap_only_path": gradcam_payload.get("heatmap_only_path"),
            "explainability": gradcam_payload.get("explainability"),
            "disease_severity": severity,
            "yield_estimate": yield_est,
            "advanced_recommendations": adv_recs,
            "farmer_insights": insights,
            "treatment_recommendations": treatment_recs,
        }

        if growth.get("main_class") is None:
            fallback_reason = (
                "Growth stage model unavailable."
                if yolo_model is None
                else "Cotton growth stage could not be detected."
            )
            result["warnings"] = [
                fallback_reason,
                "Disease analysis is still provided, but comparison may be less reliable.",
            ]

        return result
    except Exception as exc:
        logger.error("Unexpected error in image analysis: %s", exc)
        return {
            "error": "The AI model encountered an error. Please verify the image file."
        }


def build_comparison_result(
    old_results: Dict[str, Any], new_results: Dict[str, Any]
) -> Dict[str, Any]:
    if not isinstance(old_results, dict) or not isinstance(new_results, dict):
        raise ValueError("Invalid result objects.")

    old_disease = old_results.get("disease")
    new_disease = new_results.get("disease")
    if old_disease is None or new_disease is None:
        raise ValueError("Valid crop analysis missing in one or both images.")

    old_score = float(old_disease.get("health_score", 0.0))
    new_score = float(new_disease.get("health_score", 0.0))
    change = new_score - old_score
    abs_change = abs(change)

    if change > 1:
        trend = {
            "status": "improved",
            "label": "Improved",
            "icon": "fa-arrow-trend-up",
            "direction": "up",
        }
        headline = f"Crop health improved by {abs_change:.1f}%"
        recommendation = "Continue current treatment plan."
    elif change < -1:
        trend = {
            "status": "declined",
            "label": "Declined",
            "icon": "fa-arrow-trend-down",
            "direction": "down",
        }
        headline = f"Crop health declined by {abs_change:.1f}%"
        recommendation = "Increase inspection frequency."
    else:
        trend = {
            "status": "stable",
            "label": "Stable",
            "icon": "fa-arrows-left-right",
            "direction": "flat",
        }
        headline = "Crop health remained stable"
        recommendation = "Maintain current routine."

    old_predicted = old_disease.get("predicted_class", "Unknown")
    new_predicted = new_disease.get("predicted_class", "Unknown")
    disease_reduced = old_predicted != "Healthy" and new_predicted == "Healthy"
    disease_changed = old_predicted != new_predicted

    summary = [
        headline,
        (
            "Disease spread reduced"
            if disease_reduced
            else (
                f"Signal shifted to {new_predicted}"
                if disease_changed
                else f"Signal remains {new_predicted}"
            )
        ),
        recommendation,
    ]

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
def add_security_headers(response):
    # Existing cache headers
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    
    # --- NEW SECURITY HEADERS ---
    response.headers['X-Frame-Options'] = 'SAMEORIGIN'
    response.headers['X-Content-Type-Options'] = 'nosniff'
    
    return response

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


@app.route("/model-admin")
@login_required
def admin_dashboard():
    if not current_user.is_researcher():
        flash("Access denied. Researchers and Admins only.", "danger")
        return redirect(url_for("index"))
    return render_template("admin_dashboard.html")


# --- Model Management Admin Endpoints ---


@app.route("/admin/models", methods=["GET"])
@login_required
def list_models():
    """List all registered models with their metadata"""

    if not current_user.is_researcher():
        logger.warning(f"Unauthorized admin access attempt by user: {getattr(current_user, 'id', None)}")
        return jsonify({"error": "Access denied. Researchers/Admins only."}), 403

    model_type = request.args.get("type")
    try:
        models = registry.list_models(model_type)
        return jsonify(
            {
                "status": "success",
                "models": models,
                "ab_test_enabled": registry.ab_test_enabled,
                "rollback_threshold": registry.rollback_threshold,
            }
        )
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/admin/models/active", methods=["GET"])
@login_required
def get_active_models():
    """Get currently active models"""
    if not current_user.is_researcher():
        logger.warning(f"Unauthorized admin access attempt by user: {getattr(current_user, 'id', None)}")
        return jsonify({"error": "Access denied. Researchers/Admins only."}), 403

    try:
        active_resnet = registry.get_active_model("resnet")
        active_yolo = registry.get_active_model("yolo")
        return jsonify(
            {
                "status": "success",
                "active_models": {
                    "resnet": active_resnet.to_dict() if active_resnet else None,
                    "yolo": active_yolo.to_dict() if active_yolo else None,
                },
            }
        )
    except Exception as e:
        logger.error(f"Error getting active models: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/admin/models/register", methods=["POST"])
@login_required
def register_model():
    """Register a new model version"""
    if not current_user.is_researcher():
        logger.warning(f"Unauthorized admin access attempt by user: {getattr(current_user, 'id', None)}")
        return jsonify({"error": "Access denied. Researchers/Admins only."}), 403

    try:
        data = request.get_json()
        required_fields = ["model_type", "version", "path"]
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400
        provided_path = data["path"]
        models_base = os.path.abspath("models")
        abs_provided = os.path.abspath(provided_path)
        if not abs_provided.startswith(models_base + os.sep) and abs_provided != models_base:
            return (
                jsonify({"error": "Invalid model path: must be inside the server 'models/' directory."}),
                400,
            )

        metadata = registry.register_model(
            model_type=data["model_type"],
            version=data["version"],
            path=provided_path,
            accuracy=data.get("accuracy", 0.0),
            dataset_version=data.get("dataset_version", "unknown"),
            parameters=data.get("parameters", 0),
            is_active=data.get("is_active", False),
            ab_test_ratio=data.get("ab_test_ratio", 0.0),
        )
        return jsonify(
            {
                "status": "success",
                "message": f"Model {data['model_type']} version {data['version']} registered successfully",
                "metadata": metadata.to_dict(),
            }
        )
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        logger.error(f"Error registering model: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/admin/models/activate", methods=["POST"])
@login_required
def activate_model():
    """Set a model version as active"""
    if not current_user.is_researcher():
        logger.warning(f"Unauthorized admin access attempt by user: {getattr(current_user, 'id', None)}")
        return jsonify({"error": "Access denied. Researchers/Admins only."}), 403

    try:
        data = request.get_json()
        required_fields = ["model_type", "version"]
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400

        registry.set_active_model(data["model_type"], data["version"])
        return jsonify(
            {
                "status": "success",
                "message": f"Model {data['model_type']} version {data['version']} activated successfully",
            }
        )
    except ValueError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        logger.error(f"Error activating model: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/admin/models/delete", methods=["DELETE"])
@login_required
def delete_model():
    """Delete a model version"""
    if not current_user.is_researcher():
        logger.warning(f"Unauthorized admin access attempt by user: {getattr(current_user, 'id', None)}")
        return jsonify({"error": "Access denied. Researchers/Admins only."}), 403

    try:
        data = request.get_json()
        required_fields = ["model_type", "version"]
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400

        registry.delete_model(data["model_type"], data["version"])
        return jsonify(
            {
                "status": "success",
                "message": f"Model {data['model_type']} version {data['version']} deleted successfully",
            }
        )
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Error deleting model: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/admin/models/ab-testing", methods=["POST"])
@login_required
def toggle_ab_testing():
    """Enable or disable A/B testing"""
    if not current_user.is_researcher():
        logger.warning(f"Unauthorized admin access attempt by user: {getattr(current_user, 'id', None)}")
        return jsonify({"error": "Access denied. Researchers/Admins only."}), 403

    try:
        data = request.get_json()
        enabled = data.get("enabled", True)
        registry.enable_ab_testing(enabled)
        return jsonify(
            {
                "status": "success",
                "message": f"A/B testing {'enabled' if enabled else 'disabled'}",
                "ab_test_enabled": registry.ab_test_enabled,
            }
        )
    except Exception as e:
        logger.error(f"Error toggling A/B testing: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/admin/models/ab-ratio", methods=["POST"])
@login_required
def set_ab_ratio():
    """Set A/B testing ratio for a model version"""
    if not current_user.is_researcher():
        logger.warning(f"Unauthorized admin access attempt by user: {getattr(current_user, 'id', None)}")
        return jsonify({"error": "Access denied. Researchers/Admins only."}), 403

    try:
        data = request.get_json()
        required_fields = ["model_type", "version", "ratio"]
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400

        registry.set_ab_test_ratio(data["model_type"], data["version"], data["ratio"])
        return jsonify(
            {
                "status": "success",
                "message": f"A/B test ratio for {data['model_type']} version {data['version']} set to {data['ratio']}",
            }
        )
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Error setting A/B ratio: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/admin/models/metrics", methods=["GET"])
@login_required
def get_model_metrics():
    """Get performance metrics for all models"""
    if not current_user.is_researcher():
        logger.warning(f"Unauthorized admin access attempt by user: {getattr(current_user, 'id', None)}")
        return jsonify({"error": "Access denied. Researchers/Admins only."}), 403

    try:
        models = registry.list_models()
        return jsonify({"status": "success", "metrics": models})
    except Exception as e:
        logger.error(f"Error getting model metrics: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/admin/models/rollback-threshold", methods=["POST"])
@login_required
def set_rollback_threshold():
    """Set automatic rollback threshold"""
    if not current_user.is_researcher():
        logger.warning(f"Unauthorized admin access attempt by user: {getattr(current_user, 'id', None)}")
        return jsonify({"error": "Access denied. Researchers/Admins only."}), 403

    try:
        data = request.get_json()
        threshold = data.get("threshold")
        if threshold is None:
            return jsonify({"error": "Missing required field: threshold"}), 400

        if not 0.0 <= threshold <= 1.0:
            return jsonify({"error": "Threshold must be between 0.0 and 1.0"}), 400

        registry.rollback_threshold = threshold
        registry.save_config()
        return jsonify(
            {
                "status": "success",
                "message": f"Rollback threshold set to {threshold}",
                "rollback_threshold": registry.rollback_threshold,
            }
        )
    except Exception as e:
        logger.error(f"Error setting rollback threshold: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/admin/models/export/pdf", methods=["GET"])
@login_required
def export_pdf():
    """Export model metrics as PDF"""
    if not current_user.is_researcher():
        logger.warning(f"Unauthorized admin access attempt by user: {getattr(current_user, 'id', None)}")
        return jsonify({"error": "Access denied. Researchers/Admins only."}), 403

    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import (
            SimpleDocTemplate,
            Table,
            TableStyle,
            Paragraph,
            Spacer,
        )
        from reportlab.lib.styles import getSampleStyleSheet

        models = registry.list_models()

        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        elements = []
        styles = getSampleStyleSheet()

        title = Paragraph("Model Performance Report", styles["Title"])
        elements.append(title)
        elements.append(Spacer(1, 12))

        date = Paragraph(
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            styles["Normal"],
        )
        elements.append(date)
        elements.append(Spacer(1, 12))

        table_data = [
            [
                "Model Type",
                "Version",
                "Accuracy",
                "Requests",
                "Success Rate",
                "Avg Confidence",
                "Avg Time",
                "Status",
            ]
        ]

        if models:
            for model_type in models:
                for model in models[model_type]:
                    metrics = model.performance_metrics
                    success_rate = (
                        (metrics.successful_predictions / metrics.total_requests * 100)
                        if metrics.total_requests > 0
                        else 0
                    )
                    table_data.append(
                        [
                            model_type.capitalize(),
                            model.version,
                            f"{model.accuracy * 100:.2f}%",
                            str(metrics.total_requests),
                            f"{success_rate:.1f}%",
                            f"{metrics.avg_confidence * 100:.1f}%",
                            f"{metrics.avg_inference_time:.3f}s",
                            "Active" if model.is_active else "Inactive",
                        ]
                    )

        table = Table(table_data)
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, 0), 12),
                    ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                    ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                    ("GRID", (0, 0), (-1, -1), 1, colors.black),
                ]
            )
        )

        elements.append(table)
        doc.build(elements)

        buffer.seek(0)
        return send_file(
            buffer,
            as_attachment=True,
            download_name=f'model_metrics_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf',
            mimetype="application/pdf",
        )
    except ImportError:
        return (
            jsonify(
                {
                    "error": "reportlab not installed. Install with: pip install reportlab"
                }
            ),
            500,
        )
    except Exception as e:
        logger.error(f"Error generating PDF: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/analyze/download-report", methods=["POST"])
@login_required
def download_analysis_report():
    """Generate and download a professional PDF crop analysis report"""
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.units import inch
        from reportlab.platypus import (
            SimpleDocTemplate,
            Paragraph,
            Spacer,
            Table,
            TableStyle,
            Image as RLImage,
        )
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.enums import TA_CENTER
        from io import BytesIO
        import base64
        from PIL import Image as PILImage

        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        disease_detected = data.get("disease_detected", "Unknown")
        disease_confidence = data.get("disease_confidence", 0)
        health_score = data.get("health_score", 0)
        growth_stage = data.get("growth_stage", "Unknown")
        growth_confidence = data.get("growth_confidence", 0)
        image_b64 = data.get("image_b64", "")
        recommendations = data.get("recommendations", [])
        timestamp = data.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        weather_data = data.get("weather_data", {})
        yield_data = data.get("yield_estimate", {})

        pdf_buffer = BytesIO()
        doc = SimpleDocTemplate(
            pdf_buffer, pagesize=letter, topMargin=0.5 * inch, bottomMargin=0.5 * inch
        )
        elements = []

        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            "CustomTitle",
            parent=styles["Heading1"],
            fontSize=24,
            textColor=colors.HexColor("#2c3e50"),
            spaceAfter=6,
            alignment=TA_CENTER,
            fontName="Helvetica-Bold",
        )
        subtitle_style = ParagraphStyle(
            "CustomSubtitle",
            parent=styles["Normal"],
            fontSize=12,
            textColor=colors.HexColor("#7f8c8d"),
            spaceAfter=12,
            alignment=TA_CENTER,
            fontName="Helvetica",
        )
        header_style = ParagraphStyle(
            "SectionHeader",
            parent=styles["Heading2"],
            fontSize=14,
            textColor=colors.HexColor("#27ae60"),
            spaceAfter=8,
            spaceBefore=12,
            fontName="Helvetica-Bold",
        )
        normal_style = ParagraphStyle(
            "Normal",
            parent=styles["Normal"],
            fontSize=10,
            textColor=colors.HexColor("#2c3e50"),
            spaceAfter=6,
        )

        title = Paragraph("🌾 Agri-Vision Crop Analysis Report", title_style)
        elements.append(title)
        subtitle = Paragraph("Professional Analysis Insights", subtitle_style)
        elements.append(subtitle)
        elements.append(Spacer(1, 0.15 * inch))

        metadata_data = [
            ["Report Generated", timestamp],
            ["Analysis ID", f"AGRI-{datetime.now().strftime('%Y%m%d%H%M%S')}"],
        ]
        metadata_table = Table(metadata_data, colWidths=[2 * inch, 4 * inch])
        metadata_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#ecf0f1")),
                    ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#2c3e50")),
                    ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                    ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                    ("TOPPADDING", (0, 0), (-1, -1), 8),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#bdc3c7")),
                ]
            )
        )
        elements.append(metadata_table)
        elements.append(Spacer(1, 0.2 * inch))

        if image_b64:
            try:
                image_data = base64.b64decode(
                    image_b64.split(",")[-1] if "," in image_b64 else image_b64
                )
                img_pil = PILImage.open(BytesIO(image_data))
                img_width = 6 * inch
                img_height = img_width * img_pil.height / img_pil.width
                if img_height > 3.5 * inch:
                    img_height = 3.5 * inch
                    img_width = img_height * img_pil.width / img_pil.height

                img_buffer = BytesIO(image_data)
                rl_image = RLImage(img_buffer, width=img_width, height=img_height)
                elements.append(rl_image)
                elements.append(Spacer(1, 0.15 * inch))
            except Exception as e:
                logger.warning(f"Could not embed image in PDF: {e}")

        elements.append(Paragraph("DISEASE HEALTH ANALYSIS", header_style))
        disease_data = [
            ["Detected Issue", str(disease_detected)],
            ["Confidence Score", f"{float(disease_confidence):.1f}%"],
            ["Health Score", f"{float(health_score):.1f}%"],
        ]
        disease_table = Table(disease_data, colWidths=[2 * inch, 4 * inch])
        disease_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#e8f5e9")),
                    ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#1b5e20")),
                    ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                    ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 10),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                    ("TOPPADDING", (0, 0), (-1, -1), 8),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#a5d6a7")),
                ]
            )
        )
        elements.append(disease_table)
        elements.append(Spacer(1, 0.15 * inch))

        elements.append(Paragraph("GROWTH STAGE DETECTION", header_style))
        growth_data = [
            ["Current Stage", str(growth_stage)],
            ["Stage Confidence", f"{float(growth_confidence):.1f}%"],
        ]
        growth_table = Table(growth_data, colWidths=[2 * inch, 4 * inch])
        growth_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#e3f2fd")),
                    ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#1565c0")),
                    ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                    ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 10),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                    ("TOPPADDING", (0, 0), (-1, -1), 8),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#90caf9")),
                ]
            )
        )
        elements.append(growth_table)
        elements.append(Spacer(1, 0.15 * inch))

        if weather_data:
            elements.append(Paragraph("WEATHER CONDITIONS", header_style))
            weather_rows = []
            if weather_data.get("temperature") is not None:
                weather_rows.append(
                    ["Temperature", f"{weather_data.get('temperature')}°C"]
                )
            if weather_data.get("humidity") is not None:
                weather_rows.append(["Humidity", f"{weather_data.get('humidity')}%"])
            if weather_data.get("precipitation") is not None:
                weather_rows.append(
                    ["Precipitation", f"{weather_data.get('precipitation')} mm"]
                )
            if weather_data.get("description"):
                weather_rows.append(["Condition", weather_data.get("description")])

            if weather_rows:
                weather_table = Table(weather_rows, colWidths=[2 * inch, 4 * inch])
                weather_table.setStyle(
                    TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#fff3e0")),
                            ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#e65100")),
                            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                            ("FONTSIZE", (0, 0), (-1, -1), 10),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                            ("TOPPADDING", (0, 0), (-1, -1), 8),
                            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#ffe0b2")),
                        ]
                    )
                )
                elements.append(weather_table)
                elements.append(Spacer(1, 0.15 * inch))

        if yield_data:
            elements.append(Paragraph("YIELD ESTIMATE", header_style))
            yield_rows = []
            if yield_data.get("yield_min_acre") and yield_data.get("yield_max_acre"):
                yield_rows.append(
                    [
                        "Per Acre Estimate",
                        f"{yield_data.get('yield_min_acre')}–{yield_data.get('yield_max_acre')} q/acre",
                    ]
                )
            if yield_data.get("yield_min_total") and yield_data.get("yield_max_total"):
                yield_rows.append(
                    [
                        "Total Field Estimate",
                        f"{yield_data.get('yield_min_total')}–{yield_data.get('yield_max_total')} quintals",
                    ]
                )

            if yield_rows:
                yield_table = Table(yield_rows, colWidths=[2 * inch, 4 * inch])
                yield_table.setStyle(
                    TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f3e5f5")),
                            ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#4a148c")),
                            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                            ("FONTSIZE", (0, 0), (-1, -1), 10),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                            ("TOPPADDING", (0, 0), (-1, -1), 8),
                            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e1bee7")),
                        ]
                    )
                )
                elements.append(yield_table)
                elements.append(Spacer(1, 0.15 * inch))

        if recommendations:
            elements.append(Paragraph("RECOMMENDATIONS", header_style))
            rec_text = ""
            for i, rec in enumerate(recommendations[:10], 1):
                rec_text += f"• {rec}<br/>"
            elements.append(Paragraph(rec_text, normal_style))
            elements.append(Spacer(1, 0.1 * inch))

        elements.append(Spacer(1, 0.2 * inch))
        footer_text = "Generated by Agri-Vision Cotton Analysis System | For professional agricultural guidance, consult local extension officers"
        elements.append(
            Paragraph(
                footer_text,
                ParagraphStyle(
                    "Footer",
                    parent=styles["Normal"],
                    fontSize=8,
                    textColor=colors.HexColor("#95a5a6"),
                    alignment=TA_CENTER,
                ),
            )
        )

        doc.build(elements)
        pdf_buffer.seek(0)

        return send_file(
            pdf_buffer,
            mimetype="application/pdf",
            as_attachment=True,
            download_name=f'agri_vision_crop_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf',
        )

    except Exception as e:
        logger.error(f"Error generating analysis PDF: {e}")
        return jsonify({"error": f"Failed to generate PDF: {str(e)}"}), 500


@app.route("/history")
@login_required
def history():
    from models import AnalysisHistory
    records = AnalysisHistory.query.filter_by(user_id=current_user.id).order_by(AnalysisHistory.created_at.desc()).all()
    return render_template("history.html", history_records=records)


@app.route("/health")
def health():
    ensure_models_loaded()
    diagnostics = model_manager.diagnostics()
    model_loaded = diagnostics["resnet"]["loaded"] and diagnostics["yolo"]["loaded"]
    status_code = 200 if model_loaded else 503
    return (
        jsonify(
            {
                "status": "healthy" if model_loaded else "degraded",
                "model_loaded": model_loaded,
                "models": diagnostics,
            }
        ),
        status_code,
    )


@app.route("/analyze", methods=["GET", "POST"])
@login_required
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
            flash(
                "Invalid file type. Please upload an image (PNG, JPG, JPEG, GIF)",
                "error",
            )
            return redirect(request.url)

        try:
            safe_filename, image, image_rgb = read_uploaded_image(file)
            image_quality, _quality_fallback = safe_validate_image_quality(image)
            if image_quality.get("is_blocking"):
                flash(
                    "Image quality check failed: "
                    + "; ".join(image_quality.get("warnings", [])),
                    "error",
                )
                return render_template(
                    "upload.html",
                    image_quality=image_quality,
                )
            if image_quality.get("warnings") and not should_continue_after_quality_warning(image_quality):
                return render_template(
                    "upload.html",
                    image_quality=image_quality,
                    quality_requires_confirmation=True,
                )

            compressed_rgb = resize_image(image_rgb, MAX_INFERENCE_DIMENSION)
            results = analyze_image(compressed_rgb)
            add_quality_warning_to_results(results, image_quality)

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
                results["recommendations"] = (
                    results.get("recommendations", [])
                    + generate_weather_recommendations(weather)
                )[:6]
                results["weather"] = weather
                
                # Recalculate yield estimate using the fetched weather data
                if "yield_estimate" in results:
                    results["yield_estimate"] = estimate_yield(
                        results["disease"],
                        results["growth"],
                        weather=weather,
                        field_acres=results["yield_estimate"].get("field_acres", 1.0)
                    )

            # Add forecast data if location provided
            forecast_data = None
            if lat and lon and weather:
                try:
                    from services.disease_prediction_service import DiseasePredictor, HistoricalPatternAnalyzer
                    from models import DiseaseOccurrence
                    
                    forecast_data = {
                        'weather': weather,
                        'location': city or f"{lat:.4f}, {lon:.4f}"
                    }
                    
                    # Get disease prediction based on weather
                    predictor = DiseasePredictor()
                    detected_disease = results.get("disease", {}).get("predicted_class", "")
                    if detected_disease:
                        # DiseasePredictor uses fuzzy matching so pass names directly
                        disease_name = detected_disease

                        # Get weather-based risk
                        weather_risk = predictor.predict_disease_risk([weather], disease_name)
                        if weather_risk:
                            forecast_data['weather_risk'] = weather_risk[0] if weather_risk else None
                        
                        # Get historical insights
                        try:
                            occurrences = DiseaseOccurrence.query.limit(1000).all()
                            occurrences_data = [o.to_dict() for o in occurrences]
                            
                            analyzer = HistoricalPatternAnalyzer()
                            analyzer.train(occurrences_data)
                            
                            # Get peak season for detected disease
                            peak_season = analyzer.get_peak_season(disease_name)
                            if peak_season:
                                forecast_data['peak_season'] = peak_season
                            
                            # Get current month risk
                            current_month = datetime.now().month
                            seasonal_patterns = analyzer.seasonal_patterns
                            seasonal_match = DiseasePredictor._fuzzy_match(disease_name, set(seasonal_patterns.keys()))
                            if seasonal_match:
                                monthly_risk = seasonal_patterns[seasonal_match].get(current_month, 0)
                                forecast_data['seasonal_risk'] = {
                                    'month': current_month,
                                    'month_name': datetime(2024, current_month, 1).strftime('%B'),
                                    'risk_percentage': monthly_risk
                                }
                            
                            # Get weather recommendations
                            if weather_risk and weather_risk[0]:
                                risk_level = weather_risk[0].get('risk_level', 'moderate')
                                recommendations = predictor.generate_recommendations(disease_name, risk_level)
                                forecast_data['recommendations'] = recommendations
                        except Exception as e:
                            logger.warning(f"Could not get historical insights: {e}")
                except Exception as e:
                    logger.warning(f"Could not fetch forecast data: {e}")

            if results.get("error"):
                raise ValueError(results["error"])

            predicted_class = results.get("disease", {}).get("predicted_class", "")
            disease_info = disease_info_map.get(predicted_class, {})

            from models import AnalysisHistory, db
            if current_user.is_authenticated:
                import time
                unique_filename = f"{int(time.time())}_{safe_filename}"
                file_path = os.path.join("static", "uploads", unique_filename)
                cv2.imwrite(file_path, image)
                
                history_entry = AnalysisHistory(
                    user_id=current_user.id,
                    image_path=unique_filename,
                    disease_result=results.get("disease"),
                    growth_result=results.get("growth"),
                    confidence=results.get("disease", {}).get("confidence"),
                    health_score=results.get("disease", {}).get("health_score")
                )
                db.session.add(history_entry)
                db.session.commit()

            return render_template(
                "results.html",
                results=results,
                filename=safe_filename,
                image_b64=encode_image_for_display(image_rgb),
                img_shape={"width": image.shape[1], "height": image.shape[0]},
                raw_json=json.dumps(results, indent=2),
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                weather=weather,
                forecast=forecast_data,
                grad_cam_image_b64=results.get("grad_cam_image_b64"),
                heatmap_only_b64=results.get("heatmap_only_b64"),
                heatmap_image_path=results.get("heatmap_image_path"),
                heatmap_only_path=results.get("heatmap_only_path"),
                disease_info=disease_info,
                treatment_recommendations=results.get("treatment_recommendations", {}),
            )
        except Exception as exc:
            logger.error("Analysis error: %s", exc)
            flash(f"Error during analysis: {str(exc)}", "error")
            return redirect(request.url)

    return render_template("upload.html")


@app.route("/api/explain", methods=["POST"])
def api_explain():
    if "file" not in request.files:
        return jsonify({"status": "error", "error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"status": "error", "error": "No file selected"}), 400

    if not is_allowed_image(file.filename):
        return (
            jsonify(
                {
                    "status": "error",
                    "error": "Invalid file type. Please upload an image.",
                }
            ),
            400,
        )

    try:
        _, image, image_rgb = read_uploaded_image(file)
        image_quality, _quality_fallback = safe_validate_image_quality(image)
        if image_quality.get("is_blocking"):
            return (
                jsonify(
                    {
                        "status": "error",
                        "error": "Image quality check failed.",
                        "image_quality": image_quality,
                    }
                ),
                400,
            )
        compressed_rgb = resize_image(image_rgb, MAX_INFERENCE_DIMENSION)

        results = analyze_image(compressed_rgb)
        add_quality_warning_to_results(results, image_quality)

        if "error" in results:
            return jsonify({"status": "error", "error": results["error"]}), 500

        disease_result = results.get("disease", {})

        return jsonify(
            {
                "status": "success",
                "heatmap_b64": results.get("grad_cam_image_b64"),
                "heatmap_only_b64": results.get("heatmap_only_b64"),
                "target_layer": "ResNet50 layer4[-1]",
                "image_b64": encode_image_for_display(compressed_rgb),
                "predicted_class": disease_result.get("predicted_class", "Unknown"),
                "confidence": disease_result.get("confidence", 0.0),
                "image_quality": image_quality,
            }
        )
    except Exception as exc:
        logger.error("Error in API explain endpoint: %s", exc)
        return jsonify({"status": "error", "error": str(exc)}), 500


@app.route("/comparison", methods=["GET", "POST"])
def comparison():
    error_message = None
    old_filename, new_filename, old_image, new_image = None, None, None, None

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
                flash(
                    f"Invalid file type for {label}. Please upload PNG, JPG, JPEG, or GIF.",
                    "error",
                )
                return redirect(request.url)

        try:
            last_week_file = request.files["last_week_image"]
            current_week_file = request.files["current_week_image"]

            last_week_hash = calculate_file_hash(last_week_file)
            current_week_hash = calculate_file_hash(current_week_file)

            if last_week_hash == current_week_hash:
                error_message = "Duplicate field images detected. Please upload two different images for meaningful comparison analysis."
                return render_template("comparison.html", error_message=error_message)
        except Exception as exc:
            logger.error("Hashing error: %s", exc)

        try:
            old_filename, old_image, old_rgb = read_uploaded_image(
                request.files["last_week_image"]
            )
            new_filename, new_image, new_rgb = read_uploaded_image(
                request.files["current_week_image"]
            )

            old_results = analyze_image(old_rgb)
            new_results = analyze_image(new_rgb)

            _, yolo_model = model_manager.load_models()
            if old_results.get("disease") is None or new_results.get("disease") is None:
                error_message = "Unable to analyze one or both uploaded images. Please upload valid field images and try again."
            elif (
                old_results.get("warnings")
                and new_results.get("warnings")
                and yolo_model is not None
            ):
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
                old_image_b64=(
                    encode_image_for_display(old_image)
                    if old_image is not None
                    else None
                ),
                new_image_b64=(
                    encode_image_for_display(new_image)
                    if new_image is not None
                    else None
                ),
            )
    return render_template("comparison.html")


@app.route("/demo")
def demo():
    try:
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
            "is_uncertain": False,
            "is_ambiguous": False,
            "interpretation_message": "Healthy crop detected with moderate confidence.",
        }
        demo_growth_boxes = [
            {
                "class_id": 3,
                "class_name": "Matured Cotton Boll",
                "confidence": 0.91,
                "bbox": [120, 80, 210, 155],
            },
            {
                "class_id": 4,
                "class_name": "Split Cotton Boll",
                "confidence": 0.70,
                "bbox": [300, 120, 390, 210],
            },
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

        # generate mock fallback heatmap
        from services.gradcam import generate_pure_heatmap

        mock_heatmap_np = generate_mock_heatmap(synthetic_rgb)
        pure_heatmap_rgb = generate_pure_heatmap(synthetic_rgb, mock_heatmap_np)
        mock_overlay = cv2.addWeighted(synthetic_rgb, 0.6, pure_heatmap_rgb, 0.4, 0)

        image_b64 = encode_image_for_display(synthetic_rgb)
        grad_cam_image_b64 = encode_image_for_display(mock_overlay)
        heatmap_only_b64 = encode_image_for_display(pure_heatmap_rgb)

        demo_treatment_recs = get_treatment_recommendations(
            crop_type="cotton",
            disease_name=demo_disease.get("predicted_class", "Healthy"),
            confidence=demo_disease.get("confidence"),
        )

        example_json = {
            "disease": demo_disease,
            "growth": demo_growth,
            "recommendations": generate_recommendations(demo_disease, demo_growth),
            "grad_cam_image_b64": grad_cam_image_b64,
            "heatmap_only_b64": heatmap_only_b64,
            "heatmap_image_path": None,
            "heatmap_only_path": None,
            "disease_severity": calculate_disease_severity(
                demo_disease["health_score"]
            ),
            "yield_estimate": estimate_yield(
                demo_disease, demo_growth, weather=None, field_acres=1.0
            ),
            "advanced_recommendations": generate_advanced_recommendations(
                demo_disease, demo_growth
            ),
            "farmer_insights": generate_farmer_insights(demo_disease, demo_growth),
            "treatment_recommendations": demo_treatment_recs,
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
            heatmap_only_b64=heatmap_only_b64,
            heatmap_image_path=None,
            heatmap_only_path=None,
            yield_estimate=example_json["yield_estimate"],
            disease_info=disease_info_map.get("Healthy", {}),
            treatment_recommendations=demo_treatment_recs,
            weather=None,
        )
    except Exception as e:
        logger.error(f"Demo route failed: {e}")
        return redirect(url_for("index"))


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
        r"\b(hello|hi|hey|howdy|greetings)\b": [
            "Hello there! How can I assist you with your cotton crop today?",
            "Hi! Need any help analyzing your farm data?",
        ],
        r"\b(disease|diseases|sick|spots?|rot|blight)\b": [
            "If you're noticing leaf spots or rotting, it could be Bacterial Blight or Target Spot. I highly recommend taking a picture and uploading it to our Analyze tab for an AI diagnosis."
        ],
        r"\b(yield|yields|harvest|harvests|produce)\b": [
            "Yield depends heavily on the crop's health score and current growth stage. Check out the Dashboard for predictions across your fields!"
        ],
        r"\b(fertilizer|fertilizers|nutrient|nutrients|npk|potassium)\b": [
            "Cotton responds well to a balanced NPK fertilizer. During the blooming and early boll stages, potassium is critical to maximize yield."
        ],
        r"\b(water|watering|irrigation|dry|drought)\b": [
            "Maintain regular watering during the blossom phase. However, once bolls mature and start splitting, you should reduce irrigation to prevent rot."
        ],
        r"\b(pest|pests|worm|worms|aphid|aphids|bug|bugs|insect|insects|bollworm)\b": [
            "Pests like Pink Bollworm and Aphids are common enemies of cotton. I recommend deploying pheromone traps and scouting the fields twice a week."
        ],
        r"\b(weather|temperature|rain|rainfall|humidity|climate)\b": [
            "Weather plays a huge role in cotton health. Hot, dry spells stress bolls while excess rain can encourage fungal diseases. Use our weather tab to monitor conditions."
        ],
        r"\b(soil|soils|ph|minerals|clay|loam|sandy)\b": [
            "Cotton thrives in well-draining loamy soil with a pH of 5.8–8.0. Conduct a soil test before the season to identify any nutrient deficiencies."
        ],
        r"\b(grow|growth|growing|stage|stages|seedling|boll|bolls|flower|flowering)\b": [
            "Cotton growth has 5 key stages: germination, seedling, vegetative, flowering/boll formation, and maturity. Each stage has unique care needs — the flowering stage is most critical!"
        ],
        r"\b(spray|spraying|pesticide|pesticides|fungicide|herbicide|chemical)\b": [
            "When spraying, always follow label rates and avoid spraying during peak heat or wind. Consider integrated pest management (IPM) to reduce chemical dependency."
        ],
        r"\b(thank(?:s|s you)?|awesome|great|perfect)\b": [
            "You're welcome! Feel free to ask any time. Happy farming! 🌱",
            "Glad I could help! Let me know if you have more questions about your cotton crop.",
        ],
        r"\b(help|assist|support|guide|advice|tips?)\b": [
            "I'm here to help! You can ask me about crop diseases, yield optimization, pest control, irrigation, fertilization, weather impacts, or soil health.",
            "Sure! Try asking about cotton diseases, pest control, yield estimates, or upload an image in the Analyze tab for an instant AI diagnosis.",
        ],
        r"\b(cotton|crop|crops|farm|farming|field|fields)\b": [
            "Agri-Vision specializes in cotton crop analysis. Upload a field image in the Analyze tab for disease detection, yield prediction, and health scoring!"
        ],
    }

    reply = "I'm your Agri-Vision AI assistant. I specialize in cotton farming, crop diseases, and yield optimization. How can I help you?"

    for pattern, reply_options in responses.items():
        if re.search(pattern, message):
            reply = random.choice(reply_options)
            break
    else:
        logger.info(f"Unmatched chat query: {message}")

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


# --- core api with redis cache & rate limiting ---
@app.route("/api/analyze", methods=["POST"])
@limiter.limit("10 per minute")
def api_analyze():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    try:
        file_bytes = np.frombuffer(file.read(), np.uint8)
        file_hash = hashlib.sha256(file_bytes).hexdigest()
        cache_key = f"inference_cache:{file_hash}"

        if redis_client:
            cached = redis_client.get(cache_key)
            if cached:
                logger.info("cache hit - skipping model inference")
                res = make_response(cached)
                res.headers["Content-Type"] = "application/json"
                res.headers["X-Cache-Hit"] = "1"
                return res

        logger.info("cache miss - running inference")
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if image is None:
            return jsonify({"error": "Invalid image file"}), 400

        image_quality, _quality_fallback = safe_validate_image_quality(image)
        if image_quality.get("is_blocking"):
            return (
                jsonify(
                    {
                        "error": "Image quality check failed.",
                        "image_quality": image_quality,
                    }
                ),
                400,
            )

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = analyze_image(image_rgb)
        add_quality_warning_to_results(results, image_quality)

        resp_data = {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "results": results,
            "image_quality": image_quality,
        }
        resp_json = json.dumps(resp_data)

        if redis_client:
            redis_client.setex(cache_key, 86400, resp_json)

        res = make_response(resp_json)
        res.headers["Content-Type"] = "application/json"
        res.headers["X-Cache-Hit"] = "0"
        return res

    except Exception as e:
        logger.error(f"API analysis error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/analyze_stream", methods=["POST"])
def api_analyze_stream():
    """Streaming endpoint for real-time analysis progress"""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    def generate():
        try:
            yield f"data: {json.dumps({'status': 'uploading', 'progress': 25})}\n\n"

            file_bytes = np.frombuffer(file.read(), np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if image is None:
                yield f"data: {json.dumps({'status': 'error', 'message': 'Unable to process this image. Please upload a clear crop photo and try again.'})}\n\n"
                return

            yield f"data: {json.dumps({'status': 'analyzing', 'progress': 50})}\n\n"

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            compressed_rgb = resize_image(image_rgb, MAX_INFERENCE_DIMENSION)
            yield f"data: {json.dumps({'step': 'upload_received', 'progress': 20, 'message': 'Image received, starting analysis...'})}\n\n"
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(analyze_image, compressed_rgb)
                try:
                    results = future.result(timeout=60)
                except concurrent.futures.TimeoutError:
                    yield f"data: {json.dumps({'step': 'error', 'progress': 100, 'message': 'The request exceeded the expected processing time. Please try again later.'})}\n\n"
                    return
            if results.get("error"):
                yield f"data: {json.dumps({'status': 'error', 'message': results['error']})}\n\n"
                return

            yield f"data: {json.dumps({'status': 'generating', 'progress': 75})}\n\n"
            yield f"data: {json.dumps({'status': 'complete', 'progress': 100, 'results': results})}\n\n"
        except Exception as e:
            logger.error(f"Streaming analysis error: {e}")
            yield f"data: {json.dumps({'status': 'error', 'message': 'Analysis is taking longer than expected. Please try again after some time.'})}\n\n" 
    def generate():
        try:
            yield f"data: {json.dumps({'status': 'uploading', 'step': 'upload_received', 'progress': 25, 'message': 'Upload received.'})}\n\n"

            file_bytes = np.frombuffer(file.read(), np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if image is None:
                yield f"data: {json.dumps({'status': 'error', 'step': 'error', 'message': 'Invalid image file'})}\n\n"
                return

            image_quality, _quality_fallback = safe_validate_image_quality(image)
            if image_quality.get("is_blocking"):
                yield f"data: {json.dumps({'status': 'error', 'step': 'error', 'message': 'Image quality check failed.', 'image_quality': image_quality})}\n\n"
                return
            if image_quality.get("warnings") and not should_continue_after_quality_warning(image_quality):
                yield f"data: {json.dumps({'status': 'quality_warning', 'step': 'quality_warning', 'progress': 35, 'message': 'Image quality warnings found.', 'image_quality': image_quality})}\n\n"
                return

            yield f"data: {json.dumps({'status': 'analyzing', 'step': 'preprocessing', 'progress': 50, 'message': 'Validating and preparing image.'})}\n\n"

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            compressed_rgb = resize_image(image_rgb, MAX_INFERENCE_DIMENSION)
            results = analyze_image(compressed_rgb)
            add_quality_warning_to_results(results, image_quality)
            if results.get("error"):
                yield f"data: {json.dumps({'status': 'error', 'step': 'error', 'message': results['error']})}\n\n"
                return

            yield f"data: {json.dumps({'status': 'generating', 'step': 'recommendations', 'progress': 75, 'message': 'Generating recommendations.'})}\n\n"
            yield f"data: {json.dumps({'status': 'complete', 'step': 'complete', 'progress': 100, 'message': 'Analysis complete.', 'results': results, 'data': {'results': results, 'image_quality': image_quality}})}\n\n"
        except Exception as e:
            logger.error(f"Streaming analysis error: {e}")
            yield f"data: {json.dumps({'status': 'error', 'step': 'error', 'message': str(e)})}\n\n"

    return Response(stream_with_context(generate()), mimetype="text/event-stream")


# --- Batch Processing Endpoints ---
@app.route("/api/batch_upload", methods=["POST"])
def api_batch_upload():
    """Upload multiple images for batch analysis"""
    try:
        if "files" not in request.files:
            return jsonify({"error": "No files uploaded"}), 400

        files = request.files.getlist("files")
        if not files or files[0].filename == "":
            return jsonify({"error": "No files selected"}), 400

        valid_files = []
        for file in files:
            if file and allowed_file(file.filename):
                valid_files.append(file)

        if not valid_files:
            return jsonify({"error": "No valid image files"}), 400

        from models import BatchJob, db, AnalysisResult

        job = BatchJob(total_images=len(valid_files), status="pending")
        db.session.add(job)
        db.session.commit()

        import base64
        images_data = []
        for file in valid_files:
            file_data = file.read()
            b64_data = base64.b64encode(file_data).decode('utf-8')
            images_data.append((file.filename, b64_data))

        celery_enabled = False
        try:
            from celery_tasks import process_batch_job, CELERY_AVAILABLE, MAX_BATCH_SIZE

            if len(images_data) > int(MAX_BATCH_SIZE):
                return (
                    jsonify({"error": f"Batch size exceeds maximum of {MAX_BATCH_SIZE}"}),
                    400,
                )

            if CELERY_AVAILABLE:
                try:
                    process_batch_job.delay(job.id, images_data)
                    celery_enabled = True
                except Exception as e:
                    logger.exception("Failed to dispatch Celery job, falling back: %s", e)
                    celery_enabled = False
            else:
                try:
                    process_batch_job(job.id, images_data)
                    celery_enabled = False
                except Exception as e:
                    logger.exception("Fallback batch processing failed to start: %s", e)
                    return jsonify({"error": str(e)}), 500

        except ImportError:
            logger.warning("celery_tasks module not available; will attempt fallback processing")
            try:
                from celery_tasks import process_batch_job
                process_batch_job(job.id, images_data)
            except Exception:
                logger.exception("Fallback batch processing failed to start")
                return jsonify({"error": "Batch processing unavailable"}), 500

        return jsonify(
            {
                "status": "success",
                "job_id": job.id,
                "total_images": len(valid_files),
                "celery_enabled": celery_enabled,
                "message": f"Batch job {job.id} started with {len(valid_files)} images",
            }
        )

    except Exception as e:
        logger.error(f"Batch upload error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/batch_status/<job_id>", methods=["GET"])
def api_batch_status(job_id):
    """Get status of a batch job"""
    from models import BatchJob, db

    job = BatchJob.query.get(job_id)
    if not job:
        return jsonify({"error": "Batch job not found"}), 404

    job.completed_images = len([r for r in job.results if r.status == "complete"])
    job.failed_images = len([r for r in job.results if r.status == "error"])

    if job.completed_images + job.failed_images >= job.total_images:
        job.status = "completed"
        job.completed_at = datetime.utcnow()
        db.session.commit()

    return jsonify(job.to_dict())


@app.route("/api/batch_results/<job_id>", methods=["GET"])
def api_batch_results(job_id):
    """Get all results for a batch job"""
    from models import BatchJob, db

    job = BatchJob.query.get(job_id)
    if not job:
        return jsonify({"error": "Batch job not found"}), 404

    results = [r.to_dict() for r in job.results]
    results.sort(key=lambda x: x["image_index"])

    return jsonify(
        {
            "job_id": job.id,
            "status": job.status,
            "total_images": job.total_images,
            "completed_images": job.completed_images,
            "failed_images": job.failed_images,
            "results": results,
        }
    )


@app.route("/api/batch_results/<job_id>/export/csv", methods=["GET"])
def export_batch_csv(job_id):
    """Export batch results as CSV"""
    from models import BatchJob
    import csv
    from io import StringIO

    job = BatchJob.query.get(job_id)
    if not job:
        return jsonify({"error": "Batch job not found"}), 404

    si = StringIO()
    cw = csv.writer(si)
    cw.writerow(
        [
            "Image Name",
            "Status",
            "Disease",
            "Confidence",
            "Health Score",
            "Growth Stage",
        ]
    )

    results = sorted(job.results, key=lambda x: x.image_index)

    for r in results:
        results_data = r.results_json or {}
        disease = results_data.get("disease", {})
        growth = results_data.get("growth", {})

        disease_class = disease.get("predicted_class", "N/A")
        confidence = (
            f"{disease.get('confidence', 0):.3f}"
            if disease.get("confidence") is not None
            else "N/A"
        )
        health_score = (
            f"{disease.get('health_score', 0):.1f}"
            if disease.get("health_score") is not None
            else "N/A"
        )
        growth_class = growth.get("main_class", "N/A")

        cw.writerow(
            [
                r.image_name,
                r.status,
                disease_class,
                confidence,
                health_score,
                growth_class,
            ]
        )

    output = si.getvalue()
    si.close()

    return Response(
        output,
        mimetype="text/csv",
        headers={
            "Content-disposition": f"attachment; filename=batch_results_{job_id}.csv"
        },
    )


@app.route("/api/batch_results/<job_id>/export/pdf", methods=["GET"])
def export_batch_pdf(job_id):
    """Export batch results as PDF"""
    from models import BatchJob

    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter, landscape
        from reportlab.platypus import (
            SimpleDocTemplate,
            Table,
            TableStyle,
            Paragraph,
            Spacer,
        )
        from reportlab.lib.styles import getSampleStyleSheet
        from io import BytesIO
    except ImportError:
        return (
            jsonify(
                {
                    "error": "reportlab not installed. Install with: pip install reportlab"
                }
            ),
            500,
        )

    job = BatchJob.query.get(job_id)
    if not job:
        return jsonify({"error": "Batch job not found"}), 404

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=landscape(letter))
    elements = []
    styles = getSampleStyleSheet()

    title = Paragraph(f"Batch Analysis Report (Job ID: {job_id})", styles["Title"])
    elements.append(title)
    elements.append(Spacer(1, 12))

    summary_text = f"Total Images: {job.total_images} | Completed: {job.completed_images} | Failed: {job.failed_images}"
    summary = Paragraph(summary_text, styles["Normal"])
    elements.append(summary)
    elements.append(Spacer(1, 12))

    table_data = [
        [
            "Image Name",
            "Status",
            "Disease",
            "Confidence",
            "Health Score",
            "Growth Stage",
        ]
    ]

    results = sorted(job.results, key=lambda x: x.image_index)

    for r in results:
        results_data = r.results_json or {}
        disease = results_data.get("disease", {})
        growth = results_data.get("growth", {})

        disease_class = disease.get("predicted_class", "N/A")
        confidence = (
            f"{disease.get('confidence', 0)*100:.1f}%"
            if disease.get("confidence") is not None
            else "N/A"
        )
        health_score = (
            f"{disease.get('health_score', 0):.1f}%"
            if disease.get("health_score") is not None
            else "N/A"
        )
        growth_class = growth.get("main_class", "N/A")

        table_data.append(
            [
                r.image_name,
                r.status.upper(),
                disease_class,
                confidence,
                health_score,
                growth_class,
            ]
        )

    table = Table(table_data)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 10),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                ("GRID", (0, 0), (-1, -1), 1, colors.black),
                ("FONTSIZE", (0, 1), (-1, -1), 9),
                ("WORDWRAP", (0, 0), (-1, -1), "CJK"),
            ]
        )
    )

    elements.append(table)
    doc.build(elements)

    buffer.seek(0)
    return send_file(
        buffer,
        as_attachment=True,
        download_name=f"batch_results_{job_id}.pdf",
        mimetype="application/pdf",
    )


@app.route("/batch", methods=["GET", "POST"])
@login_required
def batch_upload_page():
    """Batch upload page"""
    if request.method == "POST":
        return redirect(
            url_for("batch_results_page", job_id=request.form.get("job_id"))
        )
    return render_template("batch_upload.html")


@app.route("/batch/results/<job_id>")
@login_required
def batch_results_page(job_id):
    """Batch results page"""
    return render_template("batch_results.html", job_id=job_id)


# --- Authentication Routes ---


@app.route("/login", methods=["GET", "POST"])
def login():
    """Login page"""
    if current_user.is_authenticated:
        return redirect(url_for("index"))

    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")
        remember = request.form.get("remember")

        from models import User

        user = User.query.filter_by(email=email).first()

        if user and user.check_password(password):
            if not user.is_active:
                flash(
                    "Your account has been deactivated. Please contact support.",
                    "danger",
                )
                return render_template("login.html")

            login_user(user, remember=remember)
            user.last_login = datetime.utcnow()
            db.session.commit()

            next_page = request.args.get("next")
            return redirect(next_page) if next_page else redirect(url_for("index"))
        else:
            flash("Invalid email or password", "danger")

    return render_template("login.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    """Registration page"""
    if current_user.is_authenticated:
        return redirect(url_for("index"))

    if request.method == "POST":
        full_name = request.form.get("full_name")
        email = request.form.get("email")
        password = request.form.get("password")
        confirm_password = request.form.get("confirm_password")
        role = request.form.get("role", "farmer")

        if not full_name or not email or not password:
            flash("All fields are required", "danger")
            return render_template("register.html")

        if password != confirm_password:
            flash("Passwords do not match", "danger")
            return render_template("register.html")

        if len(password) < 8:
            flash("Password must be at least 8 characters", "danger")
            return render_template("register.html")

        from models import User

        if User.query.filter_by(email=email).first():
            flash("Email already registered", "danger")
            return render_template("register.html")

        user = User(email=email, full_name=full_name, role=role)
        user.set_password(password)

        db.session.add(user)
        db.session.commit()

        flash("Account created successfully! Please login.", "success")
        return redirect(url_for("login"))

    return render_template("register.html")


@app.route("/logout")
@login_required
def logout():
    """Logout user"""
    logout_user()
    flash("You have been logged out", "info")
    return redirect(url_for("login"))


@app.route("/profile")
@login_required
def profile():
    """User profile page"""
    return render_template("profile.html")


@app.route("/forgot_password", methods=["GET", "POST"])
def forgot_password():
    """Forgot password page"""
    if request.method == "POST":
        email = request.form.get("email")
        flash("Password reset link sent to your email (demo feature)", "info")
        return redirect(url_for("login"))
    return render_template("login.html")


# --- Geographic Disease Mapping ---


@app.route("/disease-map")
@login_required
def disease_map():
    """Disease map page"""
    return render_template("disease_map.html")


@app.route("/api/disease-map")
@login_required
def api_disease_map():
    """API endpoint for disease map data"""
    from models import AnalysisHistory

    disease_filter = request.args.get("disease", "all")
    time_filter = request.args.get("time", "all")
    confidence_filter = float(request.args.get("confidence", 0))

    if current_user.is_researcher():
        query = AnalysisHistory.query
    else:
        query = AnalysisHistory.query.filter_by(user_id=current_user.id)

    if time_filter == "today":
        query = query.filter(
            AnalysisHistory.created_at
            >= datetime.utcnow().replace(hour=0, minute=0, second=0)
        )
    elif time_filter == "week":
        query = query.filter(
            AnalysisHistory.created_at >= datetime.utcnow() - timedelta(days=7)
        )
    elif time_filter == "month":
        query = query.filter(
            AnalysisHistory.created_at >= datetime.utcnow() - timedelta(days=30)
        )
    elif time_filter == "year":
        query = query.filter(
            AnalysisHistory.created_at >= datetime.utcnow() - timedelta(days=365)
        )

    analyses = query.all()

    filtered_analyses = []
    for a in analyses:
        if disease_filter != "all":
            if (
                not a.disease_result
                or a.disease_result.get("predicted_class") != disease_filter
            ):
                continue

        if confidence_filter > 0:
            if not a.confidence or a.confidence < confidence_filter / 100:
                continue

        if a.latitude and a.longitude:
            filtered_analyses.append(a)

    total_analyses = len(filtered_analyses)
    healthy_count = sum(
        1
        for a in filtered_analyses
        if a.disease_result and a.disease_result.get("predicted_class") == "healthy"
    )
    diseased_count = total_analyses - healthy_count
    avg_health_score = (
        sum(a.health_score for a in filtered_analyses if a.health_score)
        / len([a for a in filtered_analyses if a.health_score])
        if filtered_analyses
        else 0
    )
    regions = set(a.region for a in filtered_analyses if a.region)

    return jsonify(
        {
            "analyses": [a.to_dict() for a in filtered_analyses],
            "stats": {
                "total_analyses": total_analyses,
                "healthy_count": healthy_count,
                "diseased_count": diseased_count,
                "avg_health_score": avg_health_score,
                "regions_count": len(regions),
            },
        }
    )


# --- Advanced Dashboard ---


@app.route("/dashboard")
@login_required
def dashboard():
    """Advanced dashboard page"""
    return render_template("dashboard.html")


@app.route("/api/dashboard-stats")
@login_required
def api_dashboard_stats():
    """API endpoint for dashboard statistics"""
    from models import AnalysisHistory

    if current_user.is_researcher():
        analyses = AnalysisHistory.query.all()
    else:
        analyses = AnalysisHistory.query.filter_by(user_id=current_user.id).all()

    total_analyses = len(analyses)
    healthy_count = sum(
        1
        for a in analyses
        if a.disease_result and a.disease_result.get("predicted_class") == "healthy"
    )
    diseased_count = total_analyses - healthy_count
    avg_health_score = (
        sum(a.health_score for a in analyses if a.health_score)
        / len([a for a in analyses if a.health_score])
        if analyses
        else 0
    )

    disease_counts = defaultdict(int)
    for a in analyses:
        if a.disease_result:
            disease = a.disease_result.get("predicted_class", "unknown")
            disease_counts[disease] += 1

    disease_distribution = {
        "labels": [d.replace("_", " ").title() for d in disease_counts.keys()],
        "values": list(disease_counts.values()),
    }

    trend_labels = []
    trend_data = defaultdict(list)
    for i in range(7):
        date = datetime.utcnow() - timedelta(days=6 - i)
        trend_labels.append(date.strftime("%b %d"))

        day_analyses = [a for a in analyses if a.created_at.date() == date.date()]
        for a in day_analyses:
            if a.disease_result:
                disease = a.disease_result.get("predicted_class", "unknown")
                trend_data[disease].append(1)

    trend_datasets = []
    colors = ["#22c55e", "#ef4444", "#f59e0b", "#8b5cf6", "#ec4899", "#06b6d4"]
    for idx, (disease, counts) in enumerate(trend_data.items()):
        daily_counts = []
        for i in range(7):
            date = datetime.utcnow() - timedelta(days=6 - i)
            day_analyses = [a for a in analyses if a.created_at.date() == date.date()]
            count = sum(
                1
                for a in day_analyses
                if a.disease_result
                and a.disease_result.get("predicted_class") == disease
            )
            daily_counts.append(count)

        trend_datasets.append(
            {
                "label": disease.replace("_", " ").title(),
                "data": daily_counts,
                "borderColor": colors[idx % len(colors)],
                "backgroundColor": colors[idx % len(colors)] + "20",
                "fill": False,
                "tension": 0.4,
            }
        )

    disease_trends = {"labels": trend_labels, "datasets": trend_datasets}

    growth_counts = defaultdict(int)
    for a in analyses:
        if a.growth_result:
            stage = a.growth_result.get("main_class", "unknown")
            growth_counts[stage] += 1

    growth_distribution = {
        "labels": [g.replace("_", " ").title() for g in growth_counts.keys()],
        "values": list(growth_counts.values()),
    }

    region_counts = defaultdict(int)
    for a in analyses:
        if a.region:
            region_counts[a.region] += 1

    regional_data = {
        "labels": list(region_counts.keys()),
        "values": list(region_counts.values()),
    }

    recent_analyses = sorted(analyses, key=lambda x: x.created_at, reverse=True)[:10]
    recent_activity = []
    for a in recent_analyses:
        disease = (
            a.disease_result.get("predicted_class", "unknown")
            if a.disease_result
            else "unknown"
        )
        activity_type = "disease" if disease != "healthy" else "healthy"
        icon = "exclamation-triangle" if disease != "healthy" else "check-circle"

        recent_activity.append(
            {
                "type": activity_type,
                "icon": icon,
                "title": f'{disease.replace("_", " ").title()} Detected',
                "description": (
                    f"Confidence: {(a.confidence * 100):.1f}%"
                    if a.confidence
                    else "No confidence data"
                ),
                "time": a.created_at.strftime("%b %d, %Y %H:%M"),
            }
        )

    return jsonify(
        {
            "stats": {
                "total_analyses": total_analyses,
                "healthy_count": healthy_count,
                "diseased_count": diseased_count,
                "avg_health_score": avg_health_score,
            },
            "disease_distribution": disease_distribution,
            "disease_trends": disease_trends,
            "growth_distribution": growth_distribution,
            "regional_data": regional_data,
            "recent_activity": recent_activity,
        }
    )


# --- Automated Reporting ---


@app.route("/reports")
@login_required
def reports():
    """Reports page"""
    return render_template("reports.html")


@app.route("/api/analyses")
@login_required
def api_analyses():
    """API endpoint to get list of analyses for report generation"""
    from models import AnalysisHistory

    # Get analyses for current user
    if current_user.is_researcher():
        analyses = (
            AnalysisHistory.query.order_by(AnalysisHistory.created_at.desc())
            .limit(50)
            .all()
        )
    else:
        analyses = (
            AnalysisHistory.query.filter_by(user_id=current_user.id)
            .order_by(AnalysisHistory.created_at.desc())
            .limit(50)
            .all()
        )

    analyses_list = []
    for a in analyses:
        disease = (
            a.disease_result.get("predicted_class", "unknown")
            if a.disease_result
            else "unknown"
        )
        analyses_list.append(
            {
                "id": a.id,
                "disease": disease.replace("_", " ").title(),
                "date": a.created_at.strftime("%Y-%m-%d %H:%M"),
                "health_score": a.health_score,
            }
        )

    return jsonify({"analyses": analyses_list})


@app.route("/api/generate-report/<analysis_id>")
@login_required
def generate_report(analysis_id):
    """Generate PDF report for a single analysis"""
    from models import AnalysisHistory
    from io import BytesIO

    try:
        from services.report_service import ReportGenerator
    except ImportError as e:
        logger.error(f"Failed to import ReportGenerator: {e}")
        return jsonify({"error": f"Report service not available: {str(e)}"}), 500

    analysis = AnalysisHistory.query.get(analysis_id)
    if not analysis:
        return jsonify({"error": "Analysis not found"}), 404

    if not current_user.is_researcher() and analysis.user_id != current_user.id:
        return jsonify({"error": "Access denied"}), 403

    try:
        generator = ReportGenerator()
        report_data = {
            "disease_result": analysis.disease_result,
            "growth_result": analysis.growth_result,
            "health_score": analysis.health_score,
            "confidence": analysis.confidence,
        }

        user_info = {
            "full_name": current_user.full_name,
            "email": current_user.email,
            "role": current_user.role,
        }

        pdf_bytes = generator.generate_analysis_report(report_data, user_info)

        return send_file(
            BytesIO(pdf_bytes),
            mimetype="application/pdf",
            as_attachment=True,
            download_name=f"analysis_report_{analysis_id}.pdf",
        )
    except Exception as e:
        logger.error(f"Error generating report: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route("/api/generate-summary-report")
@login_required
def generate_summary_report():
    """Generate summary PDF report for all analyses"""
    from models import AnalysisHistory
    from datetime import datetime, timedelta
    from io import BytesIO

    try:
        from services.report_service import ReportGenerator
    except ImportError as e:
        logger.error(f"Failed to import ReportGenerator: {e}")
        return jsonify({"error": f"Report service not available: {str(e)}"}), 500

    # Get date range
    days = request.args.get("days", 30, type=int)
    start_date = datetime.utcnow() - timedelta(days=days)

    if current_user.is_researcher():
        analyses = AnalysisHistory.query.filter(
            AnalysisHistory.created_at >= start_date
        ).all()
    else:
        analyses = AnalysisHistory.query.filter(
            AnalysisHistory.user_id == current_user.id,
            AnalysisHistory.created_at >= start_date,
        ).all()

    try:
        generator = ReportGenerator()
        analyses_data = [a.to_dict() for a in analyses]

        user_info = {
            "full_name": current_user.full_name,
            "email": current_user.email,
            "role": current_user.role,
        }

        date_range = f"Last {days} days"
        pdf_bytes = generator.generate_summary_report(
            analyses_data, user_info, date_range
        )

        return send_file(
            BytesIO(pdf_bytes),
            mimetype="application/pdf",
            as_attachment=True,
            download_name=f'summary_report_{datetime.now().strftime("%Y%m%d")}.pdf',
        )
    except Exception as e:
        logger.error(f"Error generating summary report: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


# --- Disease Database & Symptom Checker ---


@app.route("/disease-database")
@login_required
def disease_database():
    """Disease database page"""
    return render_template("disease_database.html")


@app.route("/symptom-checker")
@login_required
def symptom_checker():
    """Symptom checker page"""
    return render_template("symptom_checker.html")


@app.route("/api/diseases")
def api_diseases():
    """API endpoint to get list of diseases"""
    from models import Disease

    search = request.args.get("search", "")
    severity = request.args.get("severity", "")
    affected_part = request.args.get("affected_part", "")

    query = Disease.query

    if search:
        query = query.filter(Disease.name.ilike(f"%{search}%"))

    if severity:
        query = query.filter(Disease.severity == severity)

    if affected_part:
        query = query.filter(Disease.affected_parts.ilike(f"%{affected_part}%"))

    diseases = query.order_by(Disease.name).all()

    return jsonify(
        {"diseases": [d.to_dict() for d in diseases], "count": len(diseases)}
    )


@app.route("/api/diseases/<int:disease_id>")
def api_disease_detail(disease_id):
    """API endpoint to get disease details"""
    from models import Disease

    disease = Disease.query.get(disease_id)
    if not disease:
        return jsonify({"error": "Disease not found"}), 404

    return jsonify(disease.to_dict())


@app.route("/api/symptoms")
def api_symptoms():
    """API endpoint to get list of symptoms"""
    from models import Symptom

    category = request.args.get("category", "")

    query = Symptom.query

    if category:
        query = query.filter(Symptom.category == category)

    symptoms = query.order_by(Symptom.category, Symptom.name).all()

    return jsonify(
        {"symptoms": [s.to_dict() for s in symptoms], "count": len(symptoms)}
    )


@app.route("/api/symptom-check", methods=["POST"])
def api_symptom_check():
    """API endpoint to check symptoms and suggest diseases"""
    from models import Symptom, Disease, DiseaseSymptom

    data = request.get_json()
    symptom_ids = data.get("symptom_ids", [])

    if not symptom_ids:
        return jsonify({"error": "No symptoms provided"}), 400

    # Get diseases associated with the symptoms
    disease_scores = {}

    for symptom_id in symptom_ids:
        associations = DiseaseSymptom.query.filter_by(symptom_id=symptom_id).all()
        for assoc in associations:
            if assoc.disease_id not in disease_scores:
                disease_scores[assoc.disease_id] = 0
            disease_scores[assoc.disease_id] += assoc.confidence

    # Sort by score
    sorted_diseases = sorted(disease_scores.items(), key=lambda x: x[1], reverse=True)

    # Get top matches
    results = []
    for disease_id, score in sorted_diseases[:5]:
        disease = Disease.query.get(disease_id)
        if disease:
            results.append(
                {"disease": disease.to_dict(), "match_score": round(score * 100, 1)}
            )

    return jsonify({"results": results, "symptom_count": len(symptom_ids)})


# --- Disease Forecast & Weather Prediction ---


@app.route("/disease-forecast")
@login_required
def disease_forecast():
    """Disease forecast page"""
    return render_template("disease_forecast.html")


@app.route("/api/weather-forecast")
def api_weather_forecast():
    """API endpoint to get weather forecast for a location"""
    from services.weather_service import get_weather_forecast
    from services.disease_prediction_service import DiseasePredictor

    lat = request.args.get("lat", type=float)
    lon = request.args.get("lon", type=float)
    location_name = request.args.get("location", "Unknown")
    days = request.args.get("days", 14, type=int)

    if not lat or not lon:
        return jsonify({"error": "Latitude and longitude required"}), 400

    try:
        # Get weather forecast
        forecast_data = get_weather_forecast(lat, lon, days)

        if not forecast_data:
            return jsonify({"error": "Failed to fetch weather forecast"}), 500

        # Get disease predictions
        predictor = DiseasePredictor()
        predictions = predictor.get_all_disease_predictions(forecast_data["forecast"])

        return jsonify(
            {
                "location": location_name,
                "lat": lat,
                "lon": lon,
                "weather_forecast": forecast_data["forecast"],
                "disease_predictions": predictions,
            }
        )
    except Exception as e:
        logger.error(f"Error fetching weather forecast: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/disease-prediction/<disease_name>")
def api_disease_prediction(disease_name):
    """API endpoint to get prediction for a specific disease"""
    from services.weather_service import get_weather_forecast
    from services.disease_prediction_service import DiseasePredictor

    lat = request.args.get("lat", type=float)
    lon = request.args.get("lon", type=float)
    days = request.args.get("days", 14, type=int)

    if not lat or not lon:
        return jsonify({"error": "Latitude and longitude required"}), 400

    try:
        # Get weather forecast
        forecast_data = get_weather_forecast(lat, lon, days)

        if not forecast_data:
            return jsonify({"error": "Failed to fetch weather forecast"}), 500

        # Get prediction for specific disease
        predictor = DiseasePredictor()
        predictions = predictor.predict_disease_risk(
            forecast_data["forecast"], disease_name
        )

        # Get high risk days
        high_risk_days = predictor.get_high_risk_days(predictions)

        # Get recommendations
        if predictions:
            latest_risk = predictions[0]["risk_level"]
            recommendations = predictor.generate_recommendations(
                disease_name, latest_risk
            )
        else:
            recommendations = []

        return jsonify(
            {
                "disease": disease_name,
                "predictions": predictions,
                "high_risk_days": high_risk_days,
                "recommendations": recommendations,
            }
        )
    except Exception as e:
        logger.error(f"Error getting disease prediction: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/historical-patterns")
def api_historical_patterns():
    """API endpoint to analyze historical disease patterns with ML learning"""
    from models import DiseaseOccurrence, WeatherData
    from services.disease_prediction_service import HistoricalPatternAnalyzer

    location = request.args.get("location", "")
    disease_id = request.args.get("disease_id", type=int)
    include_weather = request.args.get("include_weather", "false").lower() == "true"

    try:
        query = DiseaseOccurrence.query

        if location:
            query = query.filter(DiseaseOccurrence.location_name.ilike(f"%{location}%"))

        if disease_id:
            query = query.filter(DiseaseOccurrence.disease_id == disease_id)

        occurrences = (
            query.order_by(DiseaseOccurrence.occurrence_date.desc()).limit(1000).all()
        )
        occurrences_data = [o.to_dict() for o in occurrences]

        # Get weather data if requested
        weather_data = None
        if include_weather:
            weather_query = WeatherData.query
            if location:
                weather_query = weather_query.filter(
                    WeatherData.location_name.ilike(f"%{location}%")
                )
            weather_records = weather_query.limit(1000).all()
            weather_data = [w.to_dict() for w in weather_records]

        # Initialize and train the analyzer
        analyzer = HistoricalPatternAnalyzer()
        analyzer.train(occurrences_data, weather_data)

        # Get comprehensive insights
        insights = analyzer.get_insights()

        return jsonify(
            {
                "insights": insights,
                "total_occurrences": len(occurrences_data),
                "weather_data_available": weather_data is not None
                and len(weather_data) > 0,
            }
        )
    except Exception as e:
        logger.error(f"Error analyzing historical patterns: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/historical-predict")
def api_historical_predict():
    """API endpoint to predict disease risk based on historical patterns"""
    from models import DiseaseOccurrence, WeatherData
    from services.disease_prediction_service import HistoricalPatternAnalyzer
    from datetime import datetime

    location = request.args.get("location", "")
    lat = request.args.get("lat", type=float)
    lon = request.args.get("lon", type=float)

    if not location:
        return jsonify({"error": "Location required"}), 400

    try:
        # Get historical occurrences
        occurrences = (
            DiseaseOccurrence.query.filter(
                DiseaseOccurrence.location_name.ilike(f"%{location}%")
            )
            .order_by(DiseaseOccurrence.occurrence_date.desc())
            .limit(1000)
            .all()
        )
        occurrences_data = [o.to_dict() for o in occurrences]

        # Get historical weather data
        weather_records = (
            WeatherData.query.filter(WeatherData.location_name.ilike(f"%{location}%"))
            .limit(1000)
            .all()
        )
        weather_data = [w.to_dict() for w in weather_records]

        # Train analyzer
        analyzer = HistoricalPatternAnalyzer()
        analyzer.train(occurrences_data, weather_data)

        # Get current month
        current_month = datetime.now().month

        # Get current weather if coordinates provided
        current_weather = None
        if lat and lon:
            from services.weather_service import get_current_weather

            current_weather_data = get_current_weather(lat, lon)
            if current_weather_data:
                current_weather = {
                    "temperature_avg": current_weather_data.get("temperature", 0),
                    "humidity": current_weather_data.get("humidity", 0),
                    "rainfall": current_weather_data.get("rainfall", 0),
                }

        # Predict from history
        predictions = analyzer.predict_from_history(
            location, current_month, current_weather
        )

        return jsonify(
            {
                "location": location,
                "current_month": current_month,
                "predictions": predictions,
                "trained_on_occurrences": len(occurrences_data),
            }
        )
    except Exception as e:
        logger.error(f"Error predicting from history: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/historical/peak-season/<disease_name>")
def api_peak_season(disease_name):
    """API endpoint to get peak season for a specific disease"""
    from models import DiseaseOccurrence
    from services.disease_prediction_service import HistoricalPatternAnalyzer

    try:
        occurrences = DiseaseOccurrence.query.limit(1000).all()
        occurrences_data = [o.to_dict() for o in occurrences]

        analyzer = HistoricalPatternAnalyzer()
        analyzer.train(occurrences_data)

        peak_season = analyzer.get_peak_season(disease_name)

        if not peak_season:
            return jsonify({"error": "Disease not found or insufficient data"}), 404

        return jsonify(peak_season)
    except Exception as e:
        logger.error(f"Error getting peak season: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/historical/regional-ranking")
def api_regional_ranking():
    """API endpoint to get disease risk ranking for a region"""
    from models import DiseaseOccurrence
    from services.disease_prediction_service import HistoricalPatternAnalyzer

    location = request.args.get("location", "")

    if not location:
        return jsonify({"error": "Location required"}), 400

    try:
        occurrences = (
            DiseaseOccurrence.query.filter(
                DiseaseOccurrence.location_name.ilike(f"%{location}%")
            )
            .limit(1000)
            .all()
        )
        occurrences_data = [o.to_dict() for o in occurrences]

        analyzer = HistoricalPatternAnalyzer()
        analyzer.train(occurrences_data)

        ranking = analyzer.get_regional_risk_ranking(location)

        return jsonify({"location": location, "ranking": ranking})
    except Exception as e:
        logger.error(f"Error getting regional ranking: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/historical/disease-trend/<disease_name>")
def api_disease_trend(disease_name):
    """API endpoint to get disease trend analysis"""
    from models import DiseaseOccurrence
    from services.disease_prediction_service import HistoricalPatternAnalyzer

    months = request.args.get("months", 12, type=int)

    try:
        occurrences = DiseaseOccurrence.query.limit(1000).all()
        occurrences_data = [o.to_dict() for o in occurrences]

        analyzer = HistoricalPatternAnalyzer()
        analyzer.train(occurrences_data)

        trend = analyzer.get_disease_trend(disease_name, months)

        if not trend or "trend" not in trend:
            return jsonify({"error": "Disease not found or insufficient data"}), 404

        return jsonify(trend)
    except Exception as e:
        logger.error(f"Error getting disease trend: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/report-disease-occurrence", methods=["POST"])
@login_required
def api_report_disease_occurrence():
    """API endpoint to report a disease occurrence (for ML training)"""
    from models import DiseaseOccurrence, Disease

    data = request.get_json()

    disease_id = data.get("disease_id")
    location_name = data.get("location_name")
    latitude = data.get("latitude")
    longitude = data.get("longitude")
    occurrence_date = data.get("occurrence_date")
    severity = data.get("severity", "moderate")
    affected_area = data.get("affected_area")
    notes = data.get("notes")

    if not disease_id or not location_name or not occurrence_date:
        return (
            jsonify(
                {"error": "disease_id, location_name, and occurrence_date required"}
            ),
            400,
        )

    try:
        # Validate disease exists
        disease = Disease.query.get(disease_id)
        if not disease:
            return jsonify({"error": "Disease not found"}), 404

        # Parse date
        from datetime import datetime

        try:
            occurrence_date = datetime.strptime(occurrence_date, "%Y-%m-%d").date()
        except ValueError:
            return jsonify({"error": "Invalid date format. Use YYYY-MM-DD"}), 400

        # Create occurrence record
        occurrence = DiseaseOccurrence(
            disease_id=disease_id,
            location_name=location_name,
            latitude=latitude,
            longitude=longitude,
            occurrence_date=occurrence_date,
            severity=severity,
            affected_area=affected_area,
            reported_by=current_user.id,
            notes=notes,
        )

        db.session.add(occurrence)
        db.session.commit()

        return jsonify(
            {
                "message": "Disease occurrence reported successfully",
                "occurrence_id": occurrence.id,
            }
        )
    except Exception as e:
        logger.error(f"Error reporting disease occurrence: {e}")
        db.session.rollback()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    with app.app_context():
        db.create_all()

    ensure_models_loaded()

    try:
        registry.register_model(
            model_type="resnet",
            version="v1.0",
            path="models/cotton_crop_disease_classification/full_resnet50_model.pth",
            accuracy=0.9983,
        )
        registry.register_model(
            model_type="yolo",
            version="v1.0",
            path="models/cotton_crop_growth_stage_prediction/best.pt",
            accuracy=0.6006,
        )
        registry.set_active_model("resnet", "v1.0")
        registry.set_active_model("yolo", "v1.0")
    except Exception as e:
        logger.error(f"Error registering models: {e}")

    is_debug = os.getenv("FLASK_DEBUG", "False").lower() in ("true", "1", "t")
    app.run(debug=is_debug, host="0.0.0.0", port=5000)
