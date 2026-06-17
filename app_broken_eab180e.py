"""
Agri-Vision Flask Application
Unified inference for disease classification (ResNet50) and growth stage prediction (YOLOv8)
"""
import hashlib
import logging
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for, send_file
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
import os
import random
import math
import re
import functools
import threading
from datetime import datetime
from typing import Any, Dict, Optional, Tuple
from werkzeug.utils import secure_filename
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from io import BytesIO
from services.weather_service import get_weather
from sqlalchemy import inspect, text

import redis
import base64
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
    Request,
)
from flask_cors import CORS
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO
import json
from jinja2 import Environment, FileSystemLoader
from model_registry import registry
from services.weather_service import generate_weather_recommendations
from services.yield_service import estimate_yield
from services.auth_security_service import (
    AccountLockoutService,
    get_client_ip,
    get_user_agent,
)
from security_utils import (
    UploadValidationError,
    cleanup_temp_upload,
    resolve_secret_key,
    save_temp_upload,
    validate_image_upload,
)

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder="static", template_folder="templates")

# --- Database Configuration ---
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///agri_vision.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Try dynamic package loading to prevent crash on automated CI testing rigs
redis_host = os.getenv("REDIS_HOST", "localhost")
redis_port = int(os.getenv("REDIS_PORT", "6379"))
redis_db = int(os.getenv("REDIS_DB", "0"))
limiter_storage_uri = "memory://"

try:
    redis_client = redis.Redis(
        host=redis_host,
        port=redis_port,
        db=redis_db,
        decode_responses=True,
    )

    redis_client.ping()
    limiter_storage_uri = f"redis://{redis_host}:{redis_port}/{redis_db}"
    logger.info("redis connected for caching and rate limiting")
except (redis.exceptions.ConnectionError, ModuleNotFoundError) as err:
    logger.warning(f"caching layer bypass active: {err}")
    redis_client = None

limiter = Limiter(
    get_remote_address,
    app=app,
    storage_uri=limiter_storage_uri,
    strategy="fixed-window",
)
from models import db
db.init_app(app)


_account_lockout_schema_checked = False


def ensure_account_lockout_schema() -> None:
    """Backfill account lockout columns for existing create_all-managed DBs."""
    inspector = inspect(db.engine)
    if "users" not in inspector.get_table_names():
        return

    existing_columns = {column["name"] for column in inspector.get_columns("users")}
    dialect = db.engine.dialect.name
    datetime_type = "TIMESTAMP" if dialect == "postgresql" else "DATETIME"
    columns = {
        "failed_login_attempts": "INTEGER NOT NULL DEFAULT 0",
        "last_failed_login_at": datetime_type,
        "account_locked_until": datetime_type,
        "last_successful_login_at": datetime_type,
        "last_failed_ip": "VARCHAR(64)",
        "last_successful_ip": "VARCHAR(64)",
    }

    changed = False
    with db.engine.begin() as connection:
        for column_name, ddl_type in columns.items():
            if column_name not in existing_columns:
                connection.execute(text(f"ALTER TABLE users ADD COLUMN {column_name} {ddl_type}"))
                changed = True
        connection.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_users_account_locked_until "
                "ON users (account_locked_until)"
            )
        )
    if changed:
        logger.info("Account lockout schema columns added to users table")


@app.before_request
def _ensure_account_lockout_schema_once() -> None:
    global _account_lockout_schema_checked
    if _account_lockout_schema_checked or app.config.get("TESTING"):
        return
    try:
        ensure_account_lockout_schema()
        _account_lockout_schema_checked = True
    except Exception as exc:
        logger.warning("Account lockout schema check skipped: %s", exc)

# Serialize concurrent writers on SQLite (e.g. refresh-token rotation).
with app.app_context():
    configure_sqlite_immediate_transactions(db.engine)


# --- Login Manager Configuration ---
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Please log in to access this page.'
login_manager.login_message_category = 'info'

@login_manager.user_loader
def load_user(user_id):
    from models import User
    return User.query.get(user_id)

# --- Google OAuth 2.0 Configuration (issue #626) ---
from authlib.integrations.flask_client import OAuth as _OAuth

_oauth = _OAuth(app)
_google_client_id = os.getenv("GOOGLE_CLIENT_ID", "")
_google_client_secret = os.getenv("GOOGLE_CLIENT_SECRET", "")

if _google_client_id and _google_client_secret:
    _oauth.register(
        name="google",
        client_id=_google_client_id,
        client_secret=_google_client_secret,
        server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
        client_kwargs={"scope": "openid email profile"},
    )
    GOOGLE_OAUTH_ENABLED = True
    logger.info("Google OAuth 2.0 enabled.")
else:
    GOOGLE_OAUTH_ENABLED = False
    logger.warning(
        "GOOGLE_CLIENT_ID / GOOGLE_CLIENT_SECRET not set — Google OAuth disabled."
    )

from functools import wraps

def api_login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not current_user.is_authenticated:
            from flask import jsonify
            return jsonify({"status": "error", "error": "Authentication required"}), 401
        return f(*args, **kwargs)
    return decorated

# --- Security Configuration ---
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # Increased to 50MB for batch uploads

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
# ------------------------------

class CustomRequest(Request):
    max_form_memory_size = 25 * 1024 * 1024  # Support larger base64-encoded forms

app.request_class = CustomRequest

swagger = Swagger(app)
CORS(app)

app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
app.jinja_env.auto_reload = True
app.jinja_env.cache = {}

try:
    secret_key = resolve_secret_key(os.environ)
except RuntimeError as exc:
    logger.critical(str(exc))
    raise SystemExit(str(exc))
app.secret_key = secret_key
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024
app.config["MAX_FORM_MEMORY_SIZE"] = 25 * 1024 * 1024
app.config.setdefault("UPLOAD_MAX_BYTES", app.config["MAX_CONTENT_LENGTH"])
app.config.setdefault("UPLOAD_RATE_LIMIT", "10 per minute")
app.config.setdefault("API_UPLOAD_RATE_LIMIT", "20 per minute")
app.config.setdefault("UPLOAD_TMP_DIR", os.path.join(app.instance_path, "uploads"))
os.makedirs(app.config["UPLOAD_TMP_DIR"], exist_ok=True)

LANG = {
    "en": {"welcome": "Welcome to Agri Vision"},
    "te": {"welcome": "అగ్రి విజన్‌కు స్వాగతం"},
}

os.makedirs("static/uploads", exist_ok=True)
os.makedirs("static/css", exist_ok=True)
os.makedirs("models", exist_ok=True)

ALLOWED_IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}
ALLOWED_IMAGE_MIME_TYPES = {"image/png", "image/jpeg", "image/gif"}
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


# -------------------------------------------------------------------
# THREAD-SAFE MODEL MANAGER
# -------------------------------------------------------------------
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
                    try:
                        self.resnet_model = torch.load(
                            RESNET_MODEL_PATH,
                            map_location=torch.device("cpu"),
                        )
                    except Exception:
                        self.resnet_model = torch.load(
                            RESNET_MODEL_PATH,
                            map_location=torch.device("cpu"),
                            weights_only=False
                        )
                    self.resnet_model.eval()
                    self.errors["resnet"] = None
                    logger.info("ResNet50 model loaded successfully")
                except Exception as exc:
                    self.errors["resnet"] = str(exc)
                    logger.warning(f"ResNet50 model not found or failed to load: {exc}")
                    self.resnet_model = None

            if self.yolo_model is None:
                try:
                    self.yolo_model = YOLO(YOLO_MODEL_PATH)
                    self.errors["yolo"] = None
                    logger.info("YOLOv8 model loaded successfully")
                except Exception as exc:
                    self.errors["yolo"] = str(exc)
                    logger.warning(f"YOLOv8 model not found or failed to load: {exc}")
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

resnet_model = None
yolo_model = None
grad_cam_instance = None


@functools.lru_cache(maxsize=1)
def load_models():
    """Delegate to ModelManager singleton for thread-safe model loading."""
    global resnet_model, yolo_model
    if resnet_model is None:
        try:
            resnet_model = torch.load(
                'models/cotton_crop_disease_classification/full_resnet50_model.pth',
                map_location=torch.device('cpu'),
            )
            logger.info("ResNet50 model loaded successfully")

            if resnet_model is not None and grad_cam_instance is None:
                grad_cam_instance = GradCAM(resnet_model, resnet_model.layer4[-1])

        except Exception as e:
            logger.warning(f"ResNet50 model not found or failed to load: {e}")
            resnet_model = None
    if yolo_model is None:
        try:
            yolo_model = YOLO('models/cotton_crop_growth_stage_prediction/best.pt')
            logger.info("YOLOv8 model loaded successfully")
        except Exception as e:
            logger.warning(f"YOLOv8 model not found or failed to load: {e}")
            yolo_model = None
    return resnet_model, yolo_model


def ensure_models_loaded() -> None:
    load_models()


# -------------------------------------------------------------------
# HELPER FUNCTIONS
# -------------------------------------------------------------------
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


def generate_pure_heatmap(image_rgb: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
    h, w, _ = image_rgb.shape
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_255 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_255, cv2.COLORMAP_JET)
    return cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)


def apply_heatmap_on_image(image_rgb: np.ndarray, heatmap: np.ndarray, alpha: float = 0.6, beta: float = 0.4) -> np.ndarray:
    heatmap_color_rgb = generate_pure_heatmap(image_rgb, heatmap)
    return cv2.addWeighted(image_rgb, alpha, heatmap_color_rgb, beta, 0)


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.heatmap_np = None
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

    def __call__(self, input_tensor: torch.Tensor, target_class_idx: Optional[int], original_image_rgb: np.ndarray) -> Optional[np.ndarray]:
        if self.model is None:
            logger.warning("Grad-CAM: model is not loaded.")
            return None

        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        self.activations = None
        self.gradients = None
        self.heatmap_np = None

        try:
            device = next(self.model.parameters()).device
            input_tensor = input_tensor.to(device)

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
                self.heatmap_np = heatmap_np
                return apply_heatmap_on_image(original_image_rgb, heatmap_np)

        except Exception as exc:
            logger.error("Error generating Grad-CAM: %s", exc)
            return None
        finally:
            self.gradients = None
            self.activations = None


# -------------------------------------------------------------------
# INFERENCE PIPELINE
# -------------------------------------------------------------------

# Define ONCE at module level — built once, reused every request.
# Includes ImageNet normalization required by ResNet50 for correct inference.
RESNET_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def preprocess_image_for_resnet(image: np.ndarray) -> torch.Tensor:
    """Preprocess an RGB numpy image for ResNet50 inference.

    Uses the module-level RESNET_TRANSFORM pipeline which includes
    ImageNet normalization (mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]).
    """
    return RESNET_TRANSFORM(image).unsqueeze(0)


def infer_disease(image):
    # Returns all disease outputs, including confidences for each class
    if model_manager.resnet_model:
        processed = preprocess_image_for_resnet(image)
        with torch.no_grad():
            output = model_manager.resnet_model(processed)
            probs = F.softmax(output, dim=1)
            confidence, prediction = torch.max(probs, 1)
        probs_np = probs.numpy()  # shape: (1, 8)
        class_idx = int(prediction.item())
        confidence_value = float(confidence.item())
        predicted_class = disease_classes[class_idx]
        healthy_idx = disease_classes.index("Healthy")  
        health_score = float(probs_np[0][healthy_idx]) * 100


    else:
        # Demo fallback
        probs_np = np.random.rand(1, len(disease_classes))
        probs_np = probs_np / probs_np.sum(axis=1, keepdims=True)
        class_idx = int(np.argmax(probs_np[0]))
        confidence_value = float(np.max(probs_np[0]))
        predicted_class = disease_classes[class_idx]
        health_score = float(np.max(probs_np[0]))*100

    # Format probabilities per class
    disease_confidences = {disease_classes[i]: float(probs_np[0][i]) for i in range(len(disease_classes))}

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
            if hasattr(r, 'boxes'):
                for b in r.boxes:
                    class_id = int(b.cls[0].item()) if hasattr(b.cls[0], 'item') else int(b.cls[0])
                    conf = float(b.conf[0].item()) if hasattr(b.conf[0], 'item') else float(b.conf[0])
                    xyxy = b.xyxy[0].cpu().numpy().tolist()
                    boxes.append({
                        "class_id": class_id,
                        "class_name": growth_stage_classes[class_id] if class_id < len(growth_stage_classes) else str(class_id),
                        "confidence": conf,
                        "bbox": xyxy,  # [x1, y1, x2, y2]
                    })
            else:
                continue
        # Most confident box as main prediction
        if len(boxes):
            main = max(boxes, key=lambda x: x['confidence'])
            result.update({
                "main_class": main["class_name"],
                "main_class_idx": main["class_id"],
                "confidence": main["confidence"],
            })
            result["boxes"] = boxes
        result["raw"] = boxes
    return result


def generate_recommendations(disease_result: Dict[str, Any], growth_result: Dict[str, Any], weather: Optional[Dict[str, Any]] = None) -> list[str]:
    recs: list[str] = []
    dclass = disease_result["predicted_class"]

    instr_map = {
        "Aphids": ["Inspect leaves closely for clusters of small pests.", "Use recommended insecticides if infestation is severe."],
        "Army worm": ["Increase scouting frequency.", "Apply biological or suitable chemical controls early."],
        "Bacterial blight": ["Avoid overhead irrigation.", "Remove and destroy affected plant parts."],
        "Cotton Boll Rot": ["Improve field drainage, avoid stagnant water.", "Remove and destroy rotten bolls."],
        "Green Cotton Boll": ["Monitor bolls for signs of pests or disease.", "Maintain optimal nutrient regime."],
        "Healthy": ["Continue general crop monitoring.", "Maintain optimal fertilization and irrigation."],
        "Powdery mildew": ["Remove infected plant debris.", "Apply fungicide at recommended intervals."],
        "Target Spot": ["Monitor for spread, reduce leaf wetness.", "Apply suitable fungicide if required."],
    }
    recs.extend(instr_map.get(dclass, ["Practice general crop hygiene."]))
    
    if disease_result["health_score"] < 50:
        recs.append("Consult an agricultural expert urgently for low health score.")
        recs.append("Consult an agricultural expert if symptoms persist.")
    elif disease_result["health_score"] < 70:
        recs.append("Increase frequency of crop monitoring based on moderate health.")

    if disease_result.get("is_uncertain"):
        recs.append("Model confidence is low. Please upload a clearer image or consult an agricultural expert.")
    elif disease_result.get("is_ambiguous"):
        alt = disease_result.get("alternative_prediction", {}).get("class", "another condition")
        recs.append(f"The prediction may overlap with {alt}. Monitor the crop closely before applying treatment.")

    gmain = growth_result.get("main_class", None)
    grow_map = {
        "Cotton Blossom": ["Maintain regular watering during blossom phase.", "Scout for early flower pests."],
        "Cotton Bud": ["Ensure adequate phosphorus supply.", "Monitor for budworm."],
        "Early Boll": ["Start borer management as boll phase begins.", "Avoid excess nitrogen at this stage."],
        "Matured Cotton Boll": ["Reduce irrigation to harden bolls.", "Plan for harvest in coming weeks."],
        "Split Cotton Boll": ["Prepare for immediate harvest.", "Avoid rainfall exposure to split bolls."],
    }
    if gmain in grow_map:
        recs.extend(grow_map[gmain])

    if weather:
        recs.extend(generate_weather_recommendations(weather))

    return recs[:6]


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


def generate_treatment_recommendations(disease_result: Dict[str, Any]) -> Dict[str, Any]:
    from services.recommendation_engine import get_recommendations

    return get_recommendations(
        "cotton",
        disease_result.get("predicted_class"),
        confidence=disease_result.get("confidence"),
    )


def encode_image_for_display(image: np.ndarray) -> str:
    display_image = resize_image(image, DISPLAY_IMAGE_MAX_DIMENSION)
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), DISPLAY_JPEG_QUALITY]
    ok, buffer = cv2.imencode(".jpg", display_image, encode_params)
    if not ok:
        raise ValueError("Failed to encode image for display")
    return base64.b64encode(buffer).decode("utf-8")


def is_allowed_image(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS


def calculate_file_hash(file_storage) -> str:
    """Generate SHA-256 hash for an uploaded file using chunk reading."""
    sha256_hash = hashlib.sha256()
    file_storage.seek(0)
    for byte_block in iter(lambda: file_storage.read(4096), b""):
        sha256_hash.update(byte_block)
    file_storage.seek(0)
    return sha256_hash.hexdigest()


def get_upload_max_bytes() -> int:
    max_bytes = app.config.get("UPLOAD_MAX_BYTES") or app.config.get("MAX_CONTENT_LENGTH")
    return int(max_bytes or 10 * 1024 * 1024)


def enforce_request_size(max_bytes: int) -> None:
    content_length = request.content_length
    if content_length is not None and content_length > max_bytes:
        raise UploadValidationError("File exceeds maximum upload size.", status_code=413)


def read_validated_upload_image(file_storage) -> Tuple[str, np.ndarray, np.ndarray, str]:
    max_bytes = get_upload_max_bytes()
    safe_filename, file_bytes, _mime = validate_image_upload(
        file_storage,
        allowed_extensions=ALLOWED_IMAGE_EXTENSIONS,
        allowed_mime_types=ALLOWED_IMAGE_MIME_TYPES,
        max_bytes=max_bytes,
    )
    temp_path = save_temp_upload(file_bytes, app.config["UPLOAD_TMP_DIR"], safe_filename)

    try:
        img = Image.open(BytesIO(file_bytes))
        img.verify()
    except Exception:
        raise UploadValidationError(
            "Unable to process this image. It may be corrupt or in an unsupported format.",
            status_code=400,
        )

    image = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise UploadValidationError("Unable to process this image. It may be corrupt or in an unsupported format.", status_code=400)
    return safe_filename, image, cv2.cvtColor(image, cv2.COLOR_BGR2RGB), temp_path


def read_uploaded_image(file_storage) -> Tuple[str, np.ndarray, np.ndarray]:
    safe_filename = secure_filename(file_storage.filename)
    file_bytes = np.frombuffer(file_storage.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Error reading image file")
    return safe_filename, image, cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# -------------------------------------------------------------------
# THREAD-SAFE GRAD-CAM CACHE
# -------------------------------------------------------------------
GRAD_CAM_CACHE = {}
GRAD_CAM_CACHE_LOCK = threading.Lock()
MAX_CACHE_SIZE = 100

def get_cached_grad_cam(image_hash: str) -> Optional[Tuple[str, str]]:
    with GRAD_CAM_CACHE_LOCK:
        return GRAD_CAM_CACHE.get(image_hash)

def set_cached_grad_cam(image_hash: str, overlay_b64: str, heatmap_only_b64: str) -> None:
    with GRAD_CAM_CACHE_LOCK:
        if len(GRAD_CAM_CACHE) >= MAX_CACHE_SIZE:
            # FIFO eviction
            first_key = next(iter(GRAD_CAM_CACHE))
            GRAD_CAM_CACHE.pop(first_key, None)
        GRAD_CAM_CACHE[image_hash] = (overlay_b64, heatmap_only_b64)


def generate_gradcam_explanation(
    resnet_model: torch.nn.Module,
    image: np.ndarray,
    disease_result: Dict[str, Any],
) -> Tuple[Optional[str], Optional[str]]:
    input_tensor = preprocess_image_for_resnet(image)
    with GradCAM(resnet_model, resnet_model.layer4[-1]) as grad_cam:
        grad_cam_overlay = grad_cam(input_tensor, disease_result["predicted_class_idx"], image)
        heatmap_np = getattr(grad_cam, "heatmap_np", None)

    grad_cam_image_b64 = encode_image_for_display(grad_cam_overlay) if grad_cam_overlay is not None else None
    heatmap_only_b64 = None
    if heatmap_np is not None:
        pure_heatmap_rgb = generate_pure_heatmap(image, heatmap_np)
        heatmap_only_b64 = encode_image_for_display(pure_heatmap_rgb)

    return grad_cam_image_b64, heatmap_only_b64


def analyze_image(image: np.ndarray,*,weather:Optional[dict]=None,field_acres: float=1.0) -> Dict[str, Any]:
    import time
    start_time = time.time()
    field_acres=normalize_field_acres(field_acres)
    
    
    resnet_model, yolo_model = model_manager.load_models()
    try:
        try:
            growth = infer_growth_stage(image)
        except Exception as exc:
            logger.error("Error during growth stage inference: %s", exc)
            growth = {"main_class": None, "main_class_idx": None, "confidence": 0.0, "boxes": [], "raw": []}

        disease = infer_disease(image)
        if not isinstance(disease, dict) or "predicted_class" not in disease or "health_score" not in disease:
            raise ValueError("Invalid disease model prediction output.")

        # Track metrics in registry
        inference_time = time.time() - start_time
        try:
            # Update ResNet metrics
            if disease and disease.get("confidence"):
                registry.update_metrics(
                    model_type="resnet",
                    version="v1.0",
                    confidence=disease.get("confidence", 0.0),
                    inference_time=inference_time,
                    success=True
                )
            
            # Update YOLO metrics
            if growth and growth.get("confidence"):
                registry.update_metrics(
                    model_type="yolo",
                    version="v1.0",
                    confidence=growth.get("confidence", 0.0),
                    inference_time=inference_time,
                    success=True
                )
        except Exception as e:
            logger.error(f"Error tracking metrics: {e}")

        # Check cache first
        image_hash = hashlib.sha256(image.tobytes()).hexdigest()
        cached_result = get_cached_grad_cam(image_hash)
        
        grad_cam_image_b64 = None
        heatmap_only_b64 = None
        explainability = {"available": False, "status": "unavailable"}

        if cached_result is not None:
            grad_cam_image_b64, heatmap_only_b64 = cached_result
            explainability = {"available": True, "status": "cached"}
            logger.info("Using cached Grad-CAM heatmaps")
        else:
            if resnet_model is not None and disease.get("predicted_class_idx") is not None:
                try:
                    grad_cam_image_b64, heatmap_only_b64 = generate_gradcam_explanation(
                        resnet_model,
                        image,
                        disease,
                    )
                    if grad_cam_image_b64 and heatmap_only_b64:
                        explainability = {"available": True, "status": "generated"}
                except Exception as exc:
                    logger.error("Error generating Grad-CAM: %s", exc)
                    explainability = {"available": False, "status": "failed"}

            if grad_cam_image_b64 is None or heatmap_only_b64 is None:
                try:
                    mock_heatmap = generate_mock_heatmap(image)
                    mock_overlay = apply_heatmap_on_image(image, mock_heatmap)
                    grad_cam_image_b64 = encode_image_for_display(mock_overlay)
                    
                    pure_heatmap_rgb = generate_pure_heatmap(image, mock_heatmap)
                    heatmap_only_b64 = encode_image_for_display(pure_heatmap_rgb)
                    if explainability["status"] == "unavailable":
                        explainability = {"available": False, "status": "fallback"}
                except Exception as exc:
                    logger.error("Error generating fallback heatmap: %s", exc)
            
            if grad_cam_image_b64 and heatmap_only_b64:
                set_cached_grad_cam(image_hash, grad_cam_image_b64, heatmap_only_b64)

        disease["heatmap_b64"] = grad_cam_image_b64
        disease["heatmap_only_b64"] = heatmap_only_b64

        recs = generate_recommendations(disease, growth,weather=weather)
        severity = calculate_disease_severity(disease["health_score"])
        yield_est = estimate_yield(disease, growth, weather=weather, field_acres=field_acres)
        adv_recs = generate_advanced_recommendations(disease, growth)
        treatment_recs = generate_treatment_recommendations(disease)
        insights = generate_farmer_insights(disease, growth)

        result = {
            "disease": disease,
            "growth": growth,
            "recommendations": recs,
            "treatment_recommendations": treatment_recs,
            "grad_cam_image_b64": grad_cam_image_b64,
            "heatmap_only_b64": heatmap_only_b64,
            "explainability": explainability,
            "disease_severity": severity,
            "yield_estimate": yield_est,
            "advanced_recommendations": adv_recs,
            "farmer_insights": insights,
        }

        if weather is not None:
            result["weather"]=weather

        if growth.get("main_class") is None:
            fallback_reason = "Growth stage model unavailable in this deployment." if yolo_model is None else "Cotton growth stage could not be detected from the uploaded image."
            result["warnings"] = [
                fallback_reason,
                "Disease analysis is still provided, but comparison may be less reliable without a confirmed cotton crop detection.",
                "Grad-CAM explainability may also be affected if the primary crop is not detected.",
            ]

        return result
    except Exception as exc:
        logger.error("Unexpected error in image analysis: %s", exc)
        return {"error": "The AI model encountered an unexpected error while analyzing the image. Please verify the image file format and content and try again."}


#---helper functions------
def normalize_field_acres(value:object)->float:
    """Coerce field size to a positive float; invalid or non-positive → 1.0."""
    try:
        if value is None or value=="":
            return 1.0
        fa=float(value)
        if fa<=0 or fa!=fa:
            return 1.0
        return fa
    except (TypeError,ValueError):
        return 1.0

def parse_api_field_acres(raw:object)->Tuple[Optional[float],Optional[str]]:
    """
    For POST /api/analyze: missing or blank field_acres → (1.0, None).
    Present but invalid, non-positive, or non-finite → (None, error message).
    """
    if raw is None:
        return 1.0,None
    s=str(raw).strip()
    if s=="":
        return 1.0,None
    try:
        fa=float(s)
    except (TypeError,ValueError):
        return None,"field_acres must be a positive number"
    
    if not math.isfinite(fa) or fa<=0:
        return None,"field_acres must be a positive finite number"
    return fa,None

def resolve_weather_for_analysis(lat:Optional[float]=None,lon:Optional[float]=None,city:Optional[str]=None)->Optional[dict]:
    """
    Fetch current weather from lat/lon, or from city name via geocoding.
    Never raises; returns None if inputs missing or upstream fails.
    """
    owm_key=os.getenv("OPENWEATHER_API_KEY")
    if lat is not None and lon is not None:
        try:
            return get_weather(float(lat),float(lon),owm_key)
        except (TypeError,ValueError):
            pass
    
    if city and str(city).strip():
        try:
            geo=geocode_city(str(city).strip())
            if geo:
                return get_weather(float(geo["lat"]),float(geo["lon"]),owm_key)
        except (ValueError,KeyError):
            pass 
    
    return None

def build_comparison_result(old_results: Dict[str, Any], new_results: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(old_results, dict) or not isinstance(new_results, dict):
        raise ValueError("Comparison analysis did not produce valid result objects.")

    old_disease = old_results.get("disease")
    new_disease = new_results.get("disease")
    if old_disease is None or new_disease is None:
        raise ValueError("Unable to compare the provided images because one or both images did not contain a valid cotton crop analysis.")

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
        "Disease spread reduced" if disease_reduced else (f"Disease signal shifted from {old_predicted} to {new_predicted}" if disease_changed else f"Disease signal remains {new_predicted}"),
        recommendation,
    ]

    if new_results.get("recommendations"):
        summary.append(f"Model priority: {new_results['recommendations'][0]}")

    if isinstance(new_results.get("farmer_insights"), list):
        insight_msg = f"Crop health improved by {abs_change:.1f}% this week." if change > 0 else (f"Crop health declined by {abs_change:.1f}% this week." if change < 0 else "Crop health remained stable this week.")
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


# --- Security Headers ---
@app.after_request
def apply_security_headers(response):
    response.headers["X-Frame-Options"] = "SAMEORIGIN"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response
# -------------------------------------------------------------------
# FLASK ROUTES
# -------------------------------------------------------------------
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


@app.route("/model-admin")
@login_required
def admin_dashboard():
    if not current_user.is_researcher():
        flash('Access denied. Researchers and Admins only.', 'danger')
        return redirect(url_for('index'))
    return render_template("admin.html")


# --- Model Management Admin Endpoints ---

@app.route('/admin/models', methods=['GET'])
@login_required
def list_models():
    """List all registered models with their metadata"""
    model_type = request.args.get('type')
    try:
        models = registry.list_models(model_type)
        return jsonify({
            "status": "success",
            "models": models,
            "ab_test_enabled": registry.ab_test_enabled,
            "rollback_threshold": registry.rollback_threshold
        })
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/admin/models/active', methods=['GET'])
def get_active_models():
    """Get currently active models"""
    try:
        active_resnet = registry.get_active_model("resnet")
        active_yolo = registry.get_active_model("yolo")
        return jsonify({
            "status": "success",
            "active_models": {
                "resnet": active_resnet.to_dict() if active_resnet else None,
                "yolo": active_yolo.to_dict() if active_yolo else None
            }
        })
    except Exception as e:
        logger.error(f"Error getting active models: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/admin/models/register', methods=['POST'])
def register_model():
    """Register a new model version"""
    try:
        data = request.get_json()
        required_fields = ['model_type', 'version', 'path']
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        metadata = registry.register_model(
            model_type=data['model_type'],
            version=data['version'],
            path=data['path'],
            accuracy=data.get('accuracy', 0.0),
            dataset_version=data.get('dataset_version', 'unknown'),
            parameters=data.get('parameters', 0),
            is_active=data.get('is_active', False),
            ab_test_ratio=data.get('ab_test_ratio', 0.0)
        )
        return jsonify({
            "status": "success",
            "message": f"Model {data['model_type']} version {data['version']} registered successfully",
            "metadata": metadata.to_dict()
        })
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        logger.error(f"Error registering model: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/admin/models/activate', methods=['POST'])
def activate_model():
    """Set a model version as active"""
    try:
        data = request.get_json()
        required_fields = ['model_type', 'version']
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        registry.set_active_model(data['model_type'], data['version'])
        return jsonify({
            "status": "success",
            "message": f"Model {data['model_type']} version {data['version']} activated successfully"
        })
    except ValueError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        logger.error(f"Error activating model: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/admin/models/delete', methods=['DELETE'])
def delete_model():
    """Delete a model version"""
    try:
        data = request.get_json()
        required_fields = ['model_type', 'version']
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        registry.delete_model(data['model_type'], data['version'])
        return jsonify({
            "status": "success",
            "message": f"Model {data['model_type']} version {data['version']} deleted successfully"
        })
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Error deleting model: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/admin/models/ab-testing', methods=['POST'])
def toggle_ab_testing():
    """Enable or disable A/B testing"""
    try:
        data = request.get_json()
        enabled = data.get('enabled', True)
        registry.enable_ab_testing(enabled)
        return jsonify({
            "status": "success",
            "message": f"A/B testing {'enabled' if enabled else 'disabled'}",
            "ab_test_enabled": registry.ab_test_enabled
        })
    except Exception as e:
        logger.error(f"Error toggling A/B testing: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/admin/models/ab-ratio', methods=['POST'])
def set_ab_ratio():
    """Set A/B testing ratio for a model version"""
    try:
        data = request.get_json()
        required_fields = ['model_type', 'version', 'ratio']
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        registry.set_ab_test_ratio(data['model_type'], data['version'], data['ratio'])
        return jsonify({
            "status": "success",
            "message": f"A/B test ratio for {data['model_type']} version {data['version']} set to {data['ratio']}"
        })
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Error setting A/B ratio: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/admin/models/metrics', methods=['GET'])
def get_model_metrics():
    """Get performance metrics for all models"""
    try:
        models = registry.list_models()
        return jsonify({
            "status": "success",
            "metrics": models
        })
    except Exception as e:
        logger.error(f"Error getting model metrics: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/admin/models/rollback-threshold', methods=['POST'])
def set_rollback_threshold():
    """Set automatic rollback threshold"""
    try:
        data = request.get_json()
        threshold = data.get('threshold')
        if threshold is None:
            return jsonify({"error": "Missing required field: threshold"}), 400
        
        if not 0.0 <= threshold <= 1.0:
            return jsonify({"error": "Threshold must be between 0.0 and 1.0"}), 400
        
        registry.rollback_threshold = threshold
        registry.save_config()
        return jsonify({
            "status": "success",
            "message": f"Rollback threshold set to {threshold}",
            "rollback_threshold": registry.rollback_threshold
        })
    except Exception as e:
        logger.error(f"Error setting rollback threshold: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/admin/models/export/pdf', methods=['GET'])
@login_required
def export_pdf():
    """Export model metrics as PDF"""
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet
        from io import BytesIO

        models = registry.list_models()
        
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        elements = []
        styles = getSampleStyleSheet()

        # Title
        title = Paragraph("Model Performance Report", styles['Title'])
        elements.append(title)
        elements.append(Spacer(1, 12))

        # Date
        date = Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal'])
        elements.append(date)
        elements.append(Spacer(1, 12))

        # Table data
        table_data = [['Model Type', 'Version', 'Accuracy', 'Requests', 'Success Rate', 'Avg Confidence', 'Avg Time', 'Status']]
        
        if models:
            for model_type in models:
                for model in models[model_type]:
                    metrics = model.performance_metrics
                    success_rate = (metrics.successful_predictions / metrics.total_requests * 100) if metrics.total_requests > 0 else 0
                    table_data.append([
                        model_type.capitalize(),
                        model.version,
                        f"{model.accuracy * 100:.2f}%",
                        str(metrics.total_requests),
                        f"{success_rate:.1f}%",
                        f"{metrics.avg_confidence * 100:.1f}%",
                        f"{metrics.avg_inference_time:.3f}s",
                        'Active' if model.is_active else 'Inactive'
                    ])

        # Create table
        table = Table(table_data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        elements.append(table)
        doc.build(elements)
        
        buffer.seek(0)
        return send_file(
            buffer,
            as_attachment=True,
            download_name=f'model_metrics_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf',
            mimetype='application/pdf'
        )
    except ImportError:
        return jsonify({"error": "reportlab not installed. Install with: pip install reportlab"}), 500
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
                # Decoded Original Image
                orig_data = base64.b64decode(
                    image_b64.split(",")[-1] if "," in image_b64 else image_b64
                )
                orig_pil = PILImage.open(BytesIO(orig_data))
                
                # If Grad-CAM overlay image is also provided
                if gradcam_image_b64:
                    gc_data = base64.b64decode(
                        gradcam_image_b64.split(",")[-1] if "," in gradcam_image_b64 else gradcam_image_b64
                    )
                    gc_pil = PILImage.open(BytesIO(gc_data))
                    
                    # Create side-by-side RLImages of 2.85 inches wide
                    w = 2.85 * inch
                    h = w * orig_pil.height / orig_pil.width
                    if h > 2.2 * inch:
                        h = 2.2 * inch
                        w = h * orig_pil.width / orig_pil.height
                        
                    rl_orig = RLImage(BytesIO(orig_data), width=w, height=h)
                    rl_gc = RLImage(BytesIO(gc_data), width=w, height=h)
                    
                    # Create a side-by-side table
                    image_table_data = [
                        [Paragraph("<b>Original Leaf Image</b>", normal_style), Paragraph("<b>Explainable AI (Grad-CAM Overlay)</b>", normal_style)],
                        [rl_orig, rl_gc]
                    ]
                    img_table = Table(image_table_data, colWidths=[3 * inch, 3 * inch])
                    img_table.setStyle(TableStyle([
                        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                        ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ]))
                    elements.append(img_table)
                else:
                    # Single original image standard display
                    img_width = 6 * inch
                    img_height = img_width * orig_pil.height / orig_pil.width
                    if img_height > 3.5 * inch:
                        img_height = 3.5 * inch
                        img_width = img_height * orig_pil.width / orig_pil.height
                    
                    rl_orig = RLImage(BytesIO(orig_data), width=img_width, height=img_height)
                    elements.append(rl_orig)
                
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


@app.route("/compare")
@login_required
def compare():
    ids_param = request.args.get('ids', '')
    if not ids_param:
        flash("No analyses selected for comparison", "warning")
        return redirect(url_for('history'))

    analysis_ids = [aid.strip() for aid in ids_param.split(',') if aid.strip()]

    from models import AnalysisHistory
    analyses = AnalysisHistory.query.filter(
        AnalysisHistory.id.in_(analysis_ids),
        AnalysisHistory.user_id == current_user.id
    ).all()

    if not analyses:
        flash("No valid analyses found", "warning")
        return redirect(url_for('history'))

    canonical_fields = [
        ('disease', 'Detected Disease'),
        ('growth_stage', 'Growth Stage'),
        ('confidence', 'Confidence'),
        ('health_score', 'Health Score'),
        ('created_at', 'Analysis Date'),
    ]

    rows = []
    for key, label in canonical_fields:
        row = {"label": label, "key": key, "values": []}
        for analysis in analyses:
            if key == 'disease':
                val = (analysis.disease_result or {}).get('predicted_class')
            elif key == 'growth_stage':
                val = (analysis.growth_result or {}).get('main_class')
            elif key == 'confidence':
                val = analysis.confidence
            elif key == 'health_score':
                val = analysis.health_score
            elif key == 'created_at':
                val = analysis.created_at.strftime('%Y-%m-%d %H:%M') if analysis.created_at else None
            else:
                val = None
            row["values"].append(val)
        rows.append(row)

    return render_template('compare.html',
        analyses=analyses,
        rows=rows,
        enumerate=enumerate,
    )


@app.route("/health")
def health():
    ensure_models_loaded()
    diagnostics = model_manager.diagnostics()
    model_loaded = diagnostics["resnet"]["loaded"] and diagnostics["yolo"]["loaded"]
    status_code = 200 if model_loaded else 503
    return jsonify({
        "status": "healthy" if model_loaded else "degraded",
        "mode": "ready" if model_loaded else "degraded",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": model_loaded,
        "models": diagnostics,
        "service": "Agri-Vision Cotton Analysis API",
    }), status_code


@app.route("/analyze", methods=["GET", "POST"])
@limiter.limit(lambda: app.config.get("UPLOAD_RATE_LIMIT", "10 per minute"))
@login_required
def analyze():
    if request.method == "POST":
        temp_path = None
        try:
            enforce_request_size(get_upload_max_bytes())

            if "file" not in request.files:
                flash("No file uploaded", "error")
                return redirect(request.url)

            file = request.files["file"]
            safe_filename, image, image_rgb, temp_path = read_validated_upload_image(file)
            compressed_rgb = resize_image(image_rgb, MAX_INFERENCE_DIMENSION)

            lat = request.form.get("lat", type=float)
            lon = request.form.get("lon", type=float)
            city = request.form.get("city", type=str)
            field_acres=normalize_field_acres(request.form.get("field_acres"))

            weather=resolve_weather_for_analysis(lat=lat,lon=lon,city=city)

            results = analyze_image(compressed_rgb,weather=weather,field_acres=field_acres)

            if results.get("error"):
                raise ValueError(results["error"])
            
            predicted_class = results.get("disease", {}).get("predicted_class", "") or ""
            disease_info = disease_info_map.get(predicted_class, {})

            import time
            unique_filename = f"{int(time.time())}_{safe_filename}"
            file_path = os.path.join("static", "uploads", unique_filename)
            cv2.imwrite(file_path, image)
            
            from models import AnalysisHistory, db
            if current_user.is_authenticated:
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
                unique_filename=unique_filename,
                disease_classes=disease_classes,
                image_b64=encode_image_for_display(image_rgb),
                img_shape={"width": image.shape[1], "height": image.shape[0]},
                raw_json=json.dumps(results, indent=2),
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                weather=weather,
                grad_cam_image_b64=results.get("grad_cam_image_b64"),
                heatmap_only_b64=results.get("heatmap_only_b64"),
                disease_info=disease_info,
            )
        except UploadValidationError as exc:
            filename = request.files.get("file", {}).filename if request.files.get("file") else "unknown"
            logger.warning("Upload rejected (user=%s, file=%s): %s", current_user.id, filename, exc)
            if exc.status_code == 413:
                return ("File too large", 413)
            flash(str(exc), "error")
            return redirect(request.url)
        except Exception as exc:
            filename = request.files.get("file", {}).filename if request.files.get("file") else "unknown"
            logger.error("Analysis error (user=%s, file=%s): %s", current_user.id, filename, exc)
            flash(f"Error during analysis: {str(exc)}", "error")
            return redirect(request.url)
        finally:
            cleanup_temp_upload(temp_path)

    return render_template("upload.html")


@app.route("/api/explain", methods=["POST"])
@api_login_required
def api_explain():
    if "file" not in request.files:
        return jsonify({"status": "error", "error": "No file uploaded"}), 400
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"status": "error", "error": "No file selected"}), 400
        
    if not is_allowed_image(file.filename):
        return jsonify({"status": "error", "error": "Invalid file type. Please upload an image."}), 400
        
    try:
        _, image, image_rgb = read_uploaded_image(file)
        compressed_rgb = resize_image(image_rgb, MAX_INFERENCE_DIMENSION)
        
        # We just need to call analyze_image to generate the Grad-CAM and get results
        results = analyze_image(compressed_rgb)
        
        if "error" in results:
            return jsonify({"status": "error", "error": results["error"]}), 500
            
        disease_result = results.get("disease", {})
        
        return jsonify({
            "status": "success",
            "heatmap_b64": results.get("grad_cam_image_b64"),
            "heatmap_only_b64": results.get("heatmap_only_b64"),
            "target_layer": "ResNet50 layer4[-1]",
            "image_b64": encode_image_for_display(compressed_rgb),
            "predicted_class": disease_result.get("predicted_class", "Unknown"),
            "confidence": disease_result.get("confidence", 0.0)
        })
    except Exception as exc:
        logger.error("Error in API explain endpoint: %s", exc)
        return jsonify({"status": "error", "error": str(exc)}), 500


@app.route("/api/explain/target", methods=["POST"])
def api_explain_target():
    """Dynamically generate Grad-CAM for a specified target disease class index"""
    try:
        data = request.get_json() or {}
        image_path = data.get("image_path")
        target_class_idx = data.get("target_class_idx")
        
        if not image_path:
            return jsonify({"status": "error", "error": "No image_path provided"}), 400
        if target_class_idx is None:
            return jsonify({"status": "error", "error": "No target_class_idx provided"}), 400
            
        try:
            target_class_idx = int(target_class_idx)
            if not (0 <= target_class_idx < len(disease_classes)):
                return jsonify({"status": "error", "error": "Invalid target_class_idx"}), 400
        except ValueError:
            return jsonify({"status": "error", "error": "target_class_idx must be an integer"}), 400
            
        from werkzeug.utils import secure_filename
        safe_filename = secure_filename(image_path)
        full_image_path = os.path.join("static", "uploads", safe_filename)
        if not os.path.exists(full_image_path):
            return jsonify({"status": "error", "error": f"Original image file not found: {safe_filename}"}), 404
            
        image = cv2.imread(full_image_path)
        if image is None:
            return jsonify({"status": "error", "error": "Failed to read original image file"}), 500
            
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if model_manager.resnet_model is None:
            return jsonify({"status": "error", "error": "Classification model is not loaded"}), 500
            
        input_tensor = preprocess_image_for_resnet(image_rgb)
        
        from services.gradcam import generate_gradcam_explanation
        import hashlib
        image_hash = hashlib.sha256(image_rgb.tobytes()).hexdigest()
        
        gradcam_result = generate_gradcam_explanation(
            model=model_manager.resnet_model,
            input_tensor=input_tensor,
            image_rgb=image_rgb,
            target_class_idx=target_class_idx,
            filename_prefix=f"{image_hash[:16]}_{target_class_idx}",
        )
        
        if (
            gradcam_result.available
            and gradcam_result.overlay_image is not None
            and gradcam_result.heatmap_image is not None
        ):
            overlay_b64 = encode_image_for_display(gradcam_result.overlay_image)
            heatmap_only_b64 = encode_image_for_display(gradcam_result.heatmap_image)
            
            return jsonify({
                "status": "success",
                "overlay_b64": overlay_b64,
                "heatmap_only_b64": heatmap_only_b64,
                "target_class": disease_classes[target_class_idx],
                "target_layer": gradcam_result.target_layer
            })
        else:
            return jsonify({"status": "error", "error": gradcam_result.error or "Failed to generate Grad-CAM explanation"}), 500
            
    except Exception as exc:
        logger.error("Dynamic target Grad-CAM generation failed: %s", exc)
        return jsonify({"status": "error", "error": str(exc)}), 500


@app.route("/comparison", methods=["GET", "POST"])
@login_required
def comparison():
    error_message = None
    old_filename, new_filename, old_image, new_image = None, None, None, None

    if request.method == "POST":
        required_files = {"last_week_image": "Last Week Field Image", "current_week_image": "Current Week Field Image"}
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
            old_filename, old_image, old_rgb = read_uploaded_image(request.files["last_week_image"])
            new_filename, new_image, new_rgb = read_uploaded_image(request.files["current_week_image"])

            old_results = analyze_image(old_rgb)
            new_results = analyze_image(new_rgb)

            _, yolo_model = model_manager.load_models()
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
    try:
        example_disease_probs = [0.08, 0.02, 0.01, 0.10, 0.04, 0.65, 0.05, 0.05]
        demo_disease = {
            "predicted_class": "Healthy",
            "predicted_class_idx": 5,
            "confidence": example_disease_probs[5],
            "model_confidence": round(example_disease_probs[5] * 100, 2),
            "detected_issue": "Healthy",
            "all_confidences": {disease_classes[i]: example_disease_probs[i] for i in range(len(disease_classes))},
            "health_score": 65.0,
            "raw": [example_disease_probs],
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
        
        # Generate high-quality synthetic cotton BGR image representing field crop
        synthetic_bgr = np.zeros((384, 512, 3), dtype=np.uint8)
        
        # Fill background with a rich soft earthy background
        synthetic_bgr[:, :] = [30, 40, 45]
        
        # Draw deep-green leaf foliage (multiple overlapping green circles)
        cv2.circle(synthetic_bgr, (200, 220), 120, (34, 139, 34), -1) # Forest Green
        cv2.circle(synthetic_bgr, (320, 260), 100, (46, 139, 87), -1) # Sea Green
        cv2.circle(synthetic_bgr, (120, 280), 90, (34, 120, 34), -1) # Darker Green
        
        # Draw organic branch structure
        cv2.line(synthetic_bgr, (256, 384), (256, 200), (42, 75, 124), 12)
        cv2.line(synthetic_bgr, (256, 260), (140, 180), (42, 75, 124), 8)
        cv2.line(synthetic_bgr, (256, 220), (380, 150), (42, 75, 124), 8)
        
        # Draw localized crop anomalies (reddish-brown leaf spots / target spot disease representation)
        cv2.circle(synthetic_bgr, (220, 200), 15, (40, 50, 139), -1)
        cv2.circle(synthetic_bgr, (215, 195), 5, (20, 30, 80), -1)
        cv2.circle(synthetic_bgr, (180, 240), 10, (40, 50, 139), -1)
        
        # Draw Matured Cotton Boll within [120, 80, 210, 155] (center is (165, 117.5))
        cv2.ellipse(synthetic_bgr, (165, 117), (40, 30), 0, 0, 360, (50, 180, 100), -1)
        cv2.ellipse(synthetic_bgr, (165, 117), (40, 30), 0, 0, 360, (40, 140, 80), 2)
        cv2.line(synthetic_bgr, (165, 87), (165, 75), (42, 75, 124), 4)
    
        # Draw Split Cotton Boll within [300, 120, 390, 210] (center is (345, 165))
        cv2.circle(synthetic_bgr, (330, 165), 20, (245, 245, 245), -1)
        cv2.circle(synthetic_bgr, (360, 165), 20, (245, 245, 245), -1)
        cv2.circle(synthetic_bgr, (345, 150), 20, (255, 255, 255), -1)
        cv2.circle(synthetic_bgr, (345, 180), 20, (230, 230, 230), -1)
        cv2.ellipse(synthetic_bgr, (345, 185), (35, 15), 0, 0, 360, (30, 50, 90), -1)
        
        # Convert from BGR to RGB
        synthetic_rgb = cv2.cvtColor(synthetic_bgr, cv2.COLOR_BGR2RGB)

        # Write synthetic image as a demo file for target explainability to locate
        os.makedirs("static/uploads", exist_ok=True)
        cv2.imwrite(os.path.join("static", "uploads", "demo_cotton.jpg"), synthetic_bgr)

        # Generate mock heatmap
        from services.gradcam import generate_pure_heatmap, apply_heatmap_on_image
        mock_heatmap = generate_mock_heatmap(synthetic_rgb)
        pure_heatmap_rgb = generate_pure_heatmap(synthetic_rgb, mock_heatmap)
        mock_overlay = apply_heatmap_on_image(synthetic_rgb, mock_heatmap)
        
        # Base64 encode both original synthetic image and XAI overlay
        image_b64 = encode_image_for_display(synthetic_rgb)
        grad_cam_image_b64 = encode_image_for_display(mock_overlay)
        heatmap_only_b64 = encode_image_for_display(pure_heatmap_rgb)
        
        # Set top-level and nested properties for robustness
        demo_disease["heatmap_b64"] = grad_cam_image_b64
        
        # Calculate Severity
        severity = calculate_disease_severity(demo_disease["health_score"])
        
        # Use estimate_yield from service
        from services.yield_service import estimate_yield
        yield_est = estimate_yield(demo_disease, demo_growth, weather=None, field_acres=1.0)
        
        # Generate advanced recommendations
        adv_recs = generate_advanced_recommendations(demo_disease, demo_growth)
        
        # Generate farmer insights
        insights = generate_farmer_insights(demo_disease, demo_growth)

        from services.recommendation_engine import get_recommendations
        demo_treatment_recs = get_recommendations(
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
            "disease_severity": severity,
            "yield_estimate": yield_est,
            "advanced_recommendations": adv_recs,
            "farmer_insights": insights,
            "treatment_recommendations": demo_treatment_recs
        }
        return render_template(
            "results.html",
            results=example_json,
            filename="demo_cotton.jpg",
            unique_filename="demo_cotton.jpg",
            image_b64=image_b64,
            img_shape={"width": 512, "height": 384},
            raw_json=json.dumps(example_json, indent=2),
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            grad_cam_image_b64=grad_cam_image_b64,
            heatmap_only_b64=heatmap_only_b64,
            heatmap_image_path=None,
            heatmap_only_path=None,
            yield_estimate=yield_est,
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
            "Hi! Need any help analyzing your farm data?"
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
            "Glad I could help! Let me know if you have more questions about your cotton crop."
        ],
        r"\b(help|assist|support|guide|advice|tips?)\b": [
            "I'm here to help! You can ask me about crop diseases, yield optimization, pest control, irrigation, fertilization, weather impacts, or soil health.",
            "Sure! Try asking about cotton diseases, pest control, yield estimates, or upload an image in the Analyze tab for an instant AI diagnosis."
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
@limiter.limit(lambda: app.config.get("API_UPLOAD_RATE_LIMIT", "20 per minute"))
def api_analyze():
    temp_path = None
    try:
        enforce_request_size(get_upload_max_bytes())

        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        _safe_filename, image, image_rgb, temp_path = read_validated_upload_image(file)
        field_acres,field_acres_error=parse_api_field_acres(request.form.get("field_acres"))
        if field_acres_error:
            return jsonify({"error":field_acres_error}),400
            
        lat=request.form.get("lat",type=float)
        lon=request.form.get("lon",type=float)
        city=request.form.get("city",type=str)
        weather=resolve_weather_for_analysis(lat=lat,lon=lon,city=city)
        compressed_rgb = resize_image(image_rgb, MAX_INFERENCE_DIMENSION)
        results = analyze_image(compressed_rgb, weather=weather, field_acres=field_acres)
        return jsonify({
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "weather": weather,
            "results": results,
            
        })
    except UploadValidationError as exc:
        filename = request.files.get("file", {}).filename if request.files.get("file") else "unknown"
        logger.warning("API upload rejected (file=%s): %s", filename, exc)
        return jsonify({"error": str(exc)}), exc.status_code
    except Exception as e:
        filename = request.files.get("file", {}).filename if request.files.get("file") else "unknown"
        logger.error("API analysis error (file=%s): %s", filename, e)
        return jsonify({"error": str(e)}), 500
    finally:
        cleanup_temp_upload(temp_path)


@app.route("/api/analyze_stream", methods=["POST"])
def api_analyze_stream():
    """Streaming endpoint for real-time analysis progress"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    image_bytes = file.read()
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    def generate():
        try:
            yield f"data: {json.dumps({'step': 'upload_received', 'progress': 25, 'message': 'Uploading image...'})}\n\n"

            image = cv2.imdecode(
                np.frombuffer(image_bytes, np.uint8),
                cv2.IMREAD_COLOR
            )

            if image is None:
                yield f"data: {json.dumps({'step': 'error', 'progress': 0, 'message': 'Invalid image file'})}\n\n"
                return

            yield f"data: {json.dumps({'step': 'preprocessing', 'progress': 50, 'message': 'Analyzing crop health...'})}\n\n"

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = analyze_image(image_rgb)

            yield f"data: {json.dumps({'step': 'recommendations', 'progress': 75, 'message': 'Generating prediction...'})}\n\n"

            yield f"data: {json.dumps({'step': 'complete', 'progress': 100, 'message': 'Analysis complete', 'data': results})}\n\n"

        except Exception as e:
            logger.error(f"Streaming analysis error: {e}")
            yield f"data: {json.dumps({'step': 'error', 'progress': 0, 'message': str(e)})}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )


# --- Batch Processing Endpoints ---

@app.route("/api/batch_upload", methods=["POST"])
def api_batch_upload():
    """Upload multiple images for batch analysis"""
    try:
        if 'files' not in request.files:
            return jsonify({'error': 'No files uploaded'}), 400
        
        files = request.files.getlist('files')
        if not files or files[0].filename == '':
            return jsonify({'error': 'No files selected'}), 400
        
        # Validate files
        valid_files = []
        for file in files:
            if file and allowed_file(file.filename):
                valid_files.append(file)
        
        if not valid_files:
            return jsonify({'error': 'No valid image files'}), 400
        
        # Create batch job
        from models import BatchJob, db
        job = BatchJob(
            total_images=len(valid_files),
            status='pending'
        )
        db.session.add(job)
        db.session.commit()
        
        # Prepare image data for Celery
        images_data = []
        for file in valid_files:
            file_data = file.read()
            images_data.append((file.filename, file_data))
        
        # Start batch processing (try to import Celery)
        try:
            from celery_tasks import process_batch_job
            process_batch_job.delay(job.id, images_data)
            celery_enabled = True
        except Exception as e:
            logger.error(f"Celery not available: {e}")
            celery_enabled = False
            # Process synchronously if Celery is not available
            job.status = 'processing'
            db.session.commit()
            
            # Process images one by one
            import cv2
            import numpy as np
            for idx, (filename, image_data) in enumerate(images_data):
                try:
                    file.seek(0)
                    file_bytes = file.read()
                    image = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)

                    file_bytes = np.frombuffer(image_data, np.uint8)
                    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                    if image is not None:
                        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        results = analyze_image(image_rgb)
                        
                        # Save result
                        from models import AnalysisResult
                        result = AnalysisResult(
                            batch_job_id=job.id,
                            image_name=filename,
                            image_index=idx,
                            status='complete',
                            disease_class=results.get('disease', {}).get('predicted_class'),
                            disease_confidence=results.get('disease', {}).get('confidence'),
                            health_score=results.get('disease', {}).get('health_score'),
                            growth_class=results.get('growth', {}).get('main_class'),
                            growth_confidence=results.get('growth', {}).get('confidence'),
                            results_json=results
                        )
                        db.session.add(result)
                except Exception as e:
                    logger.error(f"Error processing image {filename}: {e}")
                    from models import AnalysisResult
                    result = AnalysisResult(
                        batch_job_id=job.id,
                        image_name=filename,
                        image_index=idx,
                        status='error',
                        error_message=str(e)
                    )
                    db.session.add(result)
            
            job.status = 'completed'
            job.completed_at = datetime.utcnow()
            db.session.commit()
        
        return jsonify({
            'status': 'success',
            'job_id': job.id,
            'total_images': len(valid_files),
            'celery_enabled': celery_enabled,
            'message': f'Batch job {job.id} started with {len(valid_files)} images'
        })
        
    except Exception as e:
        logger.error(f"Batch upload error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route("/api/batch_status/<job_id>", methods=["GET"])
def api_batch_status(job_id):
    """Get status of a batch job"""
    from models import BatchJob, db
    
    job = BatchJob.query.get(job_id)
    if not job:
        return jsonify({'error': 'Batch job not found'}), 404
    
    # Update completed count from results
    job.completed_images = len([r for r in job.results if r.status == 'complete'])
    job.failed_images = len([r for r in job.results if r.status == 'error'])
    
    # Check if all tasks are done
    if job.completed_images + job.failed_images >= job.total_images:
        job.status = 'completed'
        job.completed_at = datetime.utcnow()
        db.session.commit()
    
    return jsonify(job.to_dict())


@app.route("/api/batch_results/<job_id>", methods=["GET"])
def api_batch_results(job_id):
    """Get all results for a batch job"""
    from models import BatchJob, db
    
    job = BatchJob.query.get(job_id)
    if not job:
        return jsonify({'error': 'Batch job not found'}), 404
    
    results = [r.to_dict() for r in job.results]
    results.sort(key=lambda x: x['image_index'])
    
    return jsonify({
        'job_id': job.id,
        'status': job.status,
        'total_images': job.total_images,
        'completed_images': job.completed_images,
        'failed_images': job.failed_images,
        'results': results
    })

@app.route("/api/batch_status/<job_id>/stream", methods=["GET"])
@login_required
def api_batch_status_stream(job_id):
    """Stream real-time status and results for a batch job using Server-Sent Events (SSE)"""
    from models import BatchJob, db
    import json
    import time
    
    # Check if job exists
    job = BatchJob.query.get(job_id)
    if not job:
        return jsonify({"error": "Batch job not found"}), 404
        
    def event_generator():
        while True:
            payload = None
            is_finished = False
            
            # We must run database queries within app_context
            with app.app_context():
                # Refresh job from DB
                db.session.expire_all()
                current_job = BatchJob.query.get(job_id)
                if not current_job:
                    payload = {"error": "Job deleted"}
                    is_finished = True
                else:
                    # Update completed/failed counts
                    results = current_job.results
                    completed = len([r for r in results if r.status in ("complete", "success")])
                    failed = len([r for r in results if r.status == "error"])
                    
                    # Check completion status
                    if completed + failed >= current_job.total_images and current_job.total_images > 0:
                        current_job.status = "completed"
                        if not current_job.completed_at:
                            current_job.completed_at = datetime.utcnow()
                        db.session.commit()
                    
                    job_dict = current_job.to_dict()
                    # Include total, completed, failed counts in job_dict
                    job_dict["completed_images"] = completed
                    job_dict["failed_images"] = failed
                    
                    # Also include per-image results that are already processed or processing so far
                    results_list = []
                    for r in results:
                        results_list.append({
                            "id": r.id,
                            "image_name": r.image_name,
                            "image_index": r.image_index,
                            "status": r.status,
                            "disease_class": r.disease_class,
                            "disease_confidence": r.disease_confidence,
                            "health_score": r.health_score,
                            "growth_class": r.growth_class,
                            "growth_confidence": r.growth_confidence,
                            "error_message": r.error_message,
                            "results": r.results_json
                        })
                    
                    results_list.sort(key=lambda x: x["image_index"])
                    
                    payload = {
                        "job": job_dict,
                        "results": results_list
                    }
                    
                    if current_job.status in ("completed", "failed"):
                        is_finished = True
            
            # Yield outside the app context block to prevent AssertionError on GeneratorExit teardown
            if "error" in payload:
                yield "event: error\ndata: " + json.dumps(payload) + "\n\n"
            else:
                yield f"data: {json.dumps(payload)}\n\n"
                
            if is_finished:
                break
                
            time.sleep(1.0)
            
    return Response(stream_with_context(event_generator()), mimetype="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no"
    })



@app.route("/api/batch_results/<job_id>/export/csv", methods=["GET"])
def export_batch_csv(job_id):
    """Export batch results as CSV"""
    from models import BatchJob
    import csv
    from io import StringIO
    
    job = BatchJob.query.get(job_id)
    if not job:
        return jsonify({'error': 'Batch job not found'}), 404

    si = StringIO()
    cw = csv.writer(si)
    cw.writerow(['Image Name', 'Status', 'Disease', 'Confidence', 'Health Score', 'Growth Stage'])
    
    # Sort results by image index
    results = sorted(job.results, key=lambda x: x.image_index)
    
    for r in results:
        results_data = r.results_json or {}
        disease = results_data.get('disease', {})
        growth = results_data.get('growth', {})
        
        disease_class = disease.get('predicted_class', 'N/A')
        confidence = f"{disease.get('confidence', 0):.3f}" if disease.get('confidence') is not None else 'N/A'
        health_score = f"{disease.get('health_score', 0):.1f}" if disease.get('health_score') is not None else 'N/A'
        growth_class = growth.get('main_class', 'N/A')
        
        cw.writerow([
            r.image_name,
            r.status,
            disease_class,
            confidence,
            health_score,
            growth_class
        ])
        
    output = si.getvalue()
    si.close()
    
    return Response(
        output,
        mimetype="text/csv",
        headers={"Content-disposition": f"attachment; filename=batch_results_{job_id}.csv"}
    )


@app.route("/api/batch_results/<job_id>/export/pdf", methods=["GET"])
def export_batch_pdf(job_id):
    """Export batch results as PDF"""
    from models import BatchJob
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter, landscape
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet
        from io import BytesIO
    except ImportError:
        return jsonify({"error": "reportlab not installed. Install with: pip install reportlab"}), 500

    job = BatchJob.query.get(job_id)
    if not job:
        return jsonify({'error': 'Batch job not found'}), 404

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=landscape(letter))
    elements = []
    styles = getSampleStyleSheet()

    # Title
    title = Paragraph(f"Batch Analysis Report (Job ID: {job_id})", styles['Title'])
    elements.append(title)
    elements.append(Spacer(1, 12))

    # Summary
    summary_text = f"Total Images: {job.total_images} | Completed: {job.completed_images} | Failed: {job.failed_images}"
    summary = Paragraph(summary_text, styles['Normal'])
    elements.append(summary)
    elements.append(Spacer(1, 12))

    # Table data
    table_data = [['Image Name', 'Status', 'Disease', 'Confidence', 'Health Score', 'Growth Stage']]
    
    results = sorted(job.results, key=lambda x: x.image_index)
    
    for r in results:
        results_data = r.results_json or {}
        disease = results_data.get('disease', {})
        growth = results_data.get('growth', {})
        
        disease_class = disease.get('predicted_class', 'N/A')
        confidence = f"{disease.get('confidence', 0)*100:.1f}%" if disease.get('confidence') is not None else 'N/A'
        health_score = f"{disease.get('health_score', 0):.1f}%" if disease.get('health_score') is not None else 'N/A'
        growth_class = growth.get('main_class', 'N/A')
        
        table_data.append([
            r.image_name,
            r.status.upper(),
            disease_class,
            confidence,
            health_score,
            growth_class
        ])

    # Create table
    table = Table(table_data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('WORDWRAP', (0, 0), (-1, -1), 'CJK'),
    ]))
    
    elements.append(table)
    doc.build(elements)
    
    buffer.seek(0)
    return send_file(
        buffer,
        as_attachment=True,
        download_name=f'batch_results_{job_id}.pdf',
        mimetype='application/pdf'
    )


@app.route("/batch", methods=["GET", "POST"])
@login_required
def batch_upload_page():
    """Batch upload page"""
    if request.method == 'POST':
        return redirect(url_for('batch_results_page', job_id=request.form.get('job_id')))
    return render_template('batch_upload.html')


@app.route("/batch/results/<job_id>")
@login_required
def batch_results_page(job_id):
    """Batch results page"""
    return render_template('batch_results.html', job_id=job_id)


# --- Authentication Routes ---

@app.route("/login", methods=["GET", "POST"])
def login():
    """Login page"""
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        email = (request.form.get('email') or '').strip().lower()
        password = request.form.get('password') or ''
        remember = request.form.get('remember')
        ip_address = get_client_ip()
        user_agent = get_user_agent()
        lockout_service = AccountLockoutService()
        
        from models import User
        user = User.query.filter_by(email=email).first()

        if user:
            lockout_state = lockout_service.check_lockout(user)
            if lockout_state.unlocked_expired_lock:
                lockout_service.record_unlock(
                    user,
                    ip=ip_address,
                    user_agent=user_agent,
                )
                db.session.commit()
            if lockout_state.locked:
                flash('Account temporarily locked. Please try again later.', 'danger')
                return render_template(
                    'login.html',
                    google_oauth_enabled=GOOGLE_OAUTH_ENABLED,
                ), 423
        
        if user and user.check_password(password):
            if not user.is_active:
                flash('Your account has been deactivated. Please contact support.', 'danger')
                return render_template('login.html', google_oauth_enabled=GOOGLE_OAUTH_ENABLED)
            
            login_user(user, remember=remember)
            lockout_service.record_successful_login(
                user,
                ip=ip_address,
                user_agent=user_agent,
            )
            user.last_login = user.last_successful_login_at
            db.session.commit()
            
            next_page = request.args.get('next')
            return redirect(next_page) if next_page else redirect(url_for('index'))
        else:
            if user:
                lockout_service.record_failed_login(
                    user,
                    ip=ip_address,
                    user_agent=user_agent,
                )
                db.session.commit()
            flash('Invalid email or password', 'danger')
    
    return render_template('login.html', google_oauth_enabled=GOOGLE_OAUTH_ENABLED)


@app.route("/auth/google")
def auth_google():
    """Redirect to Google's OAuth 2.0 consent screen."""
    if not GOOGLE_OAUTH_ENABLED:
        flash("Google Sign-In is not configured on this server.", "warning")
        return redirect(url_for("login"))
    if current_user.is_authenticated:
        return redirect(url_for("index"))
    redirect_uri = url_for("auth_google_callback", _external=True)
    return _oauth.google.authorize_redirect(redirect_uri)


@app.route("/auth/google/callback")
def auth_google_callback():
    """Handle the OAuth 2.0 callback from Google."""
    if not GOOGLE_OAUTH_ENABLED:
        flash("Google Sign-In is not configured on this server.", "warning")
        return redirect(url_for("login"))

    try:
        token = _oauth.google.authorize_access_token()
        user_info = token.get("userinfo") or _oauth.google.userinfo()
    except Exception as exc:
        logger.warning("Google OAuth callback error: %s", exc)
        flash("Google Sign-In failed. Please try again.", "danger")
        return redirect(url_for("login"))

    google_id = str(user_info["sub"])
    email = user_info.get("email", "")
    full_name = user_info.get("name", email.split("@")[0])
    picture = user_info.get("picture", "")

    from models import User

    # 1. Look up by OAuth provider + ID (most reliable)
    user = User.query.filter_by(oauth_provider="google", oauth_id=google_id).first()

    # 2. Fall back to matching by email (links existing password accounts)
    if user is None and email:
        user = User.query.filter_by(email=email).first()
        if user:
            user.oauth_provider = "google"
            user.oauth_id = google_id
            if picture:
                user.profile_picture = picture

    # 3. Auto-create a new account for first-time Google users
    if user is None:
        user = User(
            email=email,
            full_name=full_name,
            password_hash=None,
            oauth_provider="google",
            oauth_id=google_id,
            profile_picture=picture,
            role="farmer",
            is_active=True,
        )
        db.session.add(user)

    user.last_login = datetime.utcnow()
    db.session.commit()

    login_user(user)
    logger.info("User %s signed in via Google OAuth.", user.email)
    flash(f"Welcome, {user.full_name}! You are now signed in.", "success")
    return redirect(url_for("index"))


@app.route("/register", methods=["GET", "POST"])
def register():
    """Registration page"""
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        full_name = request.form.get('full_name')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        role = request.form.get('role', 'farmer')
        
        # Validation
        if not full_name or not email or not password:
            flash('All fields are required', 'danger')
            return render_template('register.html', google_oauth_enabled=GOOGLE_OAUTH_ENABLED)
        
        if password != confirm_password:
            flash('Passwords do not match', 'danger')
            return render_template('register.html', google_oauth_enabled=GOOGLE_OAUTH_ENABLED)
        
        if len(password) < 8:
            flash('Password must be at least 8 characters', 'danger')
            return render_template('register.html', google_oauth_enabled=GOOGLE_OAUTH_ENABLED)
        
        from models import User
        if User.query.filter_by(email=email).first():
            flash('Email already registered', 'danger')
            return render_template('register.html', google_oauth_enabled=GOOGLE_OAUTH_ENABLED)
        
        # Create user
        user = User(
            email=email,
            full_name=full_name,
            role=role
        )
        user.set_password(password)
        
        db.session.add(user)
        db.session.commit()
        
        flash('Account created successfully! Please login.', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html', google_oauth_enabled=GOOGLE_OAUTH_ENABLED)


@app.route("/logout")
@login_required
def logout():
    """Logout user"""
    logout_user()
    flash('You have been logged out', 'info')
    return redirect(url_for('login'))


@app.route("/profile")
@login_required
def profile():
    """User profile page"""
    return render_template('profile.html')


@app.route("/forgot_password", methods=["GET", "POST"])
def forgot_password():
    """Forgot password page"""
    if request.method == 'POST':
        email = request.form.get('email')
        flash('Password reset link sent to your email (demo feature)', 'info')
        return redirect(url_for('login'))
    return render_template('login.html')  # Reuse login template for now


# --- Geographic Disease Mapping ---

@app.route("/disease-map")
@login_required
def disease_map():
    """Disease map page"""
    return render_template('disease_map.html')


@app.route("/api/disease-map")
@login_required
def api_disease_map():
    """API endpoint for disease map data"""
    from models import AnalysisHistory
    from datetime import datetime, timedelta
    
    # Get filter parameters
    disease_filter = request.args.get('disease', 'all')
    time_filter = request.args.get('time', 'all')
    confidence_filter = float(request.args.get('confidence', 0))
    
    # Build query - get all analyses first, then filter for location
    if current_user.is_researcher():
        query = AnalysisHistory.query
    else:
        query = AnalysisHistory.query.filter_by(user_id=current_user.id)
    
    # Apply time filter
    if time_filter == 'today':
        query = query.filter(AnalysisHistory.created_at >= datetime.utcnow().replace(hour=0, minute=0, second=0))
    elif time_filter == 'week':
        query = query.filter(AnalysisHistory.created_at >= datetime.utcnow() - timedelta(days=7))
    elif time_filter == 'month':
        query = query.filter(AnalysisHistory.created_at >= datetime.utcnow() - timedelta(days=30))
    elif time_filter == 'year':
        query = query.filter(AnalysisHistory.created_at >= datetime.utcnow() - timedelta(days=365))
    
    # Get analyses
    analyses = query.all()
    
    # Filter for location data in Python (more flexible)
    filtered_analyses = []
    for a in analyses:
        # Apply disease filter
        if disease_filter != 'all':
            if not a.disease_result or a.disease_result.get('predicted_class') != disease_filter:
                continue
        
        # Apply confidence filter
        if confidence_filter > 0:
            if not a.confidence or a.confidence < confidence_filter / 100:
                continue
        
        # Only include analyses with location data
        if a.latitude and a.longitude:
            filtered_analyses.append(a)
    
    # Calculate statistics
    total_analyses = len(filtered_analyses)
    healthy_count = sum(1 for a in filtered_analyses if a.disease_result and a.disease_result.get('predicted_class') == 'healthy')
    diseased_count = total_analyses - healthy_count
    avg_health_score = sum(a.health_score for a in filtered_analyses if a.health_score) / len([a for a in filtered_analyses if a.health_score]) if filtered_analyses else 0
    regions = set(a.region for a in filtered_analyses if a.region)
    
    return jsonify({
        'analyses': [a.to_dict() for a in filtered_analyses],
        'stats': {
            'total_analyses': total_analyses,
            'healthy_count': healthy_count,
            'diseased_count': diseased_count,
            'avg_health_score': avg_health_score,
            'regions_count': len(regions)
        }
    })


# --- Advanced Dashboard ---

@app.route("/dashboard")
@login_required
def dashboard():
    """Advanced dashboard page"""
    return render_template('dashboard.html')


@app.route("/api/dashboard-stats")
@login_required
def api_dashboard_stats():
    """API endpoint for dashboard statistics"""
    from models import AnalysisHistory
    from datetime import datetime, timedelta
    from collections import defaultdict
    
    # Get all analyses for current user
    if current_user.is_researcher():
        analyses = AnalysisHistory.query.all()
    else:
        analyses = AnalysisHistory.query.filter_by(user_id=current_user.id).all()
    
    # Calculate basic statistics
    total_analyses = len(analyses)
    healthy_count = sum(1 for a in analyses if a.disease_result and a.disease_result.get('predicted_class') == 'healthy')
    diseased_count = total_analyses - healthy_count
    avg_health_score = sum(a.health_score for a in analyses if a.health_score) / len([a for a in analyses if a.health_score]) if analyses else 0
    
    # Disease distribution
    disease_counts = defaultdict(int)
    for a in analyses:
        if a.disease_result:
            disease = a.disease_result.get('predicted_class', 'unknown')
            disease_counts[disease] += 1
    
    disease_distribution = {
        'labels': [d.replace('_', ' ').title() for d in disease_counts.keys()],
        'values': list(disease_counts.values())
    }
    
    # Disease trends (last 7 days)
    trend_labels = []
    trend_data = defaultdict(list)
    for i in range(7):
        date = datetime.utcnow() - timedelta(days=6-i)
        date_str = date.strftime('%Y-%m-%d')
        trend_labels.append(date.strftime('%b %d'))
        
        day_analyses = [a for a in analyses if a.created_at.date() == date.date()]
        for a in day_analyses:
            if a.disease_result:
                disease = a.disease_result.get('predicted_class', 'unknown')
                trend_data[disease].append(1)
    
    # Create trend datasets
    trend_datasets = []
    colors = ['#22c55e', '#ef4444', '#f59e0b', '#8b5cf6', '#ec4899', '#06b6d4']
    for idx, (disease, counts) in enumerate(trend_data.items()):
        # Aggregate by day
        daily_counts = []
        for i in range(7):
            date = datetime.utcnow() - timedelta(days=6-i)
            day_analyses = [a for a in analyses if a.created_at.date() == date.date()]
            count = sum(1 for a in day_analyses if a.disease_result and a.disease_result.get('predicted_class') == disease)
            daily_counts.append(count)
        
        trend_datasets.append({
            'label': disease.replace('_', ' ').title(),
            'data': daily_counts,
            'borderColor': colors[idx % len(colors)],
            'backgroundColor': colors[idx % len(colors)] + '20',
            'fill': False,
            'tension': 0.4
        })
    
    disease_trends = {
        'labels': trend_labels,
        'datasets': trend_datasets
    }
    
    # Growth stage distribution
    growth_counts = defaultdict(int)
    for a in analyses:
        if a.growth_result:
            stage = a.growth_result.get('main_class', 'unknown')
            growth_counts[stage] += 1
    
    growth_distribution = {
        'labels': [g.replace('_', ' ').title() for g in growth_counts.keys()],
        'values': list(growth_counts.values())
    }
    
    # Regional data
    region_counts = defaultdict(int)
    for a in analyses:
        if a.region:
            region_counts[a.region] += 1
    
    regional_data = {
        'labels': list(region_counts.keys()),
        'values': list(region_counts.values())
    }
    
    # Recent activity
    recent_analyses = sorted(analyses, key=lambda x: x.created_at, reverse=True)[:10]
    recent_activity = []
    for a in recent_analyses:
        disease = a.disease_result.get('predicted_class', 'unknown') if a.disease_result else 'unknown'
        activity_type = 'disease' if disease != 'healthy' else 'healthy'
        icon = 'exclamation-triangle' if disease != 'healthy' else 'check-circle'
        
        recent_activity.append({
            'type': activity_type,
            'icon': icon,
            'title': f'{disease.replace("_", " ").title()} Detected',
            'description': f'Confidence: {(a.confidence * 100):.1f}%' if a.confidence else 'No confidence data',
            'time': a.created_at.strftime('%b %d, %Y %H:%M')
        })
    
    return jsonify({
        'stats': {
            'total_analyses': total_analyses,
            'healthy_count': healthy_count,
            'diseased_count': diseased_count,
            'avg_health_score': avg_health_score
        },
        'disease_distribution': disease_distribution,
        'disease_trends': disease_trends,
        'growth_distribution': growth_distribution,
        'regional_data': regional_data,
        'recent_activity': recent_activity
    })


# --- Automated Reporting ---

@app.route("/reports")
@login_required
def reports():
    """Reports page"""
    return render_template('reports.html')


@app.route("/api/analyses")
@login_required
def api_analyses():
    """API endpoint to get list of analyses for report generation"""
    from models import AnalysisHistory
    
    # Get analyses for current user
    if current_user.is_researcher():
        analyses = AnalysisHistory.query.order_by(AnalysisHistory.created_at.desc()).limit(50).all()
    else:
        analyses = AnalysisHistory.query.filter_by(user_id=current_user.id).order_by(AnalysisHistory.created_at.desc()).limit(50).all()
    
    analyses_list = []
    for a in analyses:
        disease = a.disease_result.get('predicted_class', 'unknown') if a.disease_result else 'unknown'
        analyses_list.append({
            'id': a.id,
            'disease': disease.replace('_', ' ').title(),
            'date': a.created_at.strftime('%Y-%m-%d %H:%M'),
            'health_score': a.health_score
        })
    
    return jsonify({'analyses': analyses_list})


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
        return jsonify({'error': f'Report service not available: {str(e)}'}), 500
    
    analysis = AnalysisHistory.query.get(analysis_id)
    if not analysis:
        return jsonify({'error': 'Analysis not found'}), 404
    
    # Check permission
    if not current_user.is_researcher() and analysis.user_id != current_user.id:
        return jsonify({'error': 'Access denied'}), 403
    
    try:
        generator = ReportGenerator()
        report_data = {
            'disease_result': analysis.disease_result,
            'growth_result': analysis.growth_result,
            'health_score': analysis.health_score,
            'confidence': analysis.confidence
        }
        
        user_info = {
            'full_name': current_user.full_name,
            'email': current_user.email,
            'role': current_user.role
        }
        
        pdf_bytes = generator.generate_analysis_report(report_data, user_info)
        
        return send_file(
            BytesIO(pdf_bytes),
            mimetype='application/pdf',
            as_attachment=True,
            download_name=f'analysis_report_{analysis_id}.pdf'
        )
    except Exception as e:
        logger.error(f"Error generating report: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


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
        return jsonify({'error': f'Report service not available: {str(e)}'}), 500
    
    # Get date range
    days = request.args.get('days', 30, type=int)
    start_date = datetime.utcnow() - timedelta(days=days)
    
    # Get analyses
    if current_user.is_researcher():
        analyses = AnalysisHistory.query.filter(AnalysisHistory.created_at >= start_date).all()
    else:
        analyses = AnalysisHistory.query.filter(
            AnalysisHistory.user_id == current_user.id,
            AnalysisHistory.created_at >= start_date
        ).all()
    
    try:
        generator = ReportGenerator()
        analyses_data = [a.to_dict() for a in analyses]
        
        user_info = {
            'full_name': current_user.full_name,
            'email': current_user.email,
            'role': current_user.role
        }
        
        date_range = f"Last {days} days"
        pdf_bytes = generator.generate_summary_report(analyses_data, user_info, date_range)
        
        return send_file(
            BytesIO(pdf_bytes),
            mimetype='application/pdf',
            as_attachment=True,
            download_name=f'summary_report_{datetime.now().strftime("%Y%m%d")}.pdf'
        )
    except Exception as e:
        logger.error(f"Error generating summary report: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


# --- Disease Database & Symptom Checker ---

@app.route("/disease-database")
@login_required
def disease_database():
    """Disease database page"""
    return render_template('disease_database.html')


@app.route("/symptom-checker")
@login_required
def symptom_checker():
    """Symptom checker page"""
    return render_template('symptom_checker.html')


@app.route("/api/diseases")
def api_diseases():
    """API endpoint to get list of diseases"""
    from models import Disease
    
    search = request.args.get('search', '')
    severity = request.args.get('severity', '')
    affected_part = request.args.get('affected_part', '')
    
    query = Disease.query
    
    if search:
        query = query.filter(Disease.name.ilike(f'%{search}%'))
    
    if severity:
        query = query.filter(Disease.severity == severity)
    
    if affected_part:
        query = query.filter(Disease.affected_parts.ilike(f'%{affected_part}%'))
    
    diseases = query.order_by(Disease.name).all()
    
    return jsonify({
        'diseases': [d.to_dict() for d in diseases],
        'count': len(diseases)
    })


@app.route("/api/diseases/<int:disease_id>")
def api_disease_detail(disease_id):
    """API endpoint to get disease details"""
    from models import Disease
    
    disease = Disease.query.get(disease_id)
    if not disease:
        return jsonify({'error': 'Disease not found'}), 404
    
    return jsonify(disease.to_dict())


@app.route("/api/symptoms")
def api_symptoms():
    """API endpoint to get list of symptoms"""
    from models import Symptom
    
    category = request.args.get('category', '')
    
    query = Symptom.query
    
    if category:
        query = query.filter(Symptom.category == category)
    
    symptoms = query.order_by(Symptom.category, Symptom.name).all()
    
    return jsonify({
        'symptoms': [s.to_dict() for s in symptoms],
        'count': len(symptoms)
    })


@app.route("/api/symptom-check", methods=['POST'])
def api_symptom_check():
    """API endpoint to check symptoms and suggest diseases"""
    from models import Symptom, Disease, DiseaseSymptom
    
    data = request.get_json()
    symptom_ids = data.get('symptom_ids', [])
    
    if not symptom_ids:
        return jsonify({'error': 'No symptoms provided'}), 400
    
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
            results.append({
                'disease': disease.to_dict(),
                'match_score': round(score * 100, 1)
            })
    
    return jsonify({
        'results': results,
        'symptom_count': len(symptom_ids)
    })


# --- Disease Forecast & Weather Prediction ---

@app.route("/disease-forecast")
@login_required
def disease_forecast():
    """Disease forecast page"""
    return render_template('disease_forecast.html')


@app.route("/api/weather-forecast")
def api_weather_forecast():
    """API endpoint to get weather forecast for a location"""
    from services.weather_service import get_weather_forecast
    from services.disease_prediction_service import DiseasePredictor
    
    lat = request.args.get('lat', type=float)
    lon = request.args.get('lon', type=float)
    location_name = request.args.get('location', 'Unknown')
    days = request.args.get('days', 14, type=int)
    
    if not lat or not lon:
        return jsonify({'error': 'Latitude and longitude required'}), 400
    
    try:
        # Get weather forecast
        forecast_data = get_weather_forecast(lat, lon, days)
        
        if not forecast_data:
            return jsonify({'error': 'Failed to fetch weather forecast'}), 500
        
        # Get disease predictions
        predictor = DiseasePredictor()
        predictions = predictor.get_all_disease_predictions(forecast_data['forecast'])
        
        return jsonify({
            'location': location_name,
            'lat': lat,
            'lon': lon,
            'weather_forecast': forecast_data['forecast'],
            'disease_predictions': predictions
        })
    except Exception as e:
        logger.error(f"Error fetching weather forecast: {e}")
        return jsonify({'error': str(e)}), 500


@app.route("/api/disease-prediction/<disease_name>")
def api_disease_prediction(disease_name):
    """API endpoint to get prediction for a specific disease"""
    from services.weather_service import get_weather_forecast
    from services.disease_prediction_service import DiseasePredictor
    
    lat = request.args.get('lat', type=float)
    lon = request.args.get('lon', type=float)
    days = request.args.get('days', 14, type=int)
    
    if not lat or not lon:
        return jsonify({'error': 'Latitude and longitude required'}), 400
    
    try:
        # Get weather forecast
        forecast_data = get_weather_forecast(lat, lon, days)
        
        if not forecast_data:
            return jsonify({'error': 'Failed to fetch weather forecast'}), 500
        
        # Get prediction for specific disease
        predictor = DiseasePredictor()
        predictions = predictor.predict_disease_risk(forecast_data['forecast'], disease_name)
        
        # Get high risk days
        high_risk_days = predictor.get_high_risk_days(predictions)
        
        # Get recommendations
        if predictions:
            latest_risk = predictions[0]['risk_level']
            recommendations = predictor.generate_recommendations(disease_name, latest_risk)
        else:
            recommendations = []
        
        return jsonify({
            'disease': disease_name,
            'predictions': predictions,
            'high_risk_days': high_risk_days,
            'recommendations': recommendations
        })
    except Exception as e:
        logger.error(f"Error getting disease prediction: {e}")
        return jsonify({'error': str(e)}), 500


@app.route("/api/historical-patterns")
def api_historical_patterns():
    """API endpoint to analyze historical disease patterns"""
    from models import DiseaseOccurrence
    from services.disease_prediction_service import HistoricalPatternAnalyzer
    
    location = request.args.get('location', '')
    disease_id = request.args.get('disease_id', type=int)
    
    try:
        query = DiseaseOccurrence.query
        
        if location:
            query = query.filter(DiseaseOccurrence.location_name.ilike(f'%{location}%'))
        
        if disease_id:
            query = query.filter(DiseaseOccurrence.disease_id == disease_id)
        
        occurrences = query.order_by(DiseaseOccurrence.occurrence_date.desc()).limit(1000).all()
        occurrences_data = [o.to_dict() for o in occurrences]
        
        analyzer = HistoricalPatternAnalyzer()
        
        # Analyze seasonal patterns
        seasonal_patterns = analyzer.analyze_seasonal_patterns(occurrences_data)
        
        # Analyze regional patterns
        regional_patterns = analyzer.get_regional_patterns(occurrences_data)
        
        return jsonify({
            'seasonal_patterns': seasonal_patterns,
            'regional_patterns': regional_patterns,
            'total_occurrences': len(occurrences_data)
        })
    except Exception as e:
        logger.error(f"Error analyzing historical patterns: {e}")
        return jsonify({'error': str(e)}), 500


@app.route("/api/report-disease-occurrence", methods=['POST'])
@login_required
def api_report_disease_occurrence():
    """API endpoint to report a disease occurrence (for ML training)"""
    from models import DiseaseOccurrence, Disease
    
    data = request.get_json()
    
    disease_id = data.get('disease_id')
    location_name = data.get('location_name')
    latitude = data.get('latitude')
    longitude = data.get('longitude')
    occurrence_date = data.get('occurrence_date')
    severity = data.get('severity', 'moderate')
    affected_area = data.get('affected_area')
    notes = data.get('notes')
    
    if not disease_id or not location_name or not occurrence_date:
        return jsonify({'error': 'disease_id, location_name, and occurrence_date required'}), 400
    
    try:
        # Validate disease exists
        disease = Disease.query.get(disease_id)
        if not disease:
            return jsonify({'error': 'Disease not found'}), 404
        
        # Parse date
        from datetime import datetime
        try:
            occurrence_date = datetime.strptime(occurrence_date, '%Y-%m-%d').date()
        except ValueError:
            return jsonify({'error': 'Invalid date format. Use YYYY-MM-DD'}), 400
        
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
            notes=notes
        )
        
        db.session.add(occurrence)
        db.session.commit()
        
        return jsonify({
            'message': 'Disease occurrence reported successfully',
            'occurrence_id': occurrence.id
        })
    except Exception as e:
        logger.error(f"Error reporting disease occurrence: {e}")
        db.session.rollback()
        return jsonify({'error': str(e)}), 500



@app.route('/analyze_result', methods=['POST'])
def analyze_result():
    payload = request.form.get('payload')

    if not payload:
        return "No analysis data received", 400

    results = json.loads(payload)

    return render_template('results.html', results=results)


if __name__ == '__main__':
    logger.info("=" * 60)
    logger.info("Agri-Vision Cotton Analysis System")
    logger.info("=" * 60)
    logger.info("Starting Flask application...")
    logger.info("Open http://localhost:5000 in your browser")
    logger.info("Endpoints:")
    logger.info("/              - Home page")
    logger.info("/analyze       - Upload and analyze image")
    logger.info("/comparison    - Compare two field images")
    logger.info("/dashboard     - Multi-farm comparison dashboard")
    logger.info("/demo          - View demo results")
    logger.info("/batch         - Batch image analysis")
    logger.info("/api/analyze   - API endpoint (POST)")
    logger.info("/health        - Health check")
    logger.info("=" * 60)

    # Initialize database tables
    with app.app_context():
        db.create_all()
        ensure_account_lockout_schema()
        logger.info("Database tables created")

        # Seed enterprise RBAC (idempotent)
        try:
            from auth.rbac_seed import seed_rbac_permissions_and_roles

            seed_rbac_permissions_and_roles()
            logger.info("RBAC seed completed")
        except Exception as exc:
            logger.warning(f"RBAC seed skipped/failed: {exc}")


    ensure_models_loaded()
    
    # Register models in the registry
    try:
        registry.register_model(
            model_type="resnet",
            version="v1.0",
            path="models/cotton_crop_disease_classification/full_resnet50_model.pth",
            accuracy=0.9983,
            dataset_version="v1.0"
        )
        registry.register_model(
            model_type="yolo",
            version="v1.0",
            path="models/cotton_crop_growth_stage_prediction/best.pt",
            accuracy=0.6006,
            dataset_version="v1.0"
        )
        registry.set_active_model("resnet", "v1.0")
        registry.set_active_model("yolo", "v1.0")
        logger.info("Models registered in model registry")
    except Exception as e:
        logger.error(f"Error registering models: {e}")
    
    is_debug = os.getenv("FLASK_DEBUG", "False").lower() in ("true", "1", "t")
    app.run(debug=is_debug, host="0.0.0.0", port=5000)
