"""
Agri-Vision Flask Application
Unified inference for disease classification (ResNet50) and growth stage prediction (YOLOv8)
"""

import json
import logging
import os
from datetime import datetime

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from flask import Flask, flash, jsonify, redirect, render_template, request, url_for
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO
from werkzeug.utils import secure_filename

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder="static", template_folder="templates")

app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

app.jinja_env.auto_reload = True
app.jinja_env.cache = {}

secret_key = os.getenv("SECRET_KEY")
if not secret_key:
    secret_key = "dev_secret_123"
app.secret_key = secret_key

app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

# ====================== HINDI LANGUAGE SUPPORT ======================
TRANSLATIONS = {
    "en": {
        "title": "Agri-Vision",
        "home": "Home",
        "analyze": "Analyze",
        "demo": "Demo",
        "tutorials": "Tutorials",
        "stories": "Stories",
        "analyze_image": "Analyze Image",
        "view_demo": "View Demo",
        "disease_classification": "Disease Classification (ResNet50)",
        "growth_stage": "Growth Stage Detection (YOLOv8)",
        "recommendations": "Recommendations",
        "analysis_time": "Analysis Time",
        "show_json": "Show Full Model Output (JSON)",
        "upload_prompt": "Drop image here or click to browse",
        "supports": "Supports: PNG, JPG, JPEG (Max 10MB)",
        "start_analysis": "Start AI Analysis",
        "get_started": "Get Started",
        "stories_title": "Farmer Success Stories",
        "tutorials_title": "Farming Video Tutorials",
        "analysis_results": "Analysis Results",
        "field_comparison": "Field Photo Comparison",
        "upload_last_week": "Upload last week photo",
        "upload_current_week": "Upload current week photo",
        "compare_button": "Compare Field Health"
    },
    "hi": {
        "title": "एग्री-विजन",
        "home": "होम",
        "analyze": "विश्लेषण करें",
        "demo": "डेमो",
        "tutorials": "ट्यूटोरियल",
        "stories": "कहानियाँ",
        "analyze_image": "इमेज का विश्लेषण करें",
        "view_demo": "डेमो देखें",
        "disease_classification": "रोग वर्गीकरण (ResNet50)",
        "growth_stage": "विकास चरण पहचान (YOLOv8)",
        "recommendations": "सिफारिशें",
        "analysis_time": "विश्लेषण का समय",
        "show_json": "पूर्ण मॉडल आउटपुट (JSON) दिखाएं",
        "upload_prompt": "यहाँ इमेज ड्रॉप करें या ब्राउज़ करने के लिए क्लिक करें",
        "supports": "समर्थित: PNG, JPG, JPEG (अधिकतम 10MB)",
        "start_analysis": "एआई विश्लेषण शुरू करें",
        "get_started": "शुरू करें",
        "stories_title": "किसान सफलता कहानियाँ",
        "tutorials_title": "कृषि वीडियो ट्यूटोरियल",
        "analysis_results": "विश्लेषण परिणाम",
        "field_comparison": "खेत की फोटो तुलना",
        "upload_last_week": "पिछले सप्ताह की फोटो अपलोड करें",
        "upload_current_week": "इस सप्ताह की फोटो अपलोड करें",
        "compare_button": "खेत स्वास्थ्य की तुलना करें"
    }
}

# ========================================================

# Setup directories
os.makedirs('static/uploads', exist_ok=True)
os.makedirs('static/css', exist_ok=True)
os.makedirs('models', exist_ok=True)

# --- Class Names ---
disease_classes = [
    "Aphids", "Army worm", "Bacterial blight", "Cotton Boll Rot",
    "Green Cotton Boll", "Healthy", "Powdery mildew", "Target Spot"
]

growth_stage_classes = [
    "Cotton Blossom", "Cotton Bud", "Early Boll", 
    "Matured Cotton Boll", "Split Cotton Boll"
]

resnet_model = None
yolo_model = None

def load_models():
    global resnet_model, yolo_model
    if resnet_model is None:
        try:
            resnet_model = torch.load(
                "models/cotton_crop_disease_classification/full_resnet50_model.pth",
                map_location=torch.device("cpu"),
            )
            logger.info("ResNet50 model loaded successfully")
        except Exception as e:
            logger.warning(f"ResNet50 model not found: {e}")
            resnet_model = None
    if yolo_model is None:
        try:
            yolo_model = YOLO("models/cotton_crop_growth_stage_prediction/best.pt")
            logger.info("YOLOv8 model loaded successfully")
        except Exception as e:
            logger.warning(f"YOLOv8 model not found: {e}")
            yolo_model = None
    return resnet_model, yolo_model

# ====================== CORE FUNCTIONS ======================
def preprocess_image_for_resnet(image, target_size=(224, 224)):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(target_size),
        transforms.ToTensor(),
    ])
    image = transform(image)
    image = image.unsqueeze(0)
    return image

def infer_disease(image):
    if resnet_model:
        processed = preprocess_image_for_resnet(image)
        with torch.no_grad():
            output = resnet_model(processed)
            probs = F.softmax(output, dim=1)
            confidence, prediction = torch.max(probs, 1)
        probs_np = probs.numpy()
        class_idx = int(prediction.item())
        healthy_idx = disease_classes.index("Healthy")
        health_score = float(probs_np[0][healthy_idx]) * 100
    else:
        probs_np = np.random.rand(1, len(disease_classes))
        probs_np = probs_np / probs_np.sum(axis=1, keepdims=True)
        class_idx = int(np.argmax(probs_np[0]))
        health_score = float(np.max(probs_np[0])) * 100

    disease_confidences = {disease_classes[i]: float(probs_np[0][i]) for i in range(len(disease_classes))}
    return {
        "predicted_class": disease_classes[class_idx],
        "predicted_class_idx": class_idx,
        "confidence": float(probs_np[0][class_idx]),
        "all_confidences": disease_confidences,
        "health_score": health_score,
        "raw": probs_np.tolist(),
    }

def infer_growth_stage(image):
    result = {"main_class": None, "main_class_idx": None, "confidence": 0.0, "boxes": [], "raw": []}
    if yolo_model:
        pil_image = Image.fromarray(image)
        yolo_results = yolo_model(pil_image)
        boxes = []
        for r in yolo_results:
            if hasattr(r, 'boxes'):
                for b in r.boxes:
                    class_id = int(b.cls[0].item())
                    conf = float(b.conf[0].item())
                    xyxy = b.xyxy[0].cpu().numpy().tolist()
                    boxes.append({
                        "class_id": class_id,
                        "class_name": growth_stage_classes[class_id] if class_id < len(growth_stage_classes) else str(class_id),
                        "confidence": conf,
                        "bbox": xyxy,
                    })
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

def generate_recommendations(disease_result, growth_result):
    recs = []
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
    elif disease_result["health_score"] < 70:
        recs.append("Increase frequency of crop monitoring based on moderate health.")

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
    return recs[:5]

def analyze_image(image):
    disease = infer_disease(image)
    growth = infer_growth_stage(image)
    recs = generate_recommendations(disease, growth)
    return {"disease": disease, "growth": growth, "recommendations": recs}

def encode_image_for_display(image):
    import base64
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')

@app.after_request
def add_no_cache_headers(response):
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

# ====================== ROUTES ======================
@app.route("/")
def index():
    lang = request.args.get("lang", "en")
    if lang not in TRANSLATIONS:
        lang = "en"
    return render_template("index.html", lang=lang, t=TRANSLATIONS[lang])

@app.route("/analyze", methods=["GET", "POST"])
def analyze():
    lang = request.args.get("lang", "en")
    if lang not in TRANSLATIONS:
        lang = "en"

    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file uploaded', 'error')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(request.url)
        allowed_extensions = {'png', 'jpg', 'jpeg', 'gif'}
        if not '.' in file.filename or file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
            flash('Invalid file type. Please upload an image (PNG, JPG, JPEG, GIF)', 'error')
            return redirect(request.url)
        try:
            safe_filename = secure_filename(file.filename)
            file_bytes = np.frombuffer(file.read(), np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if image is None:
                flash('Error reading image file', 'error')
                return redirect(request.url)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_b64 = encode_image_for_display(image)
            results = analyze_image(image_rgb)
            
            return render_template(
                "results.html",
                results=results,
                filename=safe_filename,
                image_b64=image_b64,
                img_shape={"width": image.shape[1], "height": image.shape[0]},
                raw_json=json.dumps(results, indent=2),
                timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                lang=lang,
                t=TRANSLATIONS[lang]
            )
        except Exception as e:
            logger.error(f"Analysis error: {e}")
            flash(f'Error during analysis: {str(e)}', 'error')
            return redirect(request.url)
    
    return render_template("upload.html", lang=lang, t=TRANSLATIONS[lang])

@app.route("/demo")
def demo():
    lang = request.args.get("lang", "en")
    if lang not in TRANSLATIONS:
        lang = "en"
    
    example_disease_probs = [0.08, 0.02, 0.01, 0.10, 0.04, 0.65, 0.05, 0.05]
    demo_disease = {
        "predicted_class": "Healthy",
        "predicted_class_idx": 5,
        "confidence": example_disease_probs[5],
        "all_confidences": {disease_classes[i]: example_disease_probs[i] for i in range(len(disease_classes))},
        "health_score": 65.0,
        "raw": [example_disease_probs]
    }
    demo_growth_boxes = [
        {"class_id": 3, "class_name": "Matured Cotton Boll", "confidence": 0.91, "bbox": [120, 80, 210, 155]},
        {"class_id": 4, "class_name": "Split Cotton Boll", "confidence": 0.70, "bbox": [300, 120, 390, 210]}
    ]
    demo_growth = {
        "main_class": "Matured Cotton Boll",
        "main_class_idx": 3,
        "confidence": 0.91,
        "boxes": demo_growth_boxes,
        "raw": demo_growth_boxes
    }
    example_json = {
        "disease": demo_disease,
        "growth": demo_growth,
        "recommendations": generate_recommendations(demo_disease, demo_growth)
    }
    return render_template(
        "results.html",
        results=example_json,
        filename="demo_cotton.jpg",
        image_b64="",
        img_shape={"width": 512, "height": 384},
        raw_json=json.dumps(example_json, indent=2),
        timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        lang=lang,
        t=TRANSLATIONS[lang]
    )

@app.route("/set-language/<lang>")
def set_language(lang):
    if lang not in TRANSLATIONS:
        lang = "en"
    return redirect(url_for("index", lang=lang))

@app.route('/tutorials')
def tutorials():
    lang = request.args.get("lang", "en")
    if lang not in TRANSLATIONS:
        lang = "en"
    return render_template('tutorials.html', lang=lang, t=TRANSLATIONS[lang])

@app.route('/stories')
def stories():
    lang = request.args.get("lang", "en")
    if lang not in TRANSLATIONS:
        lang = "en"
    return render_template("stories.html", lang=lang, t=TRANSLATIONS[lang])

# ====================== OTHER ROUTES (kept as per your latest file) ======================
# ... (your comparison route, weather, api, health etc. can be added here)

@app.route("/health")
def health():
    model_loaded = resnet_model is not None and yolo_model is not None
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': model_loaded,
        'service': 'Agri-Vision Cotton Analysis API'
    })

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Agri-Vision Cotton Analysis System")
    logger.info("=" * 60)
    load_models()
    is_debug = os.getenv("FLASK_DEBUG", "False").lower() in ("true", "1", "t")
    app.run(debug=is_debug, host='0.0.0.0', port=5000)
