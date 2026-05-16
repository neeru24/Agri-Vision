"""
Agri-Vision Flask Application
Simple web interface for cotton crop analysis
"""

from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
import os
import cv2
import numpy as np
import uuid
from datetime import datetime
import tensorflow as tf
from tensorflow import keras

 import json
import logging
import os
from dotenv import load_dotenv
from werkzeug.utils import secure_filename

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

app.secret_key = os.getenv("SECRET_KEY", "default-secret-key")

app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

LANG = {
    "en": {
        "welcome": "Welcome to Agri Vision"
    },
    "te": {
        "welcome": "అగ్రి విజన్‌కు స్వాగతం"
    }
}
# Create directories
os.makedirs('static/uploads', exist_ok=True)
os.makedirs('static/css', exist_ok=True)
os.makedirs('models', exist_ok=True)

# Global model variable
model = None


def load_model():
    """Load the trained model"""
    global model

    if model is None:
        try:
            model = keras.models.load_model('models/cotton_classifier.h5')
            logger.info("Model loaded successfully")

        except Exception as e:
            logger.warning(f"Model not found or failed to load: {e}")
            model = None

    return model


def preprocess_image(image, target_size=(224, 224)):
    """Preprocess image for model input"""

    # Resize
    if len(image.shape) == 2:  # Grayscale
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    image = cv2.resize(image, target_size)

    # Normalize
    image = image.astype('float32') / 255.0

    return image


def analyze_image(image):
    """Analyze cotton image and return results"""

    # Load model
    model = load_model()

    # Preprocess image
    processed = preprocess_image(image)
    processed = np.expand_dims(processed, axis=0)

    if model:
        # Use trained model
        phase_pred, health_pred, score_pred = model.predict(processed, verbose=0)

    else:
        # Demo predictions (random)
        phase_pred = np.random.rand(1, 4)
        phase_pred = phase_pred / phase_pred.sum(axis=1, keepdims=True)

        health_pred = np.random.rand(1, 4)
        health_pred = health_pred / health_pred.sum(axis=1, keepdims=True)

        score_pred = np.random.rand(1, 1) * 0.5 + 0.5  # Between 0.5 and 1.0

    # Get predictions
    phase_classes = [
        'Vegetative/Budding',
        'Flowering',
        'Bursting (Ripped)',
        'Harvest Ready'
    ]

    health_classes = [
        'Healthy',
        'Pink Bollworm Damage',
        'Discoloration',
        'Other Damage'
    ]

    phase_idx = np.argmax(phase_pred[0])
    health_idx = np.argmax(health_pred[0])
    health_score = float(score_pred[0][0] * 100)

    # Generate recommendations
    recommendations = []

    # Phase-based recommendations
    if phase_idx == 0:
        recommendations.append("Continue regular watering and fertilization")
        recommendations.append("Monitor for early signs of pests")

    elif phase_idx == 1:
        recommendations.append("Reduce nitrogen, increase phosphorus fertilizer")
        recommendations.append("Watch for flowering pests")

    elif phase_idx == 2:
        recommendations.append("Prepare for harvest in 7-10 days")
        recommendations.append("Stop irrigation to promote bursting")

    else:
        recommendations.append("Harvest immediately")
        recommendations.append("Check weather forecast for dry conditions")

    # Health-based recommendations
    if health_idx == 1:
        recommendations.append("Apply targeted pesticide for pink bollworm")
        recommendations.append("Remove infected bolls immediately")

    elif health_idx == 2:
        recommendations.append("Check soil pH and nutrient levels")
        recommendations.append("Adjust irrigation schedule")

    # Health score recommendations
    if health_score < 50:
        recommendations.append("Consult agricultural expert immediately")

    elif health_score < 70:
        recommendations.append("Increase monitoring frequency")

    # Limit recommendations
    recommendations = recommendations[:5]

    # Prepare results
    results = {
        'stage': phase_classes[phase_idx],
        'stage_confidence': float(phase_pred[0][phase_idx]),
        'health_status': health_classes[health_idx],
        'health_confidence': float(health_pred[0][health_idx]),
        'health_score': health_score,
        'is_ripped': bool(phase_idx == 2),
        'has_damage': bool(health_idx > 0),
        'recommendations': recommendations,
        'phases': {
            phase_classes[i]: float(phase_pred[0][i]) for i in range(4)
        },
        'health': {
            health_classes[i]: float(health_pred[0][i]) for i in range(4)
        }
    }

    return results


# Custom filter for datetime
@app.template_filter('datetimeformat')
def datetimeformat_filter(value):
    """Format datetime for display"""

    if value == 'now':
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    return value


@app.route('/')
def index():
    """Home page"""
    return render_template('index.html', text=LANG.get(lang, LANG["en"]), lang=lang)


@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
    """Analyze single image"""

    if request.method == 'POST':

        if 'file' not in request.files:
            flash('No file uploaded', 'error')
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(request.url)

        # Check file extension
        allowed_extensions = {'png', 'jpg', 'jpeg', 'gif'}

        if not '.' in file.filename or \
           file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:

            flash(
                'Invalid file type. Please upload an image (PNG, JPG, JPEG, GIF)',
                'error'
            )

            return redirect(request.url)

        try:
            # Secure filename handling
            safe_filename = secure_filename(file.filename)

            filename = f"{uuid.uuid4().hex[:8]}_{safe_filename}"

            filepath = os.path.join('static/uploads', filename)

            # Save file
            file.save(filepath)

            # Read image
            image = cv2.imread(filepath)

            if image is None:
                flash('Error reading image file', 'error')
                return redirect(request.url)

            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Analyze image
            results = analyze_image(image)

            flash('Analysis completed successfully!', 'success')

            return render_template(
                'results.html',
                results=results,
                filename=filename,
                timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            )

        except Exception as e:
            logger.error(f"Analysis error: {e}")

            flash(f'Error during analysis: {str(e)}', 'error')

            return redirect(request.url)

    return render_template('upload.html')


@app.route('/demo')
def demo():
    """Demo page with example results"""

    # Create demo results
    results = {
        'stage': 'Bursting (Ripped)',
        'stage_confidence': 0.87,
        'health_status': 'Pink Bollworm Damage',
        'health_confidence': 0.76,
        'health_score': 68.5,
        'is_ripped': True,
        'has_damage': True,
        'recommendations': [
            "Prepare for harvest in 7-10 days",
            "Stop irrigation to promote bursting",
            "Apply targeted pesticide for pink bollworm",
            "Remove infected bolls immediately",
            "Monitor daily for optimal harvest time"
        ],
        'phases': {
            'Vegetative/Budding': 0.05,
            'Flowering': 0.08,
            'Bursting (Ripped)': 0.87,
            'Harvest Ready': 0.0
        },
        'health': {
            'Healthy': 0.12,
            'Pink Bollworm Damage': 0.76,
            'Discoloration': 0.08,
            'Other Damage': 0.04
        }
    }

    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    return render_template(
        'results.html',
        results=results,
        filename='demo_cotton.jpg',
        timestamp=current_time
    )


@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """API endpoint for programmatic access"""

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        # Read image
        file_bytes = np.frombuffer(file.read(), np.uint8)

        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({'error': 'Invalid image file'}), 400

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Analyze
        results = analyze_image(image)

        # Prepare API response
        response = {
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
            'analysis': {
                'stage': results['stage'],
                'stage_confidence': results['stage_confidence'],
                'health_status': results['health_status'],
                'health_confidence': results['health_confidence'],
                'health_score': results['health_score'],
                'is_ripped': results['is_ripped'],
                'has_damage': results['has_damage']
            },
            'recommendations': results['recommendations']
        }

        return jsonify(response)

    except Exception as e:
        logger.error(f"API analysis error: {e}")

        return jsonify({'error': str(e)}), 500


@app.route('/health')
def health():
    """Health check endpoint"""

    model_loaded = model is not None

    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': model_loaded,
        'service': 'Agri-Vision Cotton Analysis API'
    })

@app.route("/set-language/<lang>")
def set_language(lang):
    return redirect(url_for("index", lang=lang))



if __name__ == '__main__':

    logger.info("=" * 60)
    logger.info("Agri-Vision Cotton Analysis System")
    logger.info("=" * 60)

    logger.info("Starting Flask application...")
    logger.info("Open http://localhost:5000 in your browser")

    logger.info("Endpoints:")
    logger.info("/              - Home page")
    logger.info("/analyze       - Upload and analyze image")
    logger.info("/demo          - View demo results")
    logger.info("/api/analyze   - API endpoint (POST)")
    logger.info("/health        - Health check")

    logger.info("=" * 60)

    # Try to load model
    load_model()

    # Run Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)
# fix for issue #13