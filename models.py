"""
Database models for batch processing and user authentication
"""

from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
import uuid

db = SQLAlchemy()

ROLE_FARMER = "farmer"
ROLE_RESEARCHER = "researcher"
ROLE_ADMIN = "admin"


class User(UserMixin, db.Model):
    """User model for authentication"""

    __tablename__ = "users"

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    email = db.Column(db.String(255), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    full_name = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(20), default=ROLE_FARMER)  # farmer, researcher, admin
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime, nullable=True)
    is_active = db.Column(db.Boolean, default=True)

    # Relationships
    analyses = db.relationship(
        "AnalysisHistory", backref="user", lazy=True, cascade="all, delete-orphan"
    )

    def set_password(self, password):
        """Hash and set password"""
        from bcrypt import hashpw, gensalt

        self.password_hash = hashpw(password.encode("utf-8"), gensalt()).decode("utf-8")

    def check_password(self, password):
        """Check if password matches hash"""
        from bcrypt import checkpw, hashpw

        return checkpw(password.encode("utf-8"), self.password_hash.encode("utf-8"))

    def is_admin(self):
        """Check if user is admin"""
        return self.role == ROLE_ADMIN

    def is_researcher(self):
        """Check if user is researcher or admin"""
        return self.role in [ROLE_RESEARCHER, ROLE_ADMIN]

    def to_dict(self):
        return {
            "id": self.id,
            "email": self.email,
            "full_name": self.full_name,
            "role": self.role,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_login": self.last_login.isoformat() if self.last_login else None,
            "is_active": self.is_active,
        }


class AnalysisHistory(db.Model):
    """Store individual user analyses"""

    __tablename__ = "analysis_history"

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = db.Column(
        db.String(36), db.ForeignKey("users.id"), nullable=False, index=True
    )
    image_path = db.Column(db.String(500), nullable=True)
    disease_result = db.Column(db.JSON, nullable=True)
    growth_result = db.Column(db.JSON, nullable=True)
    confidence = db.Column(db.Float, nullable=True)
    health_score = db.Column(db.Float, nullable=True)
    # Geographic location fields
    latitude = db.Column(db.Float, nullable=True)
    longitude = db.Column(db.Float, nullable=True)
    location_name = db.Column(db.String(255), nullable=True)
    region = db.Column(db.String(100), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)

    def to_dict(self):
        return {
            "id": self.id,
            "user_id": self.user_id,
            "image_path": self.image_path,
            "disease_result": self.disease_result,
            "growth_result": self.growth_result,
            "confidence": self.confidence,
            "health_score": self.health_score,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "location_name": self.location_name,
            "region": self.region,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class BatchJob(db.Model):
    """Model for batch analysis jobs"""

    __tablename__ = "batch_jobs"

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    status = db.Column(
        db.String(20), default="pending"
    )  # pending, processing, completed, failed
    total_images = db.Column(db.Integer, default=0)
    completed_images = db.Column(db.Integer, default=0)
    failed_images = db.Column(db.Integer, default=0)
    task_ids = db.Column(db.JSON, default=list)  # List of Celery task IDs
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    started_at = db.Column(db.DateTime, nullable=True)
    completed_at = db.Column(db.DateTime, nullable=True)
    error_message = db.Column(db.Text, nullable=True)

    # Relationships
    results = db.relationship(
        "AnalysisResult", backref="batch_job", lazy=True, cascade="all, delete-orphan"
    )

    def to_dict(self):
        return {
            "id": self.id,
            "status": self.status,
            "total_images": self.total_images,
            "completed_images": self.completed_images,
            "failed_images": self.failed_images,
            "progress": self.progress_percentage(),
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": (
                self.completed_at.isoformat() if self.completed_at else None
            ),
            "error_message": self.error_message,
        }

    def progress_percentage(self):
        if self.total_images == 0:
            return 0
        return int(
            (self.completed_images + self.failed_images) / self.total_images * 100
        )


class AnalysisResult(db.Model):
    """Model for individual analysis results"""

    __tablename__ = "analysis_results"

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    batch_job_id = db.Column(
        db.String(36), db.ForeignKey("batch_jobs.id"), nullable=False
    )
    image_name = db.Column(db.String(255), nullable=False)
    image_index = db.Column(db.Integer, nullable=False)
    status = db.Column(
        db.String(20), default="pending"
    )  # pending, processing, complete, error

    # Disease results
    disease_class = db.Column(db.String(100), nullable=True)
    disease_confidence = db.Column(db.Float, nullable=True)
    health_score = db.Column(db.Float, nullable=True)

    # Growth results
    growth_class = db.Column(db.String(100), nullable=True)
    growth_confidence = db.Column(db.Float, nullable=True)

    # Full results as JSON
    results_json = db.Column(db.JSON, nullable=True)

    # Error handling
    error_message = db.Column(db.Text, nullable=True)

    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "batch_job_id": self.batch_job_id,
            "image_name": self.image_name,
            "image_index": self.image_index,
            "status": self.status,
            "disease_class": self.disease_class,
            "disease_confidence": self.disease_confidence,
            "health_score": self.health_score,
            "growth_class": self.growth_class,
            "growth_confidence": self.growth_confidence,
            "results": self.results_json,
            "error_message": self.error_message,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class Disease(db.Model):
    """Model for cotton diseases"""

    __tablename__ = "diseases"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    name = db.Column(db.String(100), unique=True, nullable=False, index=True)
    scientific_name = db.Column(db.String(200), nullable=True)
    description = db.Column(db.Text, nullable=False)
    causes = db.Column(db.Text, nullable=True)
    symptoms = db.Column(db.Text, nullable=True)
    severity = db.Column(
        db.String(20), default="moderate"
    )  # low, moderate, high, severe
    spread_rate = db.Column(db.String(20), default="moderate")  # slow, moderate, fast
    affected_parts = db.Column(
        db.String(200), nullable=True
    )  # leaves, stems, bolls, roots
    favorable_conditions = db.Column(db.Text, nullable=True)
    prevention = db.Column(db.Text, nullable=True)
    image_url = db.Column(db.String(500), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(
        db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    # Relationships
    treatments = db.relationship(
        "Treatment", backref="disease", lazy=True, cascade="all, delete-orphan"
    )

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "scientific_name": self.scientific_name,
            "description": self.description,
            "causes": self.causes,
            "symptoms": self.symptoms,
            "severity": self.severity,
            "spread_rate": self.spread_rate,
            "affected_parts": self.affected_parts,
            "favorable_conditions": self.favorable_conditions,
            "prevention": self.prevention,
            "image_url": self.image_url,
            "treatments": [t.to_dict() for t in self.treatments],
        }


class Treatment(db.Model):
    """Model for disease treatments"""

    __tablename__ = "treatments"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    disease_id = db.Column(db.Integer, db.ForeignKey("diseases.id"), nullable=False)
    name = db.Column(db.String(200), nullable=False)
    type = db.Column(
        db.String(50), nullable=False
    )  # chemical, biological, cultural, integrated
    description = db.Column(db.Text, nullable=False)
    application_method = db.Column(db.Text, nullable=True)
    dosage = db.Column(db.String(200), nullable=True)
    timing = db.Column(db.String(200), nullable=True)
    effectiveness = db.Column(db.String(20), default="high")  # low, moderate, high
    cost = db.Column(db.String(20), default="moderate")  # low, moderate, high
    precautions = db.Column(db.Text, nullable=True)
    resistance_management = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(
        db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    def to_dict(self):
        return {
            "id": self.id,
            "disease_id": self.disease_id,
            "name": self.name,
            "type": self.type,
            "description": self.description,
            "application_method": self.application_method,
            "dosage": self.dosage,
            "timing": self.timing,
            "effectiveness": self.effectiveness,
            "cost": self.cost,
            "precautions": self.precautions,
            "resistance_management": self.resistance_management,
        }


class Symptom(db.Model):
    """Model for disease symptoms for symptom checker"""

    __tablename__ = "symptoms"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    name = db.Column(db.String(200), unique=True, nullable=False, index=True)
    description = db.Column(db.Text, nullable=False)
    category = db.Column(
        db.String(50), nullable=False
    )  # leaf, stem, boll, root, general
    severity_indicator = db.Column(
        db.String(20), default="moderate"
    )  # mild, moderate, severe
    image_url = db.Column(db.String(500), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Many-to-many relationship with diseases
    associated_diseases = db.relationship(
        "Disease", secondary="disease_symptoms", backref="related_symptoms"
    )

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "severity_indicator": self.severity_indicator,
            "image_url": self.image_url,
        }


class DiseaseSymptom(db.Model):
    """Association table for diseases and symptoms"""

    __tablename__ = "disease_symptoms"

    disease_id = db.Column(db.Integer, db.ForeignKey("diseases.id"), primary_key=True)
    symptom_id = db.Column(db.Integer, db.ForeignKey("symptoms.id"), primary_key=True)
    confidence = db.Column(
        db.Float, default=0.5
    )  # How strongly this symptom indicates the disease


class WeatherData(db.Model):
    """Model for storing weather data"""

    __tablename__ = "weather_data"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    location_name = db.Column(db.String(200), nullable=False, index=True)
    latitude = db.Column(db.Float, nullable=True)
    longitude = db.Column(db.Float, nullable=True)
    date = db.Column(db.Date, nullable=False, index=True)
    temperature_avg = db.Column(
        db.Float, nullable=True
    )  # Average temperature in Celsius
    temperature_max = db.Column(db.Float, nullable=True)  # Max temperature
    temperature_min = db.Column(db.Float, nullable=True)  # Min temperature
    humidity = db.Column(db.Float, nullable=True)  # Humidity percentage
    rainfall = db.Column(db.Float, nullable=True)  # Rainfall in mm
    wind_speed = db.Column(db.Float, nullable=True)  # Wind speed in km/h
    pressure = db.Column(db.Float, nullable=True)  # Atmospheric pressure
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "location_name": self.location_name,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "date": self.date.isoformat() if self.date else None,
            "temperature_avg": self.temperature_avg,
            "temperature_max": self.temperature_max,
            "temperature_min": self.temperature_min,
            "humidity": self.humidity,
            "rainfall": self.rainfall,
            "wind_speed": self.wind_speed,
            "pressure": self.pressure,
        }


class DiseasePrediction(db.Model):
    """Model for disease predictions based on weather"""

    __tablename__ = "disease_predictions"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    disease_id = db.Column(db.Integer, db.ForeignKey("diseases.id"), nullable=False)
    location_name = db.Column(db.String(200), nullable=False)
    latitude = db.Column(db.Float, nullable=True)
    longitude = db.Column(db.Float, nullable=True)
    prediction_date = db.Column(
        db.Date, nullable=False, index=True
    )  # Date the prediction is for
    risk_level = db.Column(db.String(20), nullable=False)  # low, moderate, high, severe
    risk_score = db.Column(db.Float, nullable=False)  # 0-100 risk score
    confidence = db.Column(db.Float, nullable=False)  # Model confidence
    weather_factors = db.Column(
        db.JSON, nullable=True
    )  # Which weather factors contributed
    recommended_actions = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Relationships
    disease = db.relationship("Disease", backref="predictions")

    def to_dict(self):
        return {
            "id": self.id,
            "disease_id": self.disease_id,
            "disease_name": self.disease.name if self.disease else None,
            "location_name": self.location_name,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "prediction_date": (
                self.prediction_date.isoformat() if self.prediction_date else None
            ),
            "risk_level": self.risk_level,
            "risk_score": self.risk_score,
            "confidence": self.confidence,
            "weather_factors": self.weather_factors,
            "recommended_actions": self.recommended_actions,
        }


class RefreshTokenFamily(db.Model):
    """Refresh token family tracking for rotation and replay detection."""

    __tablename__ = "refresh_token_families"

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = db.Column(db.String(36), db.ForeignKey("users.id"), nullable=False, index=True)
    session_id = db.Column(db.String(64), nullable=False, index=True)
    is_compromised = db.Column(db.Boolean, default=False, nullable=False)
    compromised_at = db.Column(db.DateTime, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    refresh_tokens = db.relationship(
        "RefreshToken",
        backref="family",
        lazy=True,
        cascade="all, delete-orphan",
    )


class RefreshToken(db.Model):
    """A single refresh token (hashed) with one-time use enforcement."""

    __tablename__ = "refresh_tokens"

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    family_id = db.Column(
        db.String(36),
        db.ForeignKey("refresh_token_families.id"),
        nullable=False,
        index=True,
    )
    user_id = db.Column(db.String(36), db.ForeignKey("users.id"), nullable=False, index=True)
    session_id = db.Column(db.String(64), nullable=False, index=True)

    token_hash = db.Column(db.String(64), nullable=False, index=True)  # sha256 hex

    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    expires_at = db.Column(db.DateTime, nullable=False, index=True)

    revoked_at = db.Column(db.DateTime, nullable=True, index=True)
    replaced_by_token_id = db.Column(db.String(36), nullable=True)

    is_compromised = db.Column(db.Boolean, default=False, nullable=False, index=True)
    last_used_at = db.Column(db.DateTime, nullable=True)

    created_ip = db.Column(db.String(64), nullable=True)
    created_user_agent = db.Column(db.String(255), nullable=True)

    __table_args__ = (
        db.Index(
            "ix_refresh_tokens_family_token_hash",
            "family_id",
            "token_hash",
        ),
    )


class DiseaseOccurrence(db.Model):
    """Model for tracking historical disease occurrences for ML training"""


    __tablename__ = "disease_occurrences"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    disease_id = db.Column(db.Integer, db.ForeignKey("diseases.id"), nullable=False)
    location_name = db.Column(db.String(200), nullable=False)
    latitude = db.Column(db.Float, nullable=True)
    longitude = db.Column(db.Float, nullable=True)
    occurrence_date = db.Column(db.Date, nullable=False, index=True)
    severity = db.Column(db.String(20), nullable=False)  # low, moderate, high, severe
    affected_area = db.Column(db.Float, nullable=True)  # Area affected in hectares
    weather_data_id = db.Column(
        db.Integer, db.ForeignKey("weather_data.id"), nullable=True
    )
    reported_by = db.Column(db.String(36), db.ForeignKey("users.id"), nullable=True)
    notes = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Relationships
    disease = db.relationship("Disease", backref="occurrences")
    weather_data = db.relationship("WeatherData", backref="disease_occurrences")
    reporter = db.relationship("User", backref="reported_diseases")

    def to_dict(self):
        return {
            "id": self.id,
            "disease_id": self.disease_id,
            "disease_name": self.disease.name if self.disease else None,
            "location_name": self.location_name,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "occurrence_date": (
                self.occurrence_date.isoformat() if self.occurrence_date else None
            ),
            "severity": self.severity,
            "affected_area": self.affected_area,
            "weather_data_id": self.weather_data_id,
            "reported_by": self.reported_by,
            "notes": self.notes,
        }
