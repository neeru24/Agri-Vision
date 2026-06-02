"""
Model Registry for Agri-Vision
Handles model versioning, A/B testing, and performance tracking
"""
import json
import os
import logging
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import torch
from ultralytics import YOLO

logger = logging.getLogger(__name__)


class ModelMetadata:
    """Metadata for a model version"""
    def __init__(self, version: str, model_type: str, path: str, 
                 accuracy: float = 0.0, training_date: str = None,
                 dataset_version: str = "unknown", parameters: int = 0,
                 is_active: bool = False, ab_test_ratio: float = 0.0):
        self.version = version
        self.model_type = model_type  # 'resnet' or 'yolo'
        self.path = path
        self.accuracy = accuracy
        self.training_date = training_date or datetime.now().isoformat()
        self.dataset_version = dataset_version
        self.parameters = parameters
        self.is_active = is_active
        self.ab_test_ratio = ab_test_ratio  # 0.0 to 1.0 for A/B testing
        self.performance_metrics = {
            "total_requests": 0,
            "successful_predictions": 0,
            "avg_confidence": 0.0,
            "avg_inference_time": 0.0,
            "error_count": 0
        }

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "model_type": self.model_type,
            "path": self.path,
            "accuracy": self.accuracy,
            "training_date": self.training_date,
            "dataset_version": self.dataset_version,
            "parameters": self.parameters,
            "is_active": self.is_active,
            "ab_test_ratio": self.ab_test_ratio,
            "performance_metrics": self.performance_metrics
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'ModelMetadata':
        metadata = cls(
            version=data["version"],
            model_type=data["model_type"],
            path=data["path"],
            accuracy=data.get("accuracy", 0.0),
            training_date=data.get("training_date"),
            dataset_version=data.get("dataset_version", "unknown"),
            parameters=data.get("parameters", 0),
            is_active=data.get("is_active", False),
            ab_test_ratio=data.get("ab_test_ratio", 0.0)
        )
        metadata.performance_metrics = data.get("performance_metrics", {
            "total_requests": 0,
            "successful_predictions": 0,
            "avg_confidence": 0.0,
            "avg_inference_time": 0.0,
            "error_count": 0
        })
        return metadata


class ModelRegistry:
    """Registry for managing multiple model versions"""
    
    def __init__(self, config_path: str = "model_config.json"):
        self.config_path = config_path
        self.models: Dict[str, List[ModelMetadata]] = {
            "resnet": [],  # Disease classification models
            "yolo": []     # Growth stage detection models
        }
        self.loaded_models: Dict[str, Dict[str, any]] = {
            "resnet": {},
            "yolo": {}
        }
        self.ab_test_enabled = False
        self.rollback_threshold = 0.7  # Accuracy threshold for automatic rollback
        self.load_config()
        
    def load_config(self):
        """Load model configuration from JSON file"""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                
                for model_type, models_data in config.get("models", {}).items():
                    self.models[model_type] = [
                        ModelMetadata.from_dict(m) for m in models_data
                    ]
                
                self.ab_test_enabled = config.get("ab_test_enabled", False)
                self.rollback_threshold = config.get("rollback_threshold", 0.7)
                logger.info(f"Loaded model registry from {self.config_path}")
            except Exception as e:
                logger.error(f"Error loading model config: {e}")
                self._initialize_default_config()
        else:
            self._initialize_default_config()
    
    def _initialize_default_config(self):
        """Initialize default configuration with existing models"""
        logger.info("Initializing default model configuration")
        
        # Add default ResNet model
        if os.path.exists("models/cotton_crop_disease_classification/full_resnet50_model.pth"):
            self.register_model(
                model_type="resnet",
                version="v1.0",
                path="models/cotton_crop_disease_classification/full_resnet50_model.pth",
                accuracy=0.9983,
                dataset_version="roboflow-v1",
                parameters=25600000,
                is_active=True
            )
        
        # Add default YOLO model
        if os.path.exists("models/cotton_crop_growth_stage_prediction/best.pt"):
            self.register_model(
                model_type="yolo",
                version="v1.0",
                path="models/cotton_crop_growth_stage_prediction/best.pt",
                accuracy=0.6006,  # mAP50
                dataset_version="roboflow-v1",
                parameters=3000000,
                is_active=True
            )
        
        self.save_config()
    
    def save_config(self):
        """Save current model configuration to JSON file"""
        config = {
            "models": {
                model_type: [m.to_dict() for m in models]
                for model_type, models in self.models.items()
            },
            "ab_test_enabled": self.ab_test_enabled,
            "rollback_threshold": self.rollback_threshold,
            "last_updated": datetime.now().isoformat()
        }
        
        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Saved model registry to {self.config_path}")
    
    def register_model(self, model_type: str, version: str, path: str,
                      accuracy: float = 0.0, dataset_version: str = "unknown",
                      parameters: int = 0, is_active: bool = False,
                      ab_test_ratio: float = 0.0) -> ModelMetadata:
        """Register a new model version"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        
        metadata = ModelMetadata(
            version=version,
            model_type=model_type,
            path=path,
            accuracy=accuracy,
            dataset_version=dataset_version,
            parameters=parameters,
            is_active=is_active,
            ab_test_ratio=ab_test_ratio
        )
        
        # Remove existing version if it exists
        self.models[model_type] = [
            m for m in self.models[model_type] if m.version != version
        ]
        
        self.models[model_type].append(metadata)
        self.save_config()
        
        logger.info(f"Registered {model_type} model version {version}")
        return metadata
    
    def get_active_model(self, model_type: str) -> Optional[ModelMetadata]:
        """Get the currently active model for a type"""
        for model in self.models.get(model_type, []):
            if model.is_active:
                return model
        return None
    
    def get_model(self, model_type: str, version: str) -> Optional[ModelMetadata]:
        """Get a specific model version"""
        for model in self.models.get(model_type, []):
            if model.version == version:
                return model
        return None
    
    def set_active_model(self, model_type: str, version: str):
        """Set a model version as active"""
        # Deactivate all models of this type
        for model in self.models.get(model_type, []):
            model.is_active = False
        
        # Activate the specified version
        target_model = self.get_model(model_type, version)
        if target_model:
            target_model.is_active = True
            # Unload the old model from memory
            if model_type in self.loaded_models and version in self.loaded_models[model_type]:
                del self.loaded_models[model_type][version]
            self.save_config()
            logger.info(f"Set {model_type} model version {version} as active")
        else:
            raise ValueError(f"Model {model_type} version {version} not found")
    
    def load_model(self, model_type: str, version: str = None) -> any:
        """Load a model into memory"""
        if version is None:
            metadata = self.get_active_model(model_type)
            if not metadata:
                raise ValueError(f"No active model found for {model_type}")
            version = metadata.version
        else:
            metadata = self.get_model(model_type, version)
            if not metadata:
                raise ValueError(f"Model {model_type} version {version} not found")
        
        # Check if already loaded
        if version in self.loaded_models.get(model_type, {}):
            return self.loaded_models[model_type][version]
        
        # Load the model
        try:
            if model_type == "resnet":
                model = torch.load(
                    metadata.path,
                    map_location=torch.device('cpu')
                )
            elif model_type == "yolo":
                model = YOLO(metadata.path)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            # Cache the loaded model
            if model_type not in self.loaded_models:
                self.loaded_models[model_type] = {}
            self.loaded_models[model_type][version] = model
            
            logger.info(f"Loaded {model_type} model version {version}")
            return model
        except Exception as e:
            logger.error(f"Error loading model {version}: {e}")
            raise
    
    def get_model_for_ab_test(self, model_type: str, request_id: str = None) -> Tuple[any, str]:
        """Get a model for A/B testing based on routing strategy"""
        if not self.ab_test_enabled:
            return self.load_model(model_type), self.get_active_model(model_type).version
        
        # Hash-based routing for consistent user experience
        if request_id:
            hash_value = int(hashlib.md5(request_id.encode()).hexdigest(), 16)
            models = self.models.get(model_type, [])
            
            # Calculate cumulative ratios
            cumulative_ratio = 0.0
            for model in models:
                if model.ab_test_ratio > 0:
                    cumulative_ratio += model.ab_test_ratio
                    if (hash_value % 100) < (cumulative_ratio * 100):
                        return self.load_model(model_type, model.version), model.version
        
        # Fallback to active model
        return self.load_model(model_type), self.get_active_model(model_type).version
    
    def update_metrics(self, model_type: str, version: str, 
                      confidence: float, inference_time: float, 
                      success: bool = True):
        """Update performance metrics for a model"""
        model = self.get_model(model_type, version)
        if not model:
            return
        
        metrics = model.performance_metrics
        metrics["total_requests"] += 1
        
        if success:
            metrics["successful_predictions"] += 1
            # Update average confidence
            old_avg = metrics["avg_confidence"]
            metrics["avg_confidence"] = (
                (old_avg * (metrics["successful_predictions"] - 1) + confidence) /
                metrics["successful_predictions"]
            )
            # Update average inference time
            old_time = metrics["avg_inference_time"]
            metrics["avg_inference_time"] = (
                (old_time * (metrics["successful_predictions"] - 1) + inference_time) /
                metrics["successful_predictions"]
            )
        else:
            metrics["error_count"] += 1
        
        # Check for automatic rollback
        if model.is_active and metrics["total_requests"] > 100:
            success_rate = metrics["successful_predictions"] / metrics["total_requests"]
            if success_rate < self.rollback_threshold:
                logger.warning(f"Model {version} performance below threshold, triggering rollback")
                self._trigger_rollback(model_type)
        
        # Save metrics periodically (every 10 requests)
        if metrics["total_requests"] % 10 == 0:
            self.save_config()
    
    def _trigger_rollback(self, model_type: str):
        """Rollback to previous stable model version"""
        models = self.models.get(model_type, [])
        if len(models) < 2:
            logger.warning("No previous model version available for rollback")
            return
        
        # Find the previous version (not currently active)
        for model in reversed(models):
            if not model.is_active and model.performance_metrics["total_requests"] > 0:
                success_rate = model.performance_metrics["successful_predictions"] / model.performance_metrics["total_requests"]
                if success_rate >= self.rollback_threshold:
                    self.set_active_model(model_type, model.version)
                    logger.info(f"Rolled back to {model_type} version {model.version}")
                    return
        
        logger.warning("No suitable previous model found for rollback")
    
    def list_models(self, model_type: str = None) -> Dict[str, List[dict]]:
        """List all registered models"""
        if model_type:
            return {model_type: [m.to_dict() for m in self.models.get(model_type, [])]}
        return {
            model_type: [m.to_dict() for m in models]
            for model_type, models in self.models.items()
        }
    
    def delete_model(self, model_type: str, version: str):
        """Delete a model version"""
        model = self.get_model(model_type, version)
        if not model:
            raise ValueError(f"Model {model_type} version {version} not found")
        
        if model.is_active:
            raise ValueError("Cannot delete active model. Set another model as active first.")
        
        # Remove from registry
        self.models[model_type] = [m for m in self.models[model_type] if m.version != version]
        
        # Unload from memory if loaded
        if version in self.loaded_models.get(model_type, {}):
            del self.loaded_models[model_type][version]
        
        self.save_config()
        logger.info(f"Deleted {model_type} model version {version}")
    
    def enable_ab_testing(self, enabled: bool = True):
        """Enable or disable A/B testing"""
        self.ab_test_enabled = enabled
        self.save_config()
        logger.info(f"A/B testing {'enabled' if enabled else 'disabled'}")
    
    def set_ab_test_ratio(self, model_type: str, version: str, ratio: float):
        """Set A/B testing ratio for a model version"""
        if not 0.0 <= ratio <= 1.0:
            raise ValueError("Ratio must be between 0.0 and 1.0")
        
        model = self.get_model(model_type, version)
        if not model:
            raise ValueError(f"Model {model_type} version {version} not found")
        
        model.ab_test_ratio = ratio
        self.save_config()
        logger.info(f"Set A/B test ratio for {model_type} version {version} to {ratio}")


# Global registry instance
registry = ModelRegistry()
