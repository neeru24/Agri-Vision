# Model Versioning and A/B Testing Framework

## Overview

The Agri-Vision project now includes a comprehensive Model Versioning and A/B Testing Framework that allows you to:

- Manage multiple model versions for ResNet50 (disease classification) and YOLOv8 (growth stage detection)
- Run A/B tests between different model versions
- Track performance metrics for each model version
- Automatically rollback to previous stable models if performance degrades
- Switch between model versions without code changes

## Architecture

### Components

1. **ModelRegistry** (`model_registry.py`): Core class managing model versions, metadata, and performance tracking
2. **ModelMetadata**: Data class storing model information (version, accuracy, training date, performance metrics)
3. **Admin API Endpoints**: REST API for model management operations
4. **Configuration File**: `model_config.json` (auto-generated) storing model registry state

### Model Metadata Structure

Each model version tracks:

- `version`: Semantic version (e.g., "v1.0", "v1.1")
- `model_type`: "resnet" or "yolo"
- `path`: File path to the model
- `accuracy`: Model accuracy metric
- `training_date`: When the model was trained
- `dataset_version`: Dataset used for training
- `parameters`: Number of model parameters
- `is_active`: Whether this is the currently active model
- `ab_test_ratio`: Traffic allocation for A/B testing (0.0 to 1.0)
- `performance_metrics`: Runtime performance data

### Performance Metrics

Each model version automatically tracks:

- `total_requests`: Total number of inference requests
- `successful_predictions`: Number of successful predictions
- `avg_confidence`: Average prediction confidence
- `avg_inference_time`: Average inference time in seconds
- `error_count`: Number of errors encountered

## Admin API Endpoints

### List All Models

```bash
GET /admin/models?type={resnet|yolo}
```

Response:
```json
{
  "status": "success",
  "models": {
    "resnet": [
      {
        "version": "v1.0",
        "model_type": "resnet",
        "path": "models/cotton_crop_disease_classification/full_resnet50_model.pth",
        "accuracy": 0.9983,
        "training_date": "2024-01-15T10:30:00",
        "dataset_version": "roboflow-v1",
        "parameters": 25600000,
        "is_active": true,
        "ab_test_ratio": 0.0,
        "performance_metrics": {
          "total_requests": 150,
          "successful_predictions": 148,
          "avg_confidence": 0.95,
          "avg_inference_time": 0.15,
          "error_count": 2
        }
      }
    ],
    "yolo": [...]
  },
  "ab_test_enabled": false,
  "rollback_threshold": 0.7
}
```

### Get Active Models

```bash
GET /admin/models/active
```

### Register New Model

```bash
POST /admin/models/register
Content-Type: application/json

{
  "model_type": "resnet",
  "version": "v2.0",
  "path": "models/cotton_crop_disease_classification/resnet_v2.pth",
  "accuracy": 0.9950,
  "dataset_version": "roboflow-v2",
  "parameters": 25600000,
  "is_active": false,
  "ab_test_ratio": 0.3
}
```

### Activate Model Version

```bash
POST /admin/models/activate
Content-Type: application/json

{
  "model_type": "resnet",
  "version": "v2.0"
}
```

### Delete Model Version

```bash
DELETE /admin/models/delete
Content-Type: application/json

{
  "model_type": "resnet",
  "version": "v1.0"
}
```

### Toggle A/B Testing

```bash
POST /admin/models/ab-testing
Content-Type: application/json

{
  "enabled": true
}
```

### Set A/B Test Ratio

```bash
POST /admin/models/ab-ratio
Content-Type: application/json

{
  "model_type": "resnet",
  "version": "v2.0",
  "ratio": 0.3
}
```

### Get Model Metrics

```bash
GET /admin/models/metrics
```

### Set Rollback Threshold

```bash
POST /admin/models/rollback-threshold
Content-Type: application/json

{
  "threshold": 0.75
}
```

## Enhanced Analysis API

The `/api/analyze` endpoint now supports optional model version selection and A/B testing:

```bash
POST /api/analyze
Content-Type: multipart/form-data

file: <image_file>
resnet_version: v2.0        # Optional: specific ResNet version
yolo_version: v1.1         # Optional: specific YOLO version
request_id: user_123        # Optional: for consistent A/B testing routing
```

Response includes model version information:
```json
{
  "status": "success",
  "timestamp": "2024-01-15T10:30:00",
  "results": {
    "disease": {
      "predicted_class": "Healthy",
      "confidence": 0.95,
      "model_version": "v2.0",
      "inference_time": 0.15
    },
    "growth": {
      "main_class": "Matured Cotton Boll",
      "confidence": 0.91,
      "model_version": "v1.0",
      "inference_time": 0.08
    }
  }
}
```

## A/B Testing

### How It Works

1. **Enable A/B Testing**: Use the admin API to enable A/B testing
2. **Set Ratios**: Allocate traffic percentages to different model versions
3. **Consistent Routing**: Provide a `request_id` for consistent user experience
4. **Hash-based Routing**: Requests are routed based on hash of `request_id` for consistency

### Example Setup

```bash
# Enable A/B testing
curl -X POST http://localhost:5000/admin/models/ab-testing \
  -H "Content-Type: application/json" \
  -d '{"enabled": true}'

# Register v2.0 model with 30% traffic
curl -X POST http://localhost:5000/admin/models/register \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "resnet",
    "version": "v2.0",
    "path": "models/resnet_v2.pth",
    "accuracy": 0.9950,
    "is_active": false,
    "ab_test_ratio": 0.3
  }'

# Keep v1.0 as active with 70% traffic (implicit)
```

### Making Requests with A/B Testing

```bash
curl -X POST http://localhost:5000/api/analyze \
  -F "file=@cotton_image.jpg" \
  -F "request_id=user_session_123"
```

The system will automatically route the request to either v1.0 or v2.0 based on the hash of `request_id` and the configured A/B test ratios.

## Automatic Rollback

### How It Works

1. **Monitor Performance**: System tracks success rate for each model version
2. **Threshold Check**: After 100+ requests, checks if success rate < threshold
3. **Automatic Rollback**: If performance degrades, automatically switches to previous stable version
4. **Logging**: All rollback events are logged

### Configuration

```bash
# Set rollback threshold to 75% success rate
curl -X POST http://localhost:5000/admin/models/rollback-threshold \
  -H "Content-Type: application/json" \
  -d '{"threshold": 0.75}'
```

## Usage Examples

### Scenario 1: Deploy New Model Version

```bash
# 1. Train and save new model
# (Your training process saves to models/resnet_v2.pth)

# 2. Register the new model
curl -X POST http://localhost:5000/admin/models/register \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "resnet",
    "version": "v2.0",
    "path": "models/resnet_v2.pth",
    "accuracy": 0.9960,
    "dataset_version": "roboflow-v2",
    "parameters": 25600000,
    "is_active": false
  }'

# 3. Enable A/B testing with 10% traffic to new model
curl -X POST http://localhost:5000/admin/models/ab-testing \
  -H "Content-Type: application/json" \
  -d '{"enabled": true}'

curl -X POST http://localhost:5000/admin/models/ab-ratio \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "resnet",
    "version": "v2.0",
    "ratio": 0.1
  }'

# 4. Monitor performance
curl http://localhost:5000/admin/models/metrics

# 5. Gradually increase traffic if performance is good
curl -X POST http://localhost:5000/admin/models/ab-ratio \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "resnet",
    "version": "v2.0",
    "ratio": 0.5
  }'

# 6. Once confident, make it the active model
curl -X POST http://localhost:5000/admin/models/activate \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "resnet",
    "version": "v2.0"
  }'
```

### Scenario 2: Emergency Rollback

```bash
# If v2.0 is having issues, immediately rollback to v1.0
curl -X POST http://localhost:5000/admin/models/activate \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "resnet",
    "version": "v1.0"
  }'
```

### Scenario 3: Compare Model Performance

```bash
# Get metrics for all versions
curl http://localhost:5000/admin/models/metrics

# Analyze the output to compare:
# - Average confidence
# - Average inference time
# - Success rate
# - Total requests handled
```

## Configuration File

The `model_config.json` file is automatically created and maintained by the system:

```json
{
  "models": {
    "resnet": [
      {
        "version": "v1.0",
        "model_type": "resnet",
        "path": "models/cotton_crop_disease_classification/full_resnet50_model.pth",
        "accuracy": 0.9983,
        "training_date": "2024-01-15T10:30:00",
        "dataset_version": "roboflow-v1",
        "parameters": 25600000,
        "is_active": true,
        "ab_test_ratio": 0.0,
        "performance_metrics": {
          "total_requests": 150,
          "successful_predictions": 148,
          "avg_confidence": 0.95,
          "avg_inference_time": 0.15,
          "error_count": 2
        }
      }
    ],
    "yolo": [...]
  },
  "ab_test_enabled": false,
  "rollback_threshold": 0.7,
  "last_updated": "2024-01-15T10:30:00"
}
```

## Best Practices

1. **Version Naming**: Use semantic versioning (v1.0, v1.1, v2.0)
2. **Gradual Rollout**: Start with small A/B test ratios (5-10%)
3. **Monitor Metrics**: Check performance metrics before full rollout
4. **Keep Backups**: Don't delete old model versions immediately
5. **Document Changes**: Note what changed between versions in training logs
6. **Test Thoroughly**: Validate new models before production deployment
7. **Set Appropriate Thresholds**: Adjust rollback thresholds based on your requirements
8. **Use Request IDs**: Provide consistent request IDs for fair A/B testing

## Security Considerations

- Admin endpoints should be protected with authentication in production
- Model files should be stored securely
- Sensitive model information should not be exposed in logs
- Consider adding rate limiting to admin endpoints
- Implement proper access control for model management operations

## Troubleshooting

### Model Not Loading

- Check that the model file path is correct
- Verify the model file exists and is accessible
- Check logs for specific error messages
- Ensure model dependencies are installed

### A/B Testing Not Working

- Verify A/B testing is enabled: `GET /admin/models`
- Check that models have non-zero `ab_test_ratio`
- Ensure `request_id` is provided in API requests
- Verify hash-based routing is working correctly

### Automatic Rollback Triggering Too Frequently

- Increase the `rollback_threshold` (default: 0.7)
- Check if there are legitimate issues with the model
- Review error logs to understand failure patterns
- Consider disabling automatic rollback temporarily

### Performance Metrics Not Updating

- Check that metrics are being saved periodically
- Verify `model_config.json` is writable
- Check logs for metric tracking errors
- Ensure sufficient disk space for config file

## Future Enhancements

Potential improvements for the framework:

- Web-based admin dashboard for model management
- Integration with MLflow for experiment tracking
- Support for distributed model serving
- Advanced A/B testing strategies (epsilon-greedy, Thompson sampling)
- Real-time alerting for performance degradation
- Model compression and optimization
- Multi-region model deployment
- Canary deployments with gradual traffic shifting
- Integration with monitoring systems (Prometheus, Grafana)
