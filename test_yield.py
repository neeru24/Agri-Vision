import sys
sys.path.insert(0, '.')
from services.yield_service import estimate_yield

print("=" * 50)
print("TEST 1 — Cotton, XGBoost path")
print("=" * 50)
disease = {"predicted_class": "Bacterial Blight", "confidence": 0.87, "health_score": 45.0}
growth  = {"main_class": "Matured Cotton Boll"}
weather = {"temperature": 32, "humidity": 65, "precipitation": 5}

result = estimate_yield(disease, growth, weather=weather, field_acres=2.5, crop_type="cotton")
print(f"  Predicted yield : {result.get('predicted_yield_kg_acre')} kg/acre")
print(f"  Model used      : {result.get('model_used')}")
print(f"  Yield range     : {result.get('yield_min_acre')} – {result.get('yield_max_acre')} kg/acre")

print()
print("=" * 50)
print("TEST 2 — Tomato, healthy crop")
print("=" * 50)
disease2 = {"predicted_class": "Healthy", "confidence": 0.95, "health_score": 90.0}
growth2  = {"main_class": "Flowering Initiation"}

result2 = estimate_yield(disease2, growth2, weather=weather, field_acres=1.0, crop_type="tomato")
print(f"  Predicted yield : {result2.get('predicted_yield_kg_acre')} kg/acre")
print(f"  Model used      : {result2.get('model_used')}")

print()
print("=" * 50)
print("TEST 3 — Fallback (wrong crop_type)")
print("=" * 50)
result3 = estimate_yield(disease, growth, weather=weather, field_acres=1.0, crop_type="banana")
print(f"  Model used      : {result3.get('model_used')}")
print(f"  Predicted yield : {result3.get('predicted_yield_kg_acre')}")

print()
print("All tests done.")
