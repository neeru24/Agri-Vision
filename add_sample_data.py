"""
Script to add sample analysis data for testing the disease map and dashboard
"""
import sys
import os
from datetime import datetime, timedelta

# Add the app directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import app, db
from models import AnalysisHistory, User

def add_sample_data():
    """Add sample analysis data with location information"""
    with app.app_context():
        # Get the first user (or create one if needed)
        user = User.query.first()
        if not user:
            print("No users found. Please register a user first.")
            return
        
        print(f"Adding sample data for user: {user.email}")
        
        # Sample locations in India (cotton growing regions)
        sample_locations = [
            {
                'latitude': 22.2587,
                'longitude': 71.1924,
                'location_name': 'Gujarat Field 1',
                'region': 'Gujarat'
            },
            {
                'latitude': 30.7333,
                'longitude': 76.7794,
                'location_name': 'Punjab Field 1',
                'region': 'Punjab'
            },
            {
                'latitude': 19.0760,
                'longitude': 72.8777,
                'location_name': 'Maharashtra Field 1',
                'region': 'Maharashtra'
            },
            {
                'latitude': 26.9124,
                'longitude': 75.7873,
                'location_name': 'Rajasthan Field 1',
                'region': 'Rajasthan'
            },
            {
                'latitude': 17.3850,
                'longitude': 78.4867,
                'location_name': 'Telangana Field 1',
                'region': 'Telangana'
            },
            {
                'latitude': 15.9129,
                'longitude': 79.7400,
                'location_name': 'Andhra Pradesh Field 1',
                'region': 'Andhra Pradesh'
            },
            {
                'latitude': 28.7041,
                'longitude': 77.1025,
                'location_name': 'Haryana Field 1',
                'region': 'Haryana'
            },
            {
                'latitude': 20.5937,
                'longitude': 78.9629,
                'location_name': 'Madhya Pradesh Field 1',
                'region': 'Madhya Pradesh'
            }
        ]
        
        # Sample disease results
        disease_types = [
            {'predicted_class': 'healthy', 'confidence': 0.95, 'class_probabilities': {'healthy': 0.95, 'bacterial_blight': 0.02, 'curl_virus': 0.01, 'fusarium_wilt': 0.01, 'target_spot': 0.01}},
            {'predicted_class': 'bacterial_blight', 'confidence': 0.88, 'class_probabilities': {'healthy': 0.05, 'bacterial_blight': 0.88, 'curl_virus': 0.03, 'fusarium_wilt': 0.02, 'target_spot': 0.02}},
            {'predicted_class': 'curl_virus', 'confidence': 0.82, 'class_probabilities': {'healthy': 0.08, 'bacterial_blight': 0.04, 'curl_virus': 0.82, 'fusarium_wilt': 0.03, 'target_spot': 0.03}},
            {'predicted_class': 'fusarium_wilt', 'confidence': 0.91, 'class_probabilities': {'healthy': 0.04, 'bacterial_blight': 0.02, 'curl_virus': 0.01, 'fusarium_wilt': 0.91, 'target_spot': 0.02}},
            {'predicted_class': 'target_spot', 'confidence': 0.85, 'class_probabilities': {'healthy': 0.06, 'bacterial_blight': 0.03, 'curl_virus': 0.02, 'fusarium_wilt': 0.04, 'target_spot': 0.85}}
        ]
        
        # Sample growth results
        growth_stages = [
            {'main_class': 'seedling', 'confidence': 0.92, 'sub_classes': {'seedling': 0.92, 'vegetative': 0.05, 'flowering': 0.02, 'boll': 0.01}},
            {'main_class': 'vegetative', 'confidence': 0.88, 'sub_classes': {'seedling': 0.03, 'vegetative': 0.88, 'flowering': 0.06, 'boll': 0.03}},
            {'main_class': 'flowering', 'confidence': 0.90, 'sub_classes': {'seedling': 0.02, 'vegetative': 0.04, 'flowering': 0.90, 'boll': 0.04}},
            {'main_class': 'boll', 'confidence': 0.87, 'sub_classes': {'seedling': 0.01, 'vegetative': 0.03, 'flowering': 0.09, 'boll': 0.87}},
            {'main_class': 'matured_boll', 'confidence': 0.94, 'sub_classes': {'seedling': 0.01, 'vegetative': 0.02, 'flowering': 0.03, 'boll': 0.94}}
        ]
        
        # Create sample analyses
        count = 0
        for i in range(20):  # Create 20 sample analyses
            location = sample_locations[i % len(sample_locations)]
            disease = disease_types[i % len(disease_types)]
            growth = growth_stages[i % len(growth_stages)]
            
            # Calculate health score based on disease
            if disease['predicted_class'] == 'Healthy':
                health_score = 85 + (i % 10)  # 85-95
            else:
                health_score = 40 + (i % 30)  # 40-70
            
            # Create analysis with random date in last 30 days
            days_ago = i % 30
            created_at = datetime.utcnow() - timedelta(days=days_ago)
            
            analysis = AnalysisHistory(
                user_id=user.id,
                image_path=f'/static/uploads/sample_{i}.jpg',
                disease_result=disease,
                growth_result=growth,
                confidence=disease['confidence'],
                health_score=health_score,
                latitude=location['latitude'],
                longitude=location['longitude'],
                location_name=location['location_name'],
                region=location['region'],
                created_at=created_at
            )
            
            db.session.add(analysis)
            count += 1
        
        db.session.commit()
        
        print(f"\n{'='*60}")
        print(f"Successfully added {count} sample analyses")
        print(f"{'='*60}")
        print(f"Sample data includes:")
        print(f"- 8 different locations across India")
        print(f"- 5 different disease types")
        print(f"- 5 different growth stages")
        print(f"- Analyses from the last 30 days")
        print(f"\nYou can now view the Disease Map and Dashboard!")
        print(f"{'='*60}")

if __name__ == "__main__":
    add_sample_data()
