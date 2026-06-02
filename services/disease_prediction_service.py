"""
Disease Prediction Service
Uses weather data and historical patterns to predict disease outbreaks
"""
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import numpy as np

logger = logging.getLogger(__name__)


class DiseasePredictor:
    """ML-based disease prediction using weather data"""
    
    def __init__(self):
        # Disease-specific weather thresholds (based on agricultural research)
        # Keys match the disease class names used in app.py for consistency
        self.disease_thresholds = self._build_thresholds()

    @staticmethod
    def _build_thresholds():
        return {
            'Aphids': {
                'temp_min': 20, 'temp_max': 30,
                'humidity_min': 50, 'rainfall_min': 0,
                'temp_weight': 0.4, 'humidity_weight': 0.3, 'rainfall_weight': 0.3
            },
            'Army worm': {
                'temp_min': 25, 'temp_max': 35,
                'humidity_min': 55, 'rainfall_min': 0,
                'temp_weight': 0.4, 'humidity_weight': 0.3, 'rainfall_weight': 0.3
            },
            'Bacterial blight': {
                'temp_min': 25, 'temp_max': 35,
                'humidity_min': 70, 'rainfall_min': 5,
                'temp_weight': 0.3, 'humidity_weight': 0.4, 'rainfall_weight': 0.3
            },
            'Cotton Boll Rot': {
                'temp_min': 25, 'temp_max': 32,
                'humidity_min': 80, 'rainfall_min': 10,
                'temp_weight': 0.3, 'humidity_weight': 0.4, 'rainfall_weight': 0.3
            },
            'Green Cotton Boll': {
                'temp_min': 22, 'temp_max': 35,
                'humidity_min': 50, 'rainfall_min': 0,
                'temp_weight': 0.3, 'humidity_weight': 0.4, 'rainfall_weight': 0.3
            },
            'Healthy': {
                'temp_min': 0, 'temp_max': 100,
                'humidity_min': 0, 'rainfall_min': 0,
                'temp_weight': 0.0, 'humidity_weight': 0.0, 'rainfall_weight': 0.0
            },
            'Powdery mildew': {
                'temp_min': 20, 'temp_max': 28,
                'humidity_min': 60, 'rainfall_min': 0,
                'temp_weight': 0.3, 'humidity_weight': 0.5, 'rainfall_weight': 0.2
            },
            'Target Spot': {
                'temp_min': 22, 'temp_max': 32,
                'humidity_min': 70, 'rainfall_min': 5,
                'temp_weight': 0.3, 'humidity_weight': 0.4, 'rainfall_weight': 0.3
            },
            'Fusarium wilt': {
                'temp_min': 25, 'temp_max': 30,
                'humidity_min': 60, 'rainfall_min': 0,
                'temp_weight': 0.4, 'humidity_weight': 0.3, 'rainfall_weight': 0.3
            },
            'Verticillium wilt': {
                'temp_min': 20, 'temp_max': 25,
                'humidity_min': 60, 'rainfall_min': 0,
                'temp_weight': 0.4, 'humidity_weight': 0.3, 'rainfall_weight': 0.3
            },
            'Cotton root rot': {
                'temp_min': 28, 'temp_max': 35,
                'humidity_min': 50, 'rainfall_min': 10,
                'temp_weight': 0.3, 'humidity_weight': 0.3, 'rainfall_weight': 0.4
            },
            'Alternaria leaf spot': {
                'temp_min': 20, 'temp_max': 30,
                'humidity_min': 70, 'rainfall_min': 5,
                'temp_weight': 0.3, 'humidity_weight': 0.4, 'rainfall_weight': 0.3
            },
            'Red leaf spot': {
                'temp_min': 22, 'temp_max': 30,
                'humidity_min': 65, 'rainfall_min': 5,
                'temp_weight': 0.3, 'humidity_weight': 0.4, 'rainfall_weight': 0.3
            },
        }
    
    @staticmethod
    def _normalize_name(name: str) -> str:
        """Convert a disease name from any common format to the canonical form used in thresholds."""
        if not name:
            return name
        normalized = name.strip().replace('_', ' ').replace('-', ' ')
        return ' '.join(
            word if word.islower() and len(word) <= 2 else word.capitalize()
            for word in normalized.split()
        )

    @staticmethod
    def _fuzzy_match(name: str, known_keys: set) -> Optional[str]:
        """
        Try to match a disease name against known keys with flexible matching.
        Returns the matched key or None.
        """
        canonical = DiseasePredictor._normalize_name(name)
        if canonical in known_keys:
            return canonical
        lower_name = canonical.lower()
        for key in known_keys:
            if key.lower() == lower_name:
                return key
            if key.replace(' ', '').lower() == canonical.replace(' ', '').lower():
                return key
        return None

    def calculate_risk_score(self, weather_data: Dict, disease_name: str) -> float:
        """
        Calculate disease risk score (0-100) based on weather conditions
        """
        matched_key = self._fuzzy_match(disease_name, set(self.disease_thresholds.keys()))
        if matched_key is None:
            return 0
        
        thresholds = self.disease_thresholds[matched_key]
        
        temp = weather_data.get('temperature_avg', weather_data.get('temperature', 0))
        humidity = weather_data.get('humidity', 0)
        rainfall = weather_data.get('rainfall', 0)
        
        # Calculate individual factor scores
        temp_score = self._calculate_factor_score(
            temp, 
            thresholds['temp_min'], 
            thresholds['temp_max']
        )
        
        humidity_score = self._calculate_factor_score(
            humidity,
            thresholds['humidity_min'],
            100  # Max humidity
        )
        
        rainfall_score = self._calculate_factor_score(
            rainfall,
            thresholds['rainfall_min'],
            50  # High rainfall threshold
        )
        
        # Weighted average
        risk_score = (
            temp_score * thresholds['temp_weight'] +
            humidity_score * thresholds['humidity_weight'] +
            rainfall_score * thresholds['rainfall_weight']
        ) * 100
        
        return min(max(risk_score, 0), 100)
    
    def _calculate_factor_score(self, value: float, min_threshold: float, max_threshold: float) -> float:
        """
        Calculate normalized score for a single factor (0-1)
        """
        if value < min_threshold:
            # Below minimum - linear decrease
            return max(0, value / min_threshold * 0.5)
        elif value > max_threshold:
            # Above maximum - cap at 1
            return 1.0
        else:
            # Within optimal range
            return 0.8 + (value - min_threshold) / (max_threshold - min_threshold) * 0.2
    
    def get_risk_level(self, risk_score: float) -> str:
        """Convert risk score to risk level"""
        if risk_score < 25:
            return 'low'
        elif risk_score < 50:
            return 'moderate'
        elif risk_score < 75:
            return 'high'
        else:
            return 'severe'
    
    def predict_disease_risk(self, weather_forecast: List[Dict], disease_name: str) -> List[Dict]:
        """
        Predict disease risk for multiple days based on weather forecast
        """
        predictions = []
        
        for day_data in weather_forecast:
            risk_score = self.calculate_risk_score(day_data, disease_name)
            risk_level = self.get_risk_level(risk_score)
            
            predictions.append({
                'date': day_data.get('date'),
                'risk_score': round(risk_score, 1),
                'risk_level': risk_level,
                'weather': {
                    'temperature_avg': day_data.get('temperature_avg'),
                    'humidity': day_data.get('humidity'),
                    'rainfall': day_data.get('rainfall')
                }
            })
        
        return predictions
    
    def get_all_disease_predictions(self, weather_forecast: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Get predictions for all diseases
        """
        predictions = {}
        
        for disease_name in self.disease_thresholds.keys():
            predictions[disease_name] = self.predict_disease_risk(weather_forecast, disease_name)
        
        return predictions
    
    def get_high_risk_days(self, predictions: List[Dict], threshold: float = 60) -> List[Dict]:
        """
        Filter predictions to show only high-risk days
        """
        return [p for p in predictions if p['risk_score'] >= threshold]
    
    def generate_recommendations(self, disease_name: str, risk_level: str) -> List[str]:
        """
        Generate preventive recommendations based on disease and risk level
        """
        recommendations = []
        display_name = self._normalize_name(disease_name)
        
        if risk_level in ['high', 'severe']:
            recommendations.append(f"Immediate action required for {display_name}.")
            recommendations.append("Apply preventive fungicides if conditions persist.")
            recommendations.append("Monitor fields daily for early symptoms.")
            recommendations.append("Consider drainage improvements if rainfall is high.")
        elif risk_level == 'moderate':
            recommendations.append(f"Monitor for {display_name} development.")
            recommendations.append("Ensure proper field ventilation and spacing.")
            recommendations.append("Avoid overhead irrigation during high humidity.")
            recommendations.append("Have treatment options ready if conditions worsen.")
        else:
            recommendations.append(f"Low risk for {display_name}.")
            recommendations.append("Continue regular monitoring.")
            recommendations.append("Maintain good cultural practices.")
        
        return recommendations


class HistoricalPatternAnalyzer:
    """Analyze historical disease patterns for ML learning"""
    
    def __init__(self):
        self.seasonal_patterns = {}
        self.regional_patterns = {}
        self.weather_correlations = {}
        self.trained = False
    
    def train(self, occurrences: List[Dict], weather_data: List[Dict] = None):
        """
        Train the analyzer on historical data
        """
        if not occurrences:
            logger.warning("No historical data provided for training")
            return
        
        # Analyze seasonal patterns
        self.seasonal_patterns = self.analyze_seasonal_patterns(occurrences)
        
        # Analyze regional patterns
        self.regional_patterns = self.analyze_regional_patterns(occurrences)
        
        # Analyze weather correlations if weather data is provided
        if weather_data:
            self.weather_correlations = self.analyze_weather_correlations(occurrences, weather_data)
        
        self.trained = True
        logger.info(f"Trained HistoricalPatternAnalyzer on {len(occurrences)} occurrences")
    
    def analyze_seasonal_patterns(self, occurrences: List[Dict]) -> Dict[str, Dict]:
        """
        Analyze seasonal patterns in disease occurrences
        Returns disease-specific seasonal risk by month
        """
        seasonal_data = {}
        
        for occurrence in occurrences:
            disease_name = occurrence.get('disease_name', 'unknown')
            date_str = occurrence.get('occurrence_date')
            
            if date_str:
                try:
                    date = datetime.strptime(date_str, '%Y-%m-%d')
                    month = date.month
                    
                    if disease_name not in seasonal_data:
                        seasonal_data[disease_name] = {i: 0 for i in range(1, 13)}
                    
                    seasonal_data[disease_name][month] += 1
                except ValueError:
                    continue
        
        # Normalize to percentages
        for disease in seasonal_data:
            total = sum(seasonal_data[disease].values())
            if total > 0:
                for month in seasonal_data[disease]:
                    seasonal_data[disease][month] = (seasonal_data[disease][month] / total) * 100
        
        return seasonal_data
    
    def analyze_regional_patterns(self, occurrences: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Analyze disease patterns by region
        """
        regional_data = {}
        
        for occurrence in occurrences:
            location = occurrence.get('location_name', 'unknown')
            disease_name = occurrence.get('disease_name', 'unknown')
            
            if location not in regional_data:
                regional_data[location] = {}
            
            if disease_name not in regional_data[location]:
                regional_data[location][disease_name] = 0
            
            regional_data[location][disease_name] += 1
        
        return regional_data
    
    def analyze_weather_correlations(self, occurrences: List[Dict], weather_data: List[Dict]) -> Dict[str, Dict]:
        """
        Analyze correlations between weather conditions and disease occurrences
        """
        correlations = {}
        
        if not weather_data:
            logger.warning("No weather data provided for correlation analysis")
            return correlations
        
        # Create a lookup for weather data by date and location
        weather_lookup = {}
        for weather in weather_data:
            weather_date = weather.get('date')
            # Handle both string and date formats
            if isinstance(weather_date, str):
                date_key = weather_date
            else:
                date_key = str(weather_date) if weather_date else None
            
            if date_key:
                location = weather.get('location_name', 'unknown')
                weather_lookup[(date_key, location)] = weather
        
        # Group occurrences by disease
        disease_occurrences = {}
        for occurrence in occurrences:
            disease_name = occurrence.get('disease_name', 'unknown')
            if disease_name not in disease_occurrences:
                disease_occurrences[disease_name] = []
            disease_occurrences[disease_name].append(occurrence)
        
        # Calculate correlations for each disease
        for disease_name, disease_occs in disease_occurrences.items():
            weather_conditions = []
            
            for occurrence in disease_occs:
                date_str = occurrence.get('occurrence_date')
                location = occurrence.get('location_name', 'unknown')
                
                if date_str and location:
                    # Try exact match first
                    key = (date_str, location)
                    if key not in weather_lookup:
                        # Try with just the date part if it's a datetime string
                        if isinstance(date_str, str) and 'T' in date_str:
                            date_only = date_str.split('T')[0]
                            key = (date_only, location)
                    
                    if key in weather_lookup:
                        weather = weather_lookup[key]
                        weather_conditions.append({
                            'temperature_avg': weather.get('temperature_avg', 0),
                            'humidity': weather.get('humidity', 0),
                            'rainfall': weather.get('rainfall', 0),
                            'wind_speed': weather.get('wind_speed', 0)
                        })
            
            if weather_conditions:
                correlations[disease_name] = {
                    'avg_temperature': np.mean([w['temperature_avg'] for w in weather_conditions]),
                    'avg_humidity': np.mean([w['humidity'] for w in weather_conditions]),
                    'avg_rainfall': np.mean([w['rainfall'] for w in weather_conditions]),
                    'avg_wind_speed': np.mean([w['wind_speed'] for w in weather_conditions]),
                    'sample_size': len(weather_conditions)
                }
        
        logger.info(f"Calculated weather correlations for {len(correlations)} diseases")
        return correlations
    
    def predict_from_history(self, current_location: str, current_month: int, 
                            weather_data: Dict = None) -> Dict[str, float]:
        """
        Predict disease risk based on historical patterns
        Combines seasonal, regional, and weather-based predictions
        """
        if not self.trained:
            logger.warning("Analyzer not trained. Call train() first.")
            return {}
        
        predictions = {}
        
        # Get seasonal risk
        for disease_name, monthly_data in self.seasonal_patterns.items():
            seasonal_risk = monthly_data.get(current_month, 0)
            
            # Get regional risk
            regional_risk = 0
            if current_location in self.regional_patterns:
                total_regional = sum(self.regional_patterns[current_location].values())
                if total_regional > 0:
                    disease_count = self.regional_patterns[current_location].get(disease_name, 0)
                    regional_risk = (disease_count / total_regional) * 100
            
            # Get weather-based risk
            weather_risk = 0
            if weather_data and disease_name in self.weather_correlations:
                weather_risk = self._calculate_weather_risk(
                    weather_data, 
                    self.weather_correlations[disease_name]
                )
            
            # Combine risks with weights
            combined_risk = (
                seasonal_risk * 0.4 +
                regional_risk * 0.3 +
                weather_risk * 0.3
            )
            
            predictions[disease_name] = min(combined_risk, 100)
        
        return predictions
    
    def _calculate_weather_risk(self, current_weather: Dict, historical_weather: Dict) -> float:
        """
        Calculate weather-based risk by comparing current conditions to historical averages
        """
        risk_score = 0
        
        # Temperature similarity
        temp_diff = abs(current_weather.get('temperature_avg', 0) - historical_weather.get('avg_temperature', 0))
        temp_risk = max(0, 100 - temp_diff * 5)  # Lower difference = higher risk
        
        # Humidity similarity
        humidity_diff = abs(current_weather.get('humidity', 0) - historical_weather.get('avg_humidity', 0))
        humidity_risk = max(0, 100 - humidity_diff * 2)
        
        # Rainfall similarity
        rainfall_diff = abs(current_weather.get('rainfall', 0) - historical_weather.get('avg_rainfall', 0))
        rainfall_risk = max(0, 100 - rainfall_diff * 3)
        
        # Weighted average
        risk_score = (temp_risk * 0.4 + humidity_risk * 0.4 + rainfall_risk * 0.2)
        
        return risk_score
    
    def get_peak_season(self, disease_name: str) -> Optional[Dict]:
        """
        Get peak season for a specific disease
        """
        if not self.trained:
            return None
        matched_key = DiseasePredictor._fuzzy_match(disease_name, set(self.seasonal_patterns.keys()))
        if matched_key is None:
            return None
        
        monthly_data = self.seasonal_patterns[matched_key]
        peak_month = max(monthly_data, key=monthly_data.get)
        
        return {
            'month': peak_month,
            'month_name': datetime(2024, peak_month, 1).strftime('%B'),
            'risk_percentage': monthly_data[peak_month]
        }
    
    def get_regional_risk_ranking(self, location: str) -> List[Dict]:
        """
        Get disease risk ranking for a specific region
        """
        if not self.trained or location not in self.regional_patterns:
            return []
        
        regional_data = self.regional_patterns[location]
        total = sum(regional_data.values())
        
        ranking = []
        for disease_name, count in regional_data.items():
            ranking.append({
                'disease_name': disease_name,
                'occurrence_count': count,
                'percentage': (count / total) * 100 if total > 0 else 0
            })
        
        # Sort by percentage descending
        ranking.sort(key=lambda x: x['percentage'], reverse=True)
        
        return ranking
    
    def get_disease_trend(self, disease_name: str, months: int = 12) -> Dict:
        """
        Get disease trend over specified number of months
        """
        if not self.trained:
            return {}
        matched_key = DiseasePredictor._fuzzy_match(disease_name, set(self.seasonal_patterns.keys()))
        if matched_key is None:
            return {}
        
        monthly_data = self.seasonal_patterns[matched_key]
        
        # Calculate trend (increasing, decreasing, stable)
        recent_months = list(monthly_data.values())[-months:]
        if len(recent_months) < 2:
            return {'trend': 'insufficient_data'}
        
        first_half_avg = np.mean(recent_months[:len(recent_months)//2])
        second_half_avg = np.mean(recent_months[len(recent_months)//2:])
        
        if second_half_avg > first_half_avg * 1.2:
            trend = 'increasing'
        elif second_half_avg < first_half_avg * 0.8:
            trend = 'decreasing'
        else:
            trend = 'stable'
        
        # Calculate percentage change
        change_percent = 0
        if first_half_avg > 0:
            change_percent = ((second_half_avg - first_half_avg) / first_half_avg) * 100
        
        return {
            'trend': trend,
            'average_risk': np.mean(recent_months),
            'peak_month': max(monthly_data, key=monthly_data.get),
            'peak_risk': monthly_data[max(monthly_data, key=monthly_data.get)],
            'change_percent': round(change_percent, 2)
        }
    
    def get_insights(self) -> Dict:
        """
        Get comprehensive insights from historical analysis
        """
        if not self.trained:
            return {'error': 'Analyzer not trained'}
        
        insights = {
            'total_diseases_analyzed': len(self.seasonal_patterns),
            'total_regions_analyzed': len(self.regional_patterns),
            'diseases_with_weather_data': len(self.weather_correlations),
            'peak_seasons': {},
            'regional_hotspots': [],
            'weather_correlations': {}
        }
        
        # Get peak seasons for all diseases with monthly risk data
        for disease_name in self.seasonal_patterns:
            peak = self.get_peak_season(disease_name)
            if peak:
                insights['peak_seasons'][disease_name] = {
                    'month': peak['month'],
                    'month_name': peak['month_name'],
                    'risk_percentage': peak['risk_percentage'],
                    'monthly_risk': self.seasonal_patterns[disease_name]
                }
        
        # Get regional hotspots as a list sorted by total occurrences
        region_totals = []
        for location in self.regional_patterns:
            total_occurrences = sum(self.regional_patterns[location].values())
            region_totals.append({
                'location': location,
                'occurrence_count': total_occurrences
            })
        
        # Sort by occurrence count descending
        region_totals.sort(key=lambda x: x['occurrence_count'], reverse=True)
        insights['regional_hotspots'] = region_totals
        
        # Get most common disease
        disease_totals = {}
        for location in self.regional_patterns:
            for disease_name, count in self.regional_patterns[location].items():
                disease_totals[disease_name] = disease_totals.get(disease_name, 0) + count
        
        if disease_totals:
            insights['most_common_disease'] = max(disease_totals, key=disease_totals.get)
        
        # Get high risk region
        if region_totals:
            insights['high_risk_region'] = region_totals[0]['location']
        
        # Format weather correlations for display
        for disease_name, correlation in self.weather_correlations.items():
            insights['weather_correlations'][disease_name] = {
                'temperature': f"{correlation.get('avg_temperature', 0):.1f}°C avg",
                'humidity': f"{correlation.get('avg_humidity', 0):.1f}% avg",
                'rainfall': f"{correlation.get('avg_rainfall', 0):.1f}mm avg"
            }
        
        return insights
