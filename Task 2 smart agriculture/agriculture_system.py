# task2_smart_agriculture/agriculture_system.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import json
import time
from datetime import datetime

class SmartAgricultureSystem:
    def __init__(self):
        self.yield_predictor = CropYieldPredictor()
        self.sensor_data = []
        self.alert_thresholds = {
            'soil_moisture': {'min': 30, 'max': 70},
            'temperature': {'min': 15, 'max': 35},
            'ph_level': {'min': 5.5, 'max': 7.0},
            'humidity': {'min': 40, 'max': 80}
        }
    
    def simulate_sensor_data(self, num_readings=100):
        """Simulate real-time sensor data"""
        np.random.seed(42)
        
        for i in range(num_readings):
            sensor_reading = {
                'timestamp': datetime.now().isoformat(),
                'sensor_id': f"sensor_{i%10}",
                'soil_moisture': np.random.uniform(20, 80),
                'temperature': np.random.uniform(10, 40),
                'humidity': np.random.uniform(30, 90),
                'ph_level': np.random.uniform(5.0, 8.0),
                'nitrogen': np.random.uniform(50, 200),
                'phosphorus': np.random.uniform(20, 100),
                'potassium': np.random.uniform(100, 300),
                'rainfall': np.random.uniform(0, 50),
                'solar_radiation': np.random.uniform(200, 800)
            }
            self.sensor_data.append(sensor_reading)
            
            # Check for alerts
            self.check_alerts(sensor_reading)
            
            # Simulate time delay
            time.sleep(0.1)
        
        return self.sensor_data
    
    def check_alerts(self, sensor_data):
        """Check sensor data against thresholds and generate alerts"""
        alerts = []
        
        if sensor_data['soil_moisture'] < self.alert_thresholds['soil_moisture']['min']:
            alerts.append(f"LOW SOIL MOISTURE: {sensor_data['soil_moisture']:.1f}%")
        
        if sensor_data['soil_moisture'] > self.alert_thresholds['soil_moisture']['max']:
            alerts.append(f"HIGH SOIL MOISTURE: {sensor_data['soil_moisture']:.1f}%")
        
        if sensor_data['temperature'] < self.alert_thresholds['temperature']['min']:
            alerts.append(f"LOW TEMPERATURE: {sensor_data['temperature']:.1f}°C")
        
        if sensor_data['temperature'] > self.alert_thresholds['temperature']['max']:
            alerts.append(f"HIGH TEMPERATURE: {sensor_data['temperature']:.1f}°C")
        
        if sensor_data['ph_level'] < self.alert_thresholds['ph_level']['min']:
            alerts.append(f"LOW pH LEVEL: {sensor_data['ph_level']:.1f}")
        
        if sensor_data['ph_level'] > self.alert_thresholds['ph_level']['max']:
            alerts.append(f"HIGH pH LEVEL: {sensor_data['ph_level']:.1f}")
        
        if alerts:
            print(f"🚨 ALERTS at {sensor_data['timestamp']}:")
            for alert in alerts:
                print(f"   - {alert}")
            print()
    
    def generate_yield_predictions(self):
        """Generate yield predictions based on sensor data"""
        if not self.sensor_data:
            print("No sensor data available. Please run simulate_sensor_data() first.")
            return
        
        # Use latest sensor reading for prediction
        latest_data = self.sensor_data[-1]
        
        features = [
            latest_data['soil_moisture'],
            latest_data['temperature'],
            latest_data['humidity'],
            latest_data['ph_level'],
            latest_data['nitrogen'],
            latest_data['phosphorus'],
            latest_data['potassium'],
            latest_data['rainfall'],
            latest_data['solar_radiation'],
            3  # growth_stage (assuming mid-growth)
        ]
        
        predicted_yield = self.yield_predictor.predict_yield(features)
        
        print("🌱 CROP YIELD PREDICTION REPORT")
        print("=" * 40)
        print(f"Timestamp: {latest_data['timestamp']}")
        print(f"Predicted Yield: {predicted_yield:.2f} kg/hectare")
        print(f"Soil Moisture: {latest_data['soil_moisture']:.1f}%")
        print(f"Temperature: {latest_data['temperature']:.1f}°C")
        print(f"pH Level: {latest_data['ph_level']:.1f}")
        print(f"Nutrients - N:{latest_data['nitrogen']:.1f}, P:{latest_data['phosphorus']:.1f}, K:{latest_data['potassium']:.1f}")
        print("=" * 40)
        
        return predicted_yield
    
    def generate_recommendations(self):
        """Generate farming recommendations based on sensor data"""
        if not self.sensor_data:
            return []
        
        latest_data = self.sensor_data[-1]
        recommendations = []
        
        # Irrigation recommendations
        if latest_data['soil_moisture'] < 40:
            recommendations.append("💧 IRRIGATION: Increase watering frequency")
        elif latest_data['soil_moisture'] > 65:
            recommendations.append("💧 IRRIGATION: Reduce watering to prevent waterlogging")
        
        # Nutrient recommendations
        if latest_data['nitrogen'] < 100:
            recommendations.append("🌿 NUTRIENTS: Apply nitrogen-rich fertilizer")
        if latest_data['phosphorus'] < 50:
            recommendations.append("🌿 NUTRIENTS: Add phosphorus fertilizer")
        if latest_data['potassium'] < 150:
            recommendations.append("🌿 NUTRIENTS: Supplement with potassium")
        
        # pH recommendations
        if latest_data['ph_level'] < 6.0:
            recommendations.append("🔄 SOIL: Add lime to increase pH level")
        elif latest_data['ph_level'] > 7.5:
            recommendations.append("🔄 SOIL: Add sulfur to decrease pH level")
        
        return recommendations

class CropYieldPredictor:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.features = ['soil_moisture', 'temperature', 'humidity', 'ph_level',
                        'nitrogen', 'phosphorus', 'potassium', 'rainfall',
                        'solar_radiation', 'growth_stage']
        self.is_trained = False
    
    def prepare_training_data(self, n_samples=1000):
        """Generate synthetic training data for crop yield prediction"""
        np.random.seed(42)
        
        data = {
            'soil_moisture': np.random.uniform(10, 80, n_samples),
            'temperature': np.random.uniform(15, 35, n_samples),
            'humidity': np.random.uniform(30, 90, n_samples),
            'ph_level': np.random.uniform(5.5, 7.5, n_samples),
            'nitrogen': np.random.uniform(50, 200, n_samples),
            'phosphorus': np.random.uniform(20, 100, n_samples),
            'potassium': np.random.uniform(100, 300, n_samples),
            'rainfall': np.random.uniform(0, 50, n_samples),
            'solar_radiation': np.random.uniform(200, 800, n_samples),
            'growth_stage': np.random.randint(1, 5, n_samples)
        }
        
        # Simulate crop yield based on optimal conditions
        data['crop_yield'] = (
            data['soil_moisture'] * 0.3 * np.where(
                (data['soil_moisture'] >= 40) & (data['soil_moisture'] <= 60), 1.2, 0.8
            ) +
            data['temperature'] * 0.2 * np.where(
                (data['temperature'] >= 20) & (data['temperature'] <= 30), 1.3, 0.7
            ) +
            data['ph_level'] * 0.15 * np.where(
                (data['ph_level'] >= 6.0) & (data['ph_level'] <= 7.0), 1.4, 0.6
            ) +
            data['nitrogen'] * 0.1 +
            data['phosphorus'] * 0.1 +
            data['potassium'] * 0.1 +
            data['solar_radiation'] * 0.05 +
            np.random.normal(0, 15, n_samples)
        )
        
        return pd.DataFrame(data)
    
    def train(self):
        """Train the crop yield prediction model"""
        print("Training crop yield prediction model...")
        df = self.prepare_training_data()
        X = df[self.features]
        y = df['crop_yield']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print("✅ Model training completed!")
        print(f"📊 Model Performance:")
        print(f"   - Mean Absolute Error: {mae:.2f}")
        print(f"   - R² Score: {r2:.4f}")
        
        self.is_trained = True
        return self.model
    
    def predict_yield(self, sensor_data):
        """Predict crop yield based on sensor inputs"""
        if not self.is_trained:
            print("Model not trained. Training now...")
            self.train()
        
        return self.model.predict([sensor_data])[0]

def main():
    """Main function to demonstrate the smart agriculture system"""
    print("🚜 SMART AGRICULTURE SYSTEM INITIALIZATION")
    print("=" * 50)
    
    # Initialize system
    agri_system = SmartAgricultureSystem()
    
    # Train the model
    agri_system.yield_predictor.train()
    
    # Simulate sensor data collection
    print("\n📡 SIMULATING SENSOR DATA COLLECTION...")
    sensor_data = agri_system.simulate_sensor_data(num_readings=50)
    print(f"✅ Collected {len(sensor_data)} sensor readings")
    
    # Generate yield prediction
    print("\n🔮 GENERATING YIELD PREDICTION...")
    predicted_yield = agri_system.generate_yield_predictions()
    
    # Generate recommendations
    print("\n💡 GENERATING RECOMMENDATIONS...")
    recommendations = agri_system.generate_recommendations()
    if recommendations:
        for rec in recommendations:
            print(f"   - {rec}")
    else:
        print("   - All parameters within optimal ranges. No action needed.")
    
    # System summary
    print("\n📈 SYSTEM SUMMARY")
    print("=" * 50)
    print(f"Total Sensor Readings: {len(agri_system.sensor_data)}")
    print(f"Predicted Yield: {predicted_yield:.2f} kg/hectare")
    print(f"Recommendations Generated: {len(recommendations)}")
    print("System Status: ✅ OPERATIONAL")

if __name__ == "__main__":
    main()
