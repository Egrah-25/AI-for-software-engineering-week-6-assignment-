# crop_yield_predictor.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

class CropYieldPredictor:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.features = ['soil_moisture', 'temperature', 'humidity', 'ph_level',
                        'nitrogen', 'phosphorus', 'potassium', 'rainfall',
                        'solar_radiation', 'growth_stage']
    
    def prepare_training_data(self):
        """Generate synthetic training data"""
        np.random.seed(42)
        n_samples = 1000
        
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
        
        # Simulate crop yield (target variable)
        data['crop_yield'] = (
            data['soil_moisture'] * 0.3 +
            data['temperature'] * 0.2 +
            data['ph_level'] * 0.15 +
            data['nitrogen'] * 0.1 +
            data['phosphorus'] * 0.1 +
            data['potassium'] * 0.1 +
            data['solar_radiation'] * 0.05 +
            np.random.normal(0, 10, n_samples)
        )
        
        return pd.DataFrame(data)
    
    def train(self):
        """Train the yield prediction model"""
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
        
        print(f"Model Performance:")
        print(f"Mean Absolute Error: {mae:.2f}")
        print(f"R² Score: {r2:.4f}")
        
        return self.model
    
    def predict_yield(self, sensor_data):
        """Predict crop yield based on sensor inputs"""
        return self.model.predict([sensor_data])[0]

# Usage example
predictor = CropYieldPredictor()
predictor.train()

# Sample prediction
sample_data = [45, 25, 65, 6.5, 120, 60, 180, 25, 500, 3]
predicted_yield = predictor.predict_yield(sample_data)
print(f"Predicted Crop Yield: {predicted_yield:.2f} kg/hectare")
