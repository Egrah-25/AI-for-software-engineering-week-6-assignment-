# tflite_testing.py
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

class TFLiteTester:
    def __init__(self, model_path):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        # Get input and output tensors
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        self.class_names = ['plastic', 'paper', 'glass', 'metal', 'organic', 
                           'other1', 'other2', 'other3', 'other4', 'other5']
    
    def preprocess_image(self, image_path):
        """Preprocess image for TFLite model"""
        image = Image.open(image_path).convert('RGB')
        image = image.resize((32, 32))
        image_array = np.array(image, dtype=np.float32) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        return image_array
    
    def predict(self, image_path):
        """Make prediction using TFLite model"""
        input_data = self.preprocess_image(image_path)
        
        # Set input tensor
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        
        # Run inference
        self.interpreter.invoke()
        
        # Get prediction results
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        prediction = np.argmax(output_data[0])
        confidence = output_data[0][prediction]
        
        return self.class_names[prediction], confidence

# Test the TFLite model
tester = TFLiteTester('recyclable_classifier.tflite')

# Performance metrics
def evaluate_tflite_model(tester, x_test, y_test, num_samples=1000):
    correct = 0
    total = min(num_samples, len(x_test))
    
    for i in range(total):
        # Simulate image processing (in real scenario, you'd use actual images)
        input_data = x_test[i:i+1]
        
        tester.interpreter.set_tensor(tester.input_details[0]['index'], input_data)
        tester.interpreter.invoke()
        output_data = tester.interpreter.get_tensor(tester.output_details[0]['index'])
        
        prediction = np.argmax(output_data[0])
        if prediction == y_test[i][0]:
            correct += 1
    
    accuracy = correct / total
    print(f"TFLite Model Accuracy: {accuracy:.4f}")
    return accuracy

# Run evaluation
tflite_accuracy = evaluate_tflite_model(tester, x_test, y_test)
