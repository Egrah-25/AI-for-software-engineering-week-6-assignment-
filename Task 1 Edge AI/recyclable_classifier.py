# recyclable_classifier.py
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

# Data preparation (using CIFAR-10 as proxy for recyclable items)
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# Map to recyclable categories: plastic, paper, glass, metal, organic
class_names = ['plastic', 'paper', 'glass', 'metal', 'organic', 
               'other1', 'other2', 'other3', 'other4', 'other5']

# Normalize pixel values
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Build lightweight CNN model
def create_lightweight_model():
    model = keras.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])
    return model

# Create and compile model
model = create_lightweight_model()
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(x_train, y_train, 
                    epochs=10, 
                    batch_size=32,
                    validation_split=0.2)

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")

# Convert to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Save the TensorFlow Lite model
with open('recyclable_classifier.tflite', 'wb') as f:
    f.write(tflite_model)

print("Model converted to TensorFlow Lite successfully!")
