"""
Agri-Vision Cotton Crop Training Script
Simplified version without complex imports
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import matplotlib.pyplot as plt
import os
import random
import cv2
from sklearn.model_selection import train_test_split

class CottonMultiTaskModel:
    def __init__(self, input_shape=(224, 224, 3), num_phases=4):
        self.input_shape = input_shape
        self.num_phases = num_phases
        
    def create_synthetic_image(self, phase, health):
        """Create synthetic cotton images for training"""
        img = np.ones((224, 224, 3), dtype=np.uint8) * 150
        
        # Phase-specific colors
        phase_colors = [
            (34, 139, 34),    # Green for vegetative
            (255, 255, 0),    # Yellow for flowering
            (255, 165, 0),    # Orange for bursting
            (255, 0, 0)       # Red for harvest ready
        ]
        
        # Draw cotton boll based on phase
        center_x, center_y = 112, 112
        
        if phase == 0:  # Vegetative
            cv2.circle(img, (center_x, center_y), 60, phase_colors[phase], -1)
        elif phase == 1:  # Flowering
            cv2.circle(img, (center_x, center_y), 70, phase_colors[phase], -1)
        elif phase == 2:  # Bursting
            cv2.circle(img, (center_x, center_y), 80, phase_colors[phase], -1)
        else:  # Harvest ready
            cv2.circle(img, (center_x, center_y), 85, phase_colors[phase], -1)
        
        # Add health anomalies
        if health == 1:  # Pink bollworm
            for _ in range(random.randint(3, 8)):
                x = random.randint(50, 174)
                y = random.randint(50, 174)
                cv2.circle(img, (x, y), 3, (255, 192, 203), -1)
        elif health == 2:  # Discoloration
            for _ in range(random.randint(2, 5)):
                x = random.randint(30, 194)
                y = random.randint(30, 194)
                cv2.circle(img, (x, y), random.randint(10, 25), 
                          (random.randint(150, 200), random.randint(100, 150), random.randint(100, 150)), -1)
        
        # Add noise for realism
        noise = np.random.normal(0, 10, (224, 224, 3))
        img = np.clip(img + noise, 0, 255).astype(np.uint8)
        
        return img
    
    def generate_synthetic_data(self, num_samples=200):
        """Generate synthetic dataset"""
        print(f"Generating {num_samples} synthetic cotton images...")
        images = []
        labels = []
        
        for i in range(num_samples):
            phase = random.randint(0, 3)
            health = random.randint(0, 3)
            
            img = self.create_synthetic_image(phase, health)
            images.append(img)
            
            labels.append({
                'phase': phase,
                'health': health,
                'health_score': max(0, 100 - (health * 25) - random.randint(0, 15))
            })
        
        return np.array(images), labels
    
    def build_model(self):
        """Build multi-task CNN model"""
        inputs = keras.Input(shape=self.input_shape)
        
        # Simple CNN architecture (no pre-trained model for simplicity)
        x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(64, (3, 3), activation='relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(128, (3, 3), activation='relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        x = layers.Flatten()(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        
        # Head 1: Phase classification
        phase_output = layers.Dense(self.num_phases, activation='softmax', name='phase_output')(x)
        
        # Head 2: Health classification
        health_output = layers.Dense(4, activation='softmax', name='health_output')(x)
        
        # Head 3: Health score regression
        score_output = layers.Dense(1, activation='sigmoid', name='health_score')(x)
        
        # Create model
        model = Model(
            inputs=inputs,
            outputs=[phase_output, health_output, score_output],
            name='cotton_classifier'
        )
        
        return model
    
    def train_model(self, epochs=10, batch_size=16):
        """Train the model"""
        print("Creating synthetic dataset...")
        images, labels = self.generate_synthetic_data(num_samples=100)
        
        print("Preparing data...")
        # Prepare labels
        phase_labels = np.array([l['phase'] for l in labels])
        health_labels = np.array([l['health'] for l in labels])
        health_scores = np.array([l['health_score'] for l in labels]) / 100.0
        
        # One-hot encode
        phase_labels_onehot = tf.keras.utils.to_categorical(phase_labels, 4)
        health_labels_onehot = tf.keras.utils.to_categorical(health_labels, 4)
        
        # Preprocess images (normalize)
        images = images.astype('float32') / 255.0
        
        # Split data
        X_train, X_val, phase_train, phase_val = train_test_split(
            images, phase_labels_onehot, test_size=0.2, random_state=42
        )
        
        _, _, health_train, health_val = train_test_split(
            images, health_labels_onehot, test_size=0.2, random_state=42
        )
        
        _, _, score_train, score_val = train_test_split(
            images, health_scores, test_size=0.2, random_state=42
        )
        
        print("Building model...")
        model = self.build_model()
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss={
                'phase_output': 'categorical_crossentropy',
                'health_output': 'categorical_crossentropy',
                'health_score': 'mse'
            },
            loss_weights={
                'phase_output': 1.0,
                'health_output': 0.8,
                'health_score': 0.5
            },
            metrics={
                'phase_output': ['accuracy'],
                'health_output': ['accuracy'],
                'health_score': ['mae']
            }
        )
        
        print("Training model...")
        history = model.fit(
            X_train, [phase_train, health_train, score_train],
            validation_data=(X_val, [phase_val, health_val, score_val]),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        
        # Save model
        os.makedirs('models', exist_ok=True)
        model.save('models/cotton_classifier.h5')
        print("✅ Model saved to 'models/cotton_classifier.h5'")
        
        # Plot training history
        self.plot_training_history(history)
        
        return model, history
    
    def plot_training_history(self, history):
        """Plot training metrics"""
        os.makedirs('results', exist_ok=True)
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Phase accuracy
        axes[0, 0].plot(history.history['phase_output_accuracy'])
        axes[0, 0].plot(history.history['val_phase_output_accuracy'])
        axes[0, 0].set_title('Phase Accuracy')
        axes[0, 0].legend(['Train', 'Val'])
        axes[0, 0].grid(True)
        
        # Health accuracy
        axes[0, 1].plot(history.history['health_output_accuracy'])
        axes[0, 1].plot(history.history['val_health_output_accuracy'])
        axes[0, 1].set_title('Health Accuracy')
        axes[0, 1].legend(['Train', 'Val'])
        axes[0, 1].grid(True)
        
        # Health score MAE
        axes[0, 2].plot(history.history['health_score_mae'])
        axes[0, 2].plot(history.history['val_health_score_mae'])
        axes[0, 2].set_title('Health Score MAE')
        axes[0, 2].legend(['Train', 'Val'])
        axes[0, 2].grid(True)
        
        # Losses
        axes[1, 0].plot(history.history['loss'])
        axes[1, 0].plot(history.history['val_loss'])
        axes[1, 0].set_title('Total Loss')
        axes[1, 0].legend(['Train', 'Val'])
        axes[1, 0].grid(True)
        
        axes[1, 1].plot(history.history['phase_output_loss'])
        axes[1, 1].plot(history.history['val_phase_output_loss'])
        axes[1, 1].set_title('Phase Loss')
        axes[1, 1].legend(['Train', 'Val'])
        axes[1, 1].grid(True)
        
        axes[1, 2].plot(history.history['health_output_loss'])
        axes[1, 2].plot(history.history['val_health_output_loss'])
        axes[1, 2].set_title('Health Loss')
        axes[1, 2].legend(['Train', 'Val'])
        axes[1, 2].grid(True)
        
        plt.tight_layout()
        plt.savefig('results/training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Training history saved to 'results/training_history.png'")

def main():
    print("=" * 60)
    print("Agri-Vision Cotton Classifier Training")
    print("=" * 60)
    
    # Create trainer and train model
    trainer = CottonMultiTaskModel()
    model, history = trainer.train_model(epochs=10, batch_size=8)
    
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("Model saved: models/cotton_classifier.h5")
    print("=" * 60)
    
    # Test the model
    print("\nTesting the model...")
    
    # Create a test image
    test_image = trainer.create_synthetic_image(phase=2, health=1)  # Bursting with pink bollworm
    test_image_norm = test_image.astype('float32') / 255.0
    test_image_norm = np.expand_dims(test_image_norm, axis=0)
    
    # Make prediction
    phase_pred, health_pred, score_pred = model.predict(test_image_norm)
    
    phase_classes = ['Vegetative/Budding', 'Flowering', 'Bursting (Ripped)', 'Harvest Ready']
    health_classes = ['Healthy', 'Pink Bollworm', 'Discoloration', 'Other Damage']
    
    phase_idx = np.argmax(phase_pred[0])
    health_idx = np.argmax(health_pred[0])
    health_score = float(score_pred[0][0] * 100)
    
    print("\nTEST PREDICTION:")
    print(f"Phase: {phase_classes[phase_idx]} ({phase_pred[0][phase_idx]:.2%})")
    print(f"Health: {health_classes[health_idx]} ({health_pred[0][health_idx]:.2%})")
    print(f"Health Score: {health_score:.1f}%")
    print(f"Is Ripped: {phase_idx == 2}")
    print("=" * 60)

if __name__ == '__main__':
    main()