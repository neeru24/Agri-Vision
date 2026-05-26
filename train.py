"""
Agri-Vision Cotton Crop Training Script
Improved Multi-Task Version
"""

import os
import cv2
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers, Model

from sklearn.model_selection import train_test_split


# ============================================================
# CONSTANTS
# ============================================================

PHASE_CLASSES = [
    "Vegetative/Budding",
    "Flowering",
    "Bursting (Ripped)",
    "Harvest Ready"
]

HEALTH_CLASSES = [
    "Healthy",
    "Pink Bollworm",
    "Discoloration",
    "Other Damage"
]


# ============================================================
# MODEL CLASS
# ============================================================

class CottonMultiTaskModel:

    def __init__(self, input_shape=(224, 224, 3), num_phases=4):
        self.input_shape = input_shape
        self.num_phases = num_phases

    # ========================================================
    # SYNTHETIC IMAGE GENERATION
    # ========================================================

    def create_synthetic_image(self, phase, health):

        img = np.ones((224, 224, 3), dtype=np.uint8) * 150

        phase_colors = [
            (34, 139, 34),
            (255, 255, 0),
            (255, 165, 0),
            (255, 0, 0)
        ]

        center_x, center_y = 112, 112

        if phase == 0:
            cv2.circle(img, (center_x, center_y), 60,
                       phase_colors[phase], -1)

        elif phase == 1:
            cv2.circle(img, (center_x, center_y), 70,
                       phase_colors[phase], -1)

        elif phase == 2:
            cv2.circle(img, (center_x, center_y), 80,
                       phase_colors[phase], -1)

        else:
            cv2.circle(img, (center_x, center_y), 85,
                       phase_colors[phase], -1)

        # Pink bollworm

        if health == 1:

            for _ in range(random.randint(3, 8)):
                x = random.randint(50, 174)
                y = random.randint(50, 174)

                cv2.circle(
                    img,
                    (x, y),
                    3,
                    (255, 192, 203),
                    -1
                )

        # Discoloration

        elif health == 2:

            for _ in range(random.randint(2, 5)):
                x = random.randint(30, 194)
                y = random.randint(30, 194)

                cv2.circle(
                    img,
                    (x, y),
                    random.randint(10, 25),
                    (
                        random.randint(150, 200),
                        random.randint(100, 150),
                        random.randint(100, 150)
                    ),
                    -1
                )

        noise = np.random.normal(
            0,
            10,
            (224, 224, 3)
        )

        img = np.clip(
            img + noise,
            0,
            255
        ).astype(np.uint8)

        return img

    # ========================================================
    # DATASET GENERATION
    # ========================================================

    def generate_synthetic_data(self, num_samples=200):

        print(f"Generating {num_samples} synthetic cotton images...")

        images = []
        labels = []

        for _ in range(num_samples):

            phase = random.randint(0, 3)
            health = random.randint(0, 3)

            img = self.create_synthetic_image(
                phase,
                health
            )

            images.append(img)

            labels.append({
                "phase": phase,
                "health": health,
                "health_score":
                    max(
                        0,
                        100 - (health * 25) -
                        random.randint(0, 15)
                    )
            })

        return np.array(images), labels

    # ========================================================
    # BUILD MODEL
    # ========================================================

    def build_model(self):

        inputs = keras.Input(
            shape=self.input_shape
        )

        # ----------------------------------------------------
        # Data Augmentation
        # ----------------------------------------------------

        augmentation = keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomFlip("vertical"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
            layers.RandomContrast(0.1)
        ])

        x = augmentation(inputs)

        # ----------------------------------------------------
        # CNN Backbone
        # ----------------------------------------------------

        x = layers.Conv2D(
            32,
            (3, 3),
            activation="relu",
            padding="same"
        )(x)

        x = layers.MaxPooling2D()(x)

        x = layers.Conv2D(
            64,
            (3, 3),
            activation="relu",
            padding="same"
        )(x)

        x = layers.MaxPooling2D()(x)

        x = layers.Conv2D(
            128,
            (3, 3),
            activation="relu",
            padding="same"
        )(x)

        x = layers.MaxPooling2D()(x)

        # ----------------------------------------------------
        # GAP Instead of Flatten
        # ----------------------------------------------------

        x = layers.GlobalAveragePooling2D()(x)

        x = layers.BatchNormalization()(x)

        shared = layers.Dense(
            256,
            activation="relu"
        )(x)

        shared = layers.Dropout(
            0.4
        )(shared)

        # ----------------------------------------------------
        # PHASE BRANCH
        # ----------------------------------------------------

        phase_branch = layers.Dense(
            128,
            activation="relu"
        )(shared)

        phase_output = layers.Dense(
            self.num_phases,
            activation="softmax",
            name="phase_output"
        )(phase_branch)

        # ----------------------------------------------------
        # HEALTH BRANCH
        # ----------------------------------------------------

        health_branch = layers.Dense(
            128,
            activation="relu"
        )(shared)

        health_output = layers.Dense(
            4,
            activation="softmax",
            name="health_output"
        )(health_branch)

        # ----------------------------------------------------
        # SCORE BRANCH
        # ----------------------------------------------------

        score_branch = layers.Dense(
            64,
            activation="relu"
        )(shared)

        score_output = layers.Dense(
            1,
            activation="sigmoid",
            name="health_score"
        )(score_branch)

        model = Model(
            inputs=inputs,
            outputs=[
                phase_output,
                health_output,
                score_output
            ],
            name="cotton_classifier"
        )

        return model

    # ========================================================
    # TRAIN MODEL
    # ========================================================

    def train_model(
        self,
        epochs=10,
        batch_size=8
    ):

        print("Creating synthetic dataset...")

        images, labels = self.generate_synthetic_data(
            num_samples=100
        )

        phase_labels = np.array(
            [l["phase"] for l in labels]
        )

        health_labels = np.array(
            [l["health"] for l in labels]
        )

        health_scores = np.array(
            [l["health_score"] for l in labels]
        ) / 100.0

        phase_labels = tf.keras.utils.to_categorical(
            phase_labels,
            4
        )

        health_labels = tf.keras.utils.to_categorical(
            health_labels,
            4
        )

        images = images.astype(
            "float32"
        ) / 255.0

        # ----------------------------------------------------
        # FIXED SPLIT
        # ----------------------------------------------------

        (
            X_train,
            X_val,
            phase_train,
            phase_val,
            health_train,
            health_val,
            score_train,
            score_val
        ) = train_test_split(
            images,
            phase_labels,
            health_labels,
            health_scores,
            test_size=0.2,
            random_state=42
        )

        print("Building model...")

        model = self.build_model()

        model.compile(
            optimizer="adam",
            loss={
                "phase_output":
                    "categorical_crossentropy",

                "health_output":
                    "categorical_crossentropy",

                "health_score":
                    "mse"
            },
            loss_weights={
                "phase_output": 1.0,
                "health_output": 0.8,
                "health_score": 0.5
            },
            metrics={
                "phase_output":
                    ["accuracy"],

                "health_output":
                    ["accuracy"],

                "health_score":
                    ["mae"]
            }
        )

        callbacks = [

            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=5,
                restore_best_weights=True
            ),

            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=2
            ),

            keras.callbacks.ModelCheckpoint(
                "models/best_model.keras",
                save_best_only=True
            )
        ]

        print("Training model...")

        history = model.fit(
            X_train,
            [
                phase_train,
                health_train,
                score_train
            ],
            validation_data=(
                X_val,
                [
                    phase_val,
                    health_val,
                    score_val
                ]
            ),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )

        os.makedirs(
            "models",
            exist_ok=True
        )

        model.save(
            "models/cotton_classifier.keras"
        )

        print(
            "\n✅ Model saved:"
            " models/cotton_classifier.keras"
        )

        self.plot_training_history(
            history
        )

        return model, history

    # ========================================================
    # TRAINING VISUALIZATION
    # ========================================================

    def plot_training_history(self, history):

        os.makedirs(
            "results",
            exist_ok=True
        )

        fig, axes = plt.subplots(
            2,
            3,
            figsize=(15, 10)
        )

        axes[0, 0].plot(
            history.history[
                "phase_output_accuracy"
            ]
        )

        axes[0, 0].plot(
            history.history[
                "val_phase_output_accuracy"
            ]
        )

        axes[0, 0].set_title(
            "Phase Accuracy"
        )

        axes[0, 0].legend(
            ["Train", "Val"]
        )

        axes[0, 0].grid(True)

        axes[0, 1].plot(
            history.history[
                "health_output_accuracy"
            ]
        )

        axes[0, 1].plot(
            history.history[
                "val_health_output_accuracy"
            ]
        )

        axes[0, 1].set_title(
            "Health Accuracy"
        )

        axes[0, 1].legend(
            ["Train", "Val"]
        )

        axes[0, 1].grid(True)

        axes[0, 2].plot(
            history.history[
                "health_score_mae"
            ]
        )

        axes[0, 2].plot(
            history.history[
                "val_health_score_mae"
            ]
        )

        axes[0, 2].set_title(
            "Health Score MAE"
        )

        axes[0, 2].legend(
            ["Train", "Val"]
        )

        axes[0, 2].grid(True)

        axes[1, 0].plot(
            history.history["loss"]
        )

        axes[1, 0].plot(
            history.history["val_loss"]
        )

        axes[1, 0].set_title(
            "Total Loss"
        )

        axes[1, 0].legend(
            ["Train", "Val"]
        )

        axes[1, 0].grid(True)

        axes[1, 1].plot(
            history.history[
                "phase_output_loss"
            ]
        )

        axes[1, 1].plot(
            history.history[
                "val_phase_output_loss"
            ]
        )

        axes[1, 1].set_title(
            "Phase Loss"
        )

        axes[1, 1].legend(
            ["Train", "Val"]
        )

        axes[1, 1].grid(True)

        axes[1, 2].plot(
            history.history[
                "health_output_loss"
            ]
        )

        axes[1, 2].plot(
            history.history[
                "val_health_output_loss"
            ]
        )

        axes[1, 2].set_title(
            "Health Loss"
        )

        axes[1, 2].legend(
            ["Train", "Val"]
        )

        axes[1, 2].grid(True)

        plt.tight_layout()

        plt.savefig(
            "results/training_history.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.show()

        print(
            "✅ Training history saved"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("Agri-Vision Cotton Classifier Training")
    print("=" * 60)

    trainer = CottonMultiTaskModel()

    model, history = trainer.train_model(
        epochs=10,
        batch_size=8
    )

    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("Model saved: models/cotton_classifier.keras")
    print("=" * 60)

    print("\nTesting the model...")

    test_image = trainer.create_synthetic_image(
        phase=2,
        health=1
    )

    test_image_norm = (
        test_image.astype("float32")
        / 255.0
    )

    test_image_norm = np.expand_dims(
        test_image_norm,
        axis=0
    )

    phase_pred, health_pred, score_pred = (
        model.predict(test_image_norm)
    )

    phase_idx = np.argmax(
        phase_pred[0]
    )

    health_idx = np.argmax(
        health_pred[0]
    )

    health_score = (
        float(score_pred[0][0]) * 100
    )

    print("\nTEST PREDICTION:")
    print(
        f"Phase: {PHASE_CLASSES[phase_idx]}"
        f" ({phase_pred[0][phase_idx]:.2%})"
    )

    print(
        f"Health: {HEALTH_CLASSES[health_idx]}"
        f" ({health_pred[0][health_idx]:.2%})"
    )

    print(
        f"Health Score: {health_score:.1f}%"
    )

    print(
        f"Is Ripped: {phase_idx == 2}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()
