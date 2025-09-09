import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

import tensorflow as tf
from tensorflow.keras import Input, regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU
from tensorflow.keras.utils import to_categorical


class BestModel:
    def __init__(self, filepath):
        """
        Initialize the KnifeClassifier with dataset path.
        """
        self.filepath = filepath
        self.df = None
        self.X_train_scaled = None
        self.X_test_scaled = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.history = None
        self.scaler = StandardScaler()

    def load_data(self):
        """
        Load dataset and preprocess features and target variable.
        Target variable 'Ra' is converted into 3 classification classes.
        """
        self.df = pd.read_excel(self.filepath)

        # Drop unused columns to keep only features
        X = self.df.drop(columns=["Number", "Name", "Linie", "Ra", "Rz", "Rq", "Rt", "Gloss"])
        y_ra = self.df["Ra"]

        # Create 3 classes based on thresholds of 'Ra'
        class_0 = np.where(y_ra < 0.13)
        class_1 = np.where((y_ra >= 0.13) & (y_ra <= 0.21))
        class_2 = np.where(y_ra > 0.21)

        y = y_ra.copy()
        y[class_0] = 0
        y[class_1] = 1
        y[class_2] = 2
        y = y.astype(int)

        # One-hot encode target variable
        y_categorical = to_categorical(y)

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_categorical, random_state=42
        )

        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(X_train)
        self.X_test_scaled = self.scaler.transform(X_test)

        self.y_train = y_train
        self.y_test = y_test

    def build_model(self, input_dim):
        """
        Build the deep learning model with L2 regularization, dropout, and LeakyReLU.
        """
        self.model = Sequential([
            Input(shape=(input_dim,)),

            # Hidden Layer 1
            Dense(128, kernel_regularizer=regularizers.l2(0.005)),
            LeakyReLU(alpha=0.01),
            Dropout(0.3),

            # Hidden Layer 2
            Dense(128, kernel_regularizer=regularizers.l2(0.005)),
            LeakyReLU(alpha=0.01),
            Dropout(0.3),

            # Hidden Layer 3
            Dense(128, kernel_regularizer=regularizers.l2(0.005)),
            LeakyReLU(alpha=0.01),
            Dropout(0.3),

            # Hidden Layer 4
            Dense(128, kernel_regularizer=regularizers.l2(0.005)),
            LeakyReLU(alpha=0.01),
            Dropout(0.3),

            # Output Layer for 3 classes
            Dense(3, activation="softmax")
        ])

        # Compile the model
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )

    def train(self, epochs=300, batch_size=32, val_split=0.15):
        """
        Train the model and store the training history.
        """
        self.history = self.model.fit(
            self.X_train_scaled, self.y_train,
            validation_split=val_split,
            epochs=epochs,
            batch_size=batch_size
        )

    def plot_training_history(self):
        """
        Plot training and validation accuracy and loss curves.
        """
        plt.figure(figsize=(12, 5))

        # Accuracy plot
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['accuracy'], label="Train Accuracy")
        plt.plot(self.history.history['val_accuracy'], label="Val Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.title("Model Accuracy")

        # Loss plot
        plt.subplot(1, 2, 2)
        plt.plot(self.history.history['loss'], label="Train Loss")
        plt.plot(self.history.history['val_loss'], label="Val Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Model Loss")

        plt.tight_layout()
        plt.show()

    def evaluate(self):
        """
        Evaluate model on test data, print accuracy, and display confusion matrix.
        """
        y_pred = self.model.predict(self.X_test_scaled)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(self.y_test, axis=1)

        # Test accuracy
        test_acc = accuracy_score(y_true_classes, y_pred_classes)
        print("\nTest Accuracy:", test_acc)

        # Confusion matrix
        cm = confusion_matrix(y_true_classes, y_pred_classes)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap="Blues")
        plt.title("Confusion Matrix - Test Data")
        plt.show()


if __name__ == "__main__":
    classifier = BestModel("chiefs_knife_dataset.xlsx")
    classifier.load_data()
    classifier.build_model(input_dim=classifier.X_train_scaled.shape[1])
    classifier.train()
    classifier.plot_training_history()
    classifier.evaluate()
