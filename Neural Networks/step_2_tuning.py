import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import Input, regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU, Activation
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping


class Step2Tuning:
    def __init__(self, raw_file, step1_file, step2_file="step_2_hyperparameter_tuning.xlsx"):
        """
        Initialize epoch tuning process.
        """
        self.raw_file = raw_file
        self.step1_file = step1_file
        self.step2_file = step2_file
        self.plot_dir = "step_2_hyperparameter_tuning"
        os.makedirs(self.plot_dir, exist_ok=True)

        # Storage
        self.df_raw = None
        self.df_best = None
        self.results = []

        # Data splits
        self.X_train_scaled = None
        self.X_test_scaled = None
        self.y_train = None
        self.y_test = None

    def load_and_prepare_data(self):
        """
        Load raw dataset, split into train/test, and scale features.
        Encode target variable into 3 classes.
        """
        self.df_raw = pd.read_excel(self.raw_file, engine="openpyxl")

        # Prepare features and target
        X = self.df_raw.drop(columns=["Number", "Name", "Linie", "Ra", "Rz", "Rq", "Rt", "Gloss"])
        y_ra = self.df_raw["Ra"]

        # Define classes
        y = y_ra.copy()
        y[np.where(y_ra < 0.13)] = 0
        y[np.where((y_ra >= 0.13) & (y_ra <= 0.21))] = 1
        y[np.where(y_ra > 0.21)] = 2
        y = y.astype(int)
        y_categorical = to_categorical(y)

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_categorical, random_state=42
        )

        # Scale features
        scaler = StandardScaler()
        self.X_train_scaled = scaler.fit_transform(X_train)
        self.X_test_scaled = scaler.transform(X_test)
        self.y_train = y_train
        self.y_test = y_test

    def load_best_models(self, top_n=1500):
        """
        Load best models from Step 1 tuning results.
        Filter only models without early stopping.
        """
        df = pd.read_excel(self.step1_file, engine="openpyxl")
        df_no_es = df[df["early_stopping"] == "No"]
        self.df_best = df_no_es.sort_values(by=["val_accuracy"], ascending=False).head(top_n)

    def build_model(self, config, activation, dropout, l2_value, lr):
        """
        Build and compile a model given the configuration and hyperparameters.
        """
        model_layers = [Input(shape=(self.X_train_scaled.shape[1],))]
        for neurons in config:
            model_layers.append(Dense(
                neurons,
                kernel_regularizer=regularizers.l2(l2_value) if l2_value > 0 else None
            ))
            if activation == "leaky_relu":
                model_layers.append(LeakyReLU(alpha=0.01))
            else:
                model_layers.append(Activation(activation))
            if dropout > 0:
                model_layers.append(Dropout(dropout))

        # Output layer
        model_layers.append(Dense(3, activation="softmax"))

        model = Sequential(model_layers)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )
        return model

    def plot_history(self, history, combination_id):
        """
        Save training and validation curves as PNG.
        """
        plt.figure(figsize=(10, 4))

        # Accuracy
        plt.subplot(1, 2, 1)
        plt.plot(history.history["accuracy"], label="Train Accuracy")
        plt.plot(history.history["val_accuracy"], label="Val Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.title("Accuracy")

        # Loss
        plt.subplot(1, 2, 2)
        plt.plot(history.history["loss"], label="Train Loss")
        plt.plot(history.history["val_loss"], label="Val Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Loss")

        plt.tight_layout()
        filename = f"{combination_id}.png"
        plt.savefig(os.path.join(self.plot_dir, filename))
        plt.close()

    def run_epoch_tuning(self, epoch_options=[100, 200], start_id=1001):
        """
        Train the top-N models with different epoch settings.
        Save results and plots.
        """
        progress_bar = tqdm(total=len(self.df_best) * len(epoch_options), desc="Epoch Tuning Progress")
        combination_id = start_id

        for _, row in self.df_best.iterrows():
            topology = row["topology"]
            num_layers = int(row["num_layers"])
            config = eval(row["layer_config"])
            batch_size = int(row["batch_size"])
            activation = row["activation"]
            lr = float(row["learning_rate"])
            dropout = float(row["dropout"])
            l2_value = float(row["l2"])

            for n_epochs in epoch_options:
                early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

                # Build and train model
                model = self.build_model(config, activation, dropout, l2_value, lr)
                history = model.fit(
                    self.X_train_scaled, self.y_train,
                    validation_split=0.15,
                    epochs=n_epochs,
                    batch_size=batch_size,
                    verbose=0
                )

                # Evaluate on test data
                test_loss, test_acc = model.evaluate(self.X_test_scaled, self.y_test, verbose=0)

                # Save results
                result = {
                    "ID": combination_id,
                    "topology": topology,
                    "num_layers": num_layers,
                    "layer_config": str(config),
                    "batch_size": batch_size,
                    "activation": activation,
                    "learning_rate": lr,
                    "dropout": dropout,
                    "l2": l2_value,
                    "epochs": n_epochs,
                    "test_accuracy": test_acc,
                    "test_loss": test_loss,
                    "early_stopping": "Yes" if early_stop.stopped_epoch > 0 else "No"
                }
                self.results.append(result)
                pd.DataFrame(self.results).to_excel(self.step2_file, index=False)

                # Save plots
                self.plot_history(history, combination_id)

                combination_id += 1
                progress_bar.update(1)

        progress_bar.close()
        print("Epoch tuning completed. Results saved to:", self.step2_file)


if __name__ == "__main__":
    tuner = Step2Tuning(
        raw_file="chiefs_knife_dataset.xlsx",
        step1_file="step_1_hyperparameter_tuning.xlsx"
    )
    tuner.load_and_prepare_data()
    tuner.load_best_models(top_n=1500)
    tuner.run_epoch_tuning(epoch_options=[100, 200])
