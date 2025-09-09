import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import Input, regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping


class Step1Tuning:
    def __init__(self, filepath, results_file="step_1_hyperparameter_tuning.xlsx"):
        """
        Initialize the grid search experiment.
        """
        self.filepath = filepath
        self.results_file = results_file
        self.df = None
        self.X_scaled = None
        self.y_categorical = None
        self.results = []

        # Hyperparameter search space
        self.layer_options = [2, 3, 4]
        self.neuron_options = [128, 64, 32, 16]
        self.topologies = ["Funnel", "Manhattan"]

        self.batch_sizes = [16, 32, 64, 128]
        self.activations = ["relu", "leaky_relu"]
        self.learning_rates = [1e-4, 5e-4, 1e-3, 5e-3]
        self.dropouts = [0, 0.1, 0.2, 0.3, 0.4]
        self.l2_options = [0, 5e-4, 1e-3, 5e-3, 1e-2]

    def load_and_prepare_data(self):
        """
        Load dataset, preprocess features, and encode target into 3 classes.
        """
        self.df = pd.read_excel(self.filepath)

        # Prepare features and target
        X = self.df.drop(columns=["Number", "Name", "Linie", "Ra", "Rz", "Rq", "Rt", "Gloss"])
        y_ra = self.df["Ra"]

        # Create class labels based on thresholds
        y = y_ra.copy()
        y[np.where(y_ra < 0.13)] = 0
        y[np.where((y_ra >= 0.13) & (y_ra <= 0.21))] = 1
        y[np.where(y_ra > 0.21)] = 2
        y = y.astype(int)

        # One-hot encode the labels
        self.y_categorical = to_categorical(y)

        # Scale features
        scaler = StandardScaler()
        self.X_scaled = scaler.fit_transform(X)

    def generate_layer_configs(self, topology, num_layers):
        """
        Generate possible layer configurations based on topology type.
        - Funnel: descending sequence of neurons
        - Manhattan: same number of neurons per layer
        """
        configs = []
        if topology == "Funnel":
            for start_idx in range(len(self.neuron_options) - num_layers + 1):
                config = self.neuron_options[start_idx:start_idx + num_layers]
                configs.append(config)
        elif topology == "Manhattan":
            for neurons in self.neuron_options:
                configs.append([neurons] * num_layers)
        return configs

    def run_grid_search(self):
        """
        Run the grid search over all hyperparameter combinations.
        """
        total_combinations = sum(
            len(self.generate_layer_configs(topology, num_layers)) *
            len(self.batch_sizes) *
            len(self.activations) *
            len(self.learning_rates) *
            len(self.dropouts) *
            len(self.l2_options)
            for topology in self.topologies for num_layers in self.layer_options
        )

        progress_bar = tqdm(total=total_combinations, desc="GridSearch Progress")
        combination_id = 1

        for topology in self.topologies:
            for num_layers in self.layer_options:
                layer_configs = self.generate_layer_configs(topology, num_layers)
                for config in layer_configs:
                    for batch_size in self.batch_sizes:
                        for activation in self.activations:
                            for lr in self.learning_rates:
                                for dropout in self.dropouts:
                                    for l2_value in self.l2_options:

                                        early_stopping = EarlyStopping(
                                            monitor="val_loss",
                                            patience=8,
                                            restore_best_weights=True
                                        )

                                        # Build model layers dynamically
                                        model_layers = [Input(shape=(self.X_scaled.shape[1],))]
                                        for neurons in config:
                                            model_layers.append(Dense(
                                                neurons,
                                                kernel_regularizer=regularizers.l2(l2_value) if l2_value > 0 else None
                                            ))
                                            if activation == "leaky_relu":
                                                model_layers.append(LeakyReLU(alpha=0.01))
                                            if dropout > 0:
                                                model_layers.append(Dropout(dropout))

                                        # Output layer
                                        model_layers.append(Dense(3, activation="softmax"))

                                        # Create and compile model
                                        model = Sequential(model_layers)
                                        model.compile(
                                            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                                            loss="categorical_crossentropy",
                                            metrics=["accuracy"]
                                        )

                                        # Train model
                                        history = model.fit(
                                            self.X_scaled, self.y_categorical,
                                            validation_split=0.15,
                                            epochs=50,
                                            batch_size=batch_size,
                                            callbacks=[early_stopping],
                                            verbose=0
                                        )

                                        # Extract validation results
                                        val_acc = round(history.history["val_accuracy"][-1], 4)
                                        val_loss = round(history.history["val_loss"][-1], 4)
                                        early_stopping_triggered = "Yes" if early_stopping.stopped_epoch > 0 else "No"

                                        # Store results
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
                                            "val_accuracy": val_acc,
                                            "val_loss": val_loss,
                                            "early_stopping": early_stopping_triggered
                                        }

                                        self.results.append(result)
                                        pd.DataFrame(self.results).to_excel(self.results_file, index=False)

                                        combination_id += 1
                                        progress_bar.update(1)

        progress_bar.close()
        print("Grid search completed. Results saved to:", self.results_file)


if __name__ == "__main__":
    st1 = Step1Tuning("chiefs_knife_dataset.xlsx")
    st1.load_and_prepare_data()
    st1.run_grid_search()
