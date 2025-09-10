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
        Initialize the epoch tuning process.

        Parameters
        ----------
        raw_file : str
            Path to the raw dataset (Excel file).
        step1_file : str
            Path to the Step 1 hyperparameter tuning results.
        step2_file : str, optional
            Output file name for saving Step 2 results (default: step_2_hyperparameter_tuning.xlsx).
        """
        self.raw_file = raw_file
        self.step1_file = step1_file
        self.step2_file = step2_file
        self.plot_dir = "step_2_hyperparameter_tuning"
        os.makedirs(self.plot_dir, exist_ok=True)

        # Data storage
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
        Load the raw dataset, split into training and test sets, 
        scale the input features, and encode the target variable into three classes.
        """
        self.df_raw = pd.read_excel(self.raw_file, engine="openpyxl")

        # Extract features and target variable
        X = self.df_raw.drop(columns=["Number", "Name", "Linie", "Ra", "Rz", "Rq", "Rt", "Gloss"])
        y_ra = self.df_raw["Ra"]

        # Encode target into three classes based on specification limits
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

        # Feature scaling
        scaler = StandardScaler()
        self.X_train_scaled = scaler.fit_transform(X_train)
        self.X_test_scaled = scaler.transform(X_test)
        self.y_train = y_train
        self.y_test = y_test

    def load_best_models(self, top_n=1500):
        """
        Load the best-performing models from Step 1 results.

        Parameters
        ----------
        top_n : int, optional
            Number of top models to consider (default: 1500).
        
        Notes
        -----
        Only models without early stopping are included.
        """
        df = pd.read_excel(self.step1_file, engine="openpyxl")
        df_no_es = df[df["early_stopping"] == "No"]
        self.df_best = df_no_es.sort_values(by=["val_accuracy"], ascending=False).head(top_n)

    def build_model(self, config, activation, dropout, l2_value, lr):
        """
        Construct and compile a feedforward neural network based on the given configuration.

        Parameters
        ----------
        config : list
            List of neuron counts for each hidden layer.
        activation : str
            Activation function to be applied ("relu" or "leaky_relu").
        dropout : float
            Dropout rate for regularization.
        l2_value : float
            L2 regularization coefficient.
        lr : float
            Learning rate for the Adam optimizer.

        Returns
        -------
        model : tensorflow.keras.models.Sequential
            Compiled Keras model.
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

        # Output layer for three classes
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
        Plot and save the training history (accuracy and loss curves).

        Parameters
        ----------
        history : tensorflow.keras.callbacks.History
            Training history returned by model.fit().
        combination_id : int
            Unique identifier for the model configuration.
        """
        plt.figure(figsize=(10, 4))

        # Accuracy plot
        plt.subplot(1, 2, 1)
        plt.plot(history.history["accuracy"], label="Train Accuracy")
        plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.title("Accuracy")

        # Loss plot
        plt.subplot(1, 2, 2)
        plt.plot(history.history["loss"], label="Train Loss")
        plt.plot(history.history["val_loss"], label="Validation Loss")
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
        Train the top-N models with different epoch values.
        Save evaluation results and learning curves.

        Parameters
        ----------
        epoch_options : list, optional
            List of epoch counts to evaluate (default: [100, 200]).
        start_id : int, optional
            Starting ID for result indexing (default: 1001).
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

                # Build and train the model
                model = self.build_model(config, activation, dropout, l2_value, lr)
                history = model.fit(
                    self.X_train_scaled, self.y_train,
                    validation_split=0.15,
                    epochs=n_epochs,
                    batch_size=batch_size,
                    verbose=0
                )

                # Evaluate on test set
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
