import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input

class LSTMPatternModel:
    """
    A class to represent an LSTM model for predicting market direction based on candlestick patterns.
    """

    def __init__(self, input_shape, lstm_units=50, dropout_rate=0.2, num_classes=3):
        """
        Initializes the LSTMPatternModel.

        Args:
            input_shape (tuple): The shape of the input data (timesteps, features).
            lstm_units (int): The number of units in the LSTM layer.
            dropout_rate (float): The dropout rate for regularization.
            num_classes (int): The number of output classes (e.g., 3 for BUY, SELL, HOLD).
        """
        self.model = Sequential([
            Input(shape=input_shape),
            LSTM(units=lstm_units, return_sequences=True),
            Dropout(dropout_rate),
            LSTM(units=lstm_units),
            Dropout(dropout_rate),
            Dense(units=num_classes, activation='softmax')
        ])
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def train(self, X_train, y_train, epochs=50, batch_size=32, validation_split=0.1):
        """
        Trains the LSTM model.

        Args:
            X_train (np.ndarray): Training data (features).
            y_train (np.ndarray): Training data (labels).
            epochs (int): The number of epochs to train for.
            batch_size (int): The batch size for training.
            validation_split (float): The fraction of data to use for validation.
        """
        # Reshape X_train for scaling
        X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
        self.scaler.fit(X_train_reshaped)
        X_train_scaled_reshaped = self.scaler.transform(X_train_reshaped)
        X_train_scaled = X_train_scaled_reshaped.reshape(X_train.shape)
        
        self.model.fit(X_train_scaled, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split, verbose=1)

    def predict(self, X_test):
        """
        Makes a prediction on new data.

        Args:
            X_test (np.ndarray): The data to make a prediction on.

        Returns:
            np.ndarray: The prediction.
        """
        X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])
        X_test_scaled_reshaped = self.scaler.transform(X_test_reshaped)
        X_test_scaled = X_test_scaled_reshaped.reshape(X_test.shape)
        
        return self.model.predict(X_test_scaled)

    def save_model(self, model_path='models/lstm_pattern_model.keras', scaler_path='models/lstm_pattern_scaler.pkl'):
        """
        Saves the model and scaler.

        Args:
            model_path (str): The path to save the model to.
            scaler_path (str): The path to save the scaler to.
        """
        self.model.save(model_path)
        import joblib
        joblib.dump(self.scaler, scaler_path)
        print(f"LSTM model saved to {model_path}")
        print(f"LSTM scaler saved to {scaler_path}")

    def load_model(self, model_path='models/lstm_pattern_model.keras', scaler_path='models/lstm_pattern_scaler.pkl'):
        """
        Loads the model and scaler.

        Args:
            model_path (str): The path to load the model from.
            scaler_path (str): The path to load the scaler from.
        """
        self.model = load_model(model_path)
        import joblib
        self.scaler = joblib.load(scaler_path)

    def evaluate(self, X_test, y_test):
        """
        Evaluates the model on test data.
        """
        X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])
        X_test_scaled_reshaped = self.scaler.transform(X_test_reshaped)
        X_test_scaled = X_test_scaled_reshaped.reshape(X_test.shape)
        loss, accuracy = self.model.evaluate(X_test_scaled, y_test, verbose=0)
        return loss, accuracy
