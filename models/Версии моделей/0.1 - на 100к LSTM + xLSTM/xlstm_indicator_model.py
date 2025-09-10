import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import RNN, Dense, Input
from tensorflow.keras.models import Sequential, load_model
from .xlstm_cell import mLSTMCell

class XLSTMIndicatorModel:
    """
    A class to represent an xLSTM model for predicting market direction based on technical indicators.
    """

    def __init__(self, input_shape, lstm_units=50, dropout_rate=0.2, num_classes=3):
        """
        Initializes the XLSTMIndicatorModel.

        Args:
            input_shape (tuple): The shape of the input data (timesteps, features).
            lstm_units (int): The number of units in the xLSTM layer.
            dropout_rate (float): The dropout rate for regularization.
            num_classes (int): The number of output classes (e.g., 3 for BUY, SELL, HOLD).
        """
        # Create two separate cell instances for each RNN layer
        # because they will have different input dimensions
        cell1 = mLSTMCell(lstm_units)
        cell2 = mLSTMCell(lstm_units)
        
        self.model = Sequential([
            Input(shape=input_shape),
            RNN(cell1, return_sequences=True),
            RNN(cell2),
            Dense(units=num_classes, activation='softmax')
        ])
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def train(self, X_train, y_train, epochs=50, batch_size=32, validation_split=0.1):
        """
        Trains the xLSTM model.
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
        """
        X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])
        X_test_scaled_reshaped = self.scaler.transform(X_test_reshaped)
        X_test_scaled = X_test_scaled_reshaped.reshape(X_test.shape)
        
        return self.model.predict(X_test_scaled)

    def save_model(self, model_path='models/xlstm_indicator_model.keras', scaler_path='models/xlstm_indicator_scaler.pkl'):
        """
        Saves the model and scaler.
        """
        self.model.save(model_path)
        import joblib
        joblib.dump(self.scaler, scaler_path)
        print(f"xLSTM model saved to {model_path}")
        print(f"xLSTM scaler saved to {scaler_path}")

    def load_model(self, model_path='models/xlstm_indicator_model.keras', scaler_path='models/xlstm_indicator_scaler.pkl'):
        """
        Loads the model and scaler.
        """
        self.model = load_model(model_path, custom_objects={'mLSTMCell': mLSTMCell})
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
