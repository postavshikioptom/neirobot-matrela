import numpy as np
import pandas as pd
from typing import Tuple, Optional
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import warnings
import os
import pickle
import traceback
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

warnings.filterwarnings('ignore')

class LSTMModel:
    """
    Реализация LSTM модели для выявления временных паттернов в рыночных данных.
    
    LSTM использует три вентиля (gate) для управления потоком информации:
    
    Вентиль забывания:
    f_t = σ(W_f * [h_{t-1}, x_t] + b_f)
    
    Вентиль входа:
    i_t = σ(W_i * [h_{t-1}, x_t] + b_i)
    
    Кандидат значения ячейки:
    C̃_t = tanh(W_C * [h_{t-1}, x_t] + b_C)
    
    Обновление состояния ячейки:
    C_t = f_t * C_{t-1} + i_t * C̃_t
    
    Вентиль выхода:
    o_t = σ(W_o * [h_{t-1}, x_t] + b_o)
    
    Скрытое состояние:
    h_t = o_t * tanh(C_t)
    
    Где:
    - f_t, i_t, o_t - значения вентилей забывания, входа и выхода
    - C_t - состояние ячейки
    - h_t - скрытое состояние
    - σ - сигмоидальная функция активации
    - tanh - гиперболический тангенс
    """
    
    def __init__(self, sequence_length: int = 60, features_count: int = 1):
        """
        Инициализация LSTM модели.
        
        Args:
            sequence_length: Длина временного окна для анализа
            features_count: Количество признаков во входных данных
        """
        self.sequence_length = sequence_length
        self.features_count = features_count
        self.model = None
        self.scaler = MinMaxScaler()
        self.is_trained = False
        
    def build_model(self) -> None:
        """
        Построение архитектуры LSTM модели.
        """
        self.model = Sequential([
            LSTM(units=50, return_sequences=True, input_shape=(self.sequence_length, self.features_count)),
            Dropout(0.2),
            LSTM(units=50, return_sequences=True),
            Dropout(0.2),
            LSTM(units=50, return_sequences=False),
            Dropout(0.2),
            Dense(units=25),
            Dense(units=1)
        ])
        
        self.model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
        
    def prepare_data(self, data: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """
        Подготовка данных для обучения LSTM модели.
        
        Args:
            data: Временной ряд данных
            
        Returns:
            Кортеж из признаков (X) и целевых значений (y)
        """
        # Нормализация данных
        # Проверяем, является ли data numpy массивом или pandas Series
        if isinstance(data, np.ndarray):
            scaled_data = self.scaler.fit_transform(data.reshape(-1, 1))
        else:
            scaled_data = self.scaler.fit_transform(data.values.reshape(-1, 1))
        
        X, y = [], []
        for i in range(self.sequence_length, len(scaled_data)):
            X.append(scaled_data[i-self.sequence_length:i, 0])
            y.append(scaled_data[i, 0])
            
        X, y = np.array(X), np.array(y)
        
        # Преобразование данных в формат, подходящий для LSTM
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        return X, y
        
    def train(self, data: pd.Series = None, X: np.ndarray = None, y: np.ndarray = None, epochs: int = 50, batch_size: int = 32) -> None:
        """
        Обучение LSTM модели.
        
        Args:
            data: Временной ряд данных для обучения (для совместимости с предыдущей версией)
            X: Готовые признаки для обучения
            y: Готовые целевые значения для обучения
            epochs: Количество эпох обучения
            batch_size: Размер батча
        """
        if self.model is None:
            self.build_model()
            
        # Если переданы готовые массивы X и y, используем их
        if X is not None and y is not None:
            pass  # X и y уже готовы к использованию
        # Если передан временной ряд данных, подготавливаем данные
        elif data is not None:
            X, y = self.prepare_data(data)
        else:
            raise ValueError("Необходимо передать либо data, либо готовые массивы X и y")
        
        # Обучение модели
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)
        self.is_trained = True
        
    def predict(self, data: pd.Series) -> np.ndarray:
        """
        Прогнозирование с помощью обученной LSTM модели.
        
        Args:
            data: Временной ряд данных для прогнозирования
            
        Returns:
            Прогнозируемые значения
        """
        if not self.is_trained:
            raise ValueError("Модель не обучена. Сначала вызовите метод train().")
            
        # Подготовка последних данных для прогнозирования
        logging.debug(f"LSTMModel.predict: data length: {len(data)}")
        logging.debug(f"LSTMModel.predict: data tail: {data.tail()}")
        logging.debug(f"LSTMModel.predict: is_trained: {self.is_trained}")
        try:
            scaled_data = self.scaler.transform(data.values.reshape(-1, 1))
            logging.debug(f"LSTMModel.predict: scaled_data (after transform): {scaled_data[-5:]}") # Печатаем последние 5 значений
        except ValueError as e:
            logging.warning(f"LSTMModel.predict: ValueError during transform: {e}")
            if "not fitted yet" in str(e):
                # Если скейлер не обучен, обучаем его на переданных данных
                logging.warning("LSTMModel.predict: Fitting scaler on provided data")
                scaled_data = self.scaler.fit_transform(data.values.reshape(-1, 1))
                logging.debug(f"LSTMModel.predict: scaled_data (after fit_transform): {scaled_data[-5:]}") # Печатаем последние 5 значений
            else:
                raise e
                
        X_test = []
        X_test.append(scaled_data[-self.sequence_length:, 0])
        X_test = np.array(X_test)
        logging.debug(f"LSTMModel.predict: X_test shape: {X_test.shape}")
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        logging.debug(f"LSTMModel.predict: X_test reshaped shape: {X_test.shape}")
        
        # Прогнозирование
        logging.debug("LSTMModel.predict: Calling model.predict")
        predicted_scaled = self.model.predict(X_test, verbose=0)
        logging.debug(f"LSTMModel.predict: predicted_scaled: {predicted_scaled}")
        
        # Обратное масштабирование
        logging.debug("LSTMModel.predict: Calling scaler.inverse_transform")
        predicted = self.scaler.inverse_transform(predicted_scaled)
        logging.debug(f"LSTMModel.predict: predicted (after inverse_transform): {predicted}")
        # Проверка на NaN в результатах
        if np.isnan(predicted).any():
            logging.warning("!!! ВНИМАНИЕ: predicted содержит NaN после inverse_transform!")
        
        return predicted.flatten()
        
    def detect_patterns(self, data: pd.Series) -> dict:
        """
        Выявление временных паттернов в данных.
        
        Args:
            data: Временной ряд данных
            
        Returns:
            Словарь с выявленными паттернами
        """
        if not self.is_trained:
            raise ValueError("Модель не обучена. Сначала вызовите метод train().")
            
        # Получаем прогноз
        logging.debug(f"LSTMModel.detect_patterns: data length: {len(data)}")
        logging.debug(f"LSTMModel.detect_patterns: data tail: {data.tail()}")
        logging.debug(f"LSTMModel.detect_patterns: is_trained: {self.is_trained}")
        prediction = self.predict(data)
        logging.debug(f"LSTMModel.detect_patterns: prediction from self.predict: {prediction}")
        
        # Получаем последние фактические значения
        actual = data.iloc[-1]
        logging.debug(f"LSTMModel.detect_patterns: actual value: {actual}")
        
        # Вычисляем направление тренда
        trend_direction = "UP" if prediction[0] > actual else "DOWN" if prediction[0] < actual else "NEUTRAL"
        logging.debug(f"LSTMModel.detect_patterns: trend_direction: {trend_direction}")
        
        # Вычисляем силу сигнала (разница между прогнозом и фактическим значением)
        signal_strength = abs(prediction[0] - actual) / actual * 100 if actual != 0 else 0
        logging.debug(f"LSTMModel.detect_patterns: signal_strength: {signal_strength}")
        
        confidence_val = min(signal_strength / 5.0, 1.0)  # Нормализуем уверенность до 0-1
        logging.debug(f"LSTMModel.detect_patterns: confidence: {confidence_val}")
        
        result = {
            "prediction": prediction[0],
            "trend_direction": trend_direction,
            "signal_strength": signal_strength,
            "confidence": confidence_val
        }
        logging.debug(f"LSTMModel.detect_patterns: returning result: {result}")
        return result
        
    def save_model(self, filepath: str) -> None:
        """
        Сохранение обученной модели.
        
        Args:
            filepath: Путь для сохранения модели
        """
        if not self.is_trained:
            raise ValueError("Модель не обучена. Нечего сохранять.")
            
        model_path = filepath
        scaler_path = os.path.splitext(filepath)[0] + "_scaler.pkl"
        
        try:
            # Сохраняем модель TensorFlow
            self.model.save(model_path)
            logging.info(f"Модель TensorFlow сохранена в {model_path}")
        except Exception as e:
            logging.error(f"!!! ОШИБКА при сохранении модели TensorFlow в файл {model_path}: {e}", exc_info=True)
            raise  # Перебрасываем исключение дальше

        try:
            # Сохраняем скейлер
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            logging.info(f"Скейлер сохранен в {scaler_path}")
        except Exception as e:
            logging.error(f"!!! ОШИБКА при сохранении скейлера в файл {scaler_path}: {e}", exc_info=True)
            raise  # Перебрасываем исключение дальше
        
    def load_model(self, filepath: str) -> None:
        """
        Загрузка обученной модели.
        
        Args:
            filepath: Путь к сохраненной модели
        """
        model_path = filepath
        scaler_path = os.path.splitext(filepath)[0] + "_scaler.pkl"

        try:
            self.model = load_model(model_path)
            logging.info(f"Модель TensorFlow загружена из {model_path}")
        except Exception as e:
            logging.error(f"!!! ОШИБКА при загрузке модели TensorFlow из файла {model_path}: {e}", exc_info=True)
            raise # Перебрасываем исключение дальше

        try:
            # Загружаем скейлер
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                logging.info(f"Скейлер загружен из {scaler_path}")
            else:
                # Если скейлер не найден, создаем новый
                logging.warning(f"--- ПРЕДУПРЕЖДЕНИЕ: Файл скейлера {scaler_path} не найден. Создан новый экземпляр MinMaxScaler. ---")
                self.scaler = MinMaxScaler()
        except Exception as e:
            logging.error(f"!!! ОШИБКА при загрузке скейлера из файла {scaler_path}: {e}", exc_info=True)
            # Если скейлер не удалось загрузить, создаем новый, чтобы избежать полного падения
            self.scaler = MinMaxScaler()

        self.is_trained = True