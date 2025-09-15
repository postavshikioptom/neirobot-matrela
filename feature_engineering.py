import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle
import os


class FeatureEngineering:
    """
    Класс для обработки и подготовки признаков для модели
    """
    def __init__(self, sequence_length=60):
        self.sequence_length = sequence_length
        self.scaler = StandardScaler()
        self.feature_columns = ['open', 'high', 'low', 'close', 'volume', 'turnover']
        
    def prepare_data(self, df):
        """
        Подготовка данных: нормализация и создание последовательностей
        """
        # Убедимся, что данные отсортированы по времени
        df = df.sort_values('timestamp')
        
        # Преобразуем признаки в числовой формат
        for col in self.feature_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Получаем только нужные колонки
        data = df[self.feature_columns].values
        
        # Обучаем скейлер на всех данных
        scaled_data = self.scaler.fit_transform(data)
        
        # Создаем последовательности
        X, y_close = self._create_sequences(scaled_data)
        
        return X, y_close, df
    
    def prepare_test_data(self, df):
        """
        Подготовка тестовых данных с использованием уже обученного скейлера
        """
        # Убедимся, что данные отсортированы по времени
        df = df.sort_values('timestamp')
        
        # Преобразуем признаки в числовой формат
        for col in self.feature_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Получаем только нужные колонки
        data = df[self.feature_columns].values
        
        # Применяем уже обученный скейлер
        scaled_data = self.scaler.transform(data)
        
        # Создаем последовательности
        X, y_close = self._create_sequences(scaled_data)
        
        return X, y_close, df
    
    def _create_sequences(self, data):
        """
        Создает последовательности для обучения
        """
        X = []
        y_close = []
        
        for i in range(len(data) - self.sequence_length):
            # Последовательность признаков
            X.append(data[i:i+self.sequence_length])
            # Целевая цена закрытия (для мониторинга)
            y_close.append(data[i+self.sequence_length, 3])  # индекс 3 - это 'close'
        
        return np.array(X), np.array(y_close)
    
    # 🔥 НОВЫЕ МЕТОДЫ ДЛЯ ТРЁХЭТАПНОГО ОБУЧЕНИЯ
    def create_trading_labels(self, df, threshold=0.01, future_window=5):
        """
        Создает метки для классификации BUY/HOLD/SELL на основе будущих цен
        
        Args:
            df: DataFrame с данными
            threshold: порог изменения цены (по умолчанию 1%)
            future_window: окно для анализа будущих цен
            
        Returns:
            labels: массив меток (0=SELL, 1=HOLD, 2=BUY)
        """
        df = df.sort_values('timestamp')
        prices = df['close'].values
        labels = []
        
        for i in range(len(prices) - future_window):
            current_price = prices[i]
            future_price = prices[i + future_window]
            
            price_change = (future_price - current_price) / current_price
            
            if price_change > threshold:
                labels.append(2)  # BUY
            elif price_change < -threshold:
                labels.append(0)  # SELL
            else:
                labels.append(1)  # HOLD
        
        return np.array(labels)
    
    def prepare_supervised_data(self, df, threshold=0.01, future_window=5):
        """
        Подготавливает данные для supervised learning (этап 1)
        """
        # Подготавливаем последовательности
        X, _, processed_df = self.prepare_data(df)
        
        # Создаем метки
        labels = self.create_trading_labels(processed_df, threshold, future_window)
        
        # Обрезаем X до длины меток
        min_len = min(len(X), len(labels))
        X = X[:min_len]
        labels = labels[:min_len]
        
        return X, labels
    
    def save_scaler(self, path='models'):
        """
        Сохраняет обученный скейлер
        """
        if not os.path.exists(path):
            os.makedirs(path)
        
        with open(os.path.join(path, 'scaler.pkl'), 'wb') as f:
            pickle.dump(self.scaler, f)
        
        print(f"Скейлер сохранен в {path}/scaler.pkl")
    
    def load_scaler(self, path='models'):
        """
        Загружает обученный скейлер
        """
        scaler_path = os.path.join(path, 'scaler.pkl')
        
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            print("Скейлер успешно загружен")
            return True
        else:
            print("Не удалось найти сохраненный скейлер")
            return False