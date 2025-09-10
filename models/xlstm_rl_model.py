import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, Attention
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
import pickle
import os
from .xlstm_memory_cell import XLSTMLayer  # Импортируем настоящий xLSTM

class XLSTMRLModel:
    """
    Настоящая xLSTM модель с расширенной памятью
    """
    
    def __init__(self, input_shape, memory_units=128, memory_size=64, attention_units=64):
        self.input_shape = input_shape
        self.memory_units = memory_units
        self.memory_size = memory_size
        self.attention_units = attention_units
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def build_model(self):
        """
        Строит настоящую xLSTM архитектуру с памятью
        """
        inputs = Input(shape=self.input_shape)
        
        # Первый xLSTM слой с внешней памятью
        xlstm1 = XLSTMLayer(
            units=self.memory_units,
            memory_size=self.memory_size,
            return_sequences=True,
            name='xlstm_memory_layer_1'
        )(inputs)
        
        # Второй xLSTM слой
        xlstm2 = XLSTMLayer(
            units=self.memory_units // 2,
            memory_size=self.memory_size // 2,
            return_sequences=True,
            name='xlstm_memory_layer_2'
        )(xlstm1)
        
        # Механизм внимания
        attention = Attention(name='attention_mechanism')([xlstm2, xlstm2])
        
        # Финальный xLSTM слой
        xlstm_final = XLSTMLayer(
            units=self.attention_units,
            memory_size=self.attention_units,
            return_sequences=False,
            name='xlstm_memory_final'
        )(attention)
        
        # Классификационные слои
        dense1 = Dense(64, activation='relu', name='dense_1')(xlstm_final)
        dropout1 = Dropout(0.3)(dense1)
        
        dense2 = Dense(32, activation='relu', name='dense_2')(dropout1)
        dropout2 = Dropout(0.2)(dense2)
        
        # Выходной слой
        outputs = Dense(3, activation='softmax', name='output_layer')(dropout2)
        
        self.model = Model(inputs=inputs, outputs=outputs, name='True_xLSTM_RL_Model')
        
        # Компиляция
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        print("✅ Настоящая xLSTM модель с памятью создана!")
        return self.model
    
    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=50, batch_size=32):
        """
        Обучает модель
        """
        if self.model is None:
            self.build_model()
            
        # Нормализация данных
        X_train_scaled = self.scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
        
        if X_val is not None:
            X_val_scaled = self.scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
            validation_data = (X_val_scaled, y_val)
        else:
            validation_data = None
        
        # Обучение
        history = self.model.fit(
            X_train_scaled, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,
            shuffle=True
        )
        
        self.is_trained = True
        return history
    
    def predict(self, X):
        """
        Предсказание
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
            
        X_scaled = self.scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
        return self.model.predict(X_scaled, verbose=0)
    
    def save_model(self, model_path='models/xlstm_rl_model.keras', scaler_path='models/xlstm_rl_scaler.pkl'):
        """
        Сохранение модели и скейлера
        """
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        self.model.save(model_path)
        
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
            
        print(f"xLSTM-RL модель сохранена: {model_path}")
        print(f"Скейлер сохранен: {scaler_path}")
    
    def load_model(self, model_path='models/xlstm_rl_model.keras', scaler_path='models/xlstm_rl_scaler.pkl'):
        """
        Загрузка модели и скейлера
        """
        self.model = tf.keras.models.load_model(model_path, custom_objects={'XLSTMLayer': XLSTMLayer})
        
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
            
        self.is_trained = True
        print(f"xLSTM-RL модель загружена: {model_path}")