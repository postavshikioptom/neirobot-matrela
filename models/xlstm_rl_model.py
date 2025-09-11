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
    Настоящая xLSTM модель с расширенной памятью - ИСПРАВЛЕННАЯ ВЕРСИЯ
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
        Строит настоящую xLSTM архитектуру с памятью - ИСПРАВЛЕННАЯ ВЕРСИЯ
        """
        # ✅ Фиксируем входную форму для стабильности
        inputs = Input(shape=self.input_shape, name='input_layer')
        
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
        
        
        print("✅ Настоящая xLSTM модель с памятью создана!")
        return self.model
    
    # ... остальные методы остаются без изменений
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32, custom_callbacks=None):
        """Обучение с улучшенной стабильностью"""
        if self.model is None:
            self.build_model()
        
        # ДОБАВЬТЕ: Обучение scaler на тренировочных данных
        X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
        self.scaler.fit(X_train_reshaped)
        
        # Применяем нормализацию
        X_train_scaled = self.scaler.transform(X_train_reshaped).reshape(X_train.shape)
        X_val_scaled = self.scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
            
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=25,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                'models/xlstm_checkpoint_epoch_{epoch:02d}.keras',
                monitor='val_loss',
                save_best_only=False,
                save_freq=10,
                verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                'models/xlstm_checkpoint_latest.keras',
                monitor='val_loss',
                save_best_only=False,
                save_freq='epoch',
                verbose=0
            ),
            tf.keras.callbacks.CSVLogger('training_log.csv', append=True)
        ]
        
        if custom_callbacks:
            callbacks.extend(custom_callbacks)
        
        # Градиентное обрезание для стабильности
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=0.001,
            clipnorm=1.0
        )
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        # Обучение с нормализованными данными
        history = self.model.fit(
            X_train_scaled, y_train,
            validation_data=(X_val_scaled, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1,
            shuffle=True
        )
        
        self.is_trained = True
        return history

    def predict(self, X):
        """
        Предсказание с правильной нормализацией
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
            
        # Применяем ту же нормализацию, что и при обучении
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