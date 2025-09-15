import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from models.xlstm_memory_cell import XLSTMMemoryCell
import os


class XLSTMRLModel:
    """
    Модель xLSTM с RL для торговли - ТРЁХЭТАПНАЯ АРХИТЕКТУРА
    """
    def __init__(self, input_shape, memory_size=64, memory_units=128):
        self.input_shape = input_shape
        self.memory_size = memory_size
        self.memory_units = memory_units
        
        # Создаем модель актора (принимает решения о действиях)
        self.actor_model = self._build_actor_model()
        
        # Создаем модель критика (оценивает действия актора)
        self.critic_model = self._build_critic_model()
        
        # Оптимизаторы для разных этапов
        self.supervised_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
        
        # Флаги этапов обучения
        self.is_supervised_trained = False
        self.is_reward_model_trained = False

    def _build_actor_model(self):
        """Создает модель актора для принятия решений"""
        inputs = layers.Input(shape=self.input_shape)
        
        # Нормализация входных данных
        x = layers.LayerNormalization()(inputs)
        
        # Первый слой xLSTM
        x = layers.RNN(XLSTMMemoryCell(units=self.memory_units, 
                                       memory_size=self.memory_size),
                      return_sequences=True)(x)
        x = layers.LayerNormalization()(x)
        x = layers.Dropout(0.2)(x)
        
        # Второй слой xLSTM
        x = layers.RNN(XLSTMMemoryCell(units=self.memory_units,
                                       memory_size=self.memory_size),
                      return_sequences=False)(x)
        x = layers.LayerNormalization()(x)
        x = layers.Dropout(0.2)(x)
        
        # Полносвязные слои
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dense(32, activation='relu')(x)
        
        # Выходной слой для действий (BUY, HOLD, SELL)
        outputs = layers.Dense(3, activation='softmax')(x)
        
        model = models.Model(inputs=inputs, outputs=outputs)
        return model

    def _build_critic_model(self):
        """Создает модель критика для оценки действий"""
        inputs = layers.Input(shape=self.input_shape)
        
        # Нормализация входных данных
        x = layers.LayerNormalization()(inputs)
        
        # Первый слой xLSTM
        x = layers.RNN(XLSTMMemoryCell(units=self.memory_units,
                                       memory_size=self.memory_size),
                      return_sequences=True)(x)
        x = layers.LayerNormalization()(x)
        x = layers.Dropout(0.2)(x)
        
        # Второй слой xLSTM
        x = layers.RNN(XLSTMMemoryCell(units=self.memory_units,
                                       memory_size=self.memory_size),
                      return_sequences=False)(x)
        x = layers.LayerNormalization()(x)
        x = layers.Dropout(0.2)(x)
        
        # Полносвязные слои
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dense(32, activation='relu')(x)
        
        # Выходной слой для значения состояния (скаляр)
        outputs = layers.Dense(1)(x)
        
        model = models.Model(inputs=inputs, outputs=outputs)
        return model

    def compile_for_supervised_learning(self):
        """Компилирует модель для этапа 1: Supervised Learning"""
        self.actor_model.compile(
            optimizer=self.supervised_optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        print("✅ Модель скомпилирована для supervised learning")

    def compile_for_reward_modeling(self):
        """Компилирует модель для этапа 2: Reward Model Training"""
        self.critic_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        print("✅ Модель скомпилирована для reward modeling")

    def save(self, path='models', stage=""):
        """Сохраняет модель с указанием этапа"""
        if not os.path.exists(path):
            os.makedirs(path)
        
        # Сохранение моделей с указанием этапа
        actor_name = f'xlstm_rl_actor{stage}.keras'
        critic_name = f'xlstm_rl_critic{stage}.keras'
        
        self.actor_model.save(os.path.join(path, actor_name))
        self.critic_model.save(os.path.join(path, critic_name))
        
        print(f"Модели сохранены в {path} (этап: {stage})")

    def load(self, path='models', stage=""):
        """Загружает модель с указанием этапа"""
        actor_name = f'xlstm_rl_actor{stage}.keras'
        critic_name = f'xlstm_rl_critic{stage}.keras'
        
        actor_path = os.path.join(path, actor_name)
        critic_path = os.path.join(path, critic_name)
        
        if os.path.exists(actor_path) and os.path.exists(critic_path):
            self.actor_model = tf.keras.models.load_model(
                actor_path, 
                custom_objects={'XLSTMMemoryCell': XLSTMMemoryCell}
            )
            self.critic_model = tf.keras.models.load_model(
                critic_path,
                custom_objects={'XLSTMMemoryCell': XLSTMMemoryCell}
            )
            print(f"Модели успешно загружены (этап: {stage})")
        else:
            print(f"Не удалось найти сохраненные модели для этапа: {stage}")

    def predict_action(self, state):
        """Предсказывает действие на основе состояния"""
        if len(state.shape) == 2:
            state = np.expand_dims(state, axis=0)
        
        if state.dtype != np.float32:
            state = state.astype(np.float32)
        
        action_probs = self.actor_model.predict(state, verbose=0)[0]
        return action_probs

    def predict_value(self, state):
        """Предсказывает значение состояния"""
        if len(state.shape) == 2:
            state = np.expand_dims(state, axis=0)
        
        if state.dtype != np.float32:
            state = state.astype(np.float32)
        
        value = self.critic_model.predict(state, verbose=0)[0]
        return value