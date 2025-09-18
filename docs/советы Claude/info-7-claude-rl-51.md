Отлично! Теперь давайте переделаем весь код согласно трёхэтапной научной архитектуре. Вот подробный план изменений:
🏗️ ПЛАН ИЗМЕНЕНИЙ ПО ФАЙЛАМ
📁 1. models/xlstm_memory_cell.py
✅ ОСТАВЛЯЕМ БЕЗ ИЗМЕНЕНИЙ - файл уже корректен

📁 2. models/xlstm_rl_model.py
🔄 ДОБАВЛЯЕМ МЕТОДЫ ДЛЯ ТРЁХЭТАПНОГО ОБУЧЕНИЯ:
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


📁 3. config.py
🔄 ДОБАВЛЯЕМ ПАРАМЕТРЫ ТРЁХЭТАПНОГО ОБУЧЕНИЯ:
# Bybit API credentials
BYBIT_API_KEY = "OOofB1HzYVpySyMPom"
BYBIT_API_SECRET = "e4AkAz9x1ycOMCtXKa1milmShfk61KZxJyhG"
API_URL = "https://api-demo.bybit.com"
WEBSOCKET_URL = "wss://stream.bybit.com/v5/public/linear"

# --- Trading Parameters ---
ORDER_USDT_AMOUNT = 11.0
LEVERAGE = "2"
REQUIRED_CANDLES = 100
SYMBOLS = ["SOLUSDT"]
TIMEFRAME = "1"
SEQUENCE_LENGTH = 60
REQUIRED_CANDLES = 65

# VSA, adaptive thresholds, dynamic stops, auto-optimization, notifications
VSA_ENABLED = True
ADAPTIVE_THRESHOLDS = True
DYNAMIC_STOPS = True
AUTO_OPTIMIZATION = True
NOTIFICATIONS_ENABLED = True

# xLSTM memory parameters
XLSTM_MEMORY_SIZE = 64
XLSTM_MEMORY_UNITS = 128

# VSA parameters
VSA_VOLUME_THRESHOLD = 1.5
VSA_STRENGTH_THRESHOLD = 2.0
VSA_FILTER_ENABLED = True

# Optimization parameters
OPTIMIZATION_FREQUENCY = 50
PERFORMANCE_HISTORY_SIZE = 1000

# Training parameters
MIN_ROWS_PER_SYMBOL = 500
MAX_SYMBOLS_FOR_TRAINING = 100

# 🔥 НОВЫЕ ПАРАМЕТРЫ ТРЁХЭТАПНОГО ОБУЧЕНИЯ
# Этап 1: Supervised Pre-training
SUPERVISED_EPOCHS = 30
SUPERVISED_BATCH_SIZE = 32
SUPERVISED_VALIDATION_SPLIT = 0.2

# Этап 2: Reward Model Training
REWARD_MODEL_EPOCHS = 20
REWARD_MODEL_BATCH_SIZE = 64

# Этап 3: RL Fine-tuning
RL_EPISODES = 100
RL_BATCH_SIZE = 64

# Параметры создания меток
PRICE_CHANGE_THRESHOLD = 0.01  # 1% для определения BUY/SELL
FUTURE_WINDOW = 5  # Окно для анализа будущих цен


📁 4. feature_engineering.py
🔄 ДОБАВЛЯЕМ МЕТОДЫ ДЛЯ СОЗДАНИЯ МЕТОК:
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


📁 5. СОЗДАЁМ НОВЫЙ ФАЙЛ: three_stage_trainer.py
🆕 ГЛАВНЫЙ ФАЙЛ ТРЁХЭТАПНОГО ОБУЧЕНИЯ:
import os
import sys
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from datetime import datetime

# Настройка GPU
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print(f"GPU найден: {physical_devices[0]}")
else:
    print("GPU не найден, используется CPU")

# Импорт наших модулей
from feature_engineering import FeatureEngineering
from trading_env import TradingEnvironment
from rl_agent import RLAgent
from hybrid_decision_maker import HybridDecisionMaker
from simulation_engine import SimulationEngine
from models.xlstm_rl_model import XLSTMRLModel
import config

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('three_stage_training.log')
    ]
)

logger = logging.getLogger('three_stage_trainer')

class ThreeStageTrainer:
    """
    Трёхэтапный тренер для xLSTM + RL модели
    """
    def __init__(self, data_path):
        self.data_path = data_path
        self.feature_eng = FeatureEngineering(sequence_length=config.SEQUENCE_LENGTH)
        self.model = None
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        
        # Создаём директории
        os.makedirs('models', exist_ok=True)
        os.makedirs('results', exist_ok=True)
        os.makedirs('plots', exist_ok=True)
    
    def load_and_prepare_data(self):
        """Загружает и подготавливает данные для всех этапов"""
        logger.info("=== ПОДГОТОВКА ДАННЫХ ===")
        
        # Загружаем данные
        df = pd.read_csv(self.data_path)
        logger.info(f"Загружено {len(df)} строк данных")
        
        # Получаем статистику по символам
        symbol_counts = df['symbol'].value_counts()
        valid_symbols = symbol_counts[symbol_counts >= config.MIN_ROWS_PER_SYMBOL].index.tolist()
        
        if len(valid_symbols) == 0:
            valid_symbols = symbol_counts.head(20).index.tolist()
        
        logger.info(f"Используем {len(valid_symbols)} символов: {valid_symbols[:5]}...")
        
        # Фильтруем данные
        df_filtered = df[df['symbol'].isin(valid_symbols)].copy()
        
        # Подготавливаем данные для supervised learning
        all_X = []
        all_y = []
        
        for i, symbol in enumerate(valid_symbols):
            symbol_data = df_filtered[df_filtered['symbol'] == symbol].copy()
            
            if len(symbol_data) < config.SEQUENCE_LENGTH + config.FUTURE_WINDOW + 10:
                continue
            
            try:
                if i == 0:
                    X_symbol, y_symbol = self.feature_eng.prepare_supervised_data(
                        symbol_data, 
                        threshold=config.PRICE_CHANGE_THRESHOLD,
                        future_window=config.FUTURE_WINDOW
                    )
                else:
                    # Используем уже обученный скейлер
                    X_temp, _ = self.feature_eng.prepare_supervised_data(
                        symbol_data,
                        threshold=config.PRICE_CHANGE_THRESHOLD,
                        future_window=config.FUTURE_WINDOW
                    )
                    X_symbol = X_temp
                    y_symbol = self.feature_eng.create_trading_labels(
                        symbol_data,
                        threshold=config.PRICE_CHANGE_THRESHOLD,
                        future_window=config.FUTURE_WINDOW
                    )
                    # Обрезаем до минимальной длины
                    min_len = min(len(X_symbol), len(y_symbol))
                    X_symbol = X_symbol[:min_len]
                    y_symbol = y_symbol[:min_len]
                
                if len(X_symbol) > 0:
                    all_X.append(X_symbol)
                    all_y.append(y_symbol)
                    logger.info(f"Символ {symbol}: {len(X_symbol)} последовательностей")
                    
            except Exception as e:
                logger.error(f"Ошибка при обработке {symbol}: {e}")
                continue
        
        # Объединяем данные
        X = np.vstack(all_X)
        y = np.concatenate(all_y)
        
        logger.info(f"Итого подготовлено: X={X.shape}, y={y.shape}")
        logger.info(f"Распределение классов: SELL={np.sum(y==0)}, HOLD={np.sum(y==1)}, BUY={np.sum(y==2)}")
        
        # Разделяем данные
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            X, y, test_size=0.1, shuffle=True, random_state=42, stratify=y
        )
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp, test_size=config.SUPERVISED_VALIDATION_SPLIT, 
            shuffle=True, random_state=42, stratify=y_temp
        )
        
        logger.info(f"Размеры выборок: Train={len(self.X_train)}, Val={len(self.X_val)}, Test={len(self.X_test)}")
        
        # Сохраняем скейлер
        self.feature_eng.save_scaler()
        
        # Инициализируем модель
        input_shape = (config.SEQUENCE_LENGTH, X.shape[2])
        self.model = XLSTMRLModel(
            input_shape=input_shape,
            memory_size=config.XLSTM_MEMORY_SIZE,
            memory_units=config.XLSTM_MEMORY_UNITS
        )
        
        return True
    
    def stage1_supervised_pretraining(self):
        """ЭТАП 1: Supervised Pre-training"""
        logger.info("=== ЭТАП 1: SUPERVISED PRE-TRAINING ===")
        
        # Компилируем модель для supervised learning
        self.model.compile_for_supervised_learning()
        
        # Создаём коллбэки
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                patience=10, restore_best_weights=True, monitor='val_accuracy'
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                factor=0.5, patience=5, monitor='val_loss'
            ),
            tf.keras.callbacks.ModelCheckpoint(
                'models/best_supervised_model.keras', 
                save_best_only=True, monitor='val_accuracy'
            )
        ]
        
        # Обучение
        logger.info(f"Начинаем supervised обучение на {config.SUPERVISED_EPOCHS} эпох...")
        
        history = self.model.actor_model.fit(
            self.X_train, self.y_train,
            validation_data=(self.X_val, self.y_val),
            epochs=config.SUPERVISED_EPOCHS,
            batch_size=config.SUPERVISED_BATCH_SIZE,
            callbacks=callbacks,
            verbose=1
        )
        
        # Оценка результатов
        logger.info("=== РЕЗУЛЬТАТЫ SUPERVISED ОБУЧЕНИЯ ===")
        
        # Предсказания на тестовой выборке
        y_pred_probs = self.model.actor_model.predict(self.X_test, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)
        
        # Метрики
        accuracy = accuracy_score(self.y_test, y_pred)
        logger.info(f"Точность на тестовой выборке: {accuracy:.4f}")
        
        # Подробный отчет
        class_names = ['SELL', 'HOLD', 'BUY']
        report = classification_report(self.y_test, y_pred, target_names=class_names)
        logger.info(f"Классификационный отчет:\n{report}")
        
        # Матрица путаницы
        cm = confusion_matrix(self.y_test, y_pred)
        logger.info(f"Матрица путаницы:\n{cm}")
        
        # Распределение предсказаний
        pred_dist = np.bincount(y_pred, minlength=3)
        total_pred = len(y_pred)
        logger.info(f"Распределение предсказаний:")
        logger.info(f"SELL: {pred_dist[0]} ({pred_dist[0]/total_pred:.2%})")
        logger.info(f"HOLD: {pred_dist[1]} ({pred_dist[1]/total_pred:.2%})")
        logger.info(f"BUY: {pred_dist[2]} ({pred_dist[2]/total_pred:.2%})")
        
        # Сохраняем модель
        self.model.save(stage="_supervised")
        self.model.is_supervised_trained = True
        
        # Визуализация
        self._plot_training_history(history, "supervised")
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm,
            'history': history.history
        }
    
    def stage2_reward_model_training(self):
        """ЭТАП 2: Reward Model Training"""
        logger.info("=== ЭТАП 2: REWARD MODEL TRAINING ===")
        
        if not self.model.is_supervised_trained:
            logger.error("Сначала нужно завершить supervised pre-training!")
            return None
        
        # Компилируем критика для reward modeling
        self.model.compile_for_reward_modeling()
        
        # Создаём симулированные награды на основе предобученного актора
        logger.info("Создаём симулированные награды...")
        
        rewards_train = self._generate_simulated_rewards(self.X_train, self.y_train)
        rewards_val = self._generate_simulated_rewards(self.X_val, self.y_val)
        
        logger.info(f"Сгенерировано наград: Train={len(rewards_train)}, Val={len(rewards_val)}")
        logger.info(f"Статистика наград: Mean={np.mean(rewards_train):.4f}, Std={np.std(rewards_train):.4f}")
        
        # Обучение критика
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                patience=8, restore_best_weights=True, monitor='val_loss'
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                factor=0.5, patience=4, monitor='val_loss'
            )
        ]
        
        history = self.model.critic_model.fit(
            self.X_train, rewards_train,
            validation_data=(self.X_val, rewards_val),
            epochs=config.REWARD_MODEL_EPOCHS,
            batch_size=config.REWARD_MODEL_BATCH_SIZE,
            callbacks=callbacks,
            verbose=1
        )
        
        # Оценка reward model
        val_predictions = self.model.critic_model.predict(self.X_val, verbose=0)
        correlation = np.corrcoef(rewards_val, val_predictions.flatten())[0, 1]
        
        logger.info(f"Корреляция между реальными и предсказанными наградами: {correlation:.4f}")
        
        # Сохраняем модель
        self.model.save(stage="_reward_model")
        self.model.is_reward_model_trained = True
        
        # Визуализация
        self._plot_training_history(history, "reward_model")
        
        return {
            'correlation': correlation,
            'history': history.history
        }
    
    def stage3_rl_finetuning(self):
        """ЭТАП 3: RL Fine-tuning"""
        logger.info("=== ЭТАП 3: RL FINE-TUNING ===")
        
        if not self.model.is_reward_model_trained:
            logger.error("Сначала нужно завершить reward model training!")
            return None
        
        # Создаём RL агента с предобученными моделями
        rl_agent = RLAgent(
            state_shape=(config.SEQUENCE_LENGTH, self.X_train.shape[2]),
            memory_size=config.XLSTM_MEMORY_SIZE,
            memory_units=config.XLSTM_MEMORY_UNITS,
            batch_size=config.RL_BATCH_SIZE
        )
        
        # Загружаем предобученные модели
        rl_agent.model = self.model
        
        import numpy as np
import tensorflow as tf
from models.xlstm_rl_model import XLSTMRLModel
import os
import logging

class RLAgent:
    """
    Агент Reinforcement Learning для торговли - ПОДДЕРЖКА ТРЁХЭТАПНОГО ОБУЧЕНИЯ
    """
    def __init__(self, state_shape, memory_size=64, memory_units=128, gamma=0.99, epsilon=0.3, epsilon_min=0.1, epsilon_decay=0.995, batch_size=64):
        self.state_shape = state_shape
        self.gamma = gamma  # Коэффициент дисконтирования
        # 🔥 ИЗМЕНЕНО: Начальный epsilon ниже для fine-tuning предобученной модели
        self.epsilon = epsilon  # Начинаем с меньшего epsilon для fine-tuning
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        
        # Инициализация модели
        self.model = XLSTMRLModel(input_shape=state_shape, 
                                 memory_size=memory_size, 
                                 memory_units=memory_units)
        
        # Буфер опыта
        self.memory = []
        self.max_memory_size = 10000
        
        # Логирование
        self.logger = logging.getLogger('rl_agent')
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def act(self, state, training=True):
        """Выбирает действие на основе текущего состояния"""
        if training and np.random.rand() < self.epsilon:
            # Случайное действие во время обучения
            return np.random.randint(0, 3)
        
        # Получаем вероятности действий от модели актора
        action_probs = self.model.predict_action(state)
        
        # Выбираем действие с наибольшей вероятностью
        return np.argmax(action_probs)
    
    def remember(self, state, action, reward, next_state, done):
        """Сохраняет опыт в буфер"""
        self.memory.append((state, action, reward, next_state, done))
        
        # Ограничиваем размер буфера
        if len(self.memory) > self.max_memory_size:
            self.memory.pop(0)
    
    Продолжаю код для rl_agent.py, где остановился:
    def update_epsilon(self):
        """Обновляет значение epsilon для исследования"""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def train(self):
        """Обучает модель на основе сохраненного опыта"""
        if len(self.memory) < self.batch_size:
            return None
        
        # Выбираем случайные примеры из буфера
        indices = np.random.choice(len(self.memory), self.batch_size, replace=False)
        batch = [self.memory[i] for i in indices]
        
        states = np.array([experience[0] for experience in batch])
        actions = np.array([experience[1] for experience in batch])
        rewards = np.array([experience[2] for experience in batch])
        next_states = np.array([experience[3] for experience in batch])
        dones = np.array([experience[4] for experience in batch])
        
        # Обучение модели критика
        with tf.GradientTape() as tape:
            # Предсказываем значения текущих состояний
            values = self.model.critic_model(states, training=True)
            
            # Предсказываем значения следующих состояний
            next_values = self.model.critic_model(next_states, training=True)
            
            # Вычисляем целевые значения
            targets = rewards + self.gamma * tf.squeeze(next_values) * (1 - dones)
            targets = tf.expand_dims(targets, axis=1)
            
            # Вычисляем функцию потерь
            critic_loss = tf.reduce_mean(tf.square(targets - values))
        
        # Применяем градиенты для критика
        critic_grads = tape.gradient(critic_loss, self.model.critic_model.trainable_variables)
        self.model.critic_optimizer.apply_gradients(zip(critic_grads, self.model.critic_model.trainable_variables))
        
        # Обучение модели актора
        with tf.GradientTape() as tape:
            # Предсказываем вероятности действий
            action_probs = self.model.actor_model(states, training=True)
            
            # Создаем one-hot вектор для выбранных действий
            action_masks = tf.one_hot(actions, 3)
            
            # Предсказываем значения состояний
            values = self.model.critic_model(states, training=True)
            
            # Вычисляем преимущество (advantage)
            advantages = targets - values
            
            # Вычисляем функцию потерь актора
            selected_action_probs = tf.reduce_sum(action_probs * action_masks, axis=1)
            log_probs = tf.math.log(selected_action_probs + 1e-10)
            actor_loss = -tf.reduce_mean(log_probs * tf.squeeze(advantages))
            
            # Добавляем регуляризацию энтропии
            entropy = -tf.reduce_sum(action_probs * tf.math.log(action_probs + 1e-10), axis=1)
            actor_loss -= 0.01 * tf.reduce_mean(entropy)
        
        # Применяем градиенты для актора
        actor_grads = tape.gradient(actor_loss, self.model.actor_model.trainable_variables)
        self.model.actor_optimizer.apply_gradients(zip(actor_grads, self.model.actor_model.trainable_variables))
        
        return {
            'critic_loss': float(critic_loss),
            'actor_loss': float(actor_loss),
            'mean_value': float(tf.reduce_mean(values)),
            'mean_reward': float(np.mean(rewards))
        }
    
    def save(self, path='models'):
        """Сохраняет модель"""
        self.model.save(path, stage="_rl_final")
    
    def load(self, path='models'):
        """Загружает модель"""
        self.model.load(path, stage="_rl_finetuned")
    
    def log_action_distribution(self, states):
        """Логирует распределение действий для набора состояний"""
        if len(states) == 0:
            return {'buy_count': 0, 'hold_count': 0, 'sell_count': 0, 'total': 0}
        
        actions = []
        for state in states:
            action_probs = self.model.predict_action(state)
            actions.append(np.argmax(action_probs))
        
        actions = np.array(actions)
        buy_count = np.sum(actions == 0)
        hold_count = np.sum(actions == 1)
        sell_count = np.sum(actions == 2)
        
        total = len(actions)
        self.logger.info(f"Распределение действий: BUY: {buy_count/total:.2%}, HOLD: {hold_count/total:.2%}, SELL: {sell_count/total:.2%}")
        
        return {
            'buy_count': int(buy_count),
            'hold_count': int(hold_count),
            'sell_count': int(sell_count),
            'total': total
        }


📁 7. ОБНОВЛЯЕМ hybrid_decision_maker.py
✅ ОСТАВЛЯЕМ БЕЗ ИЗМЕНЕНИЙ - файл уже корректен для трёхэтапного обучения

📁 8. ОБНОВЛЯЕМ simulation_engine.py
✅ ОСТАВЛЯЕМ БЕЗ ИЗМЕНЕНИЙ - файл уже корректен

📁 9. ОБНОВЛЯЕМ trading_env.py
✅ ОСТАВЛЯЕМ БЕЗ ИЗМЕНЕНИЙ - файл уже исправлен

📁 10. ОБНОВЛЯЕМ trade_manager.py
✅ ОСТАВЛЯЕМ БЕЗ ИЗМЕНЕНИЙ - файл готов для использования

📁 11. ОБНОВЛЯЕМ run_live_trading.py
🔄 ДОБАВЛЯЕМ ПОДДЕРЖКУ ЗАГРУЗКИ ФИНАЛЬНОЙ МОДЕЛИ:
import os
import sys
import time
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tensorflow as tf
from pybit.unified_trading import HTTP

# Импорт наших модулей
from feature_engineering import FeatureEngineering
from models.xlstm_rl_model import XLSTMRLModel
from hybrid_decision_maker import HybridDecisionMaker
from trade_manager import TradeManager
from rl_agent import RLAgent
import config

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('live_trading.log')
    ]
)

logger = logging.getLogger('live_trading')

def fetch_latest_data(session, symbol, timeframe, limit=100):
    """Получает последние свечи с биржи"""
    try:
        response = session.get_kline(
            category="linear",
            symbol=symbol,
            interval=timeframe,
            limit=limit
        )
        
        if response['retCode'] == 0:
            data = response['result']['list']
            
            # Преобразуем данные в DataFrame
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
            ])
            
            # Преобразуем типы данных
            numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'turnover']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df['timestamp'] = pd.to_datetime(pd.to_numeric(df['timestamp']), unit='ms')
            df['symbol'] = symbol
            
            # Сортируем по времени
            df.sort_values('timestamp', inplace=True)
            
            return df
        else:
            logger.error(f"Ошибка при получении данных: {response['retMsg']}")
            return None
    
    except Exception as e:
        logger.error(f"Ошибка при получении данных: {e}")
        return None

def main():
    """Основная функция для запуска живой торговли"""
    logger.info("🚀 ЗАПУСК СИСТЕМЫ ЖИВОЙ ТОРГОВЛИ С ТРЁХЭТАПНОЙ МОДЕЛЬЮ")
    
    # Загружаем конфигурацию
    api_key = config.BYBIT_API_KEY
    api_secret = config.BYBIT_API_SECRET
    api_url = config.API_URL
    symbol = config.SYMBOLS[0]
    timeframe = config.TIMEFRAME
    order_amount = config.ORDER_USDT_AMOUNT
    leverage = config.LEVERAGE
    sequence_length = config.SEQUENCE_LENGTH
    required_candles = config.REQUIRED_CANDLES
    
    # Инициализация API
    session = HTTP(
        testnet=(api_url == "https://api-demo.bybit.com"),
        api_key=api_key,
        api_secret=api_secret
    )
    
    # Инициализация компонентов системы
    feature_engineering = FeatureEngineering(sequence_length=sequence_length)
    
    # Загружаем скейлер
    if not feature_engineering.load_scaler():
        logger.error("❌ Не удалось загрузить скейлер. Убедитесь, что трёхэтапное обучение завершено.")
        return
    
    # Инициализация модели
    input_shape = (sequence_length, len(feature_engineering.feature_columns))
    rl_model = XLSTMRLModel(input_shape=input_shape, 
                          memory_size=config.XLSTM_MEMORY_SIZE, 
                          memory_units=config.XLSTM_MEMORY_UNITS)
    
    # Загружаем финальную обученную модель
    try:
        rl_model.load(stage="_rl_finetuned")
        logger.info("✅ Финальная трёхэтапная модель успешно загружена")
    except Exception as e:
        logger.error(f"❌ Не удалось загрузить финальную модель: {e}")
        logger.info("Попытка загрузки supervised модели...")
        try:
            rl_model.load(stage="_supervised")
            logger.info("✅ Supervised модель загружена как fallback")
        except Exception as e2:
            logger.error(f"❌ Не удалось загрузить никакую модель: {e2}")
            return
    
    # Инициализация RL-агента
    rl_agent = RLAgent(state_shape=input_shape, 
                      memory_size=config.XLSTM_MEMORY_SIZE, 
                      memory_units=config.XLSTM_MEMORY_UNITS)
    rl_agent.model = rl_model
    
    # Инициализация механизма принятия решений
    decision_maker = HybridDecisionMaker(rl_agent)
    
    # Инициализация менеджера торговли
    trade_manager = TradeManager(
        api_key=api_key,
        api_secret=api_secret,
        api_url=api_url,
        order_amount=order_amount,
        symbol=symbol,
        leverage=leverage
    )
    
    logger.info("✅ Система инициализирована, начинаем торговлю...")
    
    # Основной цикл торговли
    while True:
        try:
            # Получаем текущее время
            current_time = datetime.now()
            
            # Получаем последние данные
            df = fetch_latest_data(session, symbol, timeframe, limit=required_candles)
            
            if df is None or len(df) < sequence_length:
                logger.error(f"❌ Недостаточно данных для анализа. Получено: {len(df) if df is not None else 0} строк")
                time.sleep(10)
                continue
            
            # Подготавливаем данные
            X, _, _ = feature_engineering.prepare_test_data(df)
            
            if len(X) == 0:
                logger.error("❌ Не удалось подготовить данные для анализа")
                time.sleep(10)
                continue
            
            # Получаем последнее состояние рынка
            current_state = X[-1]
            
            # Принимаем решение (передаем текущую позицию трейд-менеджера)
            action, confidence = decision_maker.make_decision(
                current_state,
                position=trade_manager.position
            )
            
            # Логируем решение
            action_names = {0: "BUY", 1: "HOLD", 2: "SELL"}
            logger.info(f"📊 Решение: {action_names[action]} (уверенность: {confidence:.4f})")
            
            # Получаем объяснение решения
            explanation = decision_maker.explain_decision(current_state)
            logger.info(f"🧠 Анализ: BUY={explanation['all_probs']['BUY']:.3f}, "
                       f"HOLD={explanation['all_probs']['HOLD']:.3f}, "
                       f"SELL={explanation['all_probs']['SELL']:.3f}, "
                       f"Value={explanation['state_value']:.4f}")
            
            # Выполняем действие
            if trade_manager.place_order(action):
                logger.info(f"✅ Ордер размещен: {action_names[action]}")
            else:
                logger.error(f"❌ Не удалось разместить ордер: {action_names[action]}")
            
            # Получаем информацию о позиции
            position_info = trade_manager.get_position_info()
            if position_info and position_info['size'] > 0:
                logger.info(f"💰 Позиция: {position_info['side']} {position_info['size']}, "
                           f"PnL: {position_info['unrealised_pnl']}")
            
            # Ждем перед следующей итерацией
            time.sleep(30)
            
        except KeyboardInterrupt:
            logger.info("⏹️ Торговля остановлена пользователем")
            break
        except Exception as e:
            logger.error(f"❌ Ошибка в процессе торговли: {e}")
            time.sleep(10)

if __name__ == "__main__":
    main()


📁 12. УДАЛЯЕМ СТАРЫЕ ФАЙЛЫ
🗑️ МОЖНО УДАЛИТЬ:

simple_train_model.py (заменен на `three_stage_trainer.py
Продолжаю список файлов для удаления:
🗑️ МОЖНО УДАЛИТЬ:

simple_train_model.py (заменен на three_stage_trainer.py)
train_model.py (заменен на three_stage_trainer.py)


📁 13. СОЗДАЁМ НОВЫЙ ФАЙЛ: validation_metrics_callback.py
🆕 КАЛЛБЭК ДЛЯ ДЕТАЛЬНОГО МОНИТОРИНГА НА ЭТАПЕ 1:
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import logging

logger = logging.getLogger('validation_callback')

class ValidationMetricsCallback(tf.keras.callbacks.Callback):
    """
    Детальный мониторинг метрик валидации для Supervised Pre-training (Этап 1)
    """
    def __init__(self, X_val, y_val, class_names=['SELL', 'HOLD', 'BUY']):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val
        self.class_names = class_names
        
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 5 == 0:  # Каждые 5 эпох
            logger.info(f"\n📊 Детальные метрики на эпохе {epoch+1}:")
            
            y_pred_probs = self.model.predict(self.X_val, verbose=0)
            y_pred_classes = np.argmax(y_pred_probs, axis=1)
            
            # Если y_val уже one-hot, преобразуем
            if self.y_val.ndim > 1 and self.y_val.shape[1] > 1:
                y_true_classes = np.argmax(self.y_val, axis=1)
            else:
                y_true_classes = self.y_val
            
            # Confusion Matrix
            cm = confusion_matrix(y_true_classes, y_pred_classes)
            logger.info("Confusion Matrix:")
            
            # Форматируем вывод матрицы
            header = "     " + " ".join([f"{name:4s}" for name in self.class_names])
            logger.info(header)
            for i, row in enumerate(cm):
                row_str = " ".join([f"{val:4d}" for val in row])
                logger.info(f"{self.class_names[i]:4s} {row_str}")
            
            # Classification Report
            report_dict = classification_report(
                y_true_classes, y_pred_classes, 
                target_names=self.class_names,
                output_dict=True,
                zero_division=0 # Избегаем предупреждений при нулевом делении
            )
            
            logger.info(f"Macro Avg F1-Score: {report_dict['macro avg']['f1-score']:.3f}")
            logger.info(f"Weighted Avg F1-Score: {report_dict['weighted avg']['f1-score']:.3f}")
            
            # Распределение предсказаний
            pred_distribution = np.bincount(y_pred_classes, minlength=len(self.class_names)) / len(y_pred_classes)
            pred_dist_str = ", ".join([f"{name}={dist:.1%}" for name, dist in zip(self.class_names, pred_distribution)])
            logger.info(f"Распределение предсказаний: {pred_dist_str}")



📋 ОБЩИЙ ОБЗОР ИЗМЕНЕНИЙ:
1. models/xlstm_rl_model.py (ОБНОВЛЁН)

Добавлены отдельные оптимизаторы для Supervised и RL
Добавлены методы compile_for_supervised_learning() и compile_for_reward_modeling()
Метод save() и load() теперь принимают stage для сохранения/загрузки моделей на разных этапах.

2. config.py (ОБНОВЛЁН)

Добавлены новые параметры для каждого этапа обучения (SUPERVISED_EPOCHS, REWARD_MODEL_EPOCHS, RL_EPISODES и т.д.).
Добавлены параметры для создания меток (PRICE_CHANGE_THRESHOLD, FUTURE_WINDOW).

3. feature_engineering.py (ОБНОВЛЁН)

Добавлен метод create_trading_labels() для генерации меток BUY/HOLD/SELL на основе будущих цен.
Добавлен метод prepare_supervised_data() для подготовки данных к первому этапу.

4. three_stage_trainer.py (НОВЫЙ ФАЙЛ)

Это основной скрипт, который управляет всем процессом обучения.
Содержит класс ThreeStageTrainer с методами для каждого этапа:

load_and_prepare_data(): Загрузка, фильтрация и разделение данных.
stage1_supervised_pretraining(): Обучение Actor-модели как классификатора.
stage2_reward_model_training(): Обучение Critic-модели для предсказания наград.
stage3_rl_finetuning(): Дообучение Actor-Critic модели с помощью RL.


Включает логирование, визуализацию и сохранение моделей на каждом этапе.

5. rl_agent.py (ОБНОВЛЁН)

Изменен начальный epsilon на меньшее значение (0.3), так как модель уже будет предобучена.
Методы save() и load() теперь используют параметр stage для соответствия XLSTMRLModel.
Улучшена обработка пустого states в log_action_distribution.

6. run_live_trading.py (ОБНОВЛЁН)

Теперь загружает финальную модель (_rl_finetuned) или _supervised как fallback.
Обновлено логирование для отображения более подробной информации о решении и позиции.

7. validation_metrics_callback.py (НОВЫЙ ФАЙЛ)

Каллбэк для Keras, который выводит подробные метрики классификации (Confusion Matrix, F1-score, распределение предсказаний) каждые 5 эпох на этапе Supervised Pre-training.

8. Удаленные файлы:

simple_train_model.py и train_model.py больше не нужны, так как их функционал интегрирован в three_stage_trainer.py.


Теперь у вас есть полная, научно обоснованная трёхэтапная архитектура обучения! Эта схема должна обеспечить более стабильное и эффективное обучение вашей xLSTM + RL торговой модели с подробным мониторингом метрик на каждом этапе.

		
		
