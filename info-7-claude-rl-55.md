
📁 1. config.py
🔄 ДОБАВЛЯЕМ ПАРАМЕТРЫ ИНДИКАТОРОВ:
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

# 🔥 НОВЫЕ ПАРАМЕТРЫ ИНДИКАТОРОВ (TA-Lib)
RSI_PERIOD = 14
MACD_FASTPERIOD = 12
MACD_SLOWPERIOD = 26
MACD_SIGNALPERIOD = 9
STOCH_K_PERIOD = 5
STOCH_D_PERIOD = 3
WILLR_PERIOD = 14
AO_FASTPERIOD = 5  # Для Awesome Oscillator
AO_SLOWPERIOD = 34 # Для Awesome Oscillator


📁 2. feature_engineering.py
🔄 ДОБАВЛЯЕМ ЛОГИКУ РАСЧЕТА ИНДИКАТОРОВ:
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle
import os
import talib # 🔥 ДОБАВЛЕНО: Импорт TA-Lib
import config # 🔥 ДОБАВЛЕНО: Импорт config для параметров индикаторов

class FeatureEngineering:
    """
    Класс для обработки и подготовки признаков для модели, включая технические индикаторы
    """
    def __init__(self, sequence_length=60):
        self.sequence_length = sequence_length
        self.scaler = StandardScaler()
        # 🔥 ИЗМЕНЕНО: Исходные колонки для расчета индикаторов
        self.base_features = ['open', 'high', 'low', 'close', 'volume', 'turnover']
        self.feature_columns = list(self.base_features) # Будут обновлены после добавления индикаторов
        
    def _add_technical_indicators(self, df):
        """
        Добавляет технические индикаторы в DataFrame с использованием TA-Lib.
        """
        # Убедимся, что все необходимые колонки в числовом формате
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # --- RSI (Relative Strength Index) ---
        df['RSI'] = talib.RSI(df['close'], timeperiod=config.RSI_PERIOD)
        
        # --- MACD (Moving Average Convergence Divergence) ---
        macd, macdsignal, macdhist = talib.MACD(
            df['close'], 
            fastperiod=config.MACD_FASTPERIOD, 
            slowperiod=config.MACD_SLOWPERIOD, 
            signalperiod=config.MACD_SIGNALPERIOD
        )
        df['MACD'] = macd
        df['MACDSIGNAL'] = macdsignal
        df['MACDHIST'] = macdhist
        
        # --- Stochastic Oscillator ---
        stoch_k, stoch_d = talib.STOCH(
            df['high'], df['low'], df['close'],
            fastk_period=config.STOCH_K_PERIOD,
            slowk_period=config.STOCH_K_PERIOD, # Обычно FastK и SlowK имеют одинаковый период
            slowd_period=config.STOCH_D_PERIOD
        )
        df['STOCH_K'] = stoch_k
        df['STOCH_D'] = stoch_d
        
        # --- Williams %R ---
        df['WILLR'] = talib.WILLR(df['high'], df['low'], df['close'], timeperiod=config.WILLR_PERIOD)
        
        # --- Awesome Oscillator (AO) - TA-Lib не имеет AO напрямую, вычисляем вручную ---
        # AO = SMA(High+Low)/2, 5 - SMA(High+Low)/2, 34
        # Вычисляем медианную цену
        median_price = (df['high'] + df['low']) / 2
        
        # Вычисляем 5-периодную SMA медианной цены
        sma_5 = talib.SMA(median_price, timeperiod=config.AO_FASTPERIOD)
        
        # Вычисляем 34-периодную SMA медианной цены
        sma_34 = talib.SMA(median_price, timeperiod=config.AO_SLOWPERIOD)
        
        df['AO'] = sma_5 - sma_34
        
        # Обновляем список признаков
        self.feature_columns = self.base_features + [
            'RSI', 'MACD', 'MACDSIGNAL', 'MACDHIST', 
            'STOCH_K', 'STOCH_D', 'WILLR', 'AO'
        ]
        
        # Обработка NaN значений, которые появляются из-за расчета индикаторов
        # Заполняем NaN нулями, так как StandardScaler не работает с NaN
        # или можно использовать df.fillna(method='ffill').fillna(method='bfill')
        # Для начала заполним нулями, чтобы модель видела отсутствие данных как 0
        df.fillna(0, inplace=True) 
        
        return df

    def prepare_data(self, df):
        """
        Подготовка данных: добавление индикаторов, нормализация и создание последовательностей
        """
        df = df.sort_values('timestamp')
        
        # 🔥 ИЗМЕНЕНО: Добавляем индикаторы
        df_with_indicators = self._add_technical_indicators(df.copy())
        
        # Преобразуем признаки в числовой формат
        for col in self.feature_columns:
            df_with_indicators[col] = pd.to_numeric(df_with_indicators[col], errors='coerce')
        
        # Получаем только нужные колонки
        data = df_with_indicators[self.feature_columns].values
        
        # Обучаем скейлер на всех данных (теперь включая индикаторы)
        scaled_data = self.scaler.fit_transform(data)
        
        # Создаем последовательности
        X, y_close = self._create_sequences(scaled_data)
        
        return X, y_close, df_with_indicators
    
    def prepare_test_data(self, df):
        """
        Подготовка тестовых данных с использованием уже обученного скейлера и индикаторов
        """
        df = df.sort_values('timestamp')
        
        # 🔥 ИЗМЕНЕНО: Добавляем индикаторы
        df_with_indicators = self._add_technical_indicators(df.copy())
        
        # Преобразуем признаки в числовой формат
        for col in self.feature_columns:
            df_with_indicators[col] = pd.to_numeric(df_with_indicators[col], errors='coerce')
        
        # Получаем только нужные колонки
        data = df_with_indicators[self.feature_columns].values
        
        # Применяем уже обученный скейлер
        scaled_data = self.scaler.transform(data)
        
        # Создаем последовательности
        X, y_close = self._create_sequences(scaled_data)
        
        return X, y_close, df_with_indicators
    
    def _create_sequences(self, data):
        """
        Создает последовательности для обучения
        """
        X = []
        y_close = []
        
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:i+self.sequence_length])
            # Целевая цена закрытия (индекс 3 - это 'close' в base_features)
            # Если изменится порядок base_features, нужно будет скорректировать
            y_close.append(data[i+self.sequence_length, self.base_features.index('close')]) 
        
        return np.array(X), np.array(y_close)
    
    def create_trading_labels(self, df, threshold=0.01, future_window=5):
        """
        Создает метки для классификации BUY/HOLD/SELL на основе будущих цен
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
        # 🔥 ИЗМЕНЕНО: Использование prepare_data, которая теперь включает индикаторы
        X, _, processed_df = self.prepare_data(df)
        
        # Создаем метки
        labels = self.create_trading_labels(processed_df, threshold, future_window)
        
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
    
    Продолжаю код для feature_engineering.py, где остановился:
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


Нет, это был только файл feature_engineering.py и config.py.
Чтобы полностью добавить индикаторы и обновить всю схему, как мы обсуждали, мне нужно внести изменения и в другие файлы.
Вот полный список файлов, которые нужно будет обновить (помимо config.py и feature_engineering.py, которые уже были предоставлены):

models/xlstm_rl_model.py: Обновление input_shape модели.
train_model.py: Обновление input_shape для XLSTMRLModel.
run_live_trading.py: Обновление input_shape и логики получения данных.
rl_agent.py: Обновление input_shape.
trading_env.py: Обновление observation_space.

Я продолжу с этими файлами.

📁 3. models/xlstm_rl_model.py
🔄 ОБНОВЛЯЕМ input_shape:
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
        # 🔥 ИЗМЕНЕНО: input_shape теперь будет включать индикаторы
        self.input_shape = input_shape 
        self.memory_size = memory_size
        self.memory_units = memory_units
        
        self.actor_model = self._build_actor_model()
        self.critic_model = self._build_critic_model()
        
        self.supervised_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
        
        self.is_supervised_trained = False
        self.is_reward_model_trained = False

    def _build_actor_model(self):
        """Создает модель актора для принятия решений"""
        inputs = layers.Input(shape=self.input_shape)
        
        x = layers.LayerNormalization()(inputs)
        
        x = layers.RNN(XLSTMMemoryCell(units=self.memory_units, 
                                       memory_size=self.memory_size),
                      return_sequences=True)(x)
        x = layers.LayerNormalization()(x)
        x = layers.Dropout(0.2)(x)
        
        x = layers.RNN(XLSTMMemoryCell(units=self.memory_units,
                                       memory_size=self.memory_size),
                      return_sequences=False)(x)
        x = layers.LayerNormalization()(x)
        x = layers.Dropout(0.2)(x)
        
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dense(32, activation='relu')(x)
        
        outputs = layers.Dense(3, activation='softmax')(x)
        
        model = models.Model(inputs=inputs, outputs=outputs)
        return model

    def _build_critic_model(self):
        """Создает модель критика для оценки действий"""
        inputs = layers.Input(shape=self.input_shape)
        
        x = layers.LayerNormalization()(inputs)
        
        x = layers.RNN(XLSTMMemoryCell(units=self.memory_units,
                                       memory_size=self.memory_size),
                      return_sequences=True)(x)
        x = layers.LayerNormalization()(x)
        x = layers.Dropout(0.2)(x)
        
        x = layers.RNN(XLSTMMemoryCell(units=self.memory_units,
                                       memory_size=self.memory_size),
                      return_sequences=False)(x)
        x = layers.LayerNormalization()(x)
        x = layers.Dropout(0.2)(x)
        
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dense(32, activation='relu')(x)
        
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


📁 4. train_model.py
🔄 ОБНОВЛЯЕМ input_shape:
import os
import sys
import logging # Оставляем для настройки других логгеров, но не используем напрямую
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
from validation_metrics_callback import ValidationMetricsCallback

# 🔥 ИЗМЕНЕНО: Удаляем настройку logging.basicConfig и logger, используем print
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.StreamHandler(),
#         l  ogging.FileHandler('three_stage_training.log')
#     ]
# )
# logger = logging.getLogger('three_stage_trainer')

class ThreeStageTrainer:
    """
    Трёхэтапный тренер для xLSTM + RL модели
    """
    def __init__(self, data_path):
        self.data_path = data_path
        # 🔥 ИЗМЕНЕНО: Передаем feature_columns в инициализацию FeatureEngineering
        self.feature_eng = FeatureEngineering(sequence_length=config.SEQUENCE_LENGTH) 
        self.model = None
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        
        os.makedirs('models', exist_ok=True)
        os.makedirs('results', exist_ok=True)
        os.makedirs('plots', exist_ok=True)
    
    def load_and_prepare_data(self):
        """Загружает и подготавливает данные для всех этапов"""
        print("=== ПОДГОТОВКА ДАННЫХ ===")
        
        df = pd.read_csv(self.data_path)
        print(f"Загружено {len(df)} строк данных")
        
        symbol_counts = df['symbol'].value_counts()
        valid_symbols = symbol_counts[symbol_counts >= config.MIN_ROWS_PER_SYMBOL].index.tolist()
        
        if len(valid_symbols) == 0:
            valid_symbols = symbol_counts.head(20).index.tolist()
        
        print(f"Используем {len(valid_symbols)} символов: {valid_symbols[:5]}...")
        
        df_filtered = df[df['symbol'].isin(valid_symbols)].copy()
        
        all_X = []
        all_y = []
        
        for i, symbol in enumerate(valid_symbols):
            symbol_data = df_filtered[df_filtered['symbol'] == symbol].copy()
            
            if len(symbol_data) < config.SEQUENCE_LENGTH + config.FUTURE_WINDOW + 10:
                print(f"Пропускаем символ {symbol}: недостаточно данных ({len(symbol_data)} строк)")
                continue
            
            try:
                if i == 0:
                    X_scaled_sequences, labels = self.feature_eng.prepare_supervised_data(
                        symbol_data, 
                        threshold=config.PRICE_CHANGE_THRESHOLD,
                        future_window=config.FUTURE_WINDOW
                    )
                else:
                    temp_df_for_scaling = symbol_data.copy()
                    for col in self.feature_eng.feature_columns:
                        temp_df_for_scaling[col] = pd.to_numeric(temp_df_for_scaling[col], errors='coerce')
                    scaled_data = self.feature_eng.scaler.transform(temp_df_for_scaling[self.feature_eng.feature_columns].values)
                    
                    X_scaled_sequences, _ = self.feature_eng._create_sequences(scaled_data)
                    
                    labels = self.feature_eng.create_trading_labels(
                        symbol_data,
                        threshold=config.PRICE_CHANGE_THRESHOLD,
                        future_window=config.FUTURE_WINDOW
                    )
                    
                    min_len = min(len(X_scaled_sequences), len(labels))
                    X_scaled_sequences = X_scaled_sequences[:min_len]
                    labels = labels[:min_len]
                
                if len(X_scaled_sequences) > 0:
                    all_X.append(X_scaled_sequences)
                    all_y.append(labels)
                    print(f"Символ {symbol}: {len(X_scaled_sequences)} последовательностей")
                    
            except Exception as e:
                print(f"Ошибка при обработке {symbol}: {e}")
                continue
        
        X = np.vstack(all_X)
        y = np.concatenate(all_y)
        
        print(f"Итого подготовлено: X={X.shape}, y={y.shape}")
        print(f"Распределение классов: SELL={np.sum(y==0)}, HOLD={np.sum(y==1)}, BUY={np.sum(y==2)}")
        
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            X, y, test_size=0.1, shuffle=True, random_state=42, stratify=y
        )
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp, test_size=config.SUPERVISED_VALIDATION_SPLIT, 
            shuffle=True, random_state=42, stratify=y_temp
        )
        
        print(f"Размеры выборок: Train={len(self.X_train)}, Val={len(self.X_val)}, Test={len(self.X_test)}")
        
        self.feature_eng.save_scaler()
        
        # 🔥 ИЗМЕНЕНО: input_shape теперь использует длину feature_columns из feature_eng
        input_shape = (config.SEQUENCE_LENGTH, len(self.feature_eng.feature_columns)) 
        self.model = XLSTMRLModel(
            input_shape=input_shape,
            memory_size=config.XLSTM_MEMORY_SIZE,
            memory_units=config.XLSTM_MEMORY_UNITS
        )
        
        return True
    
    def stage1_supervised_pretraining(self):
        """ЭТАП 1: Supervised Pre-training"""
        print("=== ЭТАП 1: SUPERVISED PRE-TRAINING ===")
        
        self.model.compile_for_supervised_learning()
        
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
            ),
            ValidationMetricsCallback(self.X_val, self.y_val)
        ]
        
        print(f"Начинаем supervised обучение на {config.SUPERVISED_EPOCHS} эпох...")
        
        history = self.model.actor_model.fit(
            self.X_train, self.y_train,
            validation_data=(self.X_val, self.y_val),
            epochs=config.SUPERVISED_EPOCHS,
            batch_size=config.SUPERVISED_BATCH_SIZE,
            callbacks=callbacks,
            verbose=1
        )
        
        print("=== РЕЗУЛЬТАТЫ SUPERVISED ОБУЧЕНИЯ ===")
        
        y_pred_probs = self.model.actor_model.predict(self.X_test, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)
        
        accuracy = accuracy_score(self.y_test, y_pred)
        print(f"Точность на тестовой выборке: {accuracy:.4f}")
        
        class_names = ['SELL', 'HOLD', 'BUY']
        report = classification_report(self.y_test, y_pred, target_names=class_names, zero_division=0)
        print(f"Классификационный отчет:\n{report}")
        
        cm = confusion_matrix(self.y_test, y_pred)
        print(f"Матрица путаницы:\n{cm}")
        
        pred_dist = np.bincount(y_pred, minlength=3)
        total_pred = len(y_pred)
        print(f"Распределение предсказаний:")
        print(f"SELL: {pred_dist[0]} ({pred_dist[0]/total_pred:.2%})")
        print(f"HOLD: {pred_dist[1]} ({pred_dist[1]/total_pred:.2%})")
        print(f"BUY: {pred_dist[2]} ({pred_dist[2]/total_pred:.2%})")
        
        self.model.save(stage="_supervised")
        self.model.is_supervised_trained = True
        
        self._plot_training_history(history, "supervised")
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm,
            'history': history.history
        }
    
    def stage2_reward_model_training(self):
        """ЭТАП 2: Reward Model Training"""
        print("=== ЭТАП 2: REWARD MODEL TRAINING ===")
        
        if not self.model.is_supervised_trained:
            print("Сначала нужно завершить supervised pre-training!")
            return None
        
        self.model.compile_for_reward_modeling()
        
        print("Создаём симулированные награды...")
        
        rewards_train = self._generate_simulated_rewards(self.X_train, self.y_train)
        rewards_val = self._generate_simulated_rewards(self.X_val, self.y_val)
        
        print(f"Сгенерировано наград: Train={len(rewards_train)}, Val={len(rewards_val)}")
        print(f"Статистика наград: Mean={np.mean(rewards_train):.4f}, Std={np.std(rewards_train):.4f}")
        
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
        
        val_predictions = self.model.critic_model.predict(self.X_val, verbose=0)
        correlation = np.corrcoef(rewards_val, val_predictions.flatten())[0, 1]
        
        print(f"Корреляция между реальными и предсказанными наградами: {correlation:.4f}")
        
        self.model.save(stage="_reward_model")
        self.model.is_reward_model_trained = True
        
        self._plot_training_history(history, "reward_model")
        
        return {
            'correlation': correlation,
            'history': history.history
        }
    
    def stage3_rl_finetuning(self):
        """ЭТАП 3: RL Fine-tuning"""
        print("=== ЭТАП 3: RL FINE-TUNING ===")
        
        if not self.model.is_reward_model_trained:
            print("Сначала нужно завершить reward model training!")
            return None
        
        # 🔥 ИЗМЕНЕНО: input_shape для RLAgent
        rl_agent = RLAgent(
            state_shape=(config.SEQUENCE_LENGTH, len(self.feature_eng.feature_columns)), 
            memory_size=config.XLSTM_MEMORY_SIZE,
            memory_units=config.XLSTM_MEMORY_UNITS,
            batch_size=config.RL_BATCH_SIZE
        )
        
        rl_agent.model = self.model
        
        train_env = TradingEnvironment(self.X_train, sequence_length=config.SEQUENCE_LENGTH)
        val_env = TradingEnvironment(self.X_val, sequence_length=config.SEQUENCE_LENGTH)
        
        decision_maker = HybridDecisionMaker(rl_agent)
        train_sim = SimulationEngine(train_env, decision_maker)
        val_sim = SimulationEngine(val_env, decision_maker)
        
        rl_metrics = {
            'episode_rewards': [],
            'episode_profits': [],
            'val_rewards': [],
            'val_profits': [],
            'actor_losses': [],
            'critic_losses': []
        }
        
        best_val_profit = -float('inf')
        
        print(f"Начинаем RL fine-tuning на {config.RL_EPISODES} эпизодов...")
        
        for episode in range(config.RL_EPISODES):
            print(f"RL Эпизод {episode+1}/{config.RL_EPISODES}")
            
            train_results = train_sim.run_simulation(episodes=1, training=True)
            episode_reward = train_results[0]['total_reward']
            episode_profit = train_results[0]['profit_percentage']
            
            rl_metrics['episode_rewards'].append(episode_reward)
            rl_metrics['episode_profits'].append(episode_profit)
            
            if (episode + 1) % 10 == 0:
                val_results = val_sim.run_simulation(episodes=1, training=False)
                val_reward = val_results[0]['total_reward']
                val_profit = val_results[0]['profit_percentage']
                
                rl_metrics['val_rewards'].append(val_reward)
                rl_metrics['val_profits'].append(val_profit)
                
                sample_size = min(500, len(self.X_val))
                action_dist = rl_agent.log_action_distribution(self.X_val[:sample_size])
                
                print(f"Эпизод {episode+1}:")
                print(f"  Тренировка - Награда: {episode_reward:.4f}, Прибыль: {episode_profit:.2f}%")
                print(f"  Валидация - Награда: {val_reward:.4f}, Прибыль: {val_profit:.2f}%")
                print(f"  Действия - BUY: {action_dist['buy_count']}, HOLD: {action_dist['hold_count']}, SELL: {action_dist['sell_count']}")
                print(f"  Epsilon: {rl_agent.epsilon:.4f}")
                
                if val_profit > best_val_profit:
                    print(f"  Новая лучшая модель! Прибыль: {val_profit:.2f}%")
                    self.model.save(stage="_rl_finetuned")
                    best_val_profit = val_profit
        
        print("=== РЕЗУЛЬТАТЫ RL FINE-TUNING ===")
        print(f"Лучшая прибыль на валидации: {best_val_profit:.2f}%")
        print(f"Средняя награда за эпизод: {np.mean(rl_metrics['episode_rewards']):.4f}")
        print(f"Средняя прибыль за эпизод: {np.mean(rl_metrics['episode_profits']):.2f}%")
        
        self._plot_rl_metrics(rl_metrics)
        
        return rl_metrics
    
    def _generate_simulated_rewards(self, X, y_true):
        """Генерирует симулированные награды на основе предсказаний модели"""
        y_pred_probs = self.model.actor_model.predict(X, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)
        
        rewards = []
        for true_label, pred_label, pred_probs in zip(y_true, y_pred, y_pred_probs):
            if true_label == pred_label:
                reward = 1.0
            else:
                reward = -1.0
            
            confidence = pred_probs[pred_label]
            reward *= confidence
            
            rewards.append(reward)
        
        return np.array(rewards)
    
    def _plot_training_history(self, history, stage_name):
        """Визуализирует историю обучения"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        axes[0].plot(history.history['loss'], label='Training Loss')
        if 'val_loss' in history.history:
            axes[0].plot(history.history['val_loss'], label='Validation Loss')
        axes[0].set_title(f'{stage_name.capitalize()} Training Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        
        if 'accuracy' in history.history:
            axes[1].plot(history.history['accuracy'], label='Training Accuracy')
            if 'val_accuracy' in history.history:
                axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy')
            axes[1].set_title(f'{stage_name.capitalize()} Accuracy')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Accuracy')
            axes[1].legend()
        elif 'mae' in history.history:
            axes[1].plot(history.history['mae'], label='Training MAE')
            if 'val_mae' in history.history:
                axes[1].plot(history.history['val_mae'], label='Validation MAE')
            axes[1].set_title(f'{stage_name.capitalize()} MAE')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('MAE')
            axes[1].legend()
        
        plt.tight_layout()
        plt.savefig(f'plots/{stage_name}_training_history.png')
        plt.close()
    
    def _plot_rl_metrics(self, metrics):
        """Визуализирует RL метрики"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        axes[0,0].plot(metrics['episode_rewards'])
        axes[0,0].set_title('Episode Rewards')
        axes[0,0].set_xlabel('Episode')
        axes[0,0].set_ylabel('Reward')
        axes[0,0].grid(True)
        
        axes[0,1].plot(metrics['episode_profits'])
        axes[0,1].set_title('Episode Profits (%)')
        axes[0,1].set_xlabel('Episode')
        axes[0,1].set_ylabel('Profit %')
        axes[0,1].grid(True)
        
        if metrics['val_rewards']:
            val_episodes = range(10, len(metrics['val_rewards']) * 10 + 1, 10)
            axes[1,0].plot(val_episodes, metrics['val_rewards'])
            axes[1,0].set_title('Validation Rewards')
            axes[1,0].set_xlabel('Episode')
            axes[1,0].set_ylabel('Reward')
            axes[1,0].grid(True)
        
        if metrics['val_profits']:
            val_episodes = range(10, len(metrics['val_profits']) * 10 + 1, 10)
            axes[1,1].plot(val_episodes, metrics['val_profits'])
            axes[1,1].set_title('Validation Profits (%)')
            axes[1,1].set_xlabel('Episode')
            axes[1,1].set_ylabel('Profit %')
            axes[1,1].grid(True)
        
        plt.tight_layout()
        plt.savefig('plots/rl_training_metrics.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def run_full_training(self):
        """Запускает полное трёхэтапное обучение"""
        print("🚀 ЗАПУСК ТРЁХЭТАПНОГО ОБУЧЕНИЯ xLSTM + RL")
        
        if not self.load_and_prepare_data():
            print("Ошибка при подготовке данных")
            return None
        
        results = {}
        
        supervised_results = self.stage1_supervised_pretraining()
        if supervised_results is None:
            print("Ошибка на этапе supervised pre-training")
            return None
        results['supervised'] = supervised_results
        
        reward_results = self.stage2_reward_model_training()
        if reward_results is None:
            print("Ошибка на этапе reward model training")
            return None
        results['reward_model'] = reward_results
        
        rl_results = self.stage3_rl_finetuning()
        if rl_results is None:
            print("Ошибка на этапе RL fine-tuning")
            return None
        results['rl_finetuning'] = rl_results
        
        print("✅ ТРЁХЭТАПНОЕ ОБУЧЕНИЕ ЗАВЕРШЕНО УСПЕШНО!")
        return results

def main():
    """Основная функция"""
    data_path = "historical_data.csv"  
    if not os.path.exists(data_path):
        print(f"Файл данных {data_path} не найден!")
        return
    
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"Найдено GPU устройств: {len(gpus)}")
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("Настроен динамический рост памяти GPU")
        except RuntimeError as e:
            print(f"Не удалось настроить память GPU: {e}")
    else:
        print("GPU не найден, будет использоваться CPU")
    
    trainer = ThreeStageTrainer(data_path)
    results = trainer.run_full_training()
    
    if results:
        print("🎉 ВСЕ ЭТАПЫ ОБУЧЕНИЯ ЗАВЕРШЕНЫ!")
        
        print("=== ИТОГОВАЯ СТАТИСТИКА ===")
        print(f"Supervised Accuracy: {results['supervised']['accuracy']:.4f}")
        print(f"Reward Model Correlation: {results['reward_model']['correlation']:.4f}")
        if 'rl_finetuning' in results and results['rl_finetuning'] is not True and 'episode_profits' in results['rl_finetuning'] and len(results['rl_finetuning']['episode_profits']) > 0:
            print(f"RL Final Profit: {np.mean(results['rl_finetuning']['episode_profits'][-10:]):.2f}%")
        else:
            print("RL Fine-tuning не предоставил данные о прибыли (возможно, пропущен или не возвращает метрики)")
    else:
        print("❌ ОБУЧЕНИЕ ЗАВЕРШИЛОСЬ С ОШИБКАМИ!")

if __name__ == "__main__":
    main()
The following is the text file the user attached:
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
# import logging # 🔥 УДАЛЕНО: Импорт logging

# 🔥 УДАЛЕНО: Инициализация логгера
# logger = logging.getLogger('validation_callback')

class ValidationMetricsCallback(tf.keras.callbacks.Callback):
    &quot;&quot;&quot;
    Детальный мониторинг метрик валидации для Supervised Pre-training (Этап 1)
    &quot;&quot;&quot;
    def __init__(self, X_val, y_val, class_names&#x3D;[&#39;SELL&#39;, &#39;HOLD&#39;, &#39;BUY&#39;]):
        super().__init__()
        self.X_val &#x3D; X_val
        self.y_val &#x3D; y_val
        self.class_names &#x3D; class_names
        
    def on_epoch_end(self, epoch, logs&#x3D;None):
        if (epoch + 1) % 5 &#x3D;&#x3D; 0:  # Каждые 5 эпох
            # 🔥 ИЗМЕНЕНО: logger.info -&gt; print
            print(f&quot;\n📊 Детальные метрики на эпохе {epoch+1}:&quot;)
            
            y_pred_probs &#x3D; self.model.predict(self.X_val, verbose&#x3D;0)
            y_pred_classes &#x3D; np.argmax(y_pred_probs, axis&#x3D;1)
            
            # Если y_val уже one-hot, преобразуем
            if self.y_val.ndim &gt; 1 and self.y_val.shape[1] &gt; 1:
                y_true_classes &#x3D; np.argmax(self.y_val, axis&#x3D;1)
            else:
                y_true_classes &#x3D; self.y_val
            
            # Confusion Matrix
            cm &#x3D; confusion_matrix(y_true_classes, y_pred_classes)
            # 🔥 ИЗМЕНЕНО: logger.info -&gt; print
            print(&quot;Confusion Matrix:&quot;)
            
            # Форматируем вывод матрицы
            header &#x3D; &quot;     &quot; + &quot; &quot;.join([f&quot;{name:4s}&quot; for name in self.class_names])
            # 🔥 ИЗМЕНЕНО: logger.info -&gt; print
            print(header)
            for i, row in enumerate(cm):
                row_str &#x3D; &quot; &quot;.join([f&quot;{val:4d}&quot; for val in row])
                # 🔥 ИЗМЕНЕНО: logger.info -&gt; print
                print(f&quot;{self.class_names[i]:4s} {row_str}&quot;)
            
            # Classification Report
            report_dict &#x3D; classification_report(
                y_true_classes, y_pred_classes, 
                target_names&#x3D;self.class_names,
                output_dict&#x3D;True,
                zero_division&#x3D;0 # Избегаем предупреждений при нулевом делении
            )
            
            # 🔥 ИЗМЕНЕНО: logger.info -&gt; print
            print(f&quot;Macro Avg F1-Score: {report_dict[&#39;macro avg&#39;][&#39;f1-score&#39;]:.3f}&quot;)
            # 🔥 ИЗМЕНЕНО: logger.info -&gt; print
            print(f&quot;Weighted Avg F1-Score: {report_dict[&#39;weighted avg&#39;][&#39;f1-score&#39;]:.3f}&quot;)
            
            # Распределение предсказаний
            pred_distribution &#x3D; np.bincount(y_pred_classes, minlength&#x3D;len(self.class_names)) &#x2F; len(y_pred_classes)
            pred_dist_str &#x3D; &quot;, &quot;.join([f&quot;{name}&#x3D;{dist:.1%}&quot; for name, dist in zip(self.class_names, pred_distribution)])
            # 🔥 ИЗМЕНЕНО: logger.info -&gt; print
            print(f&quot;Распределение предсказаний: {pred_dist_str}&quot;)




