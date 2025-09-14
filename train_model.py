import pandas as pd
import numpy as np
import argparse
import os
import gc
import pickle  # ДОБАВЬТЕ эту строку
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

import tensorflow as tf

def configure_gpu_memory():
    """Настройка GPU для стабильной работы"""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Включаем рост памяти + ограничиваем максимум
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                # Ограничиваем память до 80% от доступной
                tf.config.experimental.set_virtual_device_configuration(
                    gpu,
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*10)]  # 10GB max
                )
            print(f"✅ GPU память настроена для {len(gpus)} устройств")
        except RuntimeError as e:
            print(f"⚠️ Ошибка настройки GPU: {e}")
    else:
        print("⚠️ GPU не найден, будет использоваться CPU")

if __name__ == "__main__":
    configure_gpu_memory()

# ✅ НАСТРОЙКИ ДЛЯ СОВМЕСТИМОСТИ С РАЗНЫМИ СРЕДАМИ
# Отключаем XLA если возникают проблемы (можно включить обратно)
# os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'

# Новые импорты
from feature_engineering import calculate_features, detect_candlestick_patterns
from models.xlstm_rl_model import XLSTMRLModel
from rl_agent import IntelligentRLAgent
from trading_env import TradingEnvRL
from hybrid_decision_maker import HybridDecisionMaker
from regularization_callback import AntiOverfittingCallback
from validation_metrics import ValidationMetricsCallback

import tensorflow.keras.backend as K

@tf.keras.utils.register_keras_serializable()
class CustomFocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=1.0, alpha=0.3, class_weights=None, name='CustomFocalLoss', **kwargs): # ИЗМЕНЕНО: Добавлен **kwargs
        super().__init__(name=name, **kwargs) # ИЗМЕНЕНО: Передаем **kwargs в super()
        self.gamma = gamma
        self.alpha = alpha
        # Убедимся, что class_weights - это tf.constant
        if class_weights is None:
            self.class_weights = tf.constant([1.2, 1.2, 0.8], dtype=tf.float32) # Default weights
        else:
            self.class_weights = tf.constant(class_weights, dtype=tf.float32)

    def call(self, y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        y_true = tf.cast(y_true, tf.float32)

        # Применяем класс-специфичные веса
        weights = tf.reduce_sum(self.class_weights * y_true, axis=-1, keepdims=True)
        
        cross_entropy = -y_true * K.log(y_pred)
        loss = self.alpha * K.pow(1 - y_pred, self.gamma) * cross_entropy * weights
        
        return K.sum(loss, axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({
            'gamma': self.gamma,
            'alpha': self.alpha,
            'class_weights': self.class_weights.numpy().tolist(), # Сохраняем как список
        })
        return config

def prepare_xlstm_rl_data(data_path, sequence_length=10):
    """
    Подготавливает данные для единой xLSTM+RL системы
    """
    print(f"Загрузка данных из {data_path}...")
    full_df = pd.read_csv(data_path)
    
    # Объединенные признаки для новой архитектуры
    feature_cols = [
        # ✅ ВСЕ ТЕХНИЧЕСКИЕ ИНДИКАТОРЫ (БЕЗ БОЛЛИНДЖЕРА И ATR_14)
        'RSI_14', 'MACD_12_26_9', 'MACD_signal', 'MACD_hist',
        'ADX_14', 'STOCHk_14_3_3', 'STOCHd_14_3_3',
        'WILLR_14', # 🔥 НОВЫЙ ИНДИКАТОР
        'AO_5_34',  # 🔥 НОВЫЙ ИНДИКАТОР
        
        # ❌ ВСЕ ПАТТЕРНЫ ЗАКОММЕНТИРОВАНЫ
        # 'CDLHAMMER', 'CDLENGULFING', 'CDLDOJI', 'CDLSHOOTINGSTAR',
        # 'CDLHANGINGMAN', 'CDLMARUBOZU',
        # 'CDLINVERTEDHAMMER', 'CDLDRAGONFLYDOJI', 'CDLBELTHOLD',
        # 'hammer_f', 'hangingman_f', 'engulfing_f', 'doji_f',
        # 'shootingstar_f', 'bullish_marubozu_f',
        # 'inverted_hammer_f', 'dragonfly_doji_f', 'bullish_pin_bar_f', 'bullish_belt_hold_f',
        
        # ✅ ОСТАВЛЯЕМ EVENT SAMPLING
        'is_event'
    ]
    
    all_X = []
    all_y = []
    processed_dfs = {}  # Сохраняем обработанные данные для RL
    
    symbols = full_df['symbol'].unique()
    print(f"Найдено {len(symbols)} символов. Обрабатываем каждый...")
    
    for symbol in symbols:
        df = full_df[full_df['symbol'] == symbol].copy()
        print(f"\nОбработка символа: {symbol}, строк: {len(df)}")
        
        if len(df) < sequence_length + 50:  # Нужно достаточно данных
            continue
            
        # === НОВАЯ ОБРАБОТКА С VSA ===
        df = calculate_features(df)
        df = detect_candlestick_patterns(df)
        # df = calculate_vsa_features(df)  # <--- ЗАКОММЕНТИРОВАНО: Временно отключаем VSA
        
        # =====================================================================
        # НОВЫЙ БЛОК: ФИЛЬТРАЦИЯ ПО 'is_event' (Event-Based Sampling)
        # =====================================================================
        initial_rows = len(df)
        df_event_filtered = df[df['is_event'] == 1].copy()
        
        if len(df_event_filtered) < sequence_length + 50:
            print(f"⚠️ Для символа {symbol} недостаточно событий ({len(df_event_filtered)}), использую все данные.")
            df_processed = df.copy() # Если событий мало, используем все данные
        else:
            print(f"✅ Для символа {symbol} отфильтровано {len(df_event_filtered)} событий из {initial_rows} баров.")
            df_processed = df_event_filtered.copy()
        
        # Сброс индекса после фильтрации, чтобы избежать проблем
        df_processed.reset_index(drop=True, inplace=True)

        if len(df_processed) < sequence_length + 50: # Проверяем после фильтрации
            continue # Пропускаем символ, если данных все еще недостаточно
        
        df = df_processed # Теперь работаем с отфильтрованным DataFrame
        # =====================================================================
        # КОНЕЦ НОВОГО БЛОКА
        # =====================================================================
        
        # Создаем целевые метки на основе будущих цен + индикаторов
        df['future_return'] = (df['close'].shift(-5) - df['close']) / df['close']
        
        # Увеличиваем пороги для генерации торговых сигналов
        df['base_threshold'] = 0.003 # 🔥 ИЗМЕНЕНО: С 0.008 до 0.003 (более мягкий порог)
        df['dynamic_threshold'] = np.maximum(
            df['base_threshold'],
            (abs(df['AO_5_34']) / df['close'] * 1.0).fillna(0.003) # 🔥 ИЗМЕНЕНО: Использование AO_5_34 вместо ATR_14
        )

        # Классические технические фильтры
        strong_trend = df['ADX_14'] > 20 # 🔥 ИЗМЕНЕНО: С 25 до 20 (более мягкий порог)
        
        # Условия для BUY
        rsi_buy_zone = df['RSI_14'] < 40 # 🔥 ИЗМЕНЕНО: С 30 до 40
        macd_buy_signal = (df['MACD_12_26_9'] > df['MACD_signal']) & \
                          (df['MACD_hist'] > 0.0005) # 🔥 ИЗМЕНЕНО: С 0.001 до 0.0005
        willr_buy_signal = df['WILLR_14'] < -80 # 🔥 НОВОЕ: WILLR_14 для BUY
        ao_buy_signal = df['AO_5_34'] > 0 # 🔥 НОВОЕ: AO выше нуля
        
        # Условия для SELL
        rsi_sell_zone = df['RSI_14'] > 60 # 🔥 ИЗМЕНЕНО: С 70 до 60
        macd_sell_signal = (df['MACD_12_26_9'] < df['MACD_signal']) & \
                           (df['MACD_hist'] < -0.0005) # 🔥 ИЗМЕНЕНО: С -0.001 до -0.0005
        willr_sell_signal = df['WILLR_14'] > -20 # 🔥 НОВОЕ: WILLR_14 для SELL
        ao_sell_signal = df['AO_5_34'] < 0 # 🔥 НОВОЕ: AO ниже нуля

        # Условия для BUY/SELL только на основе future_return и классических индикаторов
        buy_condition = (
            (df['future_return'] > df['dynamic_threshold'] * 1.0) &
            (strong_trend & (rsi_buy_zone | macd_buy_signal | willr_buy_signal | ao_buy_signal)) # 🔥 ИЗМЕНЕНО: Смешанные условия с OR
        )

        sell_condition = (
            (df['future_return'] < -df['dynamic_threshold'] * 1.0) &
            (strong_trend & (rsi_sell_zone | macd_sell_signal | willr_sell_signal | ao_sell_signal)) # 🔥 ИЗМЕНЕНО: Смешанные условия с OR
        )
        
        # Устанавливаем метки
        df['target'] = 2  # По умолчанию HOLD
        df.loc[buy_condition, 'target'] = 0  # BUY
        df.loc[sell_condition, 'target'] = 1  # SELL

        # 🔥 НОВЫЕ ЛОГИ: Количество сигналов до балансировки
        initial_buy_signals = (df['target'] == 0).sum()
        initial_sell_signals = (df['target'] == 1).sum()
        initial_hold_signals = (df['target'] == 2).sum()
        total_initial_signals = len(df)
        print(f"📊 Исходный баланс классов для {symbol} (до imblearn):")
        print(f"  BUY: {initial_buy_signals} ({initial_buy_signals/total_initial_signals*100:.2f}%)")
        print(f"  SELL: {initial_sell_signals} ({initial_sell_signals/total_initial_signals*100:.2f}%)")
        print(f"  HOLD: {initial_hold_signals} ({initial_hold_signals/total_initial_signals*100:.2f}%)")
        print(f"  Общее количество сигналов: {total_initial_signals}")

        current_buy_count = (df['target'] == 0).sum()
        current_sell_count = (df['target'] == 1).sum()
        current_hold_count = (df['target'] == 2).sum()

        # НОВЫЙ КОД - Менее агрессивная переклассификация HOLD
        if current_hold_count > (current_buy_count + current_sell_count) * 3.0:
            print(f"⚠️ Сильный дисбаланс классов. Попытка УМНОЙ переклассификации части HOLD-сигналов (с индикаторами).")
            hold_indices = df[df['target'] == 2].index
            
            import random
            random.seed(42)
            
            reclassify_count = int(current_hold_count * 0.10)
            if reclassify_count > 0:
                reclassify_indices = random.sample(list(hold_indices), min(reclassify_count, len(hold_indices)))
                
                for idx in reclassify_indices:
                    if idx < 5: continue
                    
                    rsi = df.loc[idx, 'RSI_14']
                    adx = df.loc[idx, 'ADX_14']
                    macd_hist = df.loc[idx, 'MACD_hist']
                    willr = df.loc[idx, 'WILLR_14'] # 🔥 НОВОЕ
                    ao = df.loc[idx, 'AO_5_34']     # 🔥 НОВОЕ
                    price_change_3_period = df['close'].pct_change(3).loc[idx]

                    # Условия для переклассификации (с индикаторами) - теперь с AO и WILLR
                    # 1. RSI + ADX + MACD_hist + WILLR + AO + движение цены
                    if (rsi < 40 and adx > 20 and macd_hist > 0.0005 and willr < -80 and ao > 0 and price_change_3_period > 0.003): # 🔥 ИЗМЕНЕНО
                        df.loc[idx, 'target'] = 0  # BUY
                    elif (rsi > 60 and adx > 20 and macd_hist < -0.0005 and willr > -20 and ao < 0 and price_change_3_period < -0.003): # 🔥 ИЗМЕНЕНО
                        df.loc[idx, 'target'] = 1  # SELL
                    
                    # 2. Сильный тренд по ADX + движение цены (без других индикаторов для более широкого охвата)
                    elif (adx > 30 and abs(price_change_3_period) > 0.005): # 🔥 ИЗМЕНЕНО: Порог ADX и price_change
                        df.loc[idx, 'target'] = 0 if price_change_3_period > 0 else 1
            
            print(f"Баланс классов после УМНОЙ переклассификации (с индикаторами):")
            unique, counts = np.unique(df['target'], return_counts=True)
            class_names = ['BUY', 'SELL', 'HOLD']
            for class_idx, count in zip(unique, counts):
                print(f"  {class_names[class_idx]}: {count} ({count/len(df)*100:.1f}%)")
        else:
            print(f"✅ Баланс классов приемлемый, переклассификация HOLD не требуется.")
        
        # Убираем NaN и обеспечиваем наличие всех признаков
        df.dropna(subset=['future_return'], inplace=True)
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0
        
        df = df[feature_cols + ['target', 'close', 'volume']].copy()
        
        # Сохраняем обработанные данные для RL
        processed_dfs[symbol] = df
        
        # Создаем последовательности для xLSTM
        if len(df) > sequence_length:
            for i in range(len(df) - sequence_length):
                all_X.append(df.iloc[i:i + sequence_length][feature_cols].values)
                all_y.append(df.iloc[i + sequence_length]['target'])
    
    if not all_X:
        raise ValueError("Нет данных для обучения после обработки всех символов")
        
    print(f"Создано последовательностей: {len(all_X)}")
    
    X = np.array(all_X, dtype=np.float32)
    y = to_categorical(np.array(all_y), num_classes=3)
    
    # Исправляем NaN/Inf значения
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    
    # =====================================================================
    # НОВЫЙ БЛОК: ИСПОЛЬЗОВАНИЕ IMBLEARN ДЛЯ БАЛАНСИРОВКИ КЛАССОВ
    # =====================================================================
    try:
        from imblearn.over_sampling import SMOTE
        from imblearn.under_sampling import RandomUnderSampler
        from imblearn.pipeline import Pipeline
        from collections import Counter

        print("\n🔄 Применяю Oversampling/Undersampling для балансировки классов...")
        
        # Преобразуем y обратно в одномерный массив для imblearn
        y_labels = np.argmax(y, axis=1)
        
        print(f"Баланс классов ДО imblearn: {Counter(y_labels)}")

        # Целевое соотношение:
        # Начнем с более агрессивного: 20% BUY, 20% SELL, 60% HOLD
        # Вы можете настроить эти проценты.
        # Важно: SMOTE работает с индексами классов (0, 1, 2)
        
        # Сначала oversampling меньшинства (BUY, SELL)
        # Увеличиваем BUY и SELL до 20% от общего числа примеров
        # (предполагаем, что общее число примеров будет около len(X) * (1 + oversampling_ratio))
        
        # Рассчитываем целевые количества на основе общего числа примеров
        # Целевое соотношение: 15% BUY, 15% SELL, 70% HOLD (более реалистичный oversampling)
        total_samples = len(X)
        target_buy_count = int(total_samples * 0.15) # 🔥 ИЗМЕНЕНО: с 0.20 до 0.15
        target_sell_count = int(total_samples * 0.15) # 🔥 ИЗМЕНЕНО: с 0.20 до 0.15
        
        current_buy_count = Counter(y_labels)[0]
        current_sell_count = Counter(y_labels)[1]

        sampling_strategy_smote = {
            0: max(current_buy_count, target_buy_count),
            1: max(current_sell_count, target_sell_count)
        }
        
        if current_buy_count > 0 or current_sell_count > 0:
            k_neighbors = min(5,
                              (current_buy_count - 1 if current_buy_count > 1 else 1),
                              (current_sell_count - 1 if current_sell_count > 1 else 1))
            k_neighbors = max(1, k_neighbors)

            if any(count <= k_neighbors for count in [current_buy_count, current_sell_count] if count > 0):
                print("⚠️ Недостаточно сэмплов для SMOTE с k_neighbors, использую RandomOverSampler.")
                from imblearn.over_sampling import RandomOverSampler
                oversampler = RandomOverSampler(sampling_strategy=sampling_strategy_smote, random_state=42)
            else:
                oversampler = SMOTE(sampling_strategy=sampling_strategy_smote, random_state=42, k_neighbors=k_neighbors)

            X_temp, y_temp_labels = oversampler.fit_resample(X.reshape(len(X), -1), y_labels)
            print(f"Баланс классов после Oversampling: {Counter(y_temp_labels)} (BUY/SELL увеличены)")
        else:
            X_temp, y_temp_labels = X.reshape(len(X), -1), y_labels
            print("Пропустил Oversampling, так как нет BUY/SELL сигналов.")

        # Undersampling HOLD: Цель - чтобы HOLD был примерно в 1.5 раза больше, чем сумма BUY + SELL
        current_hold_count_after_oversample = Counter(y_temp_labels)[2]
        target_hold_count = min(current_hold_count_after_oversample, int((Counter(y_temp_labels)[0] + Counter(y_temp_labels)[1]) * 1.5)) # ИЗМЕНЕНО: с 3.0 до 1.5
        
        undersampler = RandomUnderSampler(sampling_strategy={2: target_hold_count}, random_state=42)
        X_resampled, y_resampled_labels = undersampler.fit_resample(X_temp, y_temp_labels)

        # Преобразуем X обратно в 3D форму
        X = X_resampled.reshape(len(X_resampled), sequence_length, X.shape[-1])
        # Преобразуем метки обратно в one-hot
        y = to_categorical(y_resampled_labels, num_classes=3)

        print(f"✅ Балансировка завершена. Новое количество последовательностей: {len(X)}")
        print(f"Новый баланс классов ПОСЛЕ imblearn:")
        unique, counts = np.unique(np.argmax(y, axis=1), return_counts=True)
        class_names = ['BUY', 'SELL', 'HOLD']
        for class_idx, count in zip(unique, counts):
            print(f"  {class_names[class_idx]}: {count} ({count/len(y)*100:.1f}%)")

    except ImportError:
        print("⚠️ imbalanced-learn не установлен. Пропустил oversampling/undersampling. Установите: pip install imbalanced-learn")
    except Exception as e:
        print(f"❌ Ошибка при oversampling/undersampling: {e}")
    # =====================================================================
    # КОНЕЦ НОВОГО БЛОКА IMBLEARN
    # =====================================================================
    
    # ДОБАВЬТЕ: Проверка баланса классов
    print(f"Баланс классов:")
    unique, counts = np.unique(np.argmax(y, axis=1), return_counts=True)
    class_names = ['BUY', 'SELL', 'HOLD']
    for class_idx, count in zip(unique, counts):
        print(f"  {class_names[class_idx]}: {count} ({count/len(y)*100:.1f}%)")
    
    return X, y, processed_dfs, feature_cols

def train_xlstm_rl_system(X, y, processed_dfs, feature_cols):
    """
    Обучает единую систему xLSTM + RL
    """
    print("\n=== ЭТАП 1: ОБУЧЕНИЕ xLSTM МОДЕЛИ ===")
    
    # ДОБАВЬТЕ: Создание директории для моделей
    os.makedirs('models', exist_ok=True)
    
    # =====================================================================
    # =====================================================================
    # НОВЫЙ БЛОК: УЛУЧШЕННЫЙ TIME SERIES SPLIT
    # =====================================================================
    from sklearn.model_selection import StratifiedKFold # ИЗМЕНЕНО: Используем StratifiedKFold
    from collections import Counter

    print("\n🔄 Применяю СТРАТИФИЦИРОВАННЫЙ TimeSeriesSplit для валидации данных...")

    # Сначала отделим тестовую выборку (последние 20% данных)
    test_size = int(len(X) * 0.2)
    X_temp, X_test = X[:-test_size], X[-test_size:]
    y_temp, y_test = y[:-test_size], y[-test_size:]

    # Для тренировочной и валидационной выборки используем StratifiedKFold,
    # чтобы обеспечить наличие всех классов в каждом сплите.
    # Мы не можем использовать TimeSeriesSplit со стратификацией напрямую,
    # поэтому имитируем его, беря последние данные для валидации.
    n_splits_stratified = 5 # ИЗМЕНЕНО: Используем 5 сплитов для лучшего распределения
    skf = StratifiedKFold(n_splits=n_splits_stratified, shuffle=False) # shuffle=False для сохранения временного порядка

    train_indices_list = []
    val_indices_list = []

    # Сохраняем метки классов для стратификации
    y_temp_labels = np.argmax(y_temp, axis=1)

    for train_idx, val_idx in skf.split(X_temp, y_temp_labels): # ИЗМЕНЕНО: Используем y_temp_labels для стратификации
        train_indices_list.append(train_idx)
        val_indices_list.append(val_idx)

    # Берем последний сплит для тренировки и валидации, чтобы сохранить "временной" аспект
    X_train, y_train = X_temp[train_indices_list[-1]], y_temp[train_indices_list[-1]]
    X_val, y_val = X_temp[val_indices_list[-1]], y_temp[val_indices_list[-1]]

    print(f"✅ Стратифицированный TimeSeriesSplit завершен.")
    print(f"Распределение классов в X_train: {Counter(np.argmax(y_train, axis=1))}")
    print(f"Распределение классов в X_val: {Counter(np.argmax(y_val, axis=1))}")
    # =====================================================================
    # КОНЕЦ НОВОГО БЛОКА
    # =====================================================================
    
    # =====================================================================
    # НОВЫЙ БЛОК: ВЫЧИСЛЕНИЕ И ПЕРЕДАЧА ВЕСОВ КЛАССОВ
    # =====================================================================
    from sklearn.utils.class_weight import compute_class_weight
    y_integers = np.argmax(y_train, axis=1)
    class_weights_array = compute_class_weight(
        'balanced',
        classes=np.unique(y_integers),
        y=y_integers
    )
    class_weight_dict = {i: class_weights_array[i] for i in range(len(class_weights_array))}

    # НОВЫЙ КОД - БАЛАНСИРУЕМ ВЕСА ПРАВИЛЬНО
    # Проблема: HOLD имеет слишком низкий recall, нужно УВЕЛИЧИТЬ его вес
    
    # Увеличиваем веса BUY/SELL немного, чтобы модель уделяла им больше внимания,
    # но не настолько, чтобы она полностью игнорировала HOLD.
    # Уменьшаем вес HOLD, но не слишком сильно.
    # НОВЫЙ КОД - Корректируем веса классов (более сбалансированные, с акцентом на HOLD)
    if 0 in class_weight_dict:
        class_weight_dict[0] *= 1.5  # ИЗМЕНЕНО: Уменьшаем BUY (с 1.8 до 1.5)
    if 1 in class_weight_dict:
        class_weight_dict[1] *= 1.5  # ИЗМЕНЕНО: Устанавливаем SELL на 1.5 (равный BUY)
    
    if 2 in class_weight_dict:
        class_weight_dict[2] *= 2.0  # ИЗМЕНЕНО: Значительно увеличиваем HOLD (с 1.5 до 2.0)
    
    print(f"📊 ИСПРАВЛЕННЫЕ веса классов: {class_weight_dict}")
    # =====================================================================
    # КОНЕЦ НОВОГО БЛОКА
    # =====================================================================
    
    # ДОБАВЬТЕ: Принудительная очистка памяти
    gc.collect()
    tf.keras.backend.clear_session()
    
    print(f"Обучающая выборка: {len(X_train)}")
    print(f"Валидационная выборка: {len(X_val)}")
    print(f"Тестовая выборка: {len(X_test)}")
    
    # В функции train_xlstm_rl_system(), после создания xlstm_model добавьте:
    print(f"Форма данных для обучения: X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"Количество признаков: {len(feature_cols)}")

    # Проверяем наличие NaN/Inf в данных перед обучением
    if np.any(np.isnan(X_train)) or np.any(np.isinf(X_train)):
        print("⚠️ Обнаружены NaN/Inf в тренировочных данных!")
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
        print("✅ NaN/Inf исправлены")
        
    # =====================================================================
    # НОВЫЙ БЛОК: ИНИЦИАЛИЗАЦИЯ/ЗАГРУЗКА МОДЕЛИ
    # =====================================================================
    xlstm_model = XLSTMRLModel(
        input_shape=(X_train.shape[1], X_train.shape[2]),
        memory_units=128,
        attention_units=64
    )
    xlstm_model.build_model() # <--- СНАЧАЛА СТРОИМ МОДЕЛЬ!

    checkpoint_path = 'models/xlstm_checkpoint_latest.keras'
    scaler_path = 'models/xlstm_rl_scaler.pkl'

    if os.path.exists(checkpoint_path):
        print("Найдена сохраненная модель, загружаем веса...")
        try:
            # Загружаем только веса, так как архитектура уже построена
            xlstm_model.model.load_weights(checkpoint_path) # <--- ИЗМЕНЕНО: load_weights вместо load_model
            
            # Загружаем scaler если он существует
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    xlstm_model.scaler = pickle.load(f)
                xlstm_model.is_trained = True
                print("✅ Веса модели и scaler загружены, продолжаем обучение")
            else:
                print("⚠️ Scaler не найден, будет использован новый")
                
        except Exception as e:
            print(f"⚠️ Не удалось загрузить веса модели: {e}, начинаем обучение с нуля.")
            # В этом случае модель останется с инициализированными весами, что и нужно.
    else:
        print("Нет сохраненной модели, начинаем обучение с нуля.")
    # =====================================================================
    # КОНЕЦ НОВОГО БЛОКА
    # =====================================================================

    # Нормализация данных
    X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
    xlstm_model.scaler.fit(X_train_reshaped)
    X_train_scaled = xlstm_model.scaler.transform(X_train_reshaped).reshape(X_train.shape)
    X_val_scaled = xlstm_model.scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)

    # =====================================================================
    # НОВЫЙ БЛОК: ИНЪЕКЦИЯ ШУМА ВО ВХОДНЫЕ ПРИЗНАКИ (для регуляризации)
    # Этот блок теперь находится СРАЗУ ПОСЛЕ определения X_train_scaled и X_val_scaled
    # =====================================================================
    print("\n шумовые входные данные...")
    # Добавляем шум только к тренировочной выборке
    noise_std_multiplier = 0.005 # Коэффициент для стандартного отклонения шума (0.5%)

    # Более простой подход: шум на основе общего стандартного отклонения или фиксированного значения
    # Добавляем шум к масштабированным данным
    noise_level = np.std(X_train_scaled) * noise_std_multiplier # Шум пропорционален std данных

    X_train_noisy = X_train_scaled + np.random.normal(0, noise_level, X_train_scaled.shape)
    # Оставим валидацию без шума для чистоты оценки
    X_val_noisy = X_val_scaled

    # Теперь передаем зашумленные данные в модель
    X_train_to_model = X_train_noisy
    X_val_to_model = X_val_noisy
    print(f"✅ Шум добавлен к тренировочным данным (уровень шума: {noise_level:.4f})")
    # =====================================================================
    # КОНЕЦ НОВОГО БЛОКА
    # =====================================================================

    # Теперь xlstm_model.model гарантированно не None, можно компилировать
    # НОВЫЙ КОД - Инициализация Learning Rate как float
    # НОВЫЙ КОД - Уменьшаем Learning Rate
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.0002,  # ИЗМЕНЕНО: Уменьшаем LR с 0.0005 до 0.0002
        clipnorm=0.5,
        weight_decay=0.0001
    )
    xlstm_model.model.compile(
        optimizer=optimizer,
        loss=CustomFocalLoss(gamma=1.0, alpha=0.3, class_weights=[1.2, 1.2, 0.8]), # ИЗМЕНЕНО: Используем класс
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.Precision(name='precision_0', class_id=0),
            tf.keras.metrics.Precision(name='precision_1', class_id=1),
            tf.keras.metrics.Precision(name='precision_2', class_id=2),
            tf.keras.metrics.Recall(name='recall_0', class_id=0),
            tf.keras.metrics.Recall(name='recall_1', class_id=1),
            tf.keras.metrics.Recall(name='recall_2', class_id=2),
        ]
    )

    # ДОБАВЬТЕ: Кастомные колбэки для мониторинга и очистки памяти
    class MemoryCleanupCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if epoch % 10 == 0:  # Каждые 10 эпох
                gc.collect()
                # ИСПРАВЛЕНО: НЕ очищаем сессию во время обучения
                # tf.keras.backend.clear_session()  # Это может сломать обучение!
                print(f"Эпоха {epoch}: Память очищена")
    
    class DetailedProgressCallback(tf.keras.callbacks.Callback):
        def __init__(self, X_val, feature_cols, class_names=['BUY', 'SELL', 'HOLD']): # ИЗМЕНЕНО: Добавлены X_val, feature_cols
            super().__init__()
            self.X_val = X_val
            self.feature_cols = feature_cols
            self.class_names = class_names

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {} # Убедимся, что logs не None
            try:
                lr = self.model.optimizer.learning_rate.numpy()
                # ИЗМЕНЕНО: Добавлены метрики accuracy, precision, recall
                print(f"Эпоха {epoch+1}/100 - loss: {logs.get('loss', 0):.4f} - val_loss: {logs.get('val_loss', 0):.4f} - "
                      f"accuracy: {logs.get('accuracy', 0):.2f} - val_accuracy: {logs.get('val_accuracy', 0):.2f} - "
                      f"precision: {logs.get('precision', 0):.2f} - val_precision: {logs.get('val_precision', 0):.2f} - "
                      f"recall: {logs.get('recall', 0):.2f} - val_recall: {logs.get('val_recall', 0):.2f} - lr: {lr:.2e}")
            
                # ДОБАВЛЕНО: Вывод метрик по классам (если доступны)
                if 'precision_0' in logs:
                    print(f"  Class 0 (BUY): Prec={logs.get('precision_0', 0):.2f}, Rec={logs.get('recall_0', 0):.2f}")
                if 'precision_1' in logs:
                    print(f"  Class 1 (SELL): Prec={logs.get('precision_1', 0):.2f}, Rec={logs.get('recall_1', 0):.2f}")
                if 'precision_2' in logs:
                    print(f"  Class 2 (HOLD): Prec={logs.get('precision_2', 0):.2f}, Rec={logs.get('recall_2', 0):.2f}")

                # НОВЫЙ КОД - Анализ важности признаков
                if epoch % 5 == 0 and self.X_val is not None and self.feature_cols is not None: # Анализируем каждые 5 эпох
                    print("\n📈 ТОП-10 ВЛИЯТЕЛЬНЫХ ПРИЗНАКОВ (на валидации):")
                    # Получаем предсказания модели на валидационном наборе
                    val_preds = self.model.predict(self.X_val, verbose=0)
                    predicted_classes = np.argmax(val_preds, axis=1)

                    # Для каждого класса, найдем признаки, которые чаще всего были активны
                    class_influence = {0: [], 1: [], 2: []} # BUY, SELL, HOLD

                    for class_id in range(3):
                        # Выбираем только те данные валидации, где модель предсказала этот класс
                        class_indices = np.where(predicted_classes == class_id)[0]
                        if len(class_indices) == 0:
                            continue

                        # Берем соответствующие признаки из X_val
                        # Усредняем активации признаков для этого класса
                        active_features = self.X_val[class_indices, -1, :] # Берем признаки последней свечи в последовательности
                        
                        # Определяем "активность" признака (например, если его значение > 0.5 или просто его значение)
                        # Для простоты, мы будем считать среднее значение признака
                        avg_active_features = np.mean(active_features, axis=0)

                        # Создаем пары (значение, имя_признака)
                        feature_scores = [(avg_active_features[i], self.feature_cols[i]) for i in range(len(self.feature_cols))]
                        
                        # Сортируем по абсолютному значению
                        feature_scores.sort(key=lambda x: abs(x[0]), reverse=True)
                        
                        # Выводим топ-10
                        print(f"  Для класса {self.class_names[class_id]}:")
                        for score, name in feature_scores[:10]:
                            print(f"    - {name}: {score:.4f}")

                # Проверяем на переобучение
                if logs.get('val_loss', 0) > logs.get('loss', 0) * 2:
                    print("⚠️ Возможное переобучение!")
            except Exception as e:
                print(f"Эпоха {epoch+1}/100 - loss: {logs.get('loss', 0):.4f} - val_loss: {logs.get('val_loss', 0):.4f} (Ошибка в логировании: {e})")
            
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            mode='min',
            patience=15,
            restore_best_weights=True,
            verbose=1,
            min_delta=0.001
        ),
        AntiOverfittingCallback(patience=12, min_improvement=0.005),  # ИЗМЕНЕНО: patience до 12
        MemoryCleanupCallback(),
        DetailedProgressCallback(X_val_to_model, feature_cols), # ИЗМЕНЕНО: Передаем X_val_to_model и feature_cols
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.7,
            patience=8,
            min_lr=1e-6,
            verbose=1
        ),
        ValidationMetricsCallback(X_val_to_model, y_val),  # НОВЫЙ КОЛБЭК
    ]

    # Обучение с улучшенными колбэками
    history = xlstm_model.train(
        X_train_to_model, y_train, # <--- ИЗМЕНЕНО: используем X_train_to_model
        X_val_to_model, y_val,     # <--- ИЗМЕНЕНО: используем X_val_to_model
        epochs=80,      # Уменьшаем количество эпох с 100 до 80
        batch_size=32,  # Увеличиваем batch_size с 16 до 32
        class_weight=class_weight_dict,
        custom_callbacks=callbacks
    )

    # После завершения обучения xLSTM, добавьте:
    print(f"\n📊 Статистика обучения xLSTM:")
    print(f"Финальная loss: {history.history['loss'][-1]:.4f}")
    print(f"Финальная val_loss: {history.history['val_loss'][-1]:.4f}")
    print(f"Лучшая val_loss: {min(history.history['val_loss']):.4f}")
    print(f"Количество эпох: {len(history.history['loss'])}")
    
    # Оценка xLSTM
    try:
        X_test_scaled = xlstm_model.scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
        # НОВЫЙ КОД - Использование model.evaluate с return_dict=True
        evaluation_results_dict = xlstm_model.model.evaluate(X_test_scaled, y_test, verbose=0, return_dict=True) # ИЗМЕНЕНО: return_dict=True

        loss = evaluation_results_dict.get('loss', 0.0)
        accuracy = evaluation_results_dict.get('accuracy', 0.0)
        precision = evaluation_results_dict.get('precision', 0.0)
        recall = evaluation_results_dict.get('recall', 0.0)

        print(f"xLSTM Loss: {loss:.4f}")
        print(f"xLSTM Точность: {accuracy * 100:.2f}%")
        print(f"xLSTM Precision: {precision * 100:.2f}%")
        print(f"xLSTM Recall: {recall * 100:.2f}%")

        # Выводим метрики по классам, если они есть
        for i, class_name in enumerate(['BUY', 'SELL', 'HOLD']):
            prec_i = evaluation_results_dict.get(f'precision_{i}', 0.0)
            rec_i = evaluation_results_dict.get(f'recall_{i}', 0.0)
            print(f"  Class {i} ({class_name}): Prec={prec_i:.2f}, Rec={rec_i:.2f}")
    except Exception as e:
        print(f"⚠️ Ошибка при оценке модели: {e}")
    
    # Сохраняем xLSTM модель
    xlstm_model.save_model()

    # После обучения xlstm_model, обучите детектор режимов
    # Возьмите достаточно большой исторический DataFrame для обучения детектора
    # Например, объедините несколько символов или возьмите один большой
    regime_training_df = pd.concat(list(processed_dfs.values())).reset_index(drop=True)
    decision_maker_temp = HybridDecisionMaker(
        xlstm_model_path='models/xlstm_rl_model.keras',
        rl_agent_path=None,  # <--- ИЗМЕНЕНО: Передаем None, так как RL агент еще не обучен
        feature_columns=feature_cols,
        sequence_length=X.shape[1]
    )
    decision_maker_temp.fit_regime_detector(regime_training_df, xlstm_model, feature_cols)
    decision_maker_temp.regime_detector.save_detector('models/market_regime_detector.pkl')
    print("✅ Детектор режимов сохранен")
    
    print("\n=== ЭТАП 2: ОБУЧЕНИЕ RL АГЕНТА ===")
    
    # ДОБАВЬТЕ: Ограничиваем количество символов для стабильности
    rl_symbols = list(processed_dfs.keys())[:2]  # Только 2 символа вместо 3
    
    rl_agent = None
    for i, symbol in enumerate(rl_symbols):
        df = processed_dfs[symbol]
        print(f"\nОбучение RL на символе {symbol} ({i+1}/{len(rl_symbols)})")
        
        # ДОБАВЬТЕ: Очистка памяти перед каждым RL агентом
        gc.collect()
        
        # Разделяем данные
        split_idx = int(len(df) * 0.8)
        train_df = df.iloc[:split_idx].copy()
        eval_df = df.iloc[split_idx:].copy()
        
        if len(train_df) < 100 or len(eval_df) < 50:
            print(f"⚠️ Недостаточно данных для {symbol}, пропускаем")
            continue
            
        # Создаем RL агента
        rl_agent = IntelligentRLAgent(algorithm='PPO')
        
        try:
            # Создаем среды
            vec_env = rl_agent.create_training_environment(train_df, xlstm_model)
            rl_agent.create_evaluation_environment(eval_df, xlstm_model)
            
            # Строим и обучаем агента
            rl_agent.build_agent(vec_env)
            
            # ДОБАВЬТЕ: Обучение меньшими порциями с сохранениями
            for step in range(0, 50000, 10000):  # По 10k шагов
                print(f"RL обучение: шаги {step}-{min(step+10000, 50000)}")
                rl_agent.train_with_callbacks(
                    total_timesteps=10000,
                    eval_freq=2000
                )
                # Сохраняем промежуточные результаты
                rl_agent.save_agent(f'models/rl_agent_{symbol}_step_{step}')
                gc.collect()  # Очищаем память
            
            # Финальное сохранение
            rl_agent.save_agent(f'models/rl_agent_{symbol}')
            
        except Exception as e:
            print(f"⚠️ Ошибка при обучении RL для {symbol}: {e}")
            continue
    
    print("\n=== ОБУЧЕНИЕ ЗАВЕРШЕНО ===")
    
    print("\n🔍 Проверка сохраненных файлов:")
    saved_files = [
        'models/xlstm_rl_model.keras',
        'models/xlstm_rl_scaler.pkl',
        'models/market_regime_detector.pkl'
    ]

    for file_path in saved_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path) / (1024*1024)  # MB
            print(f"✅ {file_path} ({size:.1f} MB)")
        else:
            print(f"❌ {file_path} не найден")

    # Проверяем RL агентов
    for symbol in rl_symbols:
        rl_path = f'models/rl_agent_{symbol}'
        if os.path.exists(rl_path + '.zip'):
            size = os.path.getsize(rl_path + '.zip') / (1024*1024)
            print(f"✅ {rl_path}.zip ({size:.1f} MB)")
            
    return xlstm_model, rl_agent

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Обучение новой системы xLSTM + VSA + RL')
    parser.add_argument('--data', type=str, default='historical_data.csv', help='Путь к данным')
    parser.add_argument('--sequence_length', type=int, default=10, help='Длина последовательности')
    args = parser.parse_args()
    
    try:
        # Подготавливаем данные
        X, y, processed_dfs, feature_cols = prepare_xlstm_rl_data(args.data, args.sequence_length)
        
        # Обучаем систему
        xlstm_model, rl_agent = train_xlstm_rl_system(X, y, processed_dfs, feature_cols)
        
        print("✅ Новая система xLSTM + VSA + RL успешно обучена!")
        
    except Exception as e:
        print(f"❌ Ошибка при обучении: {e}")
        import traceback
        traceback.print_exc()
