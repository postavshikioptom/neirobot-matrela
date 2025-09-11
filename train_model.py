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

configure_gpu_memory()

# ✅ НАСТРОЙКИ ДЛЯ СОВМЕСТИМОСТИ С РАЗНЫМИ СРЕДАМИ
# Отключаем XLA если возникают проблемы (можно включить обратно)
# os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'

# Новые импорты
from feature_engineering import calculate_features, detect_candlestick_patterns, calculate_vsa_features
from models.xlstm_rl_model import XLSTMRLModel
from rl_agent import IntelligentRLAgent
from trading_env import TradingEnvRL
from hybrid_decision_maker import HybridDecisionMaker

def prepare_xlstm_rl_data(data_path, sequence_length=10):
    """
    Подготавливает данные для единой xLSTM+RL системы
    """
    print(f"Загрузка данных из {data_path}...")
    full_df = pd.read_csv(data_path)
    
    # Объединенные признаки для новой архитектуры
    feature_cols = [
        # Технические индикаторы
        'RSI_14', 'MACD_12_26_9', 'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0',
        'ADX_14', 'STOCHk_14_3_3', 'STOCHd_14_3_3',
        'ATR_14', # <--- ДОБАВЛЕНО
        # Паттерны
        'CDLHAMMER', 'CDLENGULFING', 'CDLDOJI', 'CDLSHOOTINGSTAR',
        'CDLHANGINGMAN', 'CDLMARUBOZU',
        # VSA признаки (новые!)
        'vsa_no_demand', 'vsa_no_supply', 'vsa_stopping_volume',
        'vsa_climactic_volume', 'vsa_test', 'vsa_effort_vs_result', 'vsa_strength',
        # Дополнительные рыночные данные
        'volume_ratio', 'spread_ratio', 'close_position',
        'is_event' # <--- ДОБАВЛЕНО: Новый признак
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
        df = calculate_vsa_features(df)  # Добавляем VSA!
        
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
        
        # Создаем целевые метки на основе будущих цен + VSA подтверждения
        df['future_return'] = (df['close'].shift(-5) - df['close']) / df['close']
        
        # =====================================================================
        # НОВЫЙ БЛОК: ДИНАМИЧЕСКИЕ ПОРОГИ И ВЗВЕШЕННЫЙ VSA-СКОР
        # =====================================================================
        # 1. Динамический порог future_return на основе ATR
        # Это позволяет уменьшить порог в высоковолатильные периоды и увеличить — в тихие.
        df['dynamic_future_threshold'] = df['ATR_14'] / df['close'] * 0.8 # СНИЖЕНО с 1.5 до 0.8
        df['dynamic_future_threshold'] = df['dynamic_future_threshold'].replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0005) # Минимальный порог 0.05% (было 0.1%)
        
        # 2. Взвешенное сочетание VSA-сигналов для более гибких условий
        # Еще больше ослабляем пороги для VSA-сисилы, чтобы больше сигналов проходило
        df['vsa_buy_score'] = (
            0.3 * (df['vsa_no_supply'] == 1) +
            0.3 * (df['vsa_stopping_volume'] == 1) +
            0.4 * (df['vsa_strength'] > 0.1) # СНИЖЕНО с 0.25 до 0.1
        )
        df['vsa_sell_score'] = (
            0.3 * (df['vsa_no_demand'] == 1) +
            0.3 * (df['vsa_climactic_volume'] == 1) +
            0.4 * (df['vsa_strength'] < -0.1) # СНИЖЕНО с -0.25 до -0.1
        )

        # BUY: положительная доходность (динамический порог) + взвешенный VSA-скор
        buy_condition = (
            (df['future_return'] > df['dynamic_future_threshold']) &
            (df['vsa_buy_score'] > 0.2) # СНИЖЕНО с 0.3 до 0.2
        )
        
        # SELL: отрицательная доходность (динамический порог) + взвешенный VSA-скор
        sell_condition = (
            (df['future_return'] < -df['dynamic_future_threshold']) &
            (df['vsa_sell_score'] > 0.2) # СНИЖЕНО с 0.3 до 0.2
        )
        # =====================================================================
        # КОНЕЦ НОВОГО БЛОКА
        # =====================================================================
        
        # Сначала устанавливаем все в HOLD, затем переписываем
        df['target'] = 2  # По умолчанию HOLD
        df.loc[buy_condition, 'target'] = 0 # BUY
        df.loc[sell_condition, 'target'] = 1 # SELL

        # ДОБАВЬТЕ: Принудительная балансировка классов (если необходимо)
        # Этот блок можно включать, если после ослабления порогов баланс все еще очень плохой.
        # Он попытается переклассифицировать часть "HOLD" в BUY/SELL на основе других индикаторов.
        # Это может быть "грязным" решением, но иногда необходимо для обучения.
        current_buy_count = (df['target'] == 0).sum()
        current_sell_count = (df['target'] == 1).sum()
        current_hold_count = (df['target'] == 2).sum()

        # =====================================================================
        # НОВЫЙ БЛОК: УЛУЧШЕННАЯ ПЕРЕКЛАССИФИКАЦИЯ HOLD-СИГНАЛОВ
        # =====================================================================
        if current_hold_count > (current_buy_count + current_sell_count) * 1.5: # Если HOLD в 1.5+ раза больше
            print(f"⚠️ Сильный дисбаланс классов. Попытка УМНОЙ переклассификации части HOLD-сигналов.")
            hold_indices = df[df['target'] == 2].index
            
            import random
            random.seed(42) # Для воспроизводимости
            
            # Переклассифицируем 40% HOLD (было 30%)
            reclassify_count = int(current_hold_count * 0.40) # УВЕЛИЧЕНО до 40%
            if reclassify_count > 0:
                reclassify_indices = random.sample(list(hold_indices), min(reclassify_count, len(hold_indices)))
                
                for idx in reclassify_indices:
                    if idx < 1: continue
                    
                    rsi = df.loc[idx, 'RSI_14']
                    adx = df.loc[idx, 'ADX_14']
                    adx_prev = df.loc[idx-1, 'ADX_14']
                    price_change_5_period = df['close'].pct_change(5).loc[idx]
                    atr_ratio = df.loc[idx, 'ATR_14'] / df.loc[idx, 'close']
                    
                    # Условия для переклассификации (ЕЩЕ БОЛЕЕ МЯГКИЕ)
                    # 1. Слабый RSI + растущий ADX + небольшое движение → BUY
                    if (rsi < 45 and adx > adx_prev and abs(price_change_5_period) > 0.0005): # RSI < 45 (было 40), price_change_5_period > 0.0005 (было 0.001)
                        df.loc[idx, 'target'] = 0  # BUY

                    # 2. RSI > 55 + растущий ADX + небольшое движение → SELL
                    elif (rsi > 55 and adx > adx_prev and abs(price_change_5_period) > 0.0005): # RSI > 55 (было 60), price_change_5_period < -0.0005 (было -0.001)
                        df.loc[idx, 'target'] = 1  # SELL

                    # 3. Подтверждение по объему (слабый объем, но есть движение)
                    elif (df['volume'].loc[idx] > df['volume'].rolling(20).quantile(0.6).loc[idx] and # Квантиль 0.6 (было 0.7)
                        ((price_change_5_period > 0.0005 and rsi < 50) or (price_change_5_period < -0.0005 and rsi > 50))): # RSI < 50 / > 50
                        df.loc[idx, 'target'] = 0 if price_change_5_period > 0 else 1

                    # 4. Смена тренда: ADX растет, RSI отходит от 50
                    elif (abs(rsi - 50) > 3 and adx > adx_prev and abs(adx - adx_prev) > 0.3): # abs(rsi-50) > 3 (было 5), abs(adx-adx_prev) > 0.3 (было 0.5)
                        df.loc[idx, 'target'] = 0 if rsi < 50 else 1 # Сделаем более агрессивным
            
            print(f"Баланс классов после УМНОЙ переклассификации:")
            unique, counts = np.unique(df['target'], return_counts=True)
            class_names = ['BUY', 'SELL', 'HOLD']
            for class_idx, count in zip(unique, counts):
                print(f"  {class_names[class_idx]}: {count} ({count/len(df)*100:.1f}%)")
        # =====================================================================
        # КОНЕЦ НОВОГО БЛОКА
        # =====================================================================
        
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
        total_samples = len(X)
        target_buy_count = int(total_samples * 0.25) # Цель 25% BUY
        target_sell_count = int(total_samples * 0.25) # Цель 25% SELL
        
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
            # Убедимся, что k_neighbors не меньше 1
            k_neighbors = max(1, k_neighbors)

            if any(count <= k_neighbors for count in [current_buy_count, current_sell_count] if count > 0):
                print("⚠️ Недостаточно сэмплов для SMOTE с k_neighbors, использую RandomOverSampler.")
                from imblearn.over_sampling import RandomOverSampler
                oversampler = RandomOverSampler(sampling_strategy=sampling_strategy_smote, random_state=42)
            else:
                oversampler = SMOTE(sampling_strategy=sampling_strategy_smote, random_state=42, k_neighbors=k_neighbors)

            X_temp, y_temp_labels = oversampler.fit_resample(X.reshape(len(X), -1), y_labels)
            print(f"Баланс классов после Oversampling: {Counter(y_temp_labels)}")
        else:
            X_temp, y_temp_labels = X.reshape(len(X), -1), y_labels
            print("Пропустил Oversampling, так как нет BUY/SELL сигналов.")

        # Undersampling HOLD: Цель - чтобы HOLD был примерно равен сумме BUY + SELL
        current_hold_count_after_oversample = Counter(y_temp_labels)[2]
        target_hold_count = min(current_hold_count_after_oversample, int((Counter(y_temp_labels)[0] + Counter(y_temp_labels)[1]) * 1.0)) # СНИЖЕНО с 1.5 до 1.0
        
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
    # НОВЫЙ БЛОК: TIME SERIES SPLIT ДЛЯ ВАЛИДАЦИИ
    # =====================================================================
    from sklearn.model_selection import TimeSeriesSplit

    print("\n🔄 Применяю TimeSeriesSplit для валидации данных...")
    # Используем одну складку для простоты, последняя часть для теста, предпоследняя для валидации
    # Количество сплитов = 2, чтобы получить 3 части: Train, Val, Test (в последнем сплите)
    tscv = TimeSeriesSplit(n_splits=2)

    train_val_indices, test_indices = list(tscv.split(X))[0] # Первый сплит: Train/Val vs Test
    train_indices, val_indices = list(tscv.split(X[train_val_indices]))[0] # Второй сплит: Train vs Val

    X_train, y_train = X[train_indices], y[train_indices]
    X_val, y_val = X[val_indices], y[val_indices]
    X_test, y_test = X[test_indices], y[test_indices] # Test берем из первого сплита

    print(f"✅ TimeSeriesSplit завершен.")
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

    # Дополнительное усиление весов BUY/SELL
    additional_weight_multiplier = 1.5
    if 0 in class_weight_dict:
        class_weight_dict[0] *= additional_weight_multiplier
    if 1 in class_weight_dict:
        class_weight_dict[1] *= additional_weight_multiplier

    print(f"📊 Веса классов для обучения: {class_weight_dict}")
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
        
    # Проверяем, есть ли сохраненная модель
    checkpoint_path = 'models/xlstm_checkpoint_latest.keras'
    scaler_path = 'models/xlstm_rl_scaler.pkl'

    xlstm_model = XLSTMRLModel(
        input_shape=(X_train.shape[1], X_train.shape[2]),
        memory_units=128,
        attention_units=64
    )

    # Нормализация данных
    X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
    xlstm_model.scaler.fit(X_train_reshaped)
    X_train_scaled = xlstm_model.scaler.transform(X_train_reshaped).reshape(X_train.shape)
    X_val_scaled = xlstm_model.scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)

    # =====================================================================
    # НОВЫЙ БЛОК: ИНЪЕКЦИЯ ШУМА ВО ВХОДНЫЕ ПРИЗНАКИ (для регуляризации)
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
    
    if os.path.exists(checkpoint_path):
        print("Найдена сохраненная модель, загружаем...")
        try:
            # ИСПРАВЛЕНО: Загружаем модель и scaler
            xlstm_model.model = tf.keras.models.load_model(checkpoint_path)
            
            # Загружаем scaler если он существует
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    # Перезагружаем scaler, так как он мог быть не сохранен с моделью
                    xlstm_model.scaler = pickle.load(f)
                xlstm_model.is_trained = True
                print("✅ Модель и scaler загружены, продолжаем обучение")
            else:
                print("⚠️ Scaler не найден, будет использован новый")
                
        except Exception as e:
            print(f"⚠️ Не удалось загрузить модель: {e}, начинаем заново")
            # Модель уже инициализирована, ничего не делаем
    
    # Градиентное обрезание для стабильности
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.001,
        clipnorm=1.0
    )
    xlstm_model.model.compile( # ИЗМЕНЕНО: теперь используем xlstm_model.model напрямую
        optimizer=optimizer,
        # ИЗМЕНЕНО: Добавляем Label Smoothing
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1), # <--- ИЗМЕНЕНО
        metrics=['accuracy', 'precision', 'recall']
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
        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {} # Убедимся, что logs не None
            try:
                lr = self.model.optimizer.learning_rate.numpy()
                # ИЗМЕНЕНО: Добавлены метрики accuracy, precision, recall
                print(f"Эпоха {epoch+1}/100 - loss: {logs.get('loss', 0):.4f} - val_loss: {logs.get('val_loss', 0):.4f} - "
                      f"accuracy: {logs.get('accuracy', 0):.2f} - val_accuracy: {logs.get('val_accuracy', 0):.2f} - " # <--- ДОБАВЛЕНО
                      f"precision: {logs.get('precision', 0):.2f} - val_precision: {logs.get('val_precision', 0):.2f} - " # <--- ДОБАВЛЕНО
                      f"recall: {logs.get('recall', 0):.2f} - val_recall: {logs.get('val_recall', 0):.2f} - lr: {lr:.2e}") # <--- ДОБАВЛЕНО
                
                # Проверяем на переобучение
                if logs.get('val_loss', 0) > logs.get('loss', 0) * 2:
                    print("⚠️ Возможное переобучение!")
            except Exception as e:
                print(f"Эпоха {epoch+1}/100 - loss: {logs.get('loss', 0):.4f} - val_loss: {logs.get('val_loss', 0):.4f} (Ошибка в логировании: {e})")
            
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=20,  # <--- ИЗМЕНЕНО с 35 на 20 (более агрессивный стоп)
            restore_best_weights=True,
            verbose=1
        ),
        MemoryCleanupCallback(),
        DetailedProgressCallback(),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=20,
            min_lr=1e-7,
            verbose=0
        )
    ]

    # Обучение с улучшенными колбэками
    history = xlstm_model.train(
        X_train_to_model, y_train, # <--- ИЗМЕНЕНО: используем X_train_to_model
        X_val_to_model, y_val,     # <--- ИЗМЕНЕНО: используем X_val_to_model
        epochs=100,
        batch_size=16,
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
        loss, accuracy, precision, recall = xlstm_model.model.evaluate(X_test_scaled, y_test, verbose=0)
        print(f"xLSTM Точность: {accuracy * 100:.2f}%")
        print(f"xLSTM Precision: {precision * 100:.2f}%")
        print(f"xLSTM Recall: {recall * 100:.2f}%")
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
        rl_agent_path='models/rl_agent_BTCUSDT', # Временно, он не будет использоваться для принятия решений
        feature_columns=feature_cols,
        sequence_length=X.shape[1] # Передаем sequence_length
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
