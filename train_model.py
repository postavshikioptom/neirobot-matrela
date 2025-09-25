import os
import sys
# import logging # 🔥 УДАЛЕНО: Импорт logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from datetime import datetime
# from sklearn.utils import class_weight # 🔥 УДАЛЕНО: Больше не используем sample_weights
import math
import psutil  # 🔥 ДОБАВЛЕНО: Импорт psutil для проверки памяти
import gc
from collections import deque
import itertools

# Импорт для настройки устройств
from device_config import DeviceConfig

# Вызываем функцию настройки в начале программы
has_gpu, num_gpus = DeviceConfig.setup()

# Удобно включать временно при отладке
tf.config.run_functions_eagerly(False) # 🔥 ИСПРАВЛЕНО: Установите False или удалите

# Импорт наших модулей
from feature_engineering import FeatureEngineering, apply_smote_to_training_data
from trading_env import TradingEnvironment
from rl_agent import RLAgent
from hybrid_decision_maker import HybridDecisionMaker
from simulation_engine import SimulationEngine
from models.xlstm_rl_model import XLSTMRLModel
import config
from validation_metrics_callback import ValidationMetricsCallback

# 🔥 УДАЛЕНО: logging.basicConfig и logger

class CosineDecayCallback(tf.keras.callbacks.Callback):
    """Кастомный Cosine Decay callback для TensorFlow 2.19.0"""
    def __init__(self, initial_learning_rate, decay_steps, alpha=0.0):
        super().__init__()
        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.alpha = alpha
        self.optimizer_ref = None 
    
    def on_train_begin(self, logs=None):
        if hasattr(self.model, 'optimizer'):
            # 🔥 ИСПРАВЛЕНО: Сохраняем прямую ссылку на атрибут learning_rate
            if hasattr(self.model.optimizer, 'learning_rate'):
                self.optimizer_ref = self.model.optimizer.learning_rate
            else:
                print(f"⚠️ Ошибка: Оптимизатор модели не имеет атрибута 'learning_rate' в on_train_begin. Тип: {type(self.model.optimizer)}")
        else:
            print("⚠️ Ошибка: Модель не имеет атрибута 'optimizer' в on_train_begin.")

    def on_epoch_begin(self, epoch, logs=None):
        if self.optimizer_ref is None:
            print(f"⚠️ Ошибка: Оптимизатор не инициализирован для CosineDecayCallback на эпохе {epoch}. Пропускаем изменение LR.")
            return

        if epoch < self.decay_steps:
            cosine_decay = 0.5 * (1 + math.cos(math.pi * epoch / self.decay_steps))
            decayed_learning_rate = (self.initial_learning_rate - self.alpha) * cosine_decay + self.alpha
            
            try:
                # 🔥 ИСПРАВЛЕНО: Устанавливаем значение напрямую в атрибут learning_rate
                self.optimizer_ref.assign(decayed_learning_rate)
                print(f"Epoch {epoch+1}: Установлена скорость обучения: {self.optimizer_ref.numpy():.6f}")
            except AttributeError as e:
                print(f"❌ Критическая ошибка в CosineDecayCallback на эпохе {epoch+1}: {e}")
                print(f"Тип self.optimizer_ref: {type(self.optimizer_ref)}")
                print(f"Атрибуты self.optimizer_ref: {dir(self.optimizer_ref)}")
            except Exception as e:
                print(f"❌ Неизвестная ошибка в CosineDecayCallback на эпохе {epoch+1}: {e}")

class ThreeStageTrainer:
    """
    Трёхэтапный тренер для xLSTM + RL модели
    """
    def __init__(self, data_path, has_gpu=False, num_gpus=0):
        self.data_path = data_path
        self.has_gpu = has_gpu
        self.num_gpus = num_gpus
        self.feature_eng = FeatureEngineering(sequence_length=config.SEQUENCE_LENGTH)
        self.model = None
        self.X_train_supervised = None # 🔥 ИЗМЕНЕНО: Для supervised
        self.X_val_supervised = None   # 🔥 ИЗМЕНЕНО: Для supervised
        self.X_test_supervised = None  # 🔥 ИЗМЕНЕНО: Для supervised
        self.y_train_supervised = None # 🔥 ИЗМЕНЕНО: Для supervised
        self.y_val_supervised = None   # 🔥 ИЗМЕНЕНО: Для supervised
        self.y_test_supervised = None  # 🔥 ИЗМЕНЕНО: Для supervised
        
        self.X_rl_train_by_symbol = {} # 🔥 ДОБАВЛЕНО: Данные для RL, сгруппированные по символам
        self.X_rl_val_by_symbol = {}   # 🔥 ДОБАВЛЕНО: Данные для RL, сгруппированные по символам
        
        # Создаём директории
        os.makedirs('models', exist_ok=True)
        os.makedirs('results', exist_ok=True)
        os.makedirs('plots', exist_ok=True)
    
    def _get_optimal_batch_size(self):
        """Определяет оптимальный размер батча в зависимости от устройства"""
        return DeviceConfig.get_optimal_batch_size(config.SUPERVISED_BATCH_SIZE)
    
    def load_and_prepare_data(self):
        """Загружает и подготавливает данные для всех этапов"""
        print("=== ПОДГОТОВКА ДАННЫХ ===")
        
        # Загружаем данные без преобразования timestamp
        df = pd.read_csv(self.data_path, dtype={
            'timestamp': np.int64,  # Явно указываем тип для timestamp
            'open': float, 
            'high': float, 
            'low': float, 
            'close': float, 
            'volume': float, 
            'turnover': float,
            'symbol': str
        })
        print(f"Загружено {len(df)} строк данных")
        
        # Проверяем тип timestamp
        print(f"Тип timestamp: {df['timestamp'].dtype}")
        
        symbol_counts = df['symbol'].value_counts()
        valid_symbols = symbol_counts[symbol_counts >= config.MIN_ROWS_PER_SYMBOL].index.tolist()
        
        if len(valid_symbols) == 0:
            valid_symbols = symbol_counts.head(20).index.tolist()
        
        print(f"Используем {len(valid_symbols)} символов: {valid_symbols[:5]}...")
        
        df_filtered = df[df['symbol'].isin(valid_symbols)].copy()
        
        all_X_supervised = [] # 🔥 ИЗМЕНЕНО
        all_y_supervised = [] # 🔥 ИЗМЕНЕНО
        
        # 🔥 ДОБАВЛЕНО: Для RL-этапа будем хранить X для каждого символа отдельно
        X_data_for_rl = {} 
        
        # Подготовим отображение символов в индексы для стратификации по символам
        symbol_to_id = {s: idx for idx, s in enumerate(valid_symbols)}
        all_symbol_ids = []  # выравнивается с all_X_supervised/all_y_supervised

        for i, symbol in enumerate(valid_symbols):
            symbol_data = df_filtered[df_filtered['symbol'] == symbol].copy()
            
            if len(symbol_data) < config.SEQUENCE_LENGTH + config.FUTURE_WINDOW + 10:
                print(f"Пропускаем символ {symbol}: недостаточно данных ({len(symbol_data)} строк)")
                continue
            
            try:
                if i == 0:
                    # Используем адаптивный порог для создания меток
                    X_scaled_sequences, labels = self.feature_eng.prepare_supervised_data(symbol_data)
                else:
                    # Для остальных символов используем уже обученный скейлер
                    # и только трансформируем данные, затем генерируем метки
                    
                    # 🔥 КЛЮЧЕВОЕ ИЗМЕНЕНИЕ ЗДЕСЬ: Сначала добавляем индикаторы
                    symbol_data_with_indicators = self.feature_eng._add_technical_indicators(symbol_data.copy())
                    
                    # Теперь преобразуем все колонки (включая индикаторы) в числовой формат
                    temp_df_for_scaling = symbol_data_with_indicators.copy()
                    for col in self.feature_eng.feature_columns:
                        temp_df_for_scaling[col] = pd.to_numeric(temp_df_for_scaling[col], errors='coerce')
                    
                    # Применяем обученный скейлер
                    scaled_data = self.feature_eng.scaler.transform(temp_df_for_scaling[self.feature_eng.feature_columns].values)
                    
                    # Создаем последовательности из трансформированных данных
                    X_scaled_sequences, _ = self.feature_eng._create_sequences(scaled_data)
                    
                    # Создаем метки на основе оригинальных цен (адаптивный порог)
                    labels = self.feature_eng.create_trading_labels(symbol_data)
                    
                    # Обрезаем до минимальной длины
                    min_len = min(len(X_scaled_sequences), len(labels))
                    X_scaled_sequences = X_scaled_sequences[:min_len]
                    labels = labels[:min_len]
                
                if len(X_scaled_sequences) > 0:
                    all_X_supervised.append(X_scaled_sequences)
                    all_y_supervised.append(labels)
                    
                    # Символьные ID для каждой последовательности этого символа
                    all_symbol_ids.append(np.full(shape=(len(X_scaled_sequences),), fill_value=symbol_to_id[symbol], dtype=np.int32))
                    
                    X_data_for_rl[symbol] = X_scaled_sequences  # для RL
                    
                    # Вывод распределения меток (опционально)
                    try:
                        if labels is not None and len(labels) > 0:
                            u, c = np.unique(labels, return_counts=True)
                            dist = {int(k): int(v) for k, v in zip(u, c)}
                        else:
                            pass
                    except Exception:
                        pass
                    
            except Exception as e:
                print(f"❌ Ошибка при обработке символа {symbol}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Проверка на наличие данных перед объединением
        if all_X_supervised and all_y_supervised:
            X_supervised = np.vstack(all_X_supervised)
            y_supervised = np.concatenate(all_y_supervised)
            # Выровняем и символы
            symbol_ids = np.concatenate(all_symbol_ids) if all_symbol_ids else np.zeros((len(y_supervised),), dtype=np.int32)
            
            # Анализ глобального распределения меток
            u, c = np.unique(y_supervised, return_counts=True)
            global_dist = {int(k): int(v) for k, v in zip(u, c)}
            total = y_supervised.shape[0]
            print(f"[GLOBAL LABELS] distribution: {global_dist}, total={total}")
            
            # Предупреждение, если HOLD все еще доминирует
            hold_count = global_dist.get(1, 0)
            if hold_count / total > 0.8:
                print(f"⚠️ [GLOBAL WARNING] HOLD fraction is very high: {hold_count}/{total} = {hold_count/total:.2%}")
                print(f"⚠️ Рекомендуется проверить данные или уменьшить базовый порог в config.py")
            
            print(f"Итого подготовлено для Supervised: X={X_supervised.shape}, y={y_supervised.shape}")
            print(f"Распределение классов: SELL={np.sum(y_supervised==0)}, HOLD={np.sum(y_supervised==1)}, BUY={np.sum(y_supervised==2)}")
        else:
            print("❌ Нет данных для обучения. Проверьте обработку символов выше.")
            return False
        
        # 3. Улучшение логирования для отладки
        # Добавьте более подробное логирование для лучшего понимания процесса:
        # В методе load_and_prepare_data в train_model.py
        # После обработки всех символов
        print(f"\n=== ИТОГОВАЯ СТАТИСТИКА ПО СИМВОЛАМ ===")
        print(f"Всего обработано символов: {len(valid_symbols)}")
        print(f"Успешно создано последовательностей: {sum(len(data) for data in all_X_supervised)}")
        print(f"Распределение меток по всем символам:")
        
        # Объединяем все метки для анализа
        all_labels = np.concatenate(all_y_supervised) if all_y_supervised else np.array([])
        if len(all_labels) > 0:
            unique, counts = np.unique(all_labels, return_counts=True)
            distribution = {int(u): int(c) for u, c in zip(unique, counts)}
            total = len(all_labels)
            print(f"SELL (0): {distribution.get(0, 0)} ({distribution.get(0, 0)/total:.2%})")
            print(f"HOLD (1): {distribution.get(1, 0)} ({distribution.get(1, 0)/total:.2%})")
            print(f"BUY (2): {distribution.get(2, 0)} ({distribution.get(2, 0)/total:.2%})")
        else:
            print("Нет данных для анализа")
        
        def augment_sequences_batched(X, y, factor=2, max_memory_gb=4.0):
            """Лёгкие аугментации с контролем памяти и опциями из config"""
            if not getattr(config, 'USE_AUGMENTATIONS', True):
                return X, y
            if len(X) == 0:
                return X, y
            
            noise_std = float(getattr(config, 'AUG_NOISE_STD', 0.01))
            max_shift = int(getattr(config, 'AUG_TIME_SHIFT', 1))
            mask_prob = float(getattr(config, 'AUG_MASK_PROB', 0.05))
            mask_max_t = int(getattr(config, 'AUG_MASK_MAX_T', 2))
            
            available_memory_gb = psutil.virtual_memory().available / (1024**3)
            if available_memory_gb < (max_memory_gb * 0.2):
                print(f"⚠️ Достигнут лимит памяти: available={available_memory_gb:.2f}GB, required buffer={max_memory_gb*0.2:.2f}GB")
                return X, y
            
            batch_size = min(500, max(64, len(X)//50))
            augmented_X = deque()
            augmented_y = deque()
            
            for start_idx in range(0, len(X), batch_size):
                end_idx = min(start_idx + batch_size, len(X))
                batch_X = X[start_idx:end_idx]
                batch_y = y[start_idx:end_idx]
                augmented_X.extend(batch_X)
                augmented_y.extend(batch_y)
                
                available_memory_gb = psutil.virtual_memory().available / (1024**3)
                if available_memory_gb < (max_memory_gb * 0.2):
                    print(f"⚠️ Достигнут лимит памяти: available={available_memory_gb:.2f}GB, required buffer={max_memory_gb*0.2:.2f}GB")
                    break
                
                for i in range(len(batch_X)):
                    x = batch_X[i].copy()
                    # 1) Небольшой временной сдвиг
                    if max_shift > 0 and x.shape[0] > 2:
                        shift = np.random.randint(-max_shift, max_shift+1)
                        if shift != 0:
                            x = np.roll(x, shift, axis=0)
                    # 2) Лёгкий гауссов шум
                    if noise_std > 0:
                        x = x + np.random.normal(0, noise_std, size=x.shape)
                    # 3) Крошечная временная маска
                    if np.random.rand() < mask_prob and x.shape[0] > 3:
                        t = np.random.randint(1, min(mask_max_t, x.shape[0]//4) + 1)
                        s = np.random.randint(0, x.shape[0]-t+1)
                        x[s:s+t, :] = 0.0
                    augmented_X.append(x)
                    augmented_y.append(batch_y[i])
                
                if start_idx % (batch_size * 5) == 0:
                    gc.collect()
            
            gc.collect()
            result_X = np.array(list(augmented_X))
            result_y = np.array(list(augmented_y))
            augmented_X.clear(); augmented_y.clear(); gc.collect()
            print(f"Аугментация завершена: {len(X)} -> {len(result_X)} образцов")
            return result_X, result_y
        
        # Применяем безопасную батчевую аугментацию
        X_supervised, y_supervised = augment_sequences_batched(X_supervised, y_supervised)
        
        # Разделяем данные для supervised learning
        # 🔥 ИСПРАВЛЕНО: Более надежная обработка stratify
        # Проверяем, есть ли хотя бы два класса для стратификации
        if len(np.unique(y_supervised)) > 1:
            X_temp, self.X_test_supervised, y_temp, self.y_test_supervised = train_test_split(
                X_supervised, y_supervised, test_size=0.1, shuffle=True, random_state=42, stratify=y_supervised
            )
            if len(np.unique(y_temp)) > 1:
                self.X_train_supervised, self.X_val_supervised, self.y_train_supervised, self.y_val_supervised = train_test_split(
                    X_temp, y_temp, test_size=config.SUPERVISED_VALIDATION_SPLIT, 
                    shuffle=True, random_state=42, stratify=y_temp
                )
            else:
                print("⚠️ Недостаточно классов для стратифицированного разделения на train/val. Используем обычное разделение.")
                self.X_train_supervised, self.X_val_supervised, self.y_train_supervised, self.y_val_supervised = train_test_split(
                    X_temp, y_temp, test_size=config.SUPERVISED_VALIDATION_SPLIT, 
                    shuffle=True, random_state=42
                )
        else:
            print("⚠️ Недостаточно классов для стратифицированного разделения на test. Используем обычное разделение.")
            X_temp, self.X_test_supervised, y_temp, self.y_test_supervised = train_test_split(
                X_supervised, y_supervised, test_size=0.1, shuffle=True, random_state=42
            )
            self.X_train_supervised, self.X_val_supervised, self.y_train_supervised, self.y_val_supervised = train_test_split(
                X_temp, y_temp, test_size=config.SUPERVISED_VALIDATION_SPLIT, 
                shuffle=True, random_state=42
            )
        
        print(f"Размеры выборок (Supervised): Train={len(self.X_train_supervised)}, Val={len(self.X_val_supervised)}, Test={len(self.X_test_supervised)}") # 🔥 ИЗМЕНЕНО

        # 🔥 ДОБАВЛЕНО: Применяем SMOTE для балансировки классов в тренировочных данных
        if config.USE_SMOTE:
            print("🔄 Применяем SMOTE для балансировки тренировочных данных...")
            try:
                if getattr(config, 'USE_CHUNKED_SMOTE', False):
                    from feature_engineering import apply_chunked_smote
                    self.X_train_supervised, self.y_train_supervised = apply_chunked_smote(
                        self.X_train_supervised,
                        self.y_train_supervised,
                        minority_classes=tuple(getattr(config, 'CHUNKED_SMOTE_MINORITY_CLASSES', [0,1])),
                        max_synth_per_class=getattr(config, 'CHUNKED_SMOTE_MAX_SYNTH_PER_CLASS', 15000),
                        memory_guard_gb=1.5,
                        chunk_size=2000,
                        verbose=True
                    )
                else:
                    # Целевое распределение: SELL=30%, HOLD=40%, BUY=30% (чтобы не переусилить minority классы)
                    target_distribution = {0: 30.0, 1: 40.0, 2: 30.0}
                    self.X_train_supervised, self.y_train_supervised = apply_smote_to_training_data(
                        self.X_train_supervised, self.y_train_supervised, target_distribution
                    )
                print(f"✅ SMOTE завершен успешно!")
                print(f"📊 После SMOTE: Train={len(self.X_train_supervised)}, Val={len(self.X_val_supervised)}, Test={len(self.X_test_supervised)}")
                
                # Проверяем финальное распределение классов
                unique, counts = np.unique(self.y_train_supervised, return_counts=True)
                total = len(self.y_train_supervised)
                print("📊 Финальное распределение тренировочных классов после SMOTE:")
                for cls, count in zip(unique, counts):
                    percentage = count / total * 100
                    print(f"   Класс {cls}: {percentage:.2f}% ({count} образцов)")
                    
            except Exception as e:
                print(f"❌ ОШИБКА при применении SMOTE: {e}")
                import traceback
                traceback.print_exc()
                print("🔄 Продолжаем без SMOTE...")
        else:
            print("⚠️ SMOTE отключен в конфигурации")

        print("🔄 Переходим к подготовке RL данных...")
        # 🔥 ДОБАВЛЕНО: Разделяем данные для RL-обучения по символам
        for symbol, data_sequences in X_data_for_rl.items():
            # Делим каждую последовательность на тренировочную и валидационную для RL
            # (RL env будет брать случайный отрезок из этих данных)
            train_size = int(len(data_sequences) * (1 - 0.1 - config.SUPERVISED_VALIDATION_SPLIT)) # 10% test, SUPERVISED_VALIDATION_SPLIT% val
            val_size = int(len(data_sequences) * config.SUPERVISED_VALIDATION_SPLIT)
            
            self.X_rl_train_by_symbol[symbol] = data_sequences[:train_size]
            self.X_rl_val_by_symbol[symbol] = data_sequences[train_size:train_size + val_size]
            
            print(f"RL данные для {symbol}: Train={len(self.X_rl_train_by_symbol[symbol])}, Val={len(self.X_rl_val_by_symbol[symbol])}")
            
        # Сохраняем скейлер
        print("🔄 Сохраняем скейлер...")
        self.feature_eng.save_scaler()
        print("✅ Скейлер сохранен")
        
        # Инициализируем модель
        print("🔄 Инициализируем модель...")
        # 🔥 ИЗМЕНЕНО: input_shape теперь использует длину feature_columns из feature_eng
        input_shape = (config.SEQUENCE_LENGTH, len(self.feature_eng.feature_columns)) 
        print(f"📊 Размер входных данных для модели: {input_shape}")
        
        self.model = XLSTMRLModel(
            input_shape=input_shape,
            memory_size=config.XLSTM_MEMORY_SIZE,
            memory_units=config.XLSTM_MEMORY_UNITS
        )
        print("✅ Модель инициализирована успешно")
        
        print("🎉 === ПОДГОТОВКА ДАННЫХ ЗАВЕРШЕНА УСПЕШНО ===")
        return True
    
    def stage1_supervised_pretraining(self):
        """ЭТАП 1: Supervised Pre-training"""
        print("=== ЭТАП 1: SUPERVISED PRE-TRAINING ===")
        
        self.model.compile_for_supervised_learning()
        
        # Используем улучшенные callbacks из модели
        callbacks = self.model.get_training_callbacks(
            total_epochs=config.SUPERVISED_EPOCHS,
            patience=10
        )
        
        # Добавляем кастомный callback для детальных метрик
        callbacks.append(ValidationMetricsCallback(self.X_val_supervised, self.y_val_supervised))
        
        print(f"Начинаем supervised обучение на {config.SUPERVISED_EPOCHS} эпох...")
        
        # Определяем оптимальный размер батча
        batch_size = self._get_optimal_batch_size()
        print(f"Используем размер батча: {batch_size}")
        
        # 🔥 КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Отключаем sample_weights, полагаемся только на AFL параметры
        # Проблема: sample_weights конфликтовали с AFL_ALPHA, создавая непредсказуемое поведение
        # Решение: Используем только AFL (Asymmetric Focal Loss) для контроля баланса классов
        print("🎯 Sample weights отключены - используем только AFL параметры из config для балансировки")
        sample_weights_base = None  # Принудительно отключаем автоматические веса

        # 🔥 ИСПРАВЛЕНО: Явное преобразование меток
        y_train_processed = np.array(self.y_train_supervised, dtype=np.int32)
        y_val_processed = np.array(self.y_val_supervised, dtype=np.int32)
        

        # Готовим данные для class-balanced batching, если включено
        train_X = self.X_train_supervised
        train_y = y_train_processed
        val_X = self.X_val_supervised
        val_y = y_val_processed

        if getattr(config, 'CLASS_BALANCED_BATCHING', False):
            print("🔄 Включен class-balanced batching по TARGET_CLASS_RATIOS")
            import math
            ratios = getattr(config, 'TARGET_CLASS_RATIOS', [0.3, 0.3, 0.4])

            # Разбиваем индексы по классам
            idx_sell = np.where(train_y == 0)[0]
            idx_hold = np.where(train_y == 1)[0]
            idx_buy  = np.where(train_y == 2)[0]
            rng = np.random.default_rng(42)

            # Подготовим индексы по символам для стратификации
            symbol_ids_train = symbol_ids[:len(train_y)] if 'symbol_ids' in locals() else np.zeros_like(train_y)
            symbol_to_indices = {}
            if getattr(config, 'SYMBOL_STRATIFIED_BATCHING', False):
                for sid in np.unique(symbol_ids_train):
                    symbol_to_indices[int(sid)] = np.where(symbol_ids_train == sid)[0]
                # Желаем распределить символы равномерно
                uniq_sids = list(symbol_to_indices.keys())
                sid_cycle = itertools.cycle(uniq_sids) if len(uniq_sids) > 0 else None

            # Hard Negative Mining (предварительный расчет трудных примеров для SELL/HOLD)
            hard_sell = None
            hard_hold = None
            if getattr(config, 'USE_HARD_NEGATIVE_MINING', False):
                try:
                    print("🔎 Подготовка hard negatives по loss")
                    probs = self.model.actor_model.predict(train_X, verbose=0)
                    true_prob = np.clip(probs[np.arange(len(train_y)), train_y], 1e-9, 1.0)
                    losses = -np.log(true_prob)
                    frac = float(getattr(config, 'HNM_TOP_K_FRACTION', 0.05))
                    k_sell = max(1, int(len(idx_sell) * frac)) if len(idx_sell) > 0 else 0
                    k_hold = max(1, int(len(idx_hold) * frac)) if len(idx_hold) > 0 else 0
                    if k_sell > 0:
                        hard_sell = idx_sell[np.argsort(losses[idx_sell])[-k_sell:]]
                    if k_hold > 0:
                        hard_hold = idx_hold[np.argsort(losses[idx_hold])[-k_hold:]]
                    print(f"✅ HNM: SELL hard={0 if hard_sell is None else len(hard_sell)}, HOLD hard={0 if hard_hold is None else len(hard_hold)}")
                except Exception as e:
                    print(f"⚠️ HNM отключен: {e}")
                    hard_sell, hard_hold = None, None

            def balanced_batch_generator(X, y, batch_size):
                q = (np.array(ratios) / np.sum(ratios)).astype(float)
                per_class = np.maximum(1, (q * batch_size).astype(int))
                # Корректируем сумму
                diff = batch_size - per_class.sum()
                if diff != 0:
                    per_class[np.argmax(q)] += diff
                pools = [idx_sell.copy(), idx_hold.copy(), idx_buy.copy()]

                # Планирование доли hard-negative внутри батча
                hard_start = float(getattr(config, 'HNM_HARD_SAMPLE_START', 0.20))
                hard_end = float(getattr(config, 'HNM_HARD_SAMPLE_END', 0.50))
                warm_epochs = max(1, int(getattr(config, 'AFL_WARMUP_EPOCHS', 5)))
                update_period = max(1, int(getattr(config, 'HNM_UPDATE_PERIOD', 5)))

                current_epoch = 0
                last_update_epoch = -1

                # Подсчитываем батчи и обновляем hard-пулы только по завершении "эпохи" генератора
                steps_per_epoch_local = max(1, math.ceil(len(y) / max(1, batch_size)))
                batch_counter = 0

                while True:
                    # Линейная интерполяция hnm_ratio в первые warm_epochs
                    t = min(1.0, current_epoch / float(warm_epochs))
                    hnm_ratio = hard_start * (1.0 - t) + hard_end * t

                    # Обновление hard-пулов не чаще, чем раз в update_period ЭПОХ (а не каждые N батчей)
                    if (getattr(config, 'USE_HARD_NEGATIVE_MINING', False)
                        and (batch_counter % steps_per_epoch_local == 0)
                        and (current_epoch - last_update_epoch >= update_period)):
                        try:
                            probs = self.model.actor_model.predict(X, verbose=0)
                            true_prob = np.clip(probs[np.arange(len(y)), y], 1e-9, 1.0)
                            losses = -np.log(true_prob)
                            frac = float(getattr(config, 'HNM_TOP_K_FRACTION', 0.05))
                            if len(idx_sell) > 0:
                                k_sell = max(1, int(len(idx_sell) * frac))
                                nonlocal hard_sell
                                hard_sell = idx_sell[np.argsort(losses[idx_sell])[-k_sell:]]
                            if len(idx_hold) > 0:
                                k_hold = max(1, int(len(idx_hold) * frac))
                                nonlocal hard_hold
                                hard_hold = idx_hold[np.argsort(losses[idx_hold])[-k_hold:]]
                            last_update_epoch = current_epoch
                        except Exception as e:
                            print(f"⚠️ HNM update skipped: {e}")

                    batch_idx = []
                    if getattr(config, 'SYMBOL_STRATIFIED_BATCHING', False) and 'sid_cycle' in locals() and sid_cycle is not None:
                        # Стратификация по символам: набираем мини-группы по символам
                        # Простейшая схема: равные доли символов на батч
                        uniq_sids = list(symbol_to_indices.keys())
                        per_symbol = max(1, batch_size // max(1, len(uniq_sids)))
                        chosen_indices = []
                        for _ in range(max(1, len(uniq_sids))):
                            sid = next(sid_cycle)
                            sid_pool = symbol_to_indices.get(int(sid), np.arange(len(y)))
                            if len(sid_pool) == 0:
                                continue
                            # внутри символа — соблюдаем классовые доли
                            sid_sel = []
                            for cls, need in enumerate(per_class):
                                pool = np.intersect1d(pools[cls], sid_pool, assume_unique=False)
                                if len(pool) == 0:
                                    continue
                                take = max(1, int(round(need * (per_symbol / float(batch_size)))))
                                take = min(take, len(pool))
                                sid_sel.append(rng.choice(pool, size=take, replace=False))
                            if sid_sel:
                                chosen_indices.append(np.concatenate(sid_sel))
                        if chosen_indices:
                            batch_idx = np.concatenate(chosen_indices)
                        else:
                            # fallback: стандартный набор без символьной стратификации
                            for cls, need in enumerate(per_class):
                                pool = pools[cls]
                                if len(pool) < need:
                                    chosen_pool = pool if len(pool) > 0 else np.arange(len(y))
                                    chosen = rng.choice(chosen_pool, size=need, replace=True)
                                else:
                                    hard_pool = hard_sell if cls == 0 else (hard_hold if cls == 1 else None)
                                    if hard_pool is not None and len(hard_pool) > 0:
                                        h = max(1, int(round(need * hnm_ratio)))
                                        h = min(h, len(hard_pool))
                                        h_idx = rng.choice(hard_pool, size=h, replace=False)
                                        rest = need - h
                                        rest_idx = rng.choice(pool, size=rest, replace=False)
                                        chosen = np.concatenate([h_idx, rest_idx])
                                    else:
                                        chosen = rng.choice(pool, size=need, replace=False)
                                batch_idx.append(chosen)
                            batch_idx = np.concatenate(batch_idx)
                    else:
                        # Обычный класс-стратифицированный отбор
                        for cls, need in enumerate(per_class):
                            pool = pools[cls]
                            if len(pool) < need:
                                # ресемпл с возвратом
                                chosen_pool = pool if len(pool) > 0 else np.arange(len(y))
                                chosen = rng.choice(chosen_pool, size=need, replace=True)
                            else:
                                # часть примеров берем из hard-пула, если доступен и класс SELL/HOLD
                                hard_pool = hard_sell if cls == 0 else (hard_hold if cls == 1 else None)
                                if hard_pool is not None and len(hard_pool) > 0:
                                    h = max(1, int(round(need * hnm_ratio)))
                                    h = min(h, len(hard_pool))
                                    h_idx = rng.choice(hard_pool, size=h, replace=False)
                                    rest = need - h
                                    rest_idx = rng.choice(pool, size=rest, replace=False)
                                    chosen = np.concatenate([h_idx, rest_idx])
                                else:
                                    chosen = rng.choice(pool, size=need, replace=False)
                            batch_idx.append(chosen)
                        batch_idx = np.concatenate(batch_idx)

                    rng.shuffle(batch_idx)

                    # Считаем батчи и обновляем счетчик эпох только после steps_per_epoch_local батчей
                    batch_counter += 1
                    if batch_counter % steps_per_epoch_local == 0:
                        current_epoch += 1

                    yield X[batch_idx], y[batch_idx]

            steps_per_epoch = math.ceil(len(train_X) / batch_size)
            train_data = balanced_batch_generator(train_X, train_y, batch_size)
            validation_data = (val_X, val_y)
            fit_kwargs = dict(x=train_data, steps_per_epoch=steps_per_epoch, validation_data=validation_data)
            sample_weight_arg = None
        else:
            fit_kwargs = dict(x=train_X, y=train_y, validation_data=(val_X, val_y), batch_size=batch_size)
            sample_weight_arg = sample_weights_base

        history = self.model.actor_model.fit(
            epochs=config.SUPERVISED_EPOCHS,
            callbacks=callbacks,
            verbose=1,
            sample_weight=sample_weight_arg,
            **fit_kwargs
        )
        
        print("=== РЕЗУЛЬТАТЫ SUPERVISED ОБУЧЕНИЯ ===")
        
        # === Валидация с TTA и калибровкой температуры ===
        def _moving_average_3(x):
            # Применяем простое сглаживание вдоль оси времени
            # x: (n, T, F)
            if x.shape[1] < 3:
                return x
            x_pad = np.pad(x, ((0,0),(1,1),(0,0)), mode='edge')
            return (x_pad[:, :-2, :] + 2*x_pad[:, 1:-1, :] + x_pad[:, 2:, :]) / 4.0
        
        def _zscore_window(x):
            # Нормализация по каждому образцу (по времени и фичам), без утечки между сэмплами
            mean = x.mean(axis=(1,2), keepdims=True) if x.ndim==3 else x.mean(axis=0, keepdims=True)
            std = x.std(axis=(1,2), keepdims=True) + 1e-6 if x.ndim==3 else x.std(axis=0, keepdims=True) + 1e-6
            return (x - mean) / std
        
        # Предсказания на валидации (для подбора температуры)
        val_probs = self.model.actor_model.predict(val_X, verbose=0)
        if getattr(config, 'USE_TTA_VALIDATION', False):
            print("🔄 Выполняем TTA на валидации")
            tta_list = getattr(config, 'TTA_TRANSFORMS', ['identity'])
            prob_stack = [val_probs]
            for t in tta_list:
                if t == 'identity':
                    continue
                elif t == 'zscore_window':
                    X_t = _zscore_window(val_X)
                elif t == 'gaussian_smooth':
                    X_t = _moving_average_3(val_X)
                else:
                    continue
                prob_stack.append(self.model.actor_model.predict(X_t, verbose=0))
            val_probs = np.mean(prob_stack, axis=0)
        
        # Температурная калибровка (подбор T по NLL на валидации)
        best_T = 1.0
        if getattr(config, 'USE_TEMPERATURE_SCALING', False):
            print("🔧 Подбираем температуру по валидации")
            def softmax_logits_scaled(probs, T):
                logits = np.log(np.clip(probs, 1e-7, 1-1e-7))
                z = logits / T
                z = z - z.max(axis=1, keepdims=True)
                ez = np.exp(z)
                return ez / ez.sum(axis=1, keepdims=True)
            def nll(probs, y):
                p = np.clip(probs[np.arange(len(y)), y], 1e-7, 1-1e-7)
                return -np.mean(np.log(p))
            grid = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
            best_val = float('inf')
            for T in grid:
                scaled = softmax_logits_scaled(val_probs, T)
                loss = nll(scaled, val_y)
                if loss < best_val:
                    best_val = loss
                    best_T = T
            print(f"✅ Лучшая температура: T={best_T}")
        
        # Предсказания на тесте + TTA + применение температуры
        test_probs = self.model.actor_model.predict(self.X_test_supervised, verbose=0)
        # Опционально TTA на тесте (используем тот же список трансформаций, что и для валидации)
        if getattr(config, 'USE_TTA_VALIDATION', False):  # при необходимости можно ввести отдельный флаг USE_TTA_TEST
            prob_stack_test = [test_probs]
            tta_list = getattr(config, 'TTA_TRANSFORMS', ['identity'])
            for t in tta_list:
                if t == 'identity':
                    continue
                elif t == 'zscore_window':
                    X_t = _zscore_window(self.X_test_supervised)
                elif t == 'gaussian_smooth':
                    X_t = _moving_average_3(self.X_test_supervised)
                else:
                    continue
                prob_stack_test.append(self.model.actor_model.predict(X_t, verbose=0))
            test_probs = np.mean(prob_stack_test, axis=0)
        
        # Затем применяем температурное масштабирование
        if getattr(config, 'USE_TEMPERATURE_SCALING', False) and best_T != 1.0:
            logits = np.log(np.clip(test_probs, 1e-7, 1-1e-7))
            z = logits / best_T
            z = z - z.max(axis=1, keepdims=True)
            ez = np.exp(z)
            test_probs = ez / ez.sum(axis=1, keepdims=True)
        
        y_pred = np.argmax(test_probs, axis=1)
        
        accuracy = accuracy_score(self.y_test_supervised, y_pred) # 🔥 ИЗМЕНЕНО
        print(f"Точность на тестовой выборке: {accuracy:.4f}")
        
        class_names = ['SELL', 'HOLD', 'BUY']
        report = classification_report(self.y_test_supervised, y_pred, target_names=class_names, labels=[0, 1, 2], zero_division=0) # 🔥 ИЗМЕНЕНО
        print(f"Классификационный отчет:\n{report}")
        
        cm = confusion_matrix(self.y_test_supervised, y_pred, labels=[0, 1, 2]) # 🔥 ИЗМЕНЕНО
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
        
        rewards_train = self._generate_simulated_rewards(self.X_train_supervised, self.y_train_supervised) # 🔥 ИЗМЕНЕНО
        rewards_val = self._generate_simulated_rewards(self.X_val_supervised, self.y_val_supervised)     # 🔥 ИЗМЕНЕНО
        
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
            self.X_train_supervised, rewards_train, # 🔥 ИЗМЕНЕНО
            validation_data=(self.X_val_supervised, rewards_val), # 🔥 ИЗМЕНЕНО
            epochs=config.REWARD_MODEL_EPOCHS,
            batch_size=config.REWARD_MODEL_BATCH_SIZE,
            callbacks=callbacks,
            verbose=1
        )
        
        val_predictions = self.model.critic_model.predict(self.X_val_supervised, verbose=0) # 🔥 ИЗМЕНЕНО
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
        
        rl_agent = RLAgent(
            state_shape=(config.SEQUENCE_LENGTH, len(self.feature_eng.feature_columns)),
            memory_size=config.XLSTM_MEMORY_SIZE,
            memory_units=config.XLSTM_MEMORY_UNITS,
            batch_size=config.RL_BATCH_SIZE
        )
        
        rl_agent.model = self.model
        
        # 🔥 ИЗМЕНЕНО: Теперь передаем словари данных для RL-среды
        train_env = TradingEnvironment(self.X_rl_train_by_symbol, sequence_length=config.SEQUENCE_LENGTH)
        val_env = TradingEnvironment(self.X_rl_val_by_symbol, sequence_length=config.SEQUENCE_LENGTH)
        
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
                
                # 🔥 ИЗМЕНЕНО: Для логирования действий RL-агента, нужно предоставить ему набор состояний
                # Здесь мы берем состояния из первой валидационной последовательности для примера
                if len(self.X_rl_val_by_symbol) > 0:
                    first_symbol_data = next(iter(self.X_rl_val_by_symbol.values()))
                    sample_size = min(500, len(first_symbol_data))
                    action_dist = rl_agent.log_action_distribution(first_symbol_data[:sample_size])
                else:
                    action_dist = {'buy_count': 0, 'hold_count': 0, 'sell_count': 0, 'total': 0}
                
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
        # Получаем предсказания от предобученной модели
        y_pred_probs = self.model.actor_model.predict(X, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)
        
        # Рассчитываем награды на основе точности предсказаний
        rewards = []
        for true_label, pred_label, pred_probs in zip(y_true, y_pred, y_pred_probs):
            if true_label == pred_label:
                # Правильное предсказание - положительная награда
                reward = 1.0
            else:
                # Неправильное предсказание - отрицательная награда
                reward = -1.0
            
            # Добавляем компоненту уверенности
            confidence = pred_probs[pred_label]
            reward *= confidence
            
            rewards.append(reward)
        
        return np.array(rewards)
    
    def _plot_training_history(self, history, stage_name):
        """Визуализирует историю обучения с безопасным доступом к ключам history"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Безопасный доступ к словарю истории
        hist = getattr(history, 'history', {}) or {}
        available_keys = set(hist.keys())

        # Потери: ищем первый доступный ключ
        loss_candidates = ['loss', 'total_loss', 'train_loss', 'training_loss', 'supervised_loss']
        loss_key = next((k for k in loss_candidates if k in available_keys), None)
        val_loss_key = 'val_loss' if 'val_loss' in available_keys else None

        if loss_key:
            axes[0].plot(hist[loss_key], label=f'{loss_key}')
            if val_loss_key:
                axes[0].plot(hist[val_loss_key], label=f'{val_loss_key}')
            axes[0].set_title(f'{stage_name.capitalize()} Training Loss')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Loss')
            axes[0].legend()
        elif val_loss_key:
            # Если есть только валидационные потери
            axes[0].plot(hist[val_loss_key], label=f'{val_loss_key}')
            axes[0].set_title(f'{stage_name.capitalize()} Validation Loss')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Loss')
            axes[0].legend()
        else:
            # Нет данных по loss — не падаем, а сообщаем
            axes[0].text(0.5, 0.5, 'Loss data not available',
                         ha='center', va='center', transform=axes[0].transAxes)
            axes[0].set_title(f'{stage_name.capitalize()} - No Loss Data')

        # Метрики: сохраняем исходную логику, используем безопасный словарь
        if 'accuracy' in available_keys:
            axes[1].plot(hist['accuracy'], label='Training Accuracy')
            if 'val_accuracy' in available_keys:
                axes[1].plot(hist['val_accuracy'], label='Validation Accuracy')
            axes[1].set_title(f'{stage_name.capitalize()} Accuracy')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Accuracy')
            axes[1].legend()
        elif 'mae' in available_keys:
            axes[1].plot(hist['mae'], label='Training MAE')
            if 'val_mae' in available_keys:
                axes[1].plot(hist['val_mae'], label='Validation MAE')
            axes[1].set_title(f'{stage_name.capitalize()} MAE')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('MAE')
            axes[1].legend()
        else:
            axes[1].text(0.5, 0.5, 'Metric data not available',
                         ha='center', va='center', transform=axes[1].transAxes)
            axes[1].set_title(f'{stage_name.capitalize()} - No Metric Data')

        plt.tight_layout()
        plt.savefig(f'plots/{stage_name}_training_history.png')
        plt.close()
    
    def _plot_rl_metrics(self, metrics):
        """Визуализирует RL метрики"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Episode rewards
        axes[0,0].plot(metrics['episode_rewards'])
        axes[0,0].set_title('Episode Rewards')
        axes[0,0].set_xlabel('Episode')
        axes[0,0].set_ylabel('Reward')
        axes[0,0].grid(True)
        
        # Episode profits
        axes[0,1].plot(metrics['episode_profits'])
        axes[0,1].set_title('Episode Profits (%)')
        axes[0,1].set_xlabel('Episode')
        axes[0,1].set_ylabel('Profit %')
        axes[0,1].grid(True)
        
        # Validation rewards (каждые 10 эпизодов)
        if metrics['val_rewards']:
            val_episodes = range(10, len(metrics['val_rewards']) * 10 + 1, 10)
            axes[1,0].plot(val_episodes, metrics['val_rewards'])
            axes[1,0].set_title('Validation Rewards')
            axes[1,0].set_xlabel('Episode')
            axes[1,0].set_ylabel('Reward')
            axes[1,0].grid(True)
        
        # Validation profits
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
        print("🚀 ЗАПУСК ТРЁХЭТАПНОГО ОБУЧЕНИЯ xLSTM + RL") # 🔥 ИЗМЕНЕНО: logger.info -> print
        
        try:
            # Подготовка данных
            print("🔄 Этап: Подготовка данных...")
            if not self.load_and_prepare_data():
                print("❌ Ошибка при подготовке данных") # 🔥 ИЗМЕНЕНО: logger.error -> print
                return None
            print("✅ Подготовка данных завершена успешно")
            
            results = {}
            
            # Этап 1: Supervised Pre-training
            print("🔄 Этап 1: Supervised Pre-training...")
            try:
                supervised_results = self.stage1_supervised_pretraining()
                if supervised_results is None:
                    print("❌ Ошибка на этапе supervised pre-training") # 🔥 ИЗМЕНЕНО: logger.error -> print
                    return None
                results['supervised'] = supervised_results
                print("✅ Этап 1 завершен успешно")
            except Exception as e:
                print(f"❌ КРИТИЧЕСКАЯ ОШИБКА на этапе 1 (Supervised): {e}")
                import traceback
                traceback.print_exc()
                return None
            
            # Этап 2: Reward Model Training
            print("🔄 Этап 2: Reward Model Training...")
            try:
                reward_results = self.stage2_reward_model_training()
                if reward_results is None:
                    print("❌ Ошибка на этапе reward model training") # 🔥 ИЗМЕНЕНО: logger.error -> print
                    return None
                results['reward_model'] = reward_results
                print("✅ Этап 2 завершен успешно")
            except Exception as e:
                print(f"❌ КРИТИЧЕСКАЯ ОШИБКА на этапе 2 (Reward Model): {e}")
                import traceback
                traceback.print_exc()
                return None
            
            # Этап 3: RL Fine-tuning
            print("🔄 Этап 3: RL Fine-tuning...")
            try:
                rl_results = self.stage3_rl_finetuning()
                if rl_results is None:
                    print("❌ Ошибка на этапе RL fine-tuning") # 🔥 ИЗМЕНЕНО: logger.error -> print
                    return None
                results['rl_finetuning'] = rl_results
                print("✅ Этап 3 завершен успешно")
            except Exception as e:
                print(f"❌ КРИТИЧЕСКАЯ ОШИБКА на этапе 3 (RL Fine-tuning): {e}")
                import traceback
                traceback.print_exc()
                return None
            
            print("✅ ТРЁХЭТАПНОЕ ОБУЧЕНИЕ ЗАВЕРШЕНО УСПЕШНО!") # 🔥 ИЗМЕНЕНО: logger.info -> print
            return results
            
        except Exception as e:
            print(f"❌ КРИТИЧЕСКАЯ ОШИБКА в run_full_training: {e}")
            import traceback
            traceback.print_exc()
            return None

def main():
    """Основная функция"""
    # 🔥 ИЗМЕНЕНО: Правильный путь к файлу данных
    data_path = "historical_data.csv"  
    if not os.path.exists(data_path):
        print(f"Файл данных {data_path} не найден!") # 🔥 ИЗМЕНЕНО: logger.error -> print
        return
    
    # Вызываем функцию настройки в начале программы
    has_gpu, num_gpus = DeviceConfig.setup()
    
    # Передаем информацию о устройствах в ThreeStageTrainer
    trainer = ThreeStageTrainer(data_path, has_gpu=has_gpu, num_gpus=num_gpus)
    results = trainer.run_full_training()
    
    if results:
        print("🎉 ВСЕ ЭТАПЫ ОБУЧЕНИЯ ЗАВЕРШЕНЫ!") # 🔥 ИЗМЕНЕНО: logger.info -> print
        
        # Выводим итоговую статистику
        print("=== ИТОГОВАЯ СТАТИСТИКА ===") # 🔥 ИЗМЕНЕНО: logger.info -> print
        print(f"Supervised Accuracy: {results['supervised']['accuracy']:.4f}") # 🔥 ИЗМЕНЕНО: logger.info -> print
        print(f"Reward Model Correlation: {results['reward_model']['correlation']:.4f}") # 🔥 ИЗМЕНЕНО: logger.info -> print
        # 🔥 ИЗМЕНЕНО: Проверка на наличие данных для RL Fine-tuning
        if 'rl_finetuning' in results and results['rl_finetuning'] is not True and 'episode_profits' in results['rl_finetuning'] and len(results['rl_finetuning']['episode_profits']) > 0:
            print(f"RL Final Profit: {np.mean(results['rl_finetuning']['episode_profits'][-10:]):.2f}%") # 🔥 ИЗМЕНЕНО: logger.info -> print
        else:
            print("RL Fine-tuning не предоставил данные о прибыли (возможно, пропущен или не возвращает метрики)") # 🔥 ИЗМЕНЕНО: logger.info -> print
    else:
        print("❌ ОБУЧЕНИЕ ЗАВЕРШИЛОСЬ С ОШИБКАМИ!") # 🔥 ИЗМЕНЕНО: logger.error -> print

if __name__ == "__main__":
    main()