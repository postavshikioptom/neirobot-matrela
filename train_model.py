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
from sklearn.utils import class_weight # 🔥 ДОБАВЛЕНО: Импорт class_weight
import math
import psutil  # 🔥 ДОБАВЛЕНО: Импорт psutil для проверки памяти
import gc
from collections import deque

# Импорт для настройки устройств
from device_config import DeviceConfig

# Вызываем функцию настройки в начале программы
has_gpu, num_gpus = DeviceConfig.setup()

# Удобно включать временно при отладке
tf.config.run_functions_eagerly(False) # 🔥 ИСПРАВЛЕНО: Установите False или удалите

# Импорт наших модулей
from feature_engineering import FeatureEngineering
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
                    # Это гарантирует, что все колонки (базовые + индикаторы) существуют
                    # перед тем, как мы пытаемся к ним обратиться или масштабировать
                    symbol_data_with_indicators = self.feature_eng._add_technical_indicators(symbol_data.copy()) # 🔥 ДОБАВЛЕНО
                    
                    # Теперь преобразуем все колонки (включая индикаторы) в числовой формат
                    temp_df_for_scaling = symbol_data_with_indicators.copy() # 🔥 ИЗМЕНЕНО: используем df с индикаторами
                    for col in self.feature_eng.feature_columns:
                        temp_df_for_scaling[col] = pd.to_numeric(temp_df_for_scaling[col], errors='coerce')
                    
                    # Применяем обученный скейлер
                    scaled_data = self.feature_eng.scaler.transform(temp_df_for_scaling[self.feature_eng.feature_columns].values)
                    
                    # Создаем последовательности из трансформированных данных
                    X_scaled_sequences, _ = self.feature_eng._create_sequences(scaled_data)
                    
                    # Создаем метки на основе оригинальных цен
                    # 🔥 ИЗМЕНЕНО: передаем original symbol_data, а не temp_df_for_scaling
                    # Используем адаптивный порог для создания меток
                    labels = self.feature_eng.create_trading_labels(
                        symbol_data  # 🔥 ИЗМЕНЕНО: Используем оригинальный df для меток
                    )
                    
                    # Обрезаем до минимальной длины
                    min_len = min(len(X_scaled_sequences), len(labels))
                    X_scaled_sequences = X_scaled_sequences[:min_len]
                    labels = labels[:min_len]
                
                if len(X_scaled_sequences) > 0:
                    all_X_supervised.append(X_scaled_sequences) # 🔥 ИЗМЕНЕНО
                    all_y_supervised.append(labels)           # 🔥 ИЗМЕНЕНО
                    
                    X_data_for_rl[symbol] = X_scaled_sequences # 🔥 ДОБАВЛЕНО: Сохраняем для RL
                    
                    # Вывод распределения меток
                    try:
                        if labels is not None and len(labels) > 0:
                            u, c = np.unique(labels, return_counts=True)
                            dist = {int(k): int(v) for k, v in zip(u, c)}
                            print(f"[SYMBOL DEBUG] {symbol} labels distribution: {dist} (threshold=adaptive)")
                        else:
                            print(f"[SYMBOL DEBUG] {symbol} produced no labels")
                    except Exception as e:
                        print(f"[SYMBOL DEBUG] error computing dist for {symbol}: {e}")
                    
            except Exception as e:
                print(f"❌ Ошибка при обработке символа {symbol}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Проверка на наличие данных перед объединением
        if all_X_supervised and all_y_supervised:
            X_supervised = np.vstack(all_X_supervised)
            y_supervised = np.concatenate(all_y_supervised)
            
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
            
            # Остальной код без изменений...
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
            """🔥 ИСПРАВЛЕНО: Батчевая аугментация с контролем памяти"""
            if len(X) == 0:
                return X, y
            
            # Перед циклом: get total and threshold once
            total_memory_gb = psutil.virtual_memory().total / (1024**3)
            # inside loop, replace current_memory_gb calculation with:
            used_memory_gb = (psutil.virtual_memory().total - psutil.virtual_memory().available) / (1024**3)
            available_memory_gb = psutil.virtual_memory().available / (1024**3)
            # Then the check:
            if available_memory_gb < (max_memory_gb * 0.2):  # stop if available < 20% of configured limit
                print(f"⚠️ Достигнут лимит памяти: available={available_memory_gb:.2f}GB, required buffer={max_memory_gb*0.2:.2f}GB")
                return X, y
            
            # Также уменьшите начальный batch_size выбор разумнее (не min(1000, len(X)//10) — это могло дать 1000). Пример:
            batch_size = min(500, max(64, len(X)//50))
            print(f"Начинаем батчевую аугментацию с размером батча {batch_size}")
            
            augmented_X = deque()  # Используем deque для эффективного добавления
            augmented_y = deque()
            
            print(f"Начинаем батчевую аугментацию с размером батча {batch_size}")
            
            for start_idx in range(0, len(X), batch_size):
                end_idx = min(start_idx + batch_size, len(X))
                batch_X = X[start_idx:end_idx]
                batch_y = y[start_idx:end_idx]
                
                # Добавляем оригинальные данные
                augmented_X.extend(batch_X)
                augmented_y.extend(batch_y)
                
                # Проверяем память перед аугментацией батча
                used_memory_gb = (psutil.virtual_memory().total - psutil.virtual_memory().available) / (1024**3)
                available_memory_gb = psutil.virtual_memory().available / (1024**3)
                # Then the check:
                if available_memory_gb < (max_memory_gb * 0.2):  # stop if available < 20% of configured limit
                    print(f"⚠️ Достигнут лимит памяти: available={available_memory_gb:.2f}GB, required buffer={max_memory_gb*0.2:.2f}GB")
                    break
                
                # Аугментируем батч
                for i in range(len(batch_X)):
                    try:
                        noise = np.random.normal(0, 0.05 * np.std(batch_X[i]), batch_X[i].shape)
                        augmented_X.append(batch_X[i] + noise)
                        augmented_y.append(batch_y[i])
                    except MemoryError:
                        print("⚠️ MemoryError при аугментации, останавливаем")
                        break
                
                # 🔥 ДОБАВЛЕНО: Принудительная очистка каждый батч
                if start_idx % (batch_size * 5) == 0:  # Каждые 5 батчей
                    gc.collect()
                    print(f"Обработано {end_idx}/{len(X)} образцов, очистка памяти")
            
            # Финальная очистка и конвертация
            gc.collect()
            result_X = np.array(list(augmented_X))
            result_y = np.array(list(augmented_y))
            
            # Очищаем deque
            augmented_X.clear()
            augmented_y.clear()
            gc.collect()
            
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
        self.feature_eng.save_scaler()
        
        # Инициализируем модель
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
        
        # Более агрессивные веса для редких классов
        # 🔥 ИСПРАВЛЕНО: Проверка на наличие уникальных классов перед расчетом class_weight
        unique_classes = np.unique(self.y_train_supervised)
        if len(unique_classes) > 1:
            class_weights_dict = class_weight.compute_class_weight(
                'balanced',
                classes=unique_classes,
                y=self.y_train_supervised
            )
            # Дополнительно увеличиваем вес для BUY (самый редкий)
            # Преобразуем словарь в массив для sample_weight
            sample_weights_base = np.array([class_weights_dict[label] for label in self.y_train_supervised], dtype=np.float32)
            
            # 🔥 ДОБАВЛЕНО: Увеличиваем вес для BUY в sample_weights_base
            # Предполагаем, что BUY класс - это 2
            buy_class_index = 2
            sample_weights_base[self.y_train_supervised == buy_class_index] *= 1.5
            
            print(f"DEBUG: sample_weights_base min={np.min(sample_weights_base):.4f}, max={np.max(sample_weights_base):.4f}, mean={np.mean(sample_weights_base):.4f}")
            print(f"Скорректированные веса классов (используются для sample_weight): {class_weights_dict}") # Логируем dict для информации
        else:
            print("⚠️ Недостаточно уникальных классов для расчета sample_weight. Используем None.")
            sample_weights_base = None # Если только один класс, веса не нужны

        # 🔥 ИСПРАВЛЕНО: Явное преобразование меток
        y_train_processed = np.array(self.y_train_supervised, dtype=np.int32)
        y_val_processed = np.array(self.y_val_supervised, dtype=np.int32)
        

        history = self.model.actor_model.fit(
            self.X_train_supervised, y_train_processed, # 🔥 ИЗМЕНЕНО
            validation_data=(self.X_val_supervised, y_val_processed), # 🔥 ИЗМЕНЕНО
            epochs=config.SUPERVISED_EPOCHS,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1,
            sample_weight=sample_weights_base
        )
        
        print("=== РЕЗУЛЬТАТЫ SUPERVISED ОБУЧЕНИЯ ===")
        
        y_pred_probs = self.model.actor_model.predict(self.X_test_supervised, verbose=0) # 🔥 ИЗМЕНЕНО
        y_pred = np.argmax(y_pred_probs, axis=1)
        
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
        """Визуализирует историю обучения"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Потери
        axes[0].plot(history.history['loss'], label='Training Loss')
        if 'val_loss' in history.history:
            axes[0].plot(history.history['val_loss'], label='Validation Loss')
        axes[0].set_title(f'{stage_name.capitalize()} Training Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        
        # Метрики
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
        
        # Подготовка данных
        if not self.load_and_prepare_data():
            print("Ошибка при подготовке данных") # 🔥 ИЗМЕНЕНО: logger.error -> print
            return None
        
        results = {}
        
        # Этап 1: Supervised Pre-training
        supervised_results = self.stage1_supervised_pretraining()
        if supervised_results is None:
            print("Ошибка на этапе supervised pre-training") # 🔥 ИЗМЕНЕНО: logger.error -> print
            return None
        results['supervised'] = supervised_results
        
        # Этап 2: Reward Model Training
        reward_results = self.stage2_reward_model_training()
        if reward_results is None:
            print("Ошибка на этапе reward model training") # 🔥 ИЗМЕНЕНО: logger.error -> print
            return None
        results['reward_model'] = reward_results
        
        # Этап 3: RL Fine-tuning
        rl_results = self.stage3_rl_finetuning()
        if rl_results is None:
            print("Ошибка на этапе RL fine-tuning") # 🔥 ИЗМЕНЕНО: logger.error -> print
            return None
        results['rl_finetuning'] = rl_results
        
        print("✅ ТРЁХЭТАПНОЕ ОБУЧЕНИЕ ЗАВЕРШЕНО УСПЕШНО!") # 🔥 ИЗМЕНЕНО: logger.info -> print
        return results

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