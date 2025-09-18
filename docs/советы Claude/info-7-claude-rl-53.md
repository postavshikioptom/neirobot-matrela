Давайте заменим все вызовы logger.info() и logger.error() на print() в файле train_model.py.

Вот обновленный train_model.py с заменой logger на print:


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
#         logging.FileHandler('three_stage_training.log')
#     ]
# )
# logger = logging.getLogger('three_stage_trainer')

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
        print("=== ПОДГОТОВКА ДАННЫХ ===") # 🔥 ИЗМЕНЕНО: logger.info -> print
        
        # Загружаем данные
        df = pd.read_csv(self.data_path)
        print(f"Загружено {len(df)} строк данных") # 🔥 ИЗМЕНЕНО: logger.info -> print
        
        # Получаем статистику по символам
        symbol_counts = df['symbol'].value_counts()
        valid_symbols = symbol_counts[symbol_counts >= config.MIN_ROWS_PER_SYMBOL].index.tolist()
        
        if len(valid_symbols) == 0:
            valid_symbols = symbol_counts.head(20).index.tolist()
        
        print(f"Используем {len(valid_symbols)} символов: {valid_symbols[:5]}...") # 🔥 ИЗМЕНЕНО: logger.info -> print
        
        # Фильтруем данные
        df_filtered = df[df['symbol'].isin(valid_symbols)].copy()
        
        # Подготавливаем данные для supervised learning
        all_X = []
        all_y = []
        
        for i, symbol in enumerate(valid_symbols):
            symbol_data = df_filtered[df_filtered['symbol'] == symbol].copy()
            
            if len(symbol_data) < config.SEQUENCE_LENGTH + config.FUTURE_WINDOW + 10:
                print(f"Пропускаем символ {symbol}: недостаточно данных ({len(symbol_data)} строк)") # 🔥 ИЗМЕНЕНО: logger.warning -> print
                continue
            
            try:
                # 🔥 ИЗМЕНЕНО: Исправлена логика использования скейлера
                if i == 0:
                    # Для первого символа обучаем скейлер и подготавливаем данные
                    X_scaled_sequences, labels = self.feature_eng.prepare_supervised_data(
                        symbol_data, 
                        threshold=config.PRICE_CHANGE_THRESHOLD,
                        future_window=config.FUTURE_WINDOW
                    )
                else:
                    # Для остальных символов используем уже обученный скейлер
                    # и только трансформируем данные, затем генерируем метки
                    # Сначала трансформируем данные
                    temp_df_for_scaling = symbol_data.copy()
                    for col in self.feature_eng.feature_columns:
                        temp_df_for_scaling[col] = pd.to_numeric(temp_df_for_scaling[col], errors='coerce')
                    scaled_data = self.feature_eng.scaler.transform(temp_df_for_scaling[self.feature_eng.feature_columns].values)
                    
                    # Создаем последовательности из трансформированных данных
                    X_scaled_sequences, _ = self.feature_eng._create_sequences(scaled_data)
                    
                    # Создаем метки на основе оригинальных цен
                    labels = self.feature_eng.create_trading_labels(
                        symbol_data,
                        threshold=config.PRICE_CHANGE_THRESHOLD,
                        future_window=config.FUTURE_WINDOW
                    )
                    
                    # Обрезаем до минимальной длины
                    min_len = min(len(X_scaled_sequences), len(labels))
                    X_scaled_sequences = X_scaled_sequences[:min_len]
                    labels = labels[:min_len]
                
                if len(X_scaled_sequences) > 0:
                    all_X.append(X_scaled_sequences)
                    all_y.append(labels)
                    print(f"Символ {symbol}: {len(X_scaled_sequences)} последовательностей") # 🔥 ИЗМЕНЕНО: logger.info -> print
                    
            except Exception as e:
                print(f"Ошибка при обработке {symbol}: {e}") # 🔥 ИЗМЕНЕНО: logger.error -> print
                continue
        
        # Объединяем данные
        X = np.vstack(all_X)
        y = np.concatenate(all_y)
        
        print(f"Итого подготовлено: X={X.shape}, y={y.shape}") # 🔥 ИЗМЕНЕНО: logger.info -> print
        print(f"Распределение классов: SELL={np.sum(y==0)}, HOLD={np.sum(y==1)}, BUY={np.sum(y==2)}") # 🔥 ИЗМЕНЕНО: logger.info -> print
        
        # Разделяем данные
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            X, y, test_size=0.1, shuffle=True, random_state=42, stratify=y
        )
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp, test_size=config.SUPERVISED_VALIDATION_SPLIT, 
            shuffle=True, random_state=42, stratify=y_temp
        )
        
        print(f"Размеры выборок: Train={len(self.X_train)}, Val={len(self.X_val)}, Test={len(self.X_test)}") # 🔥 ИЗМЕНЕНО: logger.info -> print
        
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
        print("=== ЭТАП 1: SUPERVISED PRE-TRAINING ===") # 🔥 ИЗМЕНЕНО: logger.info -> print
        
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
            ),
            ValidationMetricsCallback(self.X_val, self.y_val)
        ]
        
        # Обучение
        print(f"Начинаем supervised обучение на {config.SUPERVISED_EPOCHS} эпох...") # 🔥 ИЗМЕНЕНО: logger.info -> print
        
        history = self.model.actor_model.fit(
            self.X_train, self.y_train,
            validation_data=(self.X_val, self.y_val),
            epochs=config.SUPERVISED_EPOCHS,
            batch_size=config.SUPERVISED_BATCH_SIZE,
            callbacks=callbacks,
            verbose=1
        )
        
        # Оценка результатов
        print("=== РЕЗУЛЬТАТЫ SUPERVISED ОБУЧЕНИЯ ===") # 🔥 ИЗМЕНЕНО: logger.info -> print
        
        # Предсказания на тестовой выборке
        y_pred_probs = self.model.actor_model.predict(self.X_test, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)
        
        # Метрики
        accuracy = accuracy_score(self.y_test, y_pred)
        print(f"Точность на тестовой выборке: {accuracy:.4f}") # 🔥 ИЗМЕНЕНО: logger.info -> print
        
        # Подробный отчет
        class_names = ['SELL', 'HOLD', 'BUY']
        report = classification_report(self.y_test, y_pred, target_names=class_names, zero_division=0)
        print(f"Классификационный отчет:\n{report}") # 🔥 ИЗМЕНЕНО: logger.info -> print
        
        # Матрица путаницы
        cm = confusion_matrix(self.y_test, y_pred)
        print(f"Матрица путаницы:\n{cm}") # 🔥 ИЗМЕНЕНО: logger.info -> print
        
        # Распределение предсказаний
        pred_dist = np.bincount(y_pred, minlength=3)
        total_pred = len(y_pred)
        print(f"Распределение предсказаний:") # 🔥 ИЗМЕНЕНО: logger.info -> print
        print(f"SELL: {pred_dist[0]} ({pred_dist[0]/total_pred:.2%})") # 🔥 ИЗМЕНЕНО: logger.info -> print
        print(f"HOLD: {pred_dist[1]} ({pred_dist[1]/total_pred:.2%})") # 🔥 ИЗМЕНЕНО: logger.info -> print
        print(f"BUY: {pred_dist[2]} ({pred_dist[2]/total_pred:.2%})") # 🔥 ИЗМЕНЕНО: logger.info -> print
        
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
        print("=== ЭТАП 2: REWARD MODEL TRAINING ===") # 🔥 ИЗМЕНЕНО: logger.info -> print
        
        if not self.model.is_supervised_trained:
            print("Сначала нужно завершить supervised pre-training!") # 🔥 ИЗМЕНЕНО: logger.error -> print
            return None
        
        # Компилируем критика для reward modeling
        self.model.compile_for_reward_modeling()
        
        # Создаём симулированные награды на основе предобученного актора
        print("Создаём симулированные награды...") # 🔥 ИЗМЕНЕНО: logger.info -> print
        
        rewards_train = self._generate_simulated_rewards(self.X_train, self.y_train)
        rewards_val = self._generate_simulated_rewards(self.X_val, self.y_val)
        
        print(f"Сгенерировано наград: Train={len(rewards_train)}, Val={len(rewards_val)}") # 🔥 ИЗМЕНЕНО: logger.info -> print
        print(f"Статистика наград: Mean={np.mean(rewards_train):.4f}, Std={np.std(rewards_train):.4f}") # 🔥 ИЗМЕНЕНО: logger.info -> print
        
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
        
        print(f"Корреляция между реальными и предсказанными наградами: {correlation:.4f}") # 🔥 ИЗМЕНЕНО: logger.info -> print
        
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
        print("=== ЭТАП 3: RL FINE-TUNING ===") # 🔥 ИЗМЕНЕНО: logger.info -> print
        
        if not self.model.is_reward_model_trained:
            print("Сначала нужно завершить reward model training!") # 🔥 ИЗМЕНЕНО: logger.error -> print
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
        
        # Создаём торговые среды
        train_env = TradingEnvironment(self.X_train, sequence_length=config.SEQUENCE_LENGTH)
        val_env = TradingEnvironment(self.X_val, sequence_length=config.SEQUENCE_LENGTH)
        
        # Создаём decision maker и simulation engine
        decision_maker = HybridDecisionMaker(rl_agent)
        train_sim = SimulationEngine(train_env, decision_maker)
        val_sim = SimulationEngine(val_env, decision_maker)
        
        # Метрики для отслеживания RL
        rl_metrics = {
            'episode_rewards': [],
            'episode_profits': [],
            'val_rewards': [],
            'val_profits': [],
            'actor_losses': [],
            'critic_losses': []
        }
        
        best_val_profit = -float('inf')
        
        print(f"Начинаем RL fine-tuning на {config.RL_EPISODES} эпизодов...") # 🔥 ИЗМЕНЕНО: logger.info -> print
        
        for episode in range(config.RL_EPISODES):
            print(f"RL Эпизод {episode+1}/{config.RL_EPISODES}") # 🔥 ИЗМЕНЕНО: logger.info -> print
            
            # Обучение
            train_results = train_sim.run_simulation(episodes=1, training=True)
            episode_reward = train_results[0]['total_reward']
            episode_profit = train_results[0]['profit_percentage']
            
            rl_metrics['episode_rewards'].append(episode_reward)
            rl_metrics['episode_profits'].append(episode_profit)
            
            # Валидация каждые 10 эпизодов
            if (episode + 1) % 10 == 0:
                val_results = val_sim.run_simulation(episodes=1, training=False)
                val_reward = val_results[0]['total_reward']
                val_profit = val_results[0]['profit_percentage']
                
                rl_metrics['val_rewards'].append(val_reward)
                rl_metrics['val_profits'].append(val_profit)
                
                # Логирование распределения действий
                sample_size = min(500, len(self.X_val))
                action_dist = rl_agent.log_action_distribution(self.X_val[:sample_size])
                
                print(f"Эпизод {episode+1}:") # 🔥 ИЗМЕНЕНО: logger.info -> print
                print(f"  Тренировка - Награда: {episode_reward:.4f}, Прибыль: {episode_profit:.2f}%") # 🔥 ИЗМЕНЕНО: logger.info -> print
                print(f"  Валидация - Награда: {val_reward:.4f}, Прибыль: {val_profit:.2f}%") # 🔥 ИЗМЕНЕНО: logger.info -> print
                print(f"  Действия - BUY: {action_dist['buy_count']}, HOLD: {action_dist['hold_count']}, SELL: {action_dist['sell_count']}") # 🔥 ИЗМЕНЕНО: logger.info -> print
                print(f"  Epsilon: {rl_agent.epsilon:.4f}") # 🔥 ИЗМЕНЕНО: logger.info -> print
                
                # Сохраняем лучшую модель
                if val_profit > best_val_profit:
                    print(f"  Новая лучшая модель! Прибыль: {val_profit:.2f}%") # 🔥 ИЗМЕНЕНО: logger.info -> print
                    self.model.save(stage="_rl_finetuned")
                    best_val_profit = val_profit
        
        # Финальная оценка
        print("=== РЕЗУЛЬТАТЫ RL FINE-TUNING ===") # 🔥 ИЗМЕНЕНО: logger.info -> print
        print(f"Лучшая прибыль на валидации: {best_val_profit:.2f}%") # 🔥 ИЗМЕНЕНО: logger.info -> print
        print(f"Средняя награда за эпизод: {np.mean(rl_metrics['episode_rewards']):.4f}") # 🔥 ИЗМЕНЕНО: logger.info -> print
        print(f"Средняя прибыль за эпизод: {np.mean(rl_metrics['episode_profits']):.2f}%") # 🔥 ИЗМЕНЕНО: logger.info -> print
        
        # Визуализация RL метрик
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
    
    # Проверяем доступность GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"Найдено GPU устройств: {len(gpus)}") # 🔥 ИЗМЕНЕНО: logger.info -> print
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("Настроен динамический рост памяти GPU") # 🔥 ИЗМЕНЕНО: logger.info -> print
        except RuntimeError as e:
            print(f"Не удалось настроить память GPU: {e}") # 🔥 ИЗМЕНЕНО: logger.warning -> print
    else:
        print("GPU не найден, будет использоваться CPU") # 🔥 ИЗМЕНЕНО: logger.info -> print
    
    # Создаём и запускаем тренер
    trainer = ThreeStageTrainer(data_path)
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
