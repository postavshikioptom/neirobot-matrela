
Файлы, которые нужно изменить:

train_model.py (для разделения данных по символам)
trading_env.py (для обработки случайных отрезков данных)


1. Обновляем train_model.py
Мы изменим load_and_prepare_data() и stage3_rl_finetuning() так, чтобы X_train и X_val для RL-этапа были не объединенным массивом, а списком данных для каждого символа. Это позволит TradingEnvironment выбирать символы случайным образом.
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
from sklearn.utils import class_weight # 🔥 ДОБАВЛЕНО: Импорт class_weight

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

# Удаляем настройку logging.basicConfig и logger, используем print

class ThreeStageTrainer:
    """
    Трёхэтапный тренер для xLSTM + RL модели
    """
    def __init__(self, data_path):
        self.data_path = data_path
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
                    all_X_supervised.append(X_scaled_sequences) # 🔥 ИЗМЕНЕНО
                    all_y_supervised.append(labels)           # 🔥 ИЗМЕНЕНО
                    
                    X_data_for_rl[symbol] = X_scaled_sequences # 🔥 ДОБАВЛЕНО: Сохраняем для RL
                    
                    print(f"Символ {symbol}: {len(X_scaled_sequences)} последовательностей")
                    
            except Exception as e:
                print(f"❌ Ошибка при обработке символа {symbol}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Объединяем данные для supervised learning
        X_supervised = np.vstack(all_X_supervised) # 🔥 ИЗМЕНЕНО
        y_supervised = np.concatenate(all_y_supervised) # 🔥 ИЗМЕНЕНО
        
        print(f"Итого подготовлено для Supervised: X={X_supervised.shape}, y={y_supervised.shape}") # 🔥 ИЗМЕНЕНО
        print(f"Распределение классов: SELL={np.sum(y_supervised==0)}, HOLD={np.sum(y_supervised==1)}, BUY={np.sum(y_supervised==2)}") # 🔥 ИЗМЕНЕНО
        
        # Разделяем данные для supervised learning
        X_temp, self.X_test_supervised, y_temp, self.y_test_supervised = train_test_split(
            X_supervised, y_supervised, test_size=0.1, shuffle=True, random_state=42, stratify=y_supervised
        )
        self.X_train_supervised, self.X_val_supervised, self.y_train_supervised, self.y_val_supervised = train_test_split(
            X_temp, y_temp, test_size=config.SUPERVISED_VALIDATION_SPLIT, 
            shuffle=True, random_state=42, stratify=y_temp
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
            ValidationMetricsCallback(self.X_val_supervised, self.y_val_supervised) # 🔥 ИЗМЕНЕНО
        ]
        
        print(f"Начинаем supervised обучение на {config.SUPERVISED_EPOCHS} эпох...")
        
        class_weights_array = class_weight.compute_class_weight(
            'balanced', 
            classes=np.unique(self.y_train_supervised), # 🔥 ИЗМЕНЕНО
            y=self.y_train_supervised # 🔥 ИЗМЕНЕНО
        )
        class_weights = {i: class_weights_array[i] for i in range(len(class_weights_array))}
        print(f"Веса классов: {class_weights}")
        
        history = self.model.actor_model.fit(
            self.X_train_supervised, self.y_train_supervised, # 🔥 ИЗМЕНЕНО
            validation_data=(self.X_val_supervised, self.y_val_supervised), # 🔥 ИЗМЕНЕНО
            epochs=config.SUPERVISED_EPOCHS,
            batch_size=config.SUPERVISED_BATCH_SIZE,
            callbacks=callbacks,
            verbose=1,
            class_weight=class_weights
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


