Моя ошибка! Вы абсолютно правы. В файле three_stage_trainer.py я по ошибке указал market_data.csv вместо historical_data.csv.
Вот исправленный three_stage_trainer.py:
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
from validation_metrics_callback import ValidationMetricsCallback # 🔥 ДОБАВЛЕНО: Импорт каллбэка

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
                logger.warning(f"Пропускаем символ {symbol}: недостаточно данных ({len(symbol_data)} строк)")
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
                    logger.info(f"Символ {symbol}: {len(X_scaled_sequences)} последовательностей")
                    
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
            ),
            ValidationMetricsCallback(self.X_val, self.y_val) # 🔥 ДОБАВЛЕНО: Каллбэк для метрик
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
        report = classification_report(self.y_test, y_pred, target_names=class_names, zero_division=0) # 🔥 ДОБАВЛЕНО: zero_division
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
        
        logger.info(f"Начинаем RL fine-tuning на {config.RL_EPISODES} эпизодов...")
        
        for episode in range(config.RL_EPISODES):
            logger.info(f"RL Эпизод {episode+1}/{config.RL_EPISODES}")
            
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
                
                logger.info(f"Эпизод {episode+1}:")
                logger.info(f"  Тренировка - Награда: {episode_reward:.4f}, Прибыль: {episode_profit:.2f}%")
                logger.info(f"  Валидация - Награда: {val_reward:.4f}, Прибыль: {val_profit:.2f}%")
                logger.info(f"  Действия - BUY: {action_dist['buy_count']}, HOLD: {action_dist['hold_count']}, SELL: {action_dist['sell_count']}")
                logger.info(f"  Epsilon: {rl_agent.epsilon:.4f}")
                
                # Сохраняем лучшую модель
                if val_profit > best_val_profit:
                    logger.info(f"  Новая лучшая модель! Прибыль: {val_profit:.2f}%")
                    self.model.save(stage="_rl_finetuned")
                    best_val_profit = val_profit
        
        # Финальная оценка
        logger.info("=== РЕЗУЛЬТАТЫ RL FINE-TUNING ===")
        logger.info(f"Лучшая прибыль на валидации: {best_val_profit:.2f}%")
        logger.info(f"Средняя награда за эпизод: {np.mean(rl_metrics['episode_rewards']):.4f}")
        logger.info(f"Средняя прибыль за эпизод: {np.mean(rl_metrics['episode_profits']):.2f}%")
        
        # Визуализация RL метрик
        self._plot_rl_metrics(rl_metrics)
        
        return rl_metrics
    
    def _generate_simulated_rewards(self, X, y_true):
        &quot;&quot;&quot;Генерирует симулированные награды на основе предсказаний модели&quot;&quot;&quot;
        # Получаем предсказания от предобученной модели
        y_pred_probs &#x3D; self.model.actor_model.predict(X, verbose&#x3D;0)
        y_pred &#x3D; np.argmax(y_pred_probs, axis&#x3D;1)
        
        # Рассчитываем награды на основе точности предсказаний
        rewards &#x3D; []
        for true_label, pred_label, pred_probs in zip(y_true, y_pred, y_pred_probs):
            if true_label &#x3D;&#x3D; pred_label:
                # Правильное предсказание - положительная награда
                reward &#x3D; 1.0
            else:
                # Неправильное предсказание - отрицательная награда
                reward &#x3D; -1.0
            
            # Добавляем компоненту уверенности
            confidence &#x3D; pred_probs[pred_label]
            reward *&#x3D; confidence
            
            rewards.append(reward)
        
        return np.array(rewards)
    
    def _plot_training_history(self, history, stage_name):
        &quot;&quot;&quot;Визуализирует историю обучения&quot;&quot;&quot;
        fig, axes &#x3D; plt.subplots(1, 2, figsize&#x3D;(15, 5))
        
        # Потери
        axes[0].plot(history.history[&#39;loss&#39;], label&#x3D;&#39;Training Loss&#39;)
        if &#39;val_loss&#39; in history.history:
            axes[0].plot(history.history[&#39;val_loss&#39;], label&#x3D;&#39;Validation Loss&#39;)
        axes[0].set_title(f&#39;{stage_name.capitalize()} Training Loss&#39;)
        axes[0].set_xlabel(&#39;Epoch&#39;)
        axes[0].set_ylabel(&#39;Loss&#39;)
        axes[0].legend()
        
        # Метрики
        if &#39;accuracy&#x39; in history.history:
            axes[1].plot(history.history[&#39;accuracy&#x39;], label&#x3D;&#39;Training Accuracy&#39;)
            if &#39;val_accuracy&#x39; in history.history:
                axes[1].plot(history.history[&#39;val_accuracy&#39;], label&#x3D;&#39;Validation Accuracy&#39;)
            axes[1].set_title(f&#39;{stage_name.capitalize()} Accuracy&#39;)
            axes[1].set_xlabel(&#39;Epoch&#39;)
            axes[1].set_ylabel(&#39;Accuracy&#39;)
            axes[1].legend()
        elif &#39;mae&#x39; in history.history:
            axes[1].plot(history.history[&#39;mae&#39;], label&#x3D;&#39;Training MAE&#39;)
            if &#39;val_mae&#x39; in history.history:
                axes[1].plot(history.history[&#39;val_mae&#x39;], label&#x3D;&#39;Validation MAE&#39;)
            axes[1].set_title(f&#39;{stage_name.capitalize()} MAE&#39;)
            axes[1].set_xlabel(&#39;Epoch&#39;)
            axes[1].set_ylabel(&#39;MAE&#39;)
            axes[1].legend()
        
        plt.tight_layout()
        plt.savefig(f&#39;plots&#x2F;{stage_name}_training_history.png&#39;)
        plt.close()
    
    def _plot_rl_metrics(self, metrics):
        &quot;&quot;&quot;Визуализирует RL метрики&quot;&quot;&quot;
        fig, axes &#x3D; plt.subplots(2, 2, figsize&#x3D;(12, 8))
        
        # Episode rewards
        axes[0,0].plot(metrics[&#39;episode_rewards&#39;])
        axes[0,0].set_title(&#39;Episode Rewards&#39;)
        axes[0,0].set_xlabel(&#39;Episode&#39;)
        axes[0,0].set_ylabel(&#39;Reward&#39;)
        axes[0,0].grid(True)
        
        # Episode profits
        axes[0,1].plot(metrics[&#39;episode_profits&#39;])
        axes[0,1].set_title(&#39;Episode Profits (%)&#39;)
        axes[0,1].set_xlabel(&#39;Episode&#39;)
        axes[0,1].set_ylabel(&#39;Profit %&#39;)
        axes[0,1].grid(True)
        
        # Validation rewards (каждые 10 эпизодов)
        if metrics[&#39;val_rewards&#39;]:
            val_episodes &#x3D; range(10, len(metrics[&#39;val_rewards&#39;]) * 10 + 1, 10)
            axes[1,0].plot(val_episodes, metrics[&#39;val_rewards&#39;])
            axes[1,0].set_title(&#39;Validation Rewards&#39;)
            axes[1,0].set_xlabel(&#39;Episode&#39;)
            axes[1,0].set_ylabel(&#39;Reward&#39;)
            axes[1,0].grid(True)
        
        # Validation profits
        if metrics['val_profits']:
            val_episodes = range(10, len(metrics['val_profits']) * 10 + 1, 10)
            axes[1,1].plot(val_episodes, metrics['val_profits'])
            axes[1,1].set_title('Validation Profits (%)')
            axes[1,1].set_xlabel('Episode')
            axes[1,1].set_ylabel('Profit %') # 🔥 ИЗМЕНЕНО: Исправлена опечатка
            axes[1,1].grid(True)
        
        plt.tight_layout()
        plt.savefig('plots/rl_training_metrics.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def run_full_training(self):
        """Запускает полное трёхэтапное обучение"""
        logger.info("🚀 ЗАПУСК ТРЁХЭТАПНОГО ОБУЧЕНИЯ xLSTM + RL")
        
        # Подготовка данных
        if not self.load_and_prepare_data():
            logger.error("Ошибка при подготовке данных")
            return None
        
        results = {}
        
        # Этап 1: Supervised Pre-training
        supervised_results = self.stage1_supervised_pretraining()
        if supervised_results is None:
            logger.error("Ошибка на этапе supervised pre-training")
            return None
        results['supervised'] = supervised_results
        
        # Этап 2: Reward Model Training
        reward_results = self.stage2_reward_model_training()
        if reward_results is None:
            logger.error("Ошибка на этапе reward model training")
            return None
        results['reward_model'] = reward_results
        
        # Этап 3: RL Fine-tuning
        rl_results = self.stage3_rl_finetuning()
        if rl_results is None:
            logger.error("Ошибка на этапе RL fine-tuning")
            return None
        results['rl_finetuning'] = rl_results
        
        logger.info("✅ ТРЁХЭТАПНОЕ ОБУЧЕНИЕ ЗАВЕРШЕНО УСПЕШНО!")
        return results

def main():
    """Основная функция"""
    # 🔥 ИЗМЕНЕНО: Правильный путь к файлу данных
    data_path = "historical_data.csv"  
    if not os.path.exists(data_path):
        logger.error(f"Файл данных {data_path} не найден!")
        return
    
    # Проверяем доступность GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        logger.info(f"Найдено GPU устройств: {len(gpus)}")
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info("Настроен динамический рост памяти GPU")
        except RuntimeError as e:
            logger.warning(f"Не удалось настроить память GPU: {e}")
    else:
        logger.info("GPU не найден, будет использоваться CPU")
    
    # Создаём и запускаем тренер
    trainer = ThreeStageTrainer(data_path)
    results = trainer.run_full_training()
    
    if results:
        logger.info("🎉 ВСЕ ЭТАПЫ ОБУЧЕНИЯ ЗАВЕРШЕНЫ!")
        
        # Выводим итоговую статистику
        logger.info("=== ИТОГОВАЯ СТАТИСТИКА ===")
        logger.info(f"Supervised Accuracy: {results['supervised']['accuracy']:.4f}")
        logger.info(f"Reward Model Correlation: {results['reward_model']['correlation']:.4f}")
        # 🔥 ИЗМЕНЕНО: Проверка на наличие данных для RL Fine-tuning
        if 'rl_finetuning' in results and results['rl_finetuning'] is not True and 'episode_profits' in results['rl_finetuning'] and len(results['rl_finetuning']['episode_profits']) > 0:
            logger.info(f"RL Final Profit: {np.mean(results['rl_finetuning']['episode_profits'][-10:]):.2f}%")
        else:
            logger.info("RL Fine-tuning не предоставил данные о прибыли (возможно, пропущен или не возвращает метрики)")
    else:
        logger.error("❌ ОБУЧЕНИЕ ЗАВЕРШИЛОСЬ С ОШИБКАМИ!")

if __name__ == "__main__":
    main()
