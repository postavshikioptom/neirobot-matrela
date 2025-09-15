Вот исправленный код для train_model.py, который будет использовать все символы из файла и поддерживать GPU:
import os
import sys
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
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
        logging.FileHandler('training.log')
    ]
)

logger = logging.getLogger('train_model')

def load_data(file_path):
    """
    Загружает данные из CSV файла
    """
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Данные загружены из {file_path}, строк: {len(df)}")
        
        # Проверяем какие символы есть в данных
        unique_symbols = df['symbol'].unique()
        logger.info(f"Найдено символов: {len(unique_symbols)}")
        logger.info(f"Первые 10 символов: {unique_symbols[:10]}")
        
        return df
    except Exception as e:
        logger.error(f"Ошибка при загрузке данных: {e}")
        return None

def prepare_multi_symbol_data(df, min_rows_per_symbol=500):
    """
    Подготавливает данные для всех символов
    """
    # Получаем статистику по символам
    symbol_counts = df['symbol'].value_counts()
    logger.info(f"Статистика по символам:")
    logger.info(f"Всего уникальных символов: {len(symbol_counts)}")
    logger.info(f"Среднее количество строк на символ: {symbol_counts.mean():.1f}")
    logger.info(f"Минимум строк: {symbol_counts.min()}, Максимум строк: {symbol_counts.max()}")
    
    # Фильтруем символы с достаточным количеством данных
    valid_symbols = symbol_counts[symbol_counts >= min_rows_per_symbol].index.tolist()
    logger.info(f"Символов с минимум {min_rows_per_symbol} строк: {len(valid_symbols)}")
    
    if len(valid_symbols) == 0:
        # Если нет символов с достаточным количеством данных, берем топ символы
        valid_symbols = symbol_counts.head(50).index.tolist()
        logger.info(f"Используем топ-50 символов по количеству данных: {valid_symbols[:10]}...")
    
    # Фильтруем данные только для валидных символов
    filtered_df = df[df['symbol'].isin(valid_symbols)].copy()
    logger.info(f"Отфильтровано строк: {len(filtered_df)} для {len(valid_symbols)} символов")
    
    return filtered_df, valid_symbols

def train_model(data_path, epochs=30, batch_size=64, validation_split=0.2, test_split=0.1):
    """
    Обучает модель xLSTM RL для торговли на всех символах
    """
    logger.info(f"Начало обучения модели с {epochs} эпохами")
    
    # Загружаем данные
    df = load_data(data_path)
    if df is None:
        return
    
    # Подготавливаем данные для всех символов
    df_filtered, valid_symbols = prepare_multi_symbol_data(df)
    
    if len(df_filtered) == 0:
        logger.error("Нет подходящих данных для обучения")
        return
    
    logger.info(f"Будем использовать {len(valid_symbols)} символов для обучения")
    
    # Объединяем данные всех символов для обучения
    all_X = []
    all_y_close = []
    
    feature_eng = FeatureEngineering(sequence_length=config.SEQUENCE_LENGTH)
    
    # Обрабатываем каждый символ отдельно
    for i, symbol in enumerate(valid_symbols):
        logger.info(f"Обрабатываем символ {symbol} ({i+1}/{len(valid_symbols)})")
        
        symbol_data = df_filtered[df_filtered['symbol'] == symbol].copy()
        
        if len(symbol_data) < config.SEQUENCE_LENGTH + 10:  # Минимум данных
            logger.warning(f"Недостаточно данных для символа {symbol}: {len(symbol_data)} строк")
            continue
        
        try:
            # Для первого символа обучаем скейлер, для остальных используем тот же
            if i == 0:
                X_symbol, y_symbol, _ = feature_eng.prepare_data(symbol_data)
            else:
                X_symbol, y_symbol, _ = feature_eng.prepare_test_data(symbol_data)
            
            if len(X_symbol) > 0:
                all_X.append(X_symbol)
                all_y_close.append(y_symbol)
                logger.info(f"Символ {symbol}: подготовлено {len(X_symbol)} последовательностей")
            else:
                logger.warning(f"Не удалось подготовить данные для символа {symbol}")
                
        except Exception as e:
            logger.error(f"Ошибка при обработке символа {symbol}: {e}")
            continue
    
    if len(all_X) == 0:
        logger.error("Не удалось подготовить данные ни для одного символа")
        return
    
    # Объединяем все данные
    X = np.vstack(all_X)
    y_close = np.concatenate(all_y_close)
    
    logger.info(f"Итого подготовлено последовательностей: {len(X)}")
    logger.info(f"Форма данных X: {X.shape}")
    
    # Сохраняем скейлер для последующего использования
    feature_eng.save_scaler()
    
    # Разделение на обучающую, валидационную и тестовую выборки
    # Перемешиваем данные, так как они от разных символов
    X_temp, X_test, y_temp, y_test = train_test_split(X, y_close, test_size=test_split, shuffle=True, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=validation_split, shuffle=True, random_state=42)
    
    logger.info(f"Размеры выборок: Обучающая: {len(X_train)}, Валидационная: {len(X_val)}, Тестовая: {len(X_test)}")
    
    # Создаем среды для обучения, валидации и тестирования
    train_env = TradingEnvironment(X_train, sequence_length=config.SEQUENCE_LENGTH)
    val_env = TradingEnvironment(X_val, sequence_length=config.SEQUENCE_LENGTH)
    test_env = TradingEnvironment(X_test, sequence_length=config.SEQUENCE_LENGTH)
    
    # Инициализация RL-агента
    input_shape = (config.SEQUENCE_LENGTH, X.shape[2])
    rl_agent = RLAgent(
        state_shape=input_shape,
        memory_size=config.XLSTM_MEMORY_SIZE,
        memory_units=config.XLSTM_MEMORY_UNITS,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.1,
        epsilon_decay=0.995,
        batch_size=batch_size
    )
    
    # Инициализация механизма принятия решений
    decision_maker = HybridDecisionMaker(rl_agent)
    
    # Инициализация движка симуляции
    train_sim = SimulationEngine(train_env, decision_maker)
    val_sim = SimulationEngine(val_env, decision_maker)
    test_sim = SimulationEngine(test_env, decision_maker)
    
    # Создаем директорию для сохранения результатов
    os.makedirs('results', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Метрики для отслеживания
    train_rewards = []
    val_rewards = []
    train_profits = []
    val_profits = []
    best_val_profit = -float('inf')
    
    # Обучение модели
    logger.info("Начало процесса обучения...")
    
    for epoch in range(epochs):
        logger.info(f"Эпоха {epoch+1}/{epochs}")
        
        # Обучение на тренировочной выборке
        logger.info("Тренировка на обучающей выборке...")
        train_results = train_sim.run_simulation(episodes=1, training=True)
        train_reward = train_results[0]['total_reward']
        train_profit = train_results[0]['profit_percentage']
        train_rewards.append(train_reward)
        train_profits.append(train_profit)
        
        # Логирование распределения действий на обучающей выборке
        sample_size = min(1000, len(X_train))
        train_actions = rl_agent.log_action_distribution(X_train[:sample_size])
        
        # Валидация
        logger.info("Валидация модели...")
        val_results = val_sim.run_simulation(episodes=1, training=False)
        val_reward = val_results[0]['total_reward']
        val_profit = val_results[0]['profit_percentage']
        val_rewards.append(val_reward)
        val_profits.append(val_profit)
        
        # Логирование распределения действий на валидационной выборке
        sample_size = min(1000, len(X_val))
        val_actions = rl_agent.log_action_distribution(X_val[:sample_size])
        
        # Логирование метрик
        logger.info(f"Эпоха {epoch+1} завершена:")
        logger.info(f"  Тренировка - Награда: {train_reward:.4f}, Прибыль: {train_profit:.2f}%")
        logger.info(f"  Валидация - Награда: {val_reward:.4f}, Прибыль: {val_profit:.2f}%")
        
        # Логирование распределения действий
        logger.info(f"  Распределение действий (Тренировка) - BUY: {train_actions['buy_count']}, HOLD: {train_actions['hold_count']}, SELL: {train_actions['sell_count']}")
        logger.info(f"  Распределение действий (Валидация) - BUY: {val_actions['buy_count']}, HOLD: {val_actions['hold_count']}, SELL: {val_actions['sell_count']}")
        
        # Сохранение лучшей модели по прибыли на валидационной выборке
        if val_profit > best_val_profit:
            logger.info(f"  Улучшение на валидации! Сохраняем модель. Прибыль: {val_profit:.2f}% (предыдущая лучшая: {best_val_profit:.2f}%)")
            rl_agent.save()
            best_val_profit = val_profit
        
        # Каждые 5 эпох делаем подробный анализ модели
        if (epoch + 1) % 5 == 0:
            logger.info(f"Детальный анализ модели после эпохи {epoch+1}:")
            
            # Анализ признаков, влияющих на решения
            logger.info("Анализ признаков, влияющих на решения:")
            
            # Проверяем предсказания модели на валидационной выборке
            sample_size = min(500, len(X_val))  # Уменьшаем размер выборки для анализа
            val_sample = X_val[:sample_size]
            
            val_actions_full = []
            for i in range(len(val_sample)):
                action_probs = rl_agent.model.predict_action(val_sample[i])
                val_actions_full.append(np.argmax(action_probs))
            
            val_actions_full = np.array(val_actions_full)
            buy_indices = np.where(val_actions_full == 0)[0]
            hold_indices = np.where(val_actions_full == 1)[0]
            sell_indices = np.where(val_actions_full == 2)[0]
            
            # Анализ паттернов для каждого типа действия
            if len(buy_indices) > 0:
                buy_patterns = val_sample[buy_indices]
                logger.info(f"Паттерны BUY (среднее значение признаков):")
                for i in range(val_sample.shape[2]):
                    logger.info(f"  Признак {i}: {np.mean(buy_patterns[:, -1, i]):.4f}")
            
            if len(hold_indices) > 0:
                hold_patterns = val_sample[hold_indices]
                logger.info(f"Паттерны HOLD (среднее значение признаков):")
                for i in range(val_sample.shape[2]):
                    logger.info(f"  Признак {i}: {np.mean(hold_patterns[:, -1, i]):.4f}")
            
            if len(sell_indices) > 0:
                sell_patterns = val_sample[sell_indices]
                logger.info(f"Паттерны SELL (среднее значение признаков):")
                for i in range(val_sample.shape[2]):
                    logger.info(f"  Признак {i}: {np.mean(sell_patterns[:, -1, i]):.4f}")
            
            Продолжаю код для train_model.py, где остановился:
            # Визуализация метрик обучения
            plt.figure(figsize=(12, 10))
            
            plt.subplot(2, 1, 1)
            plt.plot(range(1, epoch+2), train_rewards, 'b-', label='Обучение')
            plt.plot(range(1, epoch+2), val_rewards, 'r-', label='Валидация')
            plt.title('Награды по эпохам')
            plt.xlabel('Эпоха')
            plt.ylabel('Награда')
            plt.legend()
            plt.grid(True)
            
            plt.subplot(2, 1, 2)
            plt.plot(range(1, epoch+2), train_profits, 'b-', label='Обучение')
            plt.plot(range(1, epoch+2), val_profits, 'r-', label='Валидация')
            plt.title('Прибыль по эпохам')
            plt.xlabel('Эпоха')
            plt.ylabel('Прибыль (%)')
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(f'results/training_metrics_epoch_{epoch+1}.png')
            plt.close()
            
            # Логируем информацию о GPU/CPU использовании
            if len(physical_devices) > 0:
                logger.info(f"Обучение выполняется на GPU: {tf.config.experimental.get_device_details(physical_devices[0])}")
            else:
                logger.info("Обучение выполняется на CPU")
    
    # Загружаем лучшую модель для финального тестирования
    logger.info("Загрузка лучшей модели для тестирования...")
    rl_agent.load()
    
    # Тестирование на тестовой выборке
    logger.info("Тестирование модели на тестовой выборке...")
    test_results = test_sim.run_simulation(episodes=1, training=False, render=True)
    test_reward = test_results[0]['total_reward']
    test_profit = test_results[0]['profit_percentage']
    
    # Логирование распределения действий на тестовой выборке
    sample_size = min(1000, len(X_test))
    test_actions = rl_agent.log_action_distribution(X_test[:sample_size])
    
    logger.info("Результаты тестирования:")
    logger.info(f"  Награда: {test_reward:.4f}")
    logger.info(f"  Прибыль: {test_profit:.2f}%")
    logger.info(f"  Распределение действий - BUY: {test_actions['buy_count']}, HOLD: {test_actions['hold_count']}, SELL: {test_actions['sell_count']}")
    logger.info(f"  Процент действий - BUY: {test_actions['buy_count']/test_actions['total']:.2%}, HOLD: {test_actions['hold_count']/test_actions['total']:.2%}, SELL: {test_actions['sell_count']/test_actions['total']:.2%}")
    
    # Финальная оценка модели
    logger.info("\nФинальная оценка модели:")
    logger.info(f"Лучшая прибыль на валидации: {best_val_profit:.2f}%")
    logger.info(f"Прибыль на тестовой выборке: {test_profit:.2f}%")
    logger.info(f"Обучение проводилось на {len(valid_symbols)} символах")
    logger.info(f"Всего обработано {len(X)} последовательностей")
    
    # Сохраняем информацию о символах для последующего использования
    with open('models/trained_symbols.txt', 'w') as f:
        for symbol in valid_symbols:
            f.write(f"{symbol}\n")
    logger.info("Список символов сохранен в models/trained_symbols.txt")
    
    return {
        'train_rewards': train_rewards,
        'val_rewards': val_rewards,
        'train_profits': train_profits,
        'val_profits': val_profits,
        'test_reward': test_reward,
        'test_profit': test_profit,
        'test_actions': test_actions,
        'symbols_used': valid_symbols,
        'total_sequences': len(X)
    }

if __name__ == "__main__":
    # Путь к файлу с данными
    data_path = "historical_data.csv"
    
    # Количество эпох обучения
    epochs = 30  # Это можно изменить на 50 позже
    
    # Проверяем доступность GPU
    logger.info("Проверка доступности GPU...")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        logger.info(f"Найдено GPU устройств: {len(gpus)}")
        for i, gpu in enumerate(gpus):
            logger.info(f"GPU {i}: {gpu}")
        
        # Настройка использования памяти GPU
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info("Настроен динамический рост памяти GPU")
        except RuntimeError as e:
            logger.warning(f"Не удалось настроить память GPU: {e}")
    else:
        logger.info("GPU не найден, будет использоваться CPU")
    
    # Запуск обучения
    logger.info(f"Запуск обучения модели на {epochs} эпох...")
    results = train_model(data_path, epochs=epochs)
    
    if results:
        logger.info("Обучение завершено успешно!")
        logger.info(f"Итоговые результаты: {results}")
    else:
        logger.error("Обучение завершилось с ошибкой!")

Также нужно обновить файл config.py, чтобы он не зависел от конкретного символа:
# Bybit API credentials
BYBIT_API_KEY = "OOofB1HzYVpySyMPom"
BYBIT_API_SECRET = "e4AkAz9x1ycOMCtXKa1milmShfk61KZxJyhG"
API_URL = "https://api-demo.bybit.com"
# WebSocket v5 для публичных данных (свечи) по споту
WEBSOCKET_URL = "wss://stream.bybit.com/v5/public/linear"

# --- Trading Parameters ---
ORDER_USDT_AMOUNT = 11.0 # Сумма одного ордера в USDT
LEVERAGE = "2"  # Устанавливаем плечо. ВАЖНО: это значение должно быть в виде строки.
REQUIRED_CANDLES = 100 # Сколько свечей нужно для анализа
# Для обучения будем использовать все символы из файла, для торговли - один символ
SYMBOLS = ["SOLUSDT"]  # Для живой торговли
TIMEFRAME = "1"  # Таймфрейм в минутах (1, 5, 15, 60, 240, "D", "W", "M")
SEQUENCE_LENGTH = 60 # Длина последовательности для моделей
REQUIRED_CANDLES = 65 # Количество свечей для загрузки (должно быть больше SEQUENCE_LENGTH)

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
OPTIMIZATION_FREQUENCY = 50  # Every 50 trades
PERFORMANCE_HISTORY_SIZE = 1000

# Training parameters
MIN_ROWS_PER_SYMBOL = 500  # Минимальное количество строк для символа
MAX_SYMBOLS_FOR_TRAINING = 100  # Максимальное количество символов для обучения


