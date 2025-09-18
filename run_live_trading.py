import os
import sys
import time
# import logging # 🔥 УДАЛЕНО: Импорт logging
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

# 🔥 УДАЛЕНО: Настройка логирования
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.StreamHandler(),
#         logging.FileHandler('live_trading.log')
#     ]
# )
# logger = logging.getLogger('live_trading')

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
            
            # НЕ преобразуем timestamp в datetime, оставляем как число
            df['timestamp'] = pd.to_numeric(df['timestamp'])
            df['symbol'] = symbol
            
            # Сортируем по времени
            df.sort_values('timestamp', inplace=True)
            
            return df
        else:
            # 🔥 ИЗМЕНЕНО: logger.error -> print
            print(f"Ошибка при получении данных: {response['retMsg']}")
            return None
    
    except Exception as e:
        # 🔥 ИЗМЕНЕНО: logger.error -> print
        print(f"Ошибка при получении данных: {e}")
        return None

def main():
    """Основная функция для запуска живой торговли"""
    # 🔥 ИЗМЕНЕНО: logger.info -> print
    print("🚀 ЗАПУСК СИСТЕМЫ ЖИВОЙ ТОРГОВЛИ С ТРЁХЭТАПНОЙ МОДЕЛЬЮ")
    
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
        # 🔥 ИЗМЕНЕНО: logger.error -> print
        print("❌ Не удалось загрузить скейлер. Убедитесь, что трёхэтапное обучение завершено.")
        return
    
    # Инициализация модели
    input_shape = (sequence_length, len(feature_engineering.feature_columns))
    rl_model = XLSTMRLModel(input_shape=input_shape, 
                          memory_size=config.XLSTM_MEMORY_SIZE, 
                          memory_units=config.XLSTM_MEMORY_UNITS)
    
    # Загружаем финальную обученную модель
    try:
        rl_model.load(stage="_rl_finetuned")
        # 🔥 ИЗМЕНЕНО: logger.info -> print
        print("✅ Финальная трёхэтапная модель успешно загружена")
    except Exception as e:
        # 🔥 ИЗМЕНЕНО: logger.error -> print
        print(f"❌ Не удалось загрузить финальную модель: {e}")
        # 🔥 ИЗМЕНЕНО: logger.info -> print
        print("Попытка загрузки supervised модели...")
        try:
            rl_model.load(stage="_supervised")
            # 🔥 ИЗМЕНЕНО: logger.info -> print
            print("✅ Supervised модель загружена как fallback")
        except Exception as e2:
            # 🔥 ИЗМЕНЕНО: logger.error -> print
            print(f"❌ Не удалось загрузить никакую модель: {e2}")
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
    
    # 🔥 ИЗМЕНЕНО: logger.info -> print
    print("✅ Система инициализирована, начинаем торговлю...")
    
    # Основной цикл торговли
    while True:
        try:
            # Получаем текущее время
            current_time = datetime.now()
            
            # Получаем последние данные
            df = fetch_latest_data(session, symbol, timeframe, limit=required_candles)
            
            if df is None or len(df) < sequence_length:
                # 🔥 ИЗМЕНЕНО: logger.error -> print
                print(f"❌ Недостаточно данных для анализа. Получено: {len(df) if df is not None else 0} строк")
                time.sleep(10)
                continue
            
            # Подготавливаем данные
            X, _, _ = feature_engineering.prepare_test_data(df)
            
            if len(X) == 0:
                # 🔥 ИЗМЕНЕНО: logger.error -> print
                print("❌ Не удалось подготовить данные для анализа")
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
            # 🔥 ИЗМЕНЕНО: logger.info -> print
            print(f"📊 Решение: {action_names[action]} (уверенность: {confidence:.4f})")
            
            # Получаем объяснение решения
            explanation = decision_maker.explain_decision(current_state)
            # 🔥 ИЗМЕНЕНО: logger.info -> print
            print(f"🧠 Анализ: BUY={explanation['all_probs']['BUY']:.3f}, "
                       f"HOLD={explanation['all_probs']['HOLD']:.3f}, "
                       f"SELL={explanation['all_probs']['SELL']:.3f}, "
                       f"Value={explanation['state_value']:.4f}")
            
            # Выполняем действие
            if trade_manager.place_order(action):
                # 🔥 ИЗМЕНЕНО: logger.info -> print
                print(f"✅ Ордер размещен: {action_names[action]}")
            else:
                # 🔥 ИЗМЕНЕНО: logger.error -> print
                print(f"❌ Не удалось разместить ордер: {action_names[action]}")
            
            # Получаем информацию о позиции
            position_info = trade_manager.get_position_info()
            if position_info and position_info['size'] > 0:
                # 🔥 ИЗМЕНЕНО: logger.info -> print
                print(f"💰 Позиция: {position_info['side']} {position_info['size']}, "
                           f"PnL: {position_info['unrealised_pnl']}")
            
            # Ждем перед следующей итерацией
            time.sleep(30)
            
        except KeyboardInterrupt:
            # 🔥 ИЗМЕНЕНО: logger.info -> print
            print("⏹️ Торговля остановлена пользователем")
            break
        except Exception as e:
            # 🔥 ИЗМЕНЕНО: logger.error -> print
            print(f"❌ Ошибка в процессе торговли: {e}")
            time.sleep(10)

if __name__ == "__main__":
    main()