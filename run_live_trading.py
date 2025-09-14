import time
import os
import json
import pandas as pd
import numpy as np
from pybit.unified_trading import HTTP
import logging
from functools import wraps
import psutil
import gc

# === НОВЫЕ ИМПОРТЫ ===
import config
import feature_engineering
import trade_manager
import trade_logger
from hybrid_decision_maker import HybridDecisionMaker
from performance_monitor import PerformanceMonitor
from notification_system import NotificationSystem

# === КОНСТАНТЫ ===
TRADER_STATUS_FILE = 'trader_status.txt'
ACTIVE_POSITIONS_FILE = 'active_positions.json'
LIVE_DATA_FILE = 'live_data.json'
HOTLIST_FILE = 'hotlist.txt'
LOG_FILE = 'trade_log.csv'
LOOP_SLEEP_SECONDS = 3
OPEN_TRADE_LIMIT = 1000
TAKE_PROFIT_PCT = 1.5  # Увеличили TP
STOP_LOSS_PCT = -1.0   # Уменьшили SL
CONFIDENCE_THRESHOLD = 0.65  # Повысили порог уверенности
#SEQUENCE_LENGTH = 10

# 🔥 НОВЫЕ FEATURE_COLUMNS - ТОЛЬКО ИНДИКАТОРЫ
FEATURE_COLUMNS = [
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

opened_trades_counter = 0

performance_monitor = None
notification_system = None

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)

def error_handler(func):
    """Декоратор для обработки ошибок"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.error(f"Ошибка в {func.__name__}: {e}")
            if 'notification_system' in globals():
                notification_system.send_system_alert(f"Ошибка в {func.__name__}: {e}")
            return None
    return wrapper

@error_handler
def manage_active_positions(session, decision_maker):
    """Управление активными позициями с новой логикой"""
    active_positions = load_active_positions()
    if not active_positions:
        return

    print(f"Открыто сделок: {opened_trades_counter}/{OPEN_TRADE_LIMIT}. Активных позиций: {len(active_positions)}")
    
    kline_cache = {}
    symbols_to_remove = []
    positions_items = list(active_positions.items())
    displayed_positions = positions_items[:5]
    remaining_count = len(positions_items) - 5 if len(positions_items) > 5 else 0
    
    if remaining_count > 0:
        print(f"  ... и еще {remaining_count} позиций (скрыто для чистоты логов)")

    for i, (symbol, pos) in enumerate(positions_items):
        try:
            if symbol not in kline_cache:
                kline_list = trade_manager.fetch_initial_data(session, symbol)
                if not kline_list:
                    continue
                kline_cache[symbol] = pd.DataFrame(kline_list, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
            
            kline_df = kline_cache[symbol].copy()
            
            # === НОВАЯ ОБРАБОТКА С VSA ===
            features_df = feature_engineering.calculate_features(kline_df.copy())
            # features_df = feature_engineering.detect_candlestick_patterns(features_df) # 🔥 ЗАКОММЕНТИРОВАНО
            # features_df = feature_engineering.calculate_vsa_features(features_df)  # <--- ЗАКОММЕНТИРОВАНО
            
            if features_df.empty or len(features_df) < config.SEQUENCE_LENGTH:
                continue

            # Принимаем решение через новую гибридную систему
            decision = decision_maker.get_decision(features_df.tail(15), confidence_threshold=CONFIDENCE_THRESHOLD)

            latest_price = float(features_df.iloc[-1]['close'])
            entry_price = float(pos['entry_price'])
            
            # Рассчитываем PnL
            if pos['side'] == 'BUY':
                pnl_pct = ((latest_price - entry_price) / entry_price) * 100
            else:  # SELL
                pnl_pct = ((entry_price - latest_price) / entry_price) * 100

            # Показываем детали только для первых 5 позиций
            if i < 5:
                print(f"  - {symbol}: PnL {pnl_pct:.2f}% | Вход: {entry_price} | Сейчас: {latest_price} | Решение: {decision}")

            # === УЛУЧШЕННАЯ ЛОГИКА ВЫХОДА С ДИНАМИЧЕСКИМИ СТОПАМИ ===
            should_close = False
            close_reason = ""
            
            # Вычисляем динамические стопы
            dynamic_sl, dynamic_tp = calculate_dynamic_stops(features_df.iloc[-1], pos['side'], entry_price)
            
            print(f"  📊 Динамические уровни для {symbol}: TP={dynamic_tp:.2f}%, SL={dynamic_sl:.2f}%")
            
            # 1. Динамические стоп-лосс и тейк-профит
            if pnl_pct >= dynamic_tp:
                should_close = True
                close_reason = f"DYNAMIC_TP ({pnl_pct:.2f}%)"
            elif pnl_pct <= dynamic_sl:
                should_close = True
                close_reason = f"DYNAMIC_SL ({pnl_pct:.2f}%)"
            
            # 2. Сигнал модели на закрытие
            elif (pos['side'] == 'BUY' and decision == 'SELL') or (pos['side'] == 'SELL' and decision == 'BUY'):
                should_close = True
                close_reason = f"MODEL_SIGNAL ({decision})"
            
            # 3. VSA сигнал на закрытие (отключен)
            # elif should_close_by_vsa(features_df.iloc[-1], pos['side']): # <--- ЗАКОММЕНТИРОВАНО
            #     should_close = True
            #     close_reason = "VSA_SIGNAL"
            
            if should_close:
                print(f"!!! {symbol}: {close_reason}. Закрываю позицию... !!!")
                
                close_result = trade_manager.close_market_position(session, symbol, pos['quantity'], pos['side'])
                if close_result.get('status') == 'SUCCESS':
                    # Логируем с объяснением решения
                    trade_logger.log_enhanced_trade_with_quality_metrics(symbol, 'CLOSE', close_result, pos, pnl_pct,
                                     decision_maker, features_df.iloc[-1], close_reason)
                    notification_system.send_trade_alert(symbol, "CLOSE", close_result['price'], pnl_pct, reason=close_reason)
                    performance_monitor.log_trade_closed(symbol, pnl_pct)
                    symbols_to_remove.append(symbol)

        except Exception as e:
            print(f"Ошибка при управлении позицией {symbol}: {e}")

    # Удаляем закрытые позиции
    if symbols_to_remove:
        current_positions = load_active_positions()
        for symbol in symbols_to_remove:
            if symbol in current_positions: 
                del current_positions[symbol]
        save_active_positions(current_positions)


@error_handler
def process_new_signal(session, symbol, decision_maker):
    """Обработка новых сигналов с VSA анализом"""
    global opened_trades_counter
    
    if opened_trades_counter >= OPEN_TRADE_LIMIT: 
        return
    
    active_positions = load_active_positions()
    if symbol in active_positions: 
        return

    print(f"--- Обработка нового сигнала для {symbol} (с VSA анализом) ---")
    
    try:
        with open(LIVE_DATA_FILE, 'r') as f: 
            live_data = json.load(f)
        
        symbol_data = live_data.get(symbol)
        if not symbol_data: 
            return

        kline_list = symbol_data.get('klines')
        if not kline_list or len(kline_list) < config.REQUIRED_CANDLES: 
            return

        # Подготавливаем данные с VSA
        kline_df = pd.DataFrame(kline_list, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
        
        # === ПОЛНАЯ ОБРАБОТКА С VSA ===
        features_df = feature_engineering.calculate_features(kline_df.copy())
        # features_df = feature_engineering.detect_candlestick_patterns(features_df) # 🔥 ЗАКОММЕНТИРОВАНО
        # features_df = feature_engineering.calculate_vsa_features(features_df) # <--- ЗАКОММЕНТИРОВАНО
        
        if features_df.empty or len(features_df) < config.SEQUENCE_LENGTH:
            return

        # Принимаем решение через гибридную систему
        decision = decision_maker.get_decision(features_df.tail(15), confidence_threshold=CONFIDENCE_THRESHOLD)
        
        print(f"--- {symbol} | Гибридное решение: {decision} ---")
        
        # Показываем объяснение решения
        if hasattr(decision_maker, 'get_decision_explanation'):
            explanation = decision_maker.get_decision_explanation()
            print(explanation)

        if decision in ['BUY', 'SELL']:
            # НОВЫЙ КОД - Открытие сделки без VSA подтверждения
    # Дополнительная проверка VSA подтверждения (отключена)
    # if validate_decision_with_vsa(features_df.iloc[-1], decision): # <--- ЗАКОММЕНТИРОВАНО
            open_result = trade_manager.open_market_position(session, decision, symbol)
    
            if open_result.get('status') == 'SUCCESS':
                performance_monitor.log_trade_opened(symbol, decision, vsa_confirmed=False) # ИЗМЕНЕНО: vsa_confirmed=False
        # Логируем с полной информацией
                notification_system.send_trade_alert(symbol, "OPEN", open_result['price'], reason=f"MODEL_DECISION_{decision}") # ИЗМЕНЕНО: Причина
                trade_logger.log_enhanced_trade_with_quality_metrics(symbol, 'OPEN', open_result, None, 0,
                                 decision_maker, features_df.iloc[-1], f"MODEL_DECISION_{decision}") # ИЗМЕНЕНО: Причина
        
        # Сохраняем позицию
                active_positions = load_active_positions()
                active_positions[symbol] = {
                    'side': decision,
                    'entry_price': open_result['price'],
                    'quantity': open_result['quantity'],
                    'timestamp': time.time(),
                    'duration': 0,
            # 'vsa_entry_strength': features_df.iloc[-1]['vsa_strength']  # <--- УДАЛЕНО: VSA сила входа
                }
                save_active_positions(active_positions)
        
                opened_trades_counter += 1
                print(f"✅ Сделка #{opened_trades_counter}/{OPEN_TRADE_LIMIT} открыта на основе решения модели.") # ИЗМЕНЕНО: Сообщение
        
                if opened_trades_counter >= OPEN_TRADE_LIMIT:
                    print("!!! ДОСТИГНУТ ЛИМИТ ОТКРЫТЫХ СДЕЛОК !!!")
                    set_trader_status('MANAGING_ONLY')
    # else: # <--- УДАЛЕНО: Блок else для VSA подтверждения
    #     print(f"❌ VSA не подтверждает решение {decision} для {symbol}")

    except Exception as e:
        print(f"!!! КРИТИЧЕСКАЯ ОШИБКА при обработке сигнала для {symbol}: {e} !!!")



def run_trading_loop():
    """Главный торговый цикл с новой архитектурой"""
    global performance_monitor, notification_system
    print("=== ЗАПУСК НОВОГО ТРЕЙДИНГ-БОТА: xLSTM + VSA + RL ===")
    
    performance_monitor = PerformanceMonitor()
    notification_system = NotificationSystem()

    # Очистка файла с активными позициями при старте
    if os.path.exists(ACTIVE_POSITIONS_FILE):
        os.remove(ACTIVE_POSITIONS_FILE)
        print(f"Файл {ACTIVE_POSITIONS_FILE} очищен для новой сессии.")

    # Подключение к бирже
    session = HTTP(testnet=True, api_key=config.BYBIT_API_KEY, api_secret=config.BYBIT_API_SECRET)
    session.endpoint = config.API_URL
    
    # === ИНИЦИАЛИЗАЦИЯ НОВОЙ ГИБРИДНОЙ СИСТЕМЫ ===
    try:
        decision_maker = HybridDecisionMaker(
            xlstm_model_path='models/xlstm_rl_model.keras',
            rl_agent_path='models/rl_agent_BTCUSDT',  # Используем лучшего агента
            feature_columns=FEATURE_COLUMNS,
            sequence_length=config.SEQUENCE_LENGTH
        )
        print("✅ Гибридная система xLSTM + VSA + RL успешно загружена!")
        
        try:
            decision_maker.regime_detector.load_detector('models/market_regime_detector.pkl')
            print("✅ Детектор режимов загружен")
        except:
            print("⚠️ Детектор режимов не найден, будет обучен заново")
        
    except Exception as e:
        print(f"❌ КРИТИЧЕСКАЯ ОШИБКА: Не удалось загрузить гибридную систему: {e}")
        print("Убедитесь, что модели обучены и файлы существуют:")
        print("- models/xlstm_rl_model.keras")
        print("- models/xlstm_rl_scaler.pkl")
        print("- models/rl_agent_BTCUSDT.zip")
        return

    # Главный торговый цикл
    loop_counter = 0
    while True:
        status = get_trader_status()
        if status == 'STOP':
            print("Трейдер остановлен. Выход.")
            break
        
        try:
            # Управление активными позициями
            manage_active_positions(session, decision_maker)
            
            # Обработка новых сигналов
            if get_trader_status() == 'BUSY':
                print("Получен сигнал 'BUSY'.")
                with open(HOTLIST_FILE, 'r') as f: 
                    symbol = f.read().strip()
                
                if symbol: 
                    process_new_signal(session, symbol, decision_maker)
                
                if opened_trades_counter < OPEN_TRADE_LIMIT:
                    set_trader_status('DONE')
                else:
                    set_trader_status('MANAGING_ONLY')
            
            # Каждые 10 циклов показываем статистику
            loop_counter += 1
            if loop_counter % 10 == 0:
                print(f"\n=== СТАТИСТИКА (Цикл {loop_counter}) ===")
                print(f"Открыто сделок: {opened_trades_counter}/{OPEN_TRADE_LIMIT}")
                print(f"Активных позиций: {len(load_active_positions())}")
                print(f"Статус: {get_trader_status()}")
                
                # Показываем последнее объяснение решения
                if hasattr(decision_maker, 'get_decision_explanation'):
                    explanation = decision_maker.get_decision_explanation()
                    print(f"Последнее решение:\n{explanation}")

        except Exception as e:
            print(f"Произошла ошибка в главном цикле: {e}")
            import traceback
            traceback.print_exc()

        if loop_counter % 100 == 0:
            system_stats = monitor_system_resources()
            if system_stats['memory'] > 85:
                notification_system.send_system_alert(f"Критическое использование памяти: {system_stats['memory']:.1f}%")
        time.sleep(LOOP_SLEEP_SECONDS)

# === ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ (остаются без изменений) ===
def get_trader_status():
    try:
        with open(TRADER_STATUS_FILE, 'r') as f: 
            return f.read().strip()
    except FileNotFoundError: 
        return 'DONE'

def set_trader_status(status):
    with open(TRADER_STATUS_FILE, 'w') as f: 
        f.write(status)

def load_active_positions():
    if not os.path.exists(ACTIVE_POSITIONS_FILE): 
        return {}
    try:
        with open(ACTIVE_POSITIONS_FILE, 'r') as f: 
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError): 
        return {}

def save_active_positions(positions):
    with open(ACTIVE_POSITIONS_FILE, 'w') as f: 
        json.dump(positions, f, indent=4)

def monitor_system_resources():
    """Мониторинг системных ресурсов"""
    memory_percent = psutil.virtual_memory().percent
    cpu_percent = psutil.cpu_percent()
    
    if memory_percent > 80:
        print(f"⚠️ Высокое использование памяти: {memory_percent:.1f}%")
        gc.collect()  # Принудительная сборка мусора
        
    if cpu_percent > 90:
        print(f"⚠️ Высокая загрузка CPU: {cpu_percent:.1f}%")
        
    return {'memory': memory_percent, 'cpu': cpu_percent}

if __name__ == '__main__':
    run_trading_loop()

def calculate_dynamic_stops(features_row, position_side, entry_price):
    """
    Вычисляет динамические стоп-лоссы на основе волатильности (с AO_5_34)
    """
    base_sl = STOP_LOSS_PCT
    base_tp = TAKE_PROFIT_PCT
    
    # Корректировка на основе моментума (AO_5_34)
    ao_value = features_row.get('AO_5_34', 0)
    close_price = features_row.get('close', entry_price)
    
    if close_price > 0:
        # Используем абсолютное значение AO для оценки моментума
        ao_abs_pct = (abs(ao_value) / close_price) * 100
    else:
        ao_abs_pct = 0

    # Если AO большой (сильный моментум), делаем стопы шире
    if ao_abs_pct > 0.1: # Порог для AO_abs_pct нужно будет подобрать
        dynamic_sl = base_sl * (1 + ao_abs_pct * 5) # Увеличиваем SL сильнее
        dynamic_tp = base_tp * (1 + ao_abs_pct * 2) # Увеличиваем TP (или уменьшаем, если AO означает перекупленность)
    else:
        dynamic_sl, dynamic_tp = base_sl, base_tp
        
    # Ограничиваем максимальные и минимальные значения
    dynamic_sl = max(dynamic_sl, -3.0)
    dynamic_tp = min(dynamic_tp, 3.0)

    return dynamic_sl, dynamic_tp
