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

# === НОВЫЕ КОЛОНКИ ПРИЗНАКОВ С VSA ===
FEATURE_COLUMNS = [
    # Технические индикаторы
    'RSI_14', 'MACD_12_26_9', 'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0', 
    'ADX_14', 'STOCHk_14_3_3', 'STOCHd_14_3_3',
    # Паттерны
    'CDLHAMMER', 'CDLENGULFING', 'CDLDOJI', 'CDLSHOOTINGSTAR', 
    'CDLHANGINGMAN', 'CDLMARUBOZU',
    # VSA признаки
    'vsa_no_demand', 'vsa_no_supply', 'vsa_stopping_volume', 
    'vsa_climactic_volume', 'vsa_test', 'vsa_effort_vs_result', 'vsa_strength',
    # Дополнительные рыночные данные
    'volume_ratio', 'spread_ratio', 'close_position'
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
            features_df = feature_engineering.detect_candlestick_patterns(features_df)
            features_df = feature_engineering.calculate_vsa_features(features_df)  # Добавляем VSA!
            
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
            
            # 3. VSA сигнал на закрытие (новая логика!)
            elif should_close_by_vsa(features_df.iloc[-1], pos['side']):
                should_close = True
                close_reason = "VSA_SIGNAL"
            
            if should_close:
                print(f"!!! {symbol}: {close_reason}. Закрываю позицию... !!!")
                
                close_result = trade_manager.close_market_position(session, symbol, pos['quantity'], pos['side'])
                if close_result.get('status') == 'SUCCESS':
                    # Логируем с объяснением решения
                    log_enhanced_trade(symbol, 'CLOSE', close_result, pos, pnl_pct,
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

def should_close_by_vsa(row, position_side):
    """Определяет, нужно ли закрывать позицию на основе VSA сигналов"""
    
    if position_side == 'BUY':
        # Закрываем лонг при медвежьих VSA сигналах
        return (
            row['vsa_no_demand'] == 1 or 
            row['vsa_climactic_volume'] == 1 or 
            (row['vsa_strength'] < -2 and row['volume_ratio'] > 1.5)
        )
    
    elif position_side == 'SELL':
        # Закрываем шорт при бычьих VSA сигналах
        return (
            row['vsa_no_supply'] == 1 or 
            row['vsa_stopping_volume'] == 1 or 
            (row['vsa_strength'] > 2 and row['volume_ratio'] > 1.5)
        )
    
    return False

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
        features_df = feature_engineering.detect_candlestick_patterns(features_df)
        features_df = feature_engineering.calculate_vsa_features(features_df)
        
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
            # Дополнительная проверка VSA подтверждения
            if validate_decision_with_vsa(features_df.iloc[-1], decision):
                open_result = trade_manager.open_market_position(session, decision, symbol)
                
                if open_result.get('status') == 'SUCCESS':
                    performance_monitor.log_trade_opened(symbol, decision, vsa_confirmed=True)
                    # Логируем с полной информацией
                    notification_system.send_trade_alert(symbol, "OPEN", open_result['price'], reason=f"VSA_CONFIRMED_{decision}")
                    log_enhanced_trade(symbol, 'OPEN', open_result, None, 0,
                                     decision_maker, features_df.iloc[-1], f"VSA_CONFIRMED_{decision}")
                    
                    # Сохраняем позицию
                    active_positions = load_active_positions()
                    active_positions[symbol] = {
                        'side': decision,
                        'entry_price': open_result['price'],
                        'quantity': open_result['quantity'],
                        'timestamp': time.time(),
                        'duration': 0,
                        'vsa_entry_strength': features_df.iloc[-1]['vsa_strength']  # Сохраняем VSA силу входа
                    }
                    save_active_positions(active_positions)
                    
                    opened_trades_counter += 1
                    print(f"✅ Сделка #{opened_trades_counter}/{OPEN_TRADE_LIMIT} открыта с VSA подтверждением.")
                    
                    if opened_trades_counter >= OPEN_TRADE_LIMIT:
                        print("!!! ДОСТИГНУТ ЛИМИТ ОТКРЫТЫХ СДЕЛОК !!!")
                        set_trader_status('MANAGING_ONLY')
            else:
                print(f"❌ VSA не подтверждает решение {decision} для {symbol}")

    except Exception as e:
        print(f"!!! КРИТИЧЕСКАЯ ОШИБКА при обработке сигнала для {symbol}: {e} !!!")

def validate_decision_with_vsa(row, decision):
    """Валидирует торговое решение с помощью VSA анализа"""
    
    if decision == 'BUY':
        # Подтверждение покупки через VSA
        vsa_confirmation = (
            row['vsa_no_supply'] == 1 or  # Нет предложения
            row['vsa_stopping_volume'] == 1 or  # Остановочный объем
            (row['vsa_strength'] > 1 and row['volume_ratio'] > 1.2)  # Общая сила + объем
        )
        
        # Анти-подтверждение (не покупаем)
        vsa_contradiction = (
            row['vsa_no_demand'] == 1 or  # Нет спроса
            row['vsa_climactic_volume'] == 1 or  # Кульминационный объем
            row['vsa_strength'] < -2  # Сильная медвежья сила
        )
        
        return vsa_confirmation and not vsa_contradiction
    
    elif decision == 'SELL':
        # Подтверждение продажи через VSA
        vsa_confirmation = (
            row['vsa_no_demand'] == 1 or  # Нет спроса
            row['vsa_climactic_volume'] == 1 or  # Кульминационный объем
            (row['vsa_strength'] < -1 and row['volume_ratio'] > 1.2)  # Медвежья сила + объем
        )
        
        # Анти-подтверждение (не продаем)
        vsa_contradiction = (
            row['vsa_no_supply'] == 1 or  # Нет предложения
            row['vsa_stopping_volume'] == 1 or  # Остановочный объем
            row['vsa_strength'] > 2  # Сильная бычья сила
        )
        
        return vsa_confirmation and not vsa_contradiction
    
    return False

def log_enhanced_trade(symbol, action, trade_result, position, pnl, decision_maker, features_row, reason):
    """Расширенное логирование сделок с VSA и RL информацией"""
    
    log_data = {
        'symbol': symbol,
        'action': action,
        'reason': reason,
        'order_type': trade_result.get('price', 'N/A'),
        'price': trade_result.get('price', 'N/A'),
        'quantity': trade_result.get('quantity', 'N/A'),
        'usdt_amount': float(trade_result.get('price', 0)) * float(trade_result.get('quantity', 0)),
        'bybit_order_id': trade_result.get('bybit_order_id', 'N/A'),
        'status': trade_result.get('status', 'UNKNOWN'),
        'pnl': pnl if pnl else 0,
        
        # VSA информация
        'vsa_no_demand': features_row.get('vsa_no_demand', 0),
        'vsa_no_supply': features_row.get('vsa_no_supply', 0),
        'vsa_stopping_volume': features_row.get('vsa_stopping_volume', 0),
        'vsa_climactic_volume': features_row.get('vsa_climactic_volume', 0),
        'vsa_test': features_row.get('vsa_test', 0),
        'vsa_effort_vs_result': features_row.get('vsa_effort_vs_result', 0),
        'vsa_strength': features_row.get('vsa_strength', 0),
        'volume_ratio': features_row.get('volume_ratio', 0),
        'spread_ratio': features_row.get('spread_ratio', 0),
        
        # Технические индикаторы
        'RSI_14': features_row.get('RSI_14', 0),
        'MACD_12_26_9': features_row.get('MACD_12_26_9', 0),
        'ADX_14': features_row.get('ADX_14', 0),
    }
    
    # Добавляем информацию о решении, если доступна
    if hasattr(decision_maker, 'decision_history') and decision_maker.decision_history:
        last_decision = decision_maker.decision_history[-1]
        log_data.update({
            'xlstm_buy_prob': last_decision['xlstm_prediction'][0],
            'xlstm_sell_prob': last_decision['xlstm_prediction'][1], 
            'xlstm_hold_prob': last_decision['xlstm_prediction'][2],
            'xlstm_confidence': last_decision['xlstm_confidence'],
            'rl_decision': last_decision['rl_decision'],
            'vsa_bullish_strength': last_decision['vsa_signals']['bullish_strength'],
            'vsa_bearish_strength': last_decision['vsa_signals']['bearish_strength'],
        })
    
    trade_logger.log_trade(log_data)

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
    Вычисляет динамические стоп-лоссы на основе VSA и волатильности
    """
    base_sl = STOP_LOSS_PCT  # -1.0%
    base_tp = TAKE_PROFIT_PCT  # 1.5%
    
    # Корректировка на основе VSA силы
    vsa_strength = features_row.get('vsa_strength', 0)
    volume_ratio = features_row.get('volume_ratio', 1)
    
    if position_side == 'BUY':
        # Для лонгов: сильные бычьи VSA = более широкие стопы (больше веры в движение)
        if vsa_strength > 2 and volume_ratio > 1.5:
            dynamic_sl = base_sl * 0.7  # Уменьшаем SL до -0.7%
            dynamic_tp = base_tp * 1.3  # Увеличиваем TP до 1.95%
        elif vsa_strength < -1:  # Слабые сигналы = тайтовые стопы
            dynamic_sl = base_sl * 1.5  # Увеличиваем SL до -1.5%
            dynamic_tp = base_tp * 0.8  # Уменьшаем TP до 1.2%
        else:
            dynamic_sl, dynamic_tp = base_sl, base_tp
            
    else:  # SELL
        if vsa_strength < -2 and volume_ratio > 1.5:
            dynamic_sl = base_sl * 0.7  # Более широкие стопы для сильных медвежьих сигналов
            dynamic_tp = base_tp * 1.3
        elif vsa_strength > 1:
            dynamic_sl = base_sl * 1.5  # Тайтовые стопы при слабых сигналах
            dynamic_tp = base_tp * 0.8
        else:
            dynamic_sl, dynamic_tp = base_sl, base_tp
    
    return dynamic_sl, dynamic_tp
