import time
import os
import json
import pandas as pd
import numpy as np
from pybit.unified_trading import HTTP

# --- Импорт наших модулей ---
import config
import feature_engineering
import trade_manager
import trade_logger
from trade_manager import ConsensusDecisionMaker

# --- Константы ---
TRADER_STATUS_FILE = 'trader_status.txt'
ACTIVE_POSITIONS_FILE = 'active_positions.json'
LIVE_DATA_FILE = 'live_data.json'
HOTLIST_FILE = 'hotlist.txt'
LOG_FILE = 'trade_log.csv'
LOOP_SLEEP_SECONDS = 3
OPEN_TRADE_LIMIT = 1000 # Лимит на открытие новых сделок
TAKE_PROFIT_PCT = 1.2
STOP_LOSS_PCT = -1.2
CONSENSUS_CONFIDENCE_THRESHOLD = 0.5 # Минимальная уверенность для открытия/закрытия сделки

PATTERN_COLUMNS = [
    'CDLHAMMER', 'CDLENGULFING', 'CDLDOJI', 'CDLSHOOTINGSTAR',
    'CDLHANGINGMAN', 'CDLMARUBOZU',  # ✅ Заменено CDL3BLACKCROWS
    'hammer_f', 'hangingman_f', 'engulfing_f', 'doji_f',
    'shootingstar_f', 'marubozu_f',  # ✅ Заменено 3blackcrows_f
    # ДОБАВИТЬ:
    'hammer_f_on_support', 'hammer_f_vol_spike',
    'hangingman_f_on_res', 'hangingman_f_vol_spike',
    'engulfing_f_strong', 'engulfing_f_vol_confirm',
    'doji_f_high_vol', 'doji_f_high_atr',
    'shootingstar_f_on_res',
    'marubozu_f_strong_body', 'marubozu_f_vol_confirm', 'marubozu_f_bullish'  # ✅ Заменено 3blackcrows_f_vol_up
]
INDICATOR_COLUMNS = ['RSI_14', 'MACD_12_26_9', 'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0', 'ADX_14', 'STOCHk_14_3_3', 'STOCHd_14_3_3']
SEQUENCE_LENGTH = 10

# --- Глобальные переменные ---
opened_trades_counter = 0

# --- Функции управления состоянием ---
def get_trader_status():
    try:
        with open(TRADER_STATUS_FILE, 'r') as f: return f.read().strip()
    except FileNotFoundError: return 'DONE'

def set_trader_status(status):
    with open(TRADER_STATUS_FILE, 'w') as f: f.write(status)

def load_active_positions():
    if not os.path.exists(ACTIVE_POSITIONS_FILE): return {}
    try:
        with open(ACTIVE_POSITIONS_FILE, 'r') as f: return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError): return {}

def save_active_positions(positions):
    with open(ACTIVE_POSITIONS_FILE, 'w') as f: json.dump(positions, f, indent=4)

# --- Основные функции ---

def manage_active_positions(session, decision_maker):
    active_positions = load_active_positions()
    if not active_positions: return

    print(f"Открыто сделок: {opened_trades_counter}/{OPEN_TRADE_LIMIT}. Активных позиций: {len(active_positions)}")
    
    symbols_to_remove = []
    
    # Ограничиваем отображение в логах только первыми 5 позициями
    positions_items = list(active_positions.items())
    displayed_positions = positions_items[:5]  # Показываем только первые 5
    remaining_count = len(positions_items) - 5 if len(positions_items) > 5 else 0
    
    # Отображаем информацию о скрытых позициях
    if remaining_count > 0:
        print(f"  ... и еще {remaining_count} позиций (скрыто для чистоты логов)")

    # Обрабатываем ВСЕ позиции, но показываем детали только для первых 5
    for i, (symbol, pos) in enumerate(positions_items):
        try:
            kline_list = trade_manager.fetch_initial_data(session, symbol)
            if not kline_list: continue
            
            kline_df = pd.DataFrame(kline_list, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
            features_df = feature_engineering.calculate_features(kline_df.copy())
            features_df = feature_engineering.detect_candlestick_patterns(features_df)
            if features_df.empty or len(features_df) < SEQUENCE_LENGTH: continue

            # Prepare features for decision making
            pattern_features = features_df[PATTERN_COLUMNS].values
            indicator_features = features_df[INDICATOR_COLUMNS].values
            
            # Take the last `SEQUENCE_LENGTH` data points
            pattern_data = pattern_features[-SEQUENCE_LENGTH:]
            indicator_data = indicator_features[-SEQUENCE_LENGTH:]

            # Reshape for the models
            pattern_features_reshaped = pattern_data.reshape(1, SEQUENCE_LENGTH, len(PATTERN_COLUMNS))
            indicator_features_reshaped = indicator_data.reshape(1, SEQUENCE_LENGTH, len(INDICATOR_COLUMNS))

            decision = decision_maker.get_decision(pattern_features_reshaped, indicator_features_reshaped, mode='Consensus', confidence_threshold=CONSENSUS_CONFIDENCE_THRESHOLD)

            latest_price = float(features_df.iloc[-1]['close'])
            entry_price = float(pos['entry_price'])
            
            if pos['side'] == 'BUY':
                pnl_pct = ((latest_price - entry_price) / entry_price) * 100
            else: # SELL
                pnl_pct = ((entry_price - latest_price) / entry_price) * 100

            # Показываем детали только для первых 5 позиций
            if i < 5:
                print(f"  - {symbol}: PnL {pnl_pct:.2f}% | Вход: {entry_price} | Сейчас: {latest_price} | Решение: {decision}")

            # Check for stop-loss or take-profit
            if pnl_pct >= TAKE_PROFIT_PCT or pnl_pct <= STOP_LOSS_PCT:
                # Для важных событий (закрытие по TP/SL) всегда показываем
                print(f"!!! {symbol}: Сработал TP/SL ({pnl_pct:.2f}%). Закрываю позицию... !!!")
                close_result = trade_manager.close_market_position(session, symbol, pos['quantity'], pos['side'])
                if close_result.get('status') == 'SUCCESS':
                    log_data = {
                        'symbol': symbol,
                        'decision': 'CLOSE_RULE',
                        'order_type': 'SELL' if pos['side'] == 'BUY' else 'BUY',
                        'price': close_result.get('price'),
                        'quantity': pos['quantity'],
                        'usdt_amount': float(close_result.get('price', 0)) * float(pos['quantity']),
                        'bybit_order_id': close_result.get('bybit_order_id'),
                        'status': 'SUCCESS',
                        'pnl': pnl_pct,
                        'xlstm_pattern_decision': decision_maker.xlstm_pattern_decision,
                        'xlstm_pattern_confidence': decision_maker.xlstm_pattern_confidence,
                        'xlstm_indicator_decision': decision_maker.xlstm_indicator_decision,
                        'xlstm_indicator_confidence': decision_maker.xlstm_indicator_confidence,
                        'consensus_decision': decision,
                        'consensus_confidence': decision_maker.consensus_confidence
                    }
                    latest_features = features_df.iloc[-1]
                    log_data.update(latest_features[['RSI_14', 'MACD_12_26_9', 'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0', 'ADX_14', 'STOCHk_14_3_3', 'STOCHd_14_3_3']].to_dict())
                    for col in PATTERN_COLUMNS:  # теперь включает детальные признаки
                        log_data[col] = latest_features.get(col, 0)
                    trade_logger.log_trade(log_data)
                    symbols_to_remove.append(symbol)
                continue

            if (pos['side'] == 'BUY' and decision == 'SELL') or (pos['side'] == 'SELL' and decision == 'BUY'):
                # Для важных событий (закрытие по сигналу модели) всегда показываем
                print(f"!!! {symbol}: Модель решила закрыть позицию с PnL {pnl_pct:.2f}%. Закрываю... !!!")
                close_result = trade_manager.close_market_position(session, symbol, pos['quantity'], pos['side'])
                if close_result.get('status') == 'SUCCESS':
                    log_data = {
                        'symbol': symbol,
                        'decision': 'CLOSE',
                        'order_type': 'SELL' if pos['side'] == 'BUY' else 'BUY',
                        'price': close_result.get('price'),
                        'quantity': pos['quantity'],
                        'usdt_amount': float(close_result.get('price', 0)) * float(pos['quantity']),
                        'bybit_order_id': close_result.get('bybit_order_id'),
                        'status': 'SUCCESS',
                        'pnl': pnl_pct,
                        'xlstm_pattern_decision': decision_maker.xlstm_pattern_decision,
                        'xlstm_pattern_confidence': decision_maker.xlstm_pattern_confidence,
                        'xlstm_indicator_decision': decision_maker.xlstm_indicator_decision,
                        'xlstm_indicator_confidence': decision_maker.xlstm_indicator_confidence,
                        'consensus_decision': decision,
                        'consensus_confidence': decision_maker.consensus_confidence
                    }
                    latest_features = features_df.iloc[-1]
                    log_data.update(latest_features[INDICATOR_COLUMNS].to_dict())
                    for col in PATTERN_COLUMNS:  # теперь включает детальные признаки
                        log_data[col] = latest_features.get(col, 0)
                    trade_logger.log_trade(log_data)
                    symbols_to_remove.append(symbol)

        except Exception as e:
            print(f"Ошибка при управлении позицией {symbol}: {e}")

    if symbols_to_remove:
        current_positions = load_active_positions()
        for symbol in symbols_to_remove:
            if symbol in current_positions: del current_positions[symbol]
        save_active_positions(current_positions)

def process_new_signal(session, symbol, decision_maker):
    global opened_trades_counter
    if opened_trades_counter >= OPEN_TRADE_LIMIT: return
    
    active_positions = load_active_positions()
    if symbol in active_positions: return

    print(f"--- Обработка нового сигнала для {symbol} ---")
    try:
        with open(LIVE_DATA_FILE, 'r') as f: live_data = json.load(f)
        symbol_data = live_data.get(symbol)
        if not symbol_data: return

        kline_list = symbol_data.get('klines')
        if not kline_list or len(kline_list) < config.REQUIRED_CANDLES: return

        kline_df = pd.DataFrame(kline_list, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
        features_df = feature_engineering.calculate_features(kline_df.copy())
        features_df = feature_engineering.detect_candlestick_patterns(features_df)
        if features_df.empty or len(features_df) < SEQUENCE_LENGTH: return

        # Prepare features for decision making
        pattern_features = features_df[PATTERN_COLUMNS].values
        indicator_features = features_df[INDICATOR_COLUMNS].values
        
        # Take the last `SEQUENCE_LENGTH` data points
        pattern_data = pattern_features[-SEQUENCE_LENGTH:]
        indicator_data = indicator_features[-SEQUENCE_LENGTH:]

        # Reshape for the models
        pattern_features_reshaped = pattern_data.reshape(1, SEQUENCE_LENGTH, len(PATTERN_COLUMNS))
        indicator_features_reshaped = indicator_data.reshape(1, SEQUENCE_LENGTH, len(INDICATOR_COLUMNS))

        decision = decision_maker.get_decision(pattern_features_reshaped, indicator_features_reshaped, mode='Consensus', confidence_threshold=CONSENSUS_CONFIDENCE_THRESHOLD)
        
        print(f"--- {symbol} | Решение: {decision} ---")

        if decision in ['BUY', 'SELL']:
            open_result = trade_manager.open_market_position(session, decision, symbol)
            if open_result.get('status') == 'SUCCESS':
                log_data = {
                    'symbol': symbol,
                    'decision': 'OPEN',
                    'order_type': decision,
                    'price': open_result.get('price'),
                    'quantity': open_result.get('quantity'),
                    'usdt_amount': float(open_result.get('price', 0)) * float(open_result.get('quantity', 0)),
                    'bybit_order_id': open_result.get('bybit_order_id'),
                    'status': 'SUCCESS',
                    'xlstm_pattern_decision': decision_maker.xlstm_pattern_decision,
                    'xlstm_pattern_confidence': decision_maker.xlstm_pattern_confidence,
                    'xlstm_indicator_decision': decision_maker.xlstm_indicator_decision,
                    'xlstm_indicator_confidence': decision_maker.xlstm_indicator_confidence,
                    'consensus_decision': decision,
                    'consensus_confidence': decision_maker.consensus_confidence
                }
                latest_features = features_df.iloc[-1]
                log_data.update(latest_features[INDICATOR_COLUMNS].to_dict())
                for col in PATTERN_COLUMNS:  # теперь включает детальные признаки
                    log_data[col] = latest_features.get(col, 0)
                trade_logger.log_trade(log_data)
                active_positions = load_active_positions()
                active_positions[symbol] = {
                    'side': decision,
                    'entry_price': open_result['price'],
                    'quantity': open_result['quantity'],
                    'timestamp': time.time(),
                    'duration': 0
                }
                save_active_positions(active_positions)
                opened_trades_counter += 1
                print(f"Сделка #{opened_trades_counter}/{OPEN_TRADE_LIMIT} открыта.")
                if opened_trades_counter >= OPEN_TRADE_LIMIT:
                    print("!!! ДОСТИГНУТ ЛИМИТ ОТКРЫТЫХ СДЕЛОК. НОВЫЕ СДЕЛКИ ОТКРЫВАТЬСЯ НЕ БУДУТ. !!!")
                    print("Бот продолжит управлять открытыми позициями.")
                    set_trader_status('MANAGING_ONLY')

    except Exception as e:
        print(f"!!! КРИТИЧЕСКАЯ ОШИБКА при обработке сигнала для {symbol}: {e} !!!")


def run_trading_loop():
    print("--- Запуск торгового бота ---")
    
    # Очистка файла с активными позициями при старте
    if os.path.exists(ACTIVE_POSITIONS_FILE):
        os.remove(ACTIVE_POSITIONS_FILE)
        print(f"Файл {ACTIVE_POSITIONS_FILE} очищен для новой сессии.")

    session = HTTP(testnet=True, api_key=config.BYBIT_API_KEY, api_secret=config.BYBIT_API_SECRET)
    session.endpoint = config.API_URL
    
    decision_maker = ConsensusDecisionMaker(
        xlstm_pattern_model_path='models/xlstm_pattern_model.keras',
        xlstm_indicator_model_path='models/xlstm_indicator_model.keras',
        xlstm_pattern_scaler_path='models/xlstm_pattern_scaler.pkl',
        xlstm_indicator_scaler_path='models/xlstm_indicator_scaler.pkl',
        sequence_length=SEQUENCE_LENGTH,
        pattern_feature_count=len(PATTERN_COLUMNS),
        indicator_feature_count=len(INDICATOR_COLUMNS)
    )

    while True:
        status = get_trader_status()
        if status == 'STOP':
            print("Трейдер остановлен. Выход.")
            break
        
        try:
            manage_active_positions(session, decision_maker)
            
            if get_trader_status() == 'BUSY':
                print("Получен сигнал 'BUSY'.")
                with open(HOTLIST_FILE, 'r') as f: symbol = f.read().strip()
                if symbol: process_new_signal(session, symbol, decision_maker)
                
                if opened_trades_counter < OPEN_TRADE_LIMIT:
                    set_trader_status('DONE')
                else:
                    set_trader_status('MANAGING_ONLY')

        except Exception as e:
            print(f"Произошла ошибка в основном цикле: {e}")

        time.sleep(LOOP_SLEEP_SECONDS)

if __name__ == '__main__':
    run_trading_loop()
