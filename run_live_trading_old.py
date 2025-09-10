import time
import os
import json
import pickle
import pandas as pd
import numpy as np
from stable_baselines3 import DQN
from pybit.unified_trading import HTTP

# --- Импорт наших модулей ---
import config
import feature_engineering
import trade_manager
import trade_logger
from trading_env import TradingEnv
from xgboost_model import XGBoostModel
from kalman_filter import KalmanFilter
from lstm_model import LSTMModel
from gpr_model import GPRModel

# --- Константы ---
TRADER_STATUS_FILE = 'trader_status.txt'
ACTIVE_POSITIONS_FILE = 'active_positions.json'
LIVE_DATA_FILE = 'live_data.json'
HOTLIST_FILE = 'hotlist.txt'
MODEL_PATH = "dqn_trading_model.zip"
LOG_FILE = 'trade_log.csv'
LOOP_SLEEP_SECONDS = 3
OPEN_TRADE_LIMIT = 1000 # Лимит на открытие новых сделок

# --- Правила управления позицией ---
TAKE_PROFIT_PCT = 0.40
STOP_LOSS_PCT = -0.40

# --- Глобальные переменные ---
opened_trades_counter = 0
xgboost_model = XGBoostModel()
kalman_filter = KalmanFilter()
# Загрузка обученной модели LSTM из файла
LSTM_MODEL_FILE = 'lstm_model.keras'
if os.path.exists(LSTM_MODEL_FILE):
    # Загружаем модель с помощью метода из LSTMModel
    try:
        lstm_model = LSTMModel()
        lstm_model.load_model(LSTM_MODEL_FILE)
        print(f"--- Модель LSTM и скейлер успешно загружены из '{LSTM_MODEL_FILE}'. ---")
        print(f"DEBUG: lstm_model.is_trained = {getattr(lstm_model, 'is_trained', 'Attribute not found')}")
    except Exception as e:
        print(f"!!! ОШИБКА при загрузке модели LSTM из файла '{LSTM_MODEL_FILE}': {e} !!!")
        import traceback
        traceback.print_exc()
        print("--- Создана новая (пустая) модель LSTM, так как загрузка не удалась. ---")
        lstm_model = LSTMModel()
        print(f"DEBUG: New lstm_model.is_trained = {getattr(lstm_model, 'is_trained', 'Attribute not found')}")
else:
    print(f"Файл модели LSTM {LSTM_MODEL_FILE} не найден. Создана новая модель.")
    lstm_model = LSTMModel()
    print(f"DEBUG: New lstm_model.is_trained = {getattr(lstm_model, 'is_trained', 'Attribute not found')}")

# Загрузка обученной модели GPR из файла
GPR_MODEL_FILE = 'gpr_model.pkl'
if os.path.exists(GPR_MODEL_FILE):
    with open(GPR_MODEL_FILE, 'rb') as f:
        gpr_model = pickle.load(f)
    print(f"Модель GPR загружена из {GPR_MODEL_FILE}")
else:
    print(f"Файл модели GPR {GPR_MODEL_FILE} не найден. Создана новая модель.")
    gpr_model = GPRModel()
# Режим обучения: если True, HOLD преобразуется в случайный выбор между BUY и SELL
LEARNING_MODE = False

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

def process_with_kalman_filter(data_series):
    """
    Обработка данных через фильтр Калмана.
    
    Args:
        data_series: Временной ряд данных (pandas Series)
        
    Returns:
        dict: Словарь с результатами обработки
    """
    try:
        # Обработка данных через фильтр Калмана
        smoothed_data = kalman_filter.process_data(data_series)
        
        # Получение тренда
        trend = kalman_filter.get_trend()
        
        # Получение текущего состояния
        state = kalman_filter.get_state()
        
        return {
            'smoothed_data': smoothed_data,
            'trend': trend,
            'state': state,
            'kalman_price': smoothed_data.iloc[-1] if len(smoothed_data) > 0 else 0.0
        }
    except Exception as e:
        print(f"Ошибка при обработке данных через Kalman Filter: {e}")
        return {
            'smoothed_data': data_series,
            'trend': 0.0,
            'state': np.array([0.0, 0.0]),
            'kalman_price': data_series.iloc[-1] if len(data_series) > 0 else 0.0
        }

def process_with_lstm(data_series):
    """
    Обработка данных через LSTM модель.
    
    Args:
        data_series: Временной ряд данных (pandas Series)
        
    Returns:
        dict: Словарь с результатами обработки
    """
    try:
        # Для LSTM нам нужно подготовить данные в правильном формате
        if len(data_series) < 60:  # Минимальная длина последовательности
            pred_val = data_series.iloc[-1] if len(data_series) > 0 else 0.0
            return {
                'prediction': pred_val,
                'trend_direction': 'NEUTRAL',
                'signal_strength': 0.0,
                'confidence': 0.0
            }
        
        # Выявление паттернов
        patterns = lstm_model.detect_patterns(data_series)
        # Проверка на NaN в результатах
        for key, value in patterns.items():
            if isinstance(value, float) and np.isnan(value):
                print(f"!!! ВНИМАНИЕ: detect_patterns вернул NaN в поле '{key}'")
        
        # Проверка на NaN в результатах и возврат значений по умолчанию при необходимости
        if any(isinstance(v, float) and np.isnan(v) for v in patterns.values()):
            print("!!! ВНИМАНИЕ: LSTM вернул NaN. Возвращаем значения по умолчанию.")
            default_pred = data_series.iloc[-1] if len(data_series) > 0 else 0.0
            # Если и default_pred это nan, возвращаем 0.0
            if isinstance(default_pred, float) and np.isnan(default_pred):
                default_pred = 0.0
                print("!!! ВНИМАНИЕ: data_series.iloc[-1] это NaN. Возвращаем 0.0 как prediction.")
            return {
                'prediction': default_pred,
                'trend_direction': 'NEUTRAL',
                'signal_strength': 0.0,
                'confidence': 0.0
            }
        
        return patterns
    except Exception as e:
        print(f"!!! ОШИБКА при обработке данных через LSTM: {e}")
        print(f"!!! Тип исключения: {type(e)}")
        import traceback
        traceback.print_exc()
        default_pred = data_series.iloc[-1] if len(data_series) > 0 else 0.0
        # Если и default_pred это nan, возвращаем 0.0
        if isinstance(default_pred, float) and np.isnan(default_pred):
            default_pred = 0.0
            print("!!! ВНИМАНИЕ: data_series.iloc[-1] это NaN при обработке исключения. Возвращаем 0.0 как prediction.")
        print(f"!!! Возвращаем значения по умолчанию. default_pred: {default_pred}")
        return {
            'prediction': default_pred,
            'trend_direction': 'NEUTRAL',
            'signal_strength': 0.0,
            'confidence': 0.0
        }

def process_with_gpr(data_series):
    """
    Обработка данных через GPR модель.
    
    Args:
        data_series: Временной ряд данных (pandas Series)
        
    Returns:
        dict: Словарь с результатами обработки
    """
    try:
        # Подготовка данных для GPR (нужно как минимум 2 точки)
        if len(data_series) < 2:
            return {
                'prediction': data_series.iloc[-1] if len(data_series) > 0 else 0.0,
                'uncertainty': 0.0,
                'coefficient_of_variation': 0.0,
                'normalized_uncertainty': 0.0,
                'confidence': 1.0
            }
        
        # Преобразование данных для GPR
        X = np.arange(len(data_series)).reshape(-1, 1)
        y = data_series.values
        
        # Обучение модели (в реальном времени)
        gpr_model.train(X, y)
        
        # Оценка неопределенности для следующей точки
        uncertainty = gpr_model.estimate_uncertainty(np.array([[len(data_series)]]))
        
        return {
            'prediction': uncertainty['prediction'][0] if len(uncertainty['prediction']) > 0 else data_series.iloc[-1],
            'uncertainty': uncertainty['uncertainty'][0] if len(uncertainty['uncertainty']) > 0 else 0.0,
            'coefficient_of_variation': uncertainty['coefficient_of_variation'][0] if len(uncertainty['coefficient_of_variation']) > 0 else 0.0,
            'normalized_uncertainty': uncertainty['normalized_uncertainty'][0] if len(uncertainty['normalized_uncertainty']) > 0 else 0.0,
            'confidence': uncertainty['confidence'][0] if len(uncertainty['confidence']) > 0 else 1.0
        }
    except Exception as e:
        print(f"Ошибка при обработке данных через GPR: {e}")
        return {
            'prediction': data_series.iloc[-1] if len(data_series) > 0 else 0.0,
            'uncertainty': 0.0,
            'coefficient_of_variation': 0.0,
            'normalized_uncertainty': 0.0,
            'confidence': 1.0
        }

def manage_active_positions(session, model):
    active_positions = load_active_positions()
    if not active_positions: return

    print(f"Управление {len(active_positions)} активными позициями...")
    # Добавляем информационное сообщение, если позиций больше 10 (это же значение должно быь и внизу в if display_counter
    if len(active_positions) > 10:
        print(f"(Отображаются только первые 10 для краткости)")
        
    symbols_to_remove = []
    display_counter = 0

    for symbol, pos in active_positions.items():
        try:
            kline_list = trade_manager.fetch_initial_data(session, symbol)
            if not kline_list: continue
            
            kline_df = pd.DataFrame(kline_list, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
            # Преобразуем числовые колонки в правильные типы
            numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'turnover']
            for col in numeric_cols:
                kline_df[col] = pd.to_numeric(kline_df[col], errors='coerce')
            latest_price = float(kline_df.iloc[-1]['close'])
            # Безопасное преобразование entry_price
            entry_price = float(pd.to_numeric(pos['entry_price'], errors='coerce'))

            if pos['side'] == 'BUY':
                pnl_pct = ((latest_price - entry_price) / entry_price) * 100
            else: # SELL
                pnl_pct = ((entry_price - latest_price) / entry_price) * 100
            
            kline_df['timestamp'] = pd.to_datetime(kline_df['timestamp'], unit='ms')
            features_df = feature_engineering.calculate_features(kline_df)
            if features_df.empty: continue

            latest_features = features_df.iloc[-1]
            
            # Ограничиваем вывод в консоль
            if display_counter < 10:
                print(f"  - {symbol}: PnL {pnl_pct:.2f}% | Вход: {entry_price} | Сейчас: {latest_price}")

                # Получаем временной ряд цен для обработки через новые модели
                price_series = kline_df['close'].astype(float)
                
                # 1. Обработка данных через Kalman Filter
                kalman_result = process_with_kalman_filter(price_series)
                
                # 2. Обработка данных через LSTM
                lstm_result = process_with_lstm(price_series)
                
                # 3. Обработка данных через GPR
                gpr_result = process_with_gpr(price_series)
                
                # 4. Передаем результаты Kalman Filter в GPR (дополнительно)
                kalman_trend = kalman_result['trend']
                
                # 5. Передаем только технические индикаторы в XGBoost
                feature_names = ["open","high","low","close","volume","turnover","BBL_20_2.0","BBM_20_2.0","BBU_20_2.0","BBB_20_2.0","BBP_20_2.0","MACD_12_26_9","MACDh_12_26_9","MACDs_12_26_9","OBV","ATRr_14","WILLR_14","RSI_14","CCI_20_0.015","ADX_14","DMP_14","DMN_14"]
                
                # 6. Получаем предсказание от XGBoost
                xgboost_features = latest_features[feature_names].to_frame().T
                for col in xgboost_features.columns:
                    xgboost_features[col] = pd.to_numeric(xgboost_features[col], errors='coerce')
                xgboost_features.fillna(0, inplace=True)
                xgboost_prediction = xgboost_model.predict(xgboost_features)
                
                print(f"--- {symbol} | Уверенность XGBoost: {xgboost_prediction} ---")
                print(f"--- {symbol} | Результаты Kalman Filter: Цена={kalman_result['kalman_price']}, Тренд={kalman_result['trend']} ---")
                print(f"--- {symbol} | Результаты LSTM: Предсказание={lstm_result['prediction']}, Уверенность={lstm_result['confidence']} ---")
                print(f"--- {symbol} | Результаты GPR: Предсказание={gpr_result['prediction']}, Уверенность={gpr_result['confidence']} ---")

            display_counter += 1

            if pnl_pct >= TAKE_PROFIT_PCT or pnl_pct <= STOP_LOSS_PCT:
                print(f"!!! {symbol}: Сработал TP/SL ({pnl_pct:.2f}%). Закрываю позицию... !!!")
                close_result = trade_manager.close_market_position(session, symbol, pos['quantity'], pos['side'])
                # Проверяем, успешно ли закрыта позиция или это ошибка "позиция уже закрыта"
                if close_result.get('status') == 'SUCCESS' or '110017' in str(close_result.get('message', '')):
                    close_result['pnl'] = pnl_pct # Добавляем PnL в словарь для логгера
                    # Добавляем информацию о решении XGBoost в лог
                    log_data = latest_features.to_dict()
                    # Обеспечиваем числовой формат timestamp
                    log_data['timestamp'] = int(time.time() * 1000)
                    log_data['xgboost_prediction'] = xgboost_prediction[0]
                    # Добавляем результаты всех моделей в log_data
                    log_data['kalman_price'] = kalman_result['kalman_price']
                    log_data['kalman_trend'] = kalman_trend
                    log_data['gpr_prediction'] = gpr_result['prediction']
                    log_data['gpr_confidence'] = gpr_result['confidence']
                    log_data['lstm_prediction'] = lstm_result['prediction']
                    log_data['lstm_confidence'] = lstm_result['confidence']
                    # Добавляем метаданные для логирования
                    log_data['symbol'] = symbol
                    log_data['dqn_decision'] = f"CLOSE_RULE"
                    log_data['order_type'] = close_result.get('order_type', 'N/A')
                    log_data['price'] = close_result.get('price', 'N/A')
                    log_data['quantity'] = close_result.get('quantity', 'N/A')
                    log_data['usdt_amount'] = close_result.get('usdt_amount', 'N/A')
                    log_data['bybit_order_id'] = close_result.get('bybit_order_id', 'N/A')
                    log_data['status'] = close_result.get('status', 'N/A')
                    log_data['error_message'] = close_result.get('message', 'N/A')
                    log_data['pnl'] = close_result.get('pnl', 'N/A')
                    
                    # Нормализуем OBV и quantity в log_data для консистентности с данными для модели
                    if 'OBV' in log_data:
                        log_data['OBV'] = float(log_data['OBV']) / 100000.0
                    if 'quantity' in log_data:
                        pass # quantity не нормализуем

                    # Разделяем данные на order_details и indicators
                    order_details = {
                        'timestamp': log_data.get('timestamp'),
                        'symbol': log_data.get('symbol'),
                        'dqn_decision': f"CLOSE_RULE",
                        'order_type': close_result.get('order_type', 'N/A'),
                        'price': close_result.get('price', 'N/A'),
                        'quantity': close_result.get('quantity', 'N/A'),
                        'usdt_amount': close_result.get('usdt_amount', 'N/A'),
                        'bybit_order_id': close_result.get('bybit_order_id', 'N/A'),
                        'status': close_result.get('status', 'N/A'),
                        'message': close_result.get('message', 'N/A'),
                        'pnl': close_result.get('pnl', 'N/A')
                    }
                    
                    # Исключаем поля order_details из indicators
                    indicators = {k: v for k, v in log_data.items() if k not in order_details}
                    
                    trade_logger.log_event(symbol, f"CLOSE_RULE", order_details, indicators)
                    symbols_to_remove.append(symbol)
                continue
            
            position_code = 1 if pos['side'] == 'BUY' else -1
            # Формируем log_data для модели и логирования
            log_data = latest_features.to_dict()
            # Обеспечиваем числовой формат timestamp
            log_data['timestamp'] = int(time.time() * 1000)
            log_data['xgboost_prediction'] = xgboost_prediction[0]
            # Добавляем результаты всех моделей в log_data
            log_data['kalman_price'] = kalman_result['kalman_price']
            log_data['kalman_trend'] = kalman_trend
            log_data['gpr_prediction'] = gpr_result['prediction']
            log_data['gpr_confidence'] = gpr_result['confidence']
            log_data['lstm_prediction'] = lstm_result['prediction']
            log_data['lstm_confidence'] = lstm_result['confidence']
            
            # Формируем obs для модели (все признаки из feature_cols в train_model.py + новые признаки)
            # В новой архитектуре все модели передают свои данные напрямую в TensorFlow DQN
            FIELDNAMES = [
                'symbol', 'dqn_decision', 'order_type', 'price', 'quantity',
                'timestamp',
                'usdt_amount', 'bybit_order_id', 'status', 'error_message', 'pnl',
                # Технические индикаторы
                'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0', 'BBB_20_2.0', 'BBP_20_2.0',
                'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9', 'OBV', 'ATRr_14',
                'WILLR_14', 'RSI_14', 'CCI_20_0.015', 'ADX_14', 'DMP_14', 'DMN_14',
                # Результаты XGBoost
                'xgboost_prediction',
                # Результаты Kalman Filter
                'kalman_price', 'kalman_trend',
                # Результаты GPR
                'gpr_prediction', 'gpr_confidence',
                # Результаты LSTM
                'lstm_prediction', 'lstm_confidence'
            ]
            obs = np.array([log_data.get(key, 0.0) for key in FIELDNAMES], dtype=np.float32)

            action, _ = model.predict(obs, deterministic=True) ## deterministic=True - это значит режим детерминированного принятия решений трейдинг бота.
            decision = {0: "SELL", 1: "BUY", 2: "HOLD"}.get(int(action))

            if (pos['side'] == 'BUY' and decision == 'SELL') or (pos['side'] == 'SELL' and decision == 'BUY'):
                print(f"!!! {symbol}: Модель решила закрыть позицию с PnL {pnl_pct:.2f}%. Закрываю... !!!")
                close_result = trade_manager.close_market_position(session, symbol, pos['quantity'], pos['side'])
                # Проверяем, успешно ли закрыта позиция или это ошибка "позиция уже закрыта"
                if close_result.get('status') == 'SUCCESS' or '110017' in str(close_result.get('message', '')):
                    close_result['pnl'] = pnl_pct # Добавляем PnL в словарь для логгера
                    # Добавляем информацию о решении XGBoost в лог
                    log_data = latest_features.to_dict()
                    # Обеспечиваем числовой формат timestamp
                    log_data['timestamp'] = int(time.time() * 1000)
                    log_data['xgboost_prediction'] = xgboost_prediction[0]
                    # Добавляем результаты всех моделей в log_data
                    log_data['kalman_price'] = kalman_result['kalman_price']
                    log_data['kalman_trend'] = kalman_trend
                    log_data['gpr_prediction'] = gpr_result['prediction']
                    log_data['gpr_confidence'] = gpr_result['confidence']
                    log_data['lstm_prediction'] = lstm_result['prediction']
                    log_data['lstm_confidence'] = lstm_result['confidence']
                    # Добавляем метаданные для логирования
                    log_data['symbol'] = symbol
                    log_data['dqn_decision'] = f"CLOSE_AI"
                    log_data['order_type'] = close_result.get('order_type', 'N/A')
                    log_data['price'] = close_result.get('price', 'N/A')
                    log_data['quantity'] = close_result.get('quantity', 'N/A')
                    log_data['usdt_amount'] = close_result.get('usdt_amount', 'N/A')
                    log_data['bybit_order_id'] = close_result.get('bybit_order_id', 'N/A')
                    log_data['status'] = close_result.get('status', 'N/A')
                    log_data['error_message'] = close_result.get('message', 'N/A')
                    log_data['pnl'] = close_result.get('pnl', 'N/A')

                    # Нормализуем OBV и quantity в log_data для консистентности с данными для модели
                    if 'OBV' in log_data:
                        log_data['OBV'] = float(log_data['OBV']) / 100000.0
                    if 'quantity' in log_data:
                        pass # quantity не нормализуем

                    # Разделяем данные на order_details и indicators
                    order_details = {
                        'timestamp': log_data.get('timestamp'),
                        'symbol': log_data.get('symbol'),
                        'dqn_decision': f"CLOSE_AI",
                        'order_type': close_result.get('order_type', 'N/A'),
                        'price': close_result.get('price', 'N/A'),
                        'quantity': close_result.get('quantity', 'N/A'),
                        'usdt_amount': close_result.get('usdt_amount', 'N/A'),
                        'bybit_order_id': close_result.get('bybit_order_id', 'N/A'),
                        'status': close_result.get('status', 'N/A'),
                        'message': close_result.get('message', 'N/A'),
                        'pnl': close_result.get('pnl', 'N/A')
                    }
                    
                    # Исключаем поля order_details из indicators
                    indicators = {k: v for k, v in log_data.items() if k not in order_details}
                    
                    trade_logger.log_event(symbol, f"CLOSE_AI", order_details, indicators)
                    symbols_to_remove.append(symbol)

        except Exception as e:
            print(f"Ошибка при управлении позицией {symbol}: {e}")

    if symbols_to_remove:
        current_positions = load_active_positions()
        for symbol in symbols_to_remove:
            if symbol in current_positions: del current_positions[symbol]
        save_active_positions(current_positions)

def process_new_signal(session, symbol, model):
    global opened_trades_counter
    if opened_trades_counter >= OPEN_TRADE_LIMIT:
        return # Ничего не делаем, если лимит достигнут

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
        if features_df.empty: return
        
        latest_features = features_df.iloc[-1]
        
        # Получаем временной ряд цен для обработки через новые модели
        price_series = kline_df['close'].astype(float)
        
        # 1. Обработка данных через Kalman Filter
        kalman_result = process_with_kalman_filter(price_series)
        
        # 2. Обработка данных через LSTM
        lstm_result = process_with_lstm(price_series)
        
        # 3. Обработка данных через GPR
        # В новой архитектуре GPR использует оригинальные данные, а не сглаженные от Kalman Filter
        gpr_result = process_with_gpr(price_series)
        
        # 4. Передаем результаты Kalman Filter в GPR (дополнительно)
        # Для этого мы можем использовать тренд от Kalman Filter как дополнительный признак
        kalman_trend = kalman_result['trend']
        
        # 5. Передаем только технические индикаторы в XGBoost
        # Модель XGBoost обучена только на технических индикаторах
        feature_names = ["open","high","low","close","volume","turnover","BBL_20_2.0","BBM_20_2.0","BBU_20_2.0","BBB_20_2.0","BBP_20_2.0","MACD_12_26_9","MACDh_12_26_9","MACDs_12_26_9","OBV","ATRr_14","WILLR_14","RSI_14","CCI_20_0.015","ADX_14","DMP_14","DMN_14"]
        
        # 6. Получаем предсказание от XGBoost
        # Выбираем только те признаки, которые ожидает модель
        xgboost_features = latest_features[feature_names].to_frame().T
        # Преобразуем типы данных для XGBoost
        for col in xgboost_features.columns:
            xgboost_features[col] = pd.to_numeric(xgboost_features[col], errors='coerce')
        xgboost_features.fillna(0, inplace=True)
        xgboost_prediction = xgboost_model.predict(xgboost_features)
        print(f"--- {symbol} | Решение XGBoost: {xgboost_prediction} ---")
        print(f"--- {symbol} | Результаты Kalman Filter: Цена={kalman_result['kalman_price']}, Тренд={kalman_result['trend']} ---")
        print(f"--- {symbol} | Результаты LSTM: Предсказание={lstm_result['prediction']}, Уверенность={lstm_result['confidence']} ---")
        print(f"--- {symbol} | Результаты GPR: Предсказание={gpr_result['prediction']}, Уверенность={gpr_result['confidence']} ---")
        
        # Формируем log_data для модели и логирования
        log_data = latest_features.to_dict()
        # Обеспечиваем числовой формат timestamp
        log_data['timestamp'] = int(time.time() * 1000)
        log_data['xgboost_prediction'] = xgboost_prediction[0]
        # Добавляем результаты всех моделей в log_data
        log_data['kalman_price'] = kalman_result['kalman_price']
        log_data['kalman_trend'] = kalman_trend
        log_data['gpr_prediction'] = gpr_result['prediction']
        log_data['gpr_confidence'] = gpr_result['confidence']
        log_data['lstm_prediction'] = lstm_result['prediction']
        log_data['lstm_confidence'] = lstm_result['confidence']
        
        # Формируем obs для модели (все признаки из feature_cols в train_model.py + новые признаки)
        # В новой архитектуре все модели передают свои данные напрямую в TensorFlow DQN
        feature_cols = [
            'symbol', 'dqn_decision', 'order_type', 'price', 'quantity',
            'timestamp',
            'usdt_amount', 'bybit_order_id', 'status', 'error_message', 'pnl',
            'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0', 'BBB_20_2.0', 'BBP_20_2.0',
            'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9', 'OBV', 'ATRr_14',
            'WILLR_14', 'RSI_14', 'CCI_20_0.015', 'ADX_14', 'DMP_14', 'DMN_14',
            'xgboost_prediction', 'kalman_price', 'kalman_trend',
            'gpr_prediction', 'gpr_confidence', 'lstm_prediction', 'lstm_confidence'
        ]
        numeric_feature_cols = [col for col in feature_cols if col not in ['symbol', 'dqn_decision', 'order_type', 'status', 'error_message', 'bybit_order_id']]

        # Используем ненормализованные данные
        obs = np.array([log_data.get(key, 0.0) if key not in ['symbol', 'dqn_decision', 'order_type', 'status', 'error_message', 'bybit_order_id'] else 0.0 for key in feature_cols], dtype=np.float32)

        action, _ = model.predict(obs, deterministic=True)
        decision = {0: "SELL", 1: "BUY", 2: "HOLD"}.get(int(action))

        print(f"--- {symbol} | Решение модели на вход: {decision} ---")
        if decision in ["BUY", "SELL"]:
            trade_result = trade_manager.open_market_position(session, decision, symbol)
            if trade_result.get('status') == 'SUCCESS':
                opened_trades_counter += 1
                print(f"Сделка #{opened_trades_counter}/{OPEN_TRADE_LIMIT} открыта.")
                # Добавляем информацию о решении XGBoost в лог
                log_data = latest_features.to_dict()
                # Обеспечиваем числовой формат timestamp
                log_data['timestamp'] = int(time.time() * 1000)
                log_data['xgboost_prediction'] = xgboost_prediction[0]
                # Добавляем результаты всех моделей в log_data
                log_data['kalman_price'] = kalman_result['kalman_price']
                log_data['kalman_trend'] = kalman_trend
                log_data['gpr_prediction'] = gpr_result['prediction']
                log_data['gpr_confidence'] = gpr_result['confidence']
                log_data['lstm_prediction'] = lstm_result['prediction']
                log_data['lstm_confidence'] = lstm_result['confidence']
                # Добавляем метаданные для логирования
                log_data['symbol'] = symbol
                log_data['dqn_decision'] = f"OPEN_{decision}"
                log_data['order_type'] = decision
                log_data['price'] = trade_result.get('price', 'N/A')
                log_data['quantity'] = trade_result.get('quantity', 'N/A')
                log_data['usdt_amount'] = trade_result.get('usdt_amount', 'N/A')
                log_data['bybit_order_id'] = trade_result.get('bybit_order_id', 'N/A')
                log_data['status'] = trade_result.get('status', 'N/A')
                log_data['error_message'] = trade_result.get('message', 'N/A')
                log_data['pnl'] = 'N/A'  # PnL будет доступен только при закрытии

                # Нормализуем OBV и quantity в log_data для консистентности с данными для модели
                # Важно: это делается один раз, перед записью в лог
                if 'OBV' in log_data:
                    log_data['OBV'] = float(log_data['OBV']) / 100000.0
                if 'quantity' in log_data:
                    pass # quantity не нормализуем

                # Разделяем данные на order_details и indicators
                order_details = {
                    'timestamp': log_data.get('timestamp'),
                    'symbol': log_data.get('symbol'),
                    'dqn_decision': f"OPEN_{decision}",
                    'order_type': decision,
                    'price': trade_result.get('price', 'N/A'),
                    'quantity': log_data.get('quantity_normalized', trade_result.get('quantity', 'N/A')), # Используем нормализованное значение
                    'usdt_amount': trade_result.get('usdt_amount', 'N/A'),
                    'bybit_order_id': trade_result.get('bybit_order_id', 'N/A'),
                    'status': trade_result.get('status', 'N/A'),
                    'message': trade_result.get('message', 'N/A'),
                    'pnl': 'N/A'  # PnL будет доступен только при закрытии
                }
                
                # Исключаем поля order_details из indicators
                indicators = {k: v for k, v in log_data.items() if k not in order_details}
                
                trade_logger.log_event(symbol, f"OPEN_{decision}", order_details, indicators)
                
                current_positions = load_active_positions()
                current_positions[symbol] = {
                    'side': decision,
                    'entry_price': str(trade_result['price']),
                    'quantity': trade_result['quantity'],
                    'bybit_order_id': trade_result['bybit_order_id'],
                    'entry_timestamp_ms': int(time.time() * 1000)
                }
                save_active_positions(current_positions)
                if opened_trades_counter >= OPEN_TRADE_LIMIT:
                    print("!!! ДОСТИГНУТ ЛИМИТ ОТКРЫТЫХ СДЕЛОК. НОВЫЕ СДЕЛКИ ОТКРЫВАТЬСЯ НЕ БУДУТ. !!!")
                    print("Бот продолжит управлять открытыми позициями.")
                    set_trader_status('MANAGING_ONLY')

    except Exception as e:
        print(f"!!! Ошибка при обработке сигнала для {symbol}: {e} !!!")
        import traceback
        traceback.print_exc()

def run_trading_loop():
    global opened_trades_counter
    print("### Запуск торгового бота с активным управлением... ###")
    if not os.path.exists(MODEL_PATH):
        print(f"Модель {MODEL_PATH} не найдена. Сначала запустите train_model.py для создания пустой модели.")
        return
    model = DQN.load(MODEL_PATH)
    session = HTTP(testnet=True, api_key=config.BYBIT_API_KEY, api_secret=config.BYBIT_API_SECRET)
    session.endpoint = config.API_URL

    if os.path.exists(ACTIVE_POSITIONS_FILE):
        os.remove(ACTIVE_POSITIONS_FILE)
        print(f"Файл {ACTIVE_POSITIONS_FILE} очищен для новой сессии.")
    set_trader_status('DONE')
    print("Трейдер готов к работе, ожидает сигнала...")

    while True:
        try:
            manage_active_positions(session, model)

            if get_trader_status() == 'BUSY':
                print("Получен сигнал 'BUSY'.")
                with open(HOTLIST_FILE, 'r') as f: symbol = f.read().strip()
                if symbol: process_new_signal(session, symbol, model)
                
                # Статус на DONE сбрасывается только если лимит сделок еще не достигнут.
                # Если лимит достигнут, статус MANAGING_ONLY должен сохраниться.
                if opened_trades_counter < OPEN_TRADE_LIMIT:
                    set_trader_status('DONE')
            
            time.sleep(LOOP_SLEEP_SECONDS)

        except Exception as e:
            print(f"!!! КРИТИЧЕСКАЯ ОШИБКА в главном цикле: {e} !!!")
            traceback.print_exc()
            set_trader_status('DONE')
            time.sleep(LOOP_SLEEP_SECONDS)

if __name__ == "__main__":
    run_trading_loop()
