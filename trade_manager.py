import numpy as np
from pybit.unified_trading import HTTP
import config
from decimal import Decimal
import time
from models.xlstm_pattern_model import XLSTMPatternModel
from models.xlstm_indicator_model import XLSTMIndicatorModel

# --- Глобальный кэш для правил торговли ---
instrument_info_cache = {}

# --- Константы ---
ORDER_USDT_AMOUNT = 11.0

def get_instrument_info(session, symbol):
    """Получает и кэширует правила торговли для символа."""
    if symbol not in instrument_info_cache:
        try:
            info = session.get_instruments_info(
                category="linear", symbol=symbol
            )['result']['list'][0]
            instrument_info_cache[symbol] = info['lotSizeFilter']
            print(f"Получены и закэшированы правила для {symbol}.")
        except Exception as e:
            print(f"Не удалось получить правила торговли для {symbol}: {e}")
            return None
    return instrument_info_cache[symbol]

def fetch_initial_data(session, symbol):
    """
    Загружает последние REQUIRED_CANDLES свечей для символа.
    Эта функция нужна для run_live_trading, чтобы получать свежие данные для анализа открытых позиций.
    """
    try:
        kline_data = session.get_kline(
            category="linear",
            symbol=symbol,
            interval=config.TIMEFRAME,
            limit=config.REQUIRED_CANDLES
        )['result']['list']

        if len(kline_data) < config.REQUIRED_CANDLES:
            return None

        kline_data.reverse()
        formatted_klines = []
        for k in kline_data:
            formatted_klines.append([
                int(k[0]), float(k[1]), float(k[2]),
                float(k[3]), float(k[4]), float(k[5]),
                float(k[6])
            ])
        return formatted_klines
    except Exception as e:
        print(f"\n!!! Ошибка при загрузке свежих данных для {symbol}: {e} !!!")
        return None

def open_market_position(session, decision, symbol):
    """
    Открывает позицию рыночным ордером.
    """
    side = "Buy" if decision == "BUY" else "Sell"
    try:
        lot_size_filter = get_instrument_info(session, symbol)
        if not lot_size_filter:
            return {"status": "error", "message": f"Нет правил торговли для {symbol}"}

        ticker_info = session.get_tickers(category="linear", symbol=symbol)['result']['list'][0]
        last_price = Decimal(ticker_info['lastPrice'])
        if last_price == 0:
            return {"status": "error", "message": "Last price is zero."}

        qty_step = Decimal(lot_size_filter['qtyStep'])
        quantity_raw = Decimal(ORDER_USDT_AMOUNT) / last_price
        final_quantity = (quantity_raw // qty_step) * qty_step

        min_order_qty = Decimal(lot_size_filter['minOrderQty'])
        if final_quantity < min_order_qty:
            return {"status": "error", "message": f"Calculated qty {final_quantity} is less than minOrderQty {min_order_qty}"}

        # --- Установка кредитного плеча ---
        try:
            leverage_response = session.set_leverage(
                category="linear",
                symbol=symbol,
                buyLeverage=config.LEVERAGE,
                sellLeverage=config.LEVERAGE,
            )
            ret_code = leverage_response.get('retCode')
            if ret_code == 0:
                print(f"  [Leverage] Успешно установлено плечо {config.LEVERAGE}x для {symbol}.")
            # Код 110043 означает "leverage not modified" - это не ошибка, а информация.
            elif ret_code == 110043:
                print(f"  [Leverage] Плечо для {symbol} уже было {config.LEVERAGE}x. Пропускаем установку.")
            else:
                # Логируем настоящее предупреждение для всех других кодов ошибок
                print(f"  [Leverage] ПРЕДУПРЕЖДЕНИЕ: Не удалось установить плечо для {symbol}. Код: {ret_code}, Ответ: {leverage_response.get('retMsg')}")
        except Exception as e:
            # Код 110043 (leverage not modified) - это не ошибка, а информация. Игнорируем.
            if '110043' in str(e):
                print(f"  [Leverage] Плечо для {symbol} уже было {config.LEVERAGE}x. Пропускаем установку.")
            else:
                print(f"  [Leverage] ПРЕДУПРЕЖДЕНИЕ: Не удалось установить плечо для {symbol}: {e}")
        # --- Конец установки плеча ---

        print(f"\n>>> ПОПЫТКА ОТКРЫТИЯ РЫНОЧНОГО ОРДЕРА: {side} {final_quantity} {symbol}...")
        response = session.place_order(
            category="linear",
            symbol=symbol,
            side=side,
            orderType="Market",
            qty=str(final_quantity),
        )

        if response.get('retCode') == 0:
            order_id = response['result'].get('orderId')
            # Получаем цену входа после исполнения
            time.sleep(0.5) # Даем ордеру время на исполнение
            trade_history = session.get_executions(category="linear", orderId=order_id, limit=1)['result']['list']
            entry_price = trade_history[0]['execPrice'] if trade_history else last_price
            print(f"УСПЕХ: Ордер на открытие {order_id} исполнен.")
            return {"status": "SUCCESS", "bybit_order_id": order_id, "quantity": str(final_quantity), "price": str(entry_price)}
        else:
            error_message = response.get('retMsg', 'Unknown error')
            print(f"ОШИБКА ОТКРЫТИЯ: {error_message}")
            return {"status": "error", "message": error_message}

    except Exception as e:
        print(f"!!! КРИТИЧЕСКАЯ ОШИБКА ОТКРЫТИЯ ОРДЕРА: {e} !!!")
        return {"status": "error", "message": str(e)}

def close_market_position(session, symbol, qty, side):
    """
    Закрывает позицию рыночным ордером.
    """
    close_side = "Sell" if side == "BUY" else "Buy"
    try:
        print(f"\n>>> ПОПЫТКА ЗАКРЫТИЯ ПОЗИЦИИ: {close_side} {qty} {symbol}...")
        response = session.place_order(
            category="linear",
            symbol=symbol,
            side=close_side,
            orderType="Market",
            qty=str(qty),
            reduceOnly=True
        )

        if response.get('retCode') == 0:
            order_id = response['result'].get('orderId')
            print(f"УСПЕХ: Ордер на закрытие {order_id} принят биржей!")
            
            # --- Улучшенный механизм получения цены закрытия ---
            close_price = 'N/A'
            # 1. Основной метод: пытаемся получить точную цену исполнения
            for i in range(4): # Пробуем 4 раза с паузой
                try:
                    time.sleep(0.25 * (i + 1)) # Паузы: 0.25, 0.5, 0.75, 1.0 сек
                    trade_history = session.get_executions(category="linear", orderId=order_id, limit=1)['result']['list']
                    if trade_history:
                        close_price = trade_history[0]['execPrice']
                        print(f"  [Executions] Точная цена закрытия {close_price} получена для ордера {order_id}.")
                        break 
                except Exception:
                    pass # Игнорируем ошибки, чтобы дойти до конца цикла или до резервного метода
            
            # 2. Резервный метод: если точная цена не получена, берем последнюю цену тикера
            if close_price == 'N/A':
                try:
                    ticker_info = session.get_tickers(category="linear", symbol=symbol)['result']['list'][0]
                    close_price = ticker_info['lastPrice']
                    print(f"  [Executions] ПРЕДУПРЕЖДЕНИЕ: Не удалось получить точную цену исполнения. Использована последняя цена тикера: {close_price}")
                except Exception as e:
                    print(f"  [Executions] КРИТИЧЕСКАЯ ОШИБКА: Не удалось получить ни цену исполнения, ни цену тикера для {symbol}: {e}")
            # --- Конец ---    

            return {"status": "SUCCESS", "bybit_order_id": order_id, "price": str(close_price)}
        else:
            error_message = response.get('retMsg', 'Unknown error')
            if '110017' in error_message:
                print(f"ИНФО: Позиция для {symbol} уже закрыта. Удаляем из активных.")
                return {"status": "SUCCESS", "bybit_order_id": "CLOSED_EXTERNALLY", "price": "N/A"}
            print(f"ОШИКА ЗАКРЫТИЯ: {error_message}")
            return {"status": "error", "message": error_message}

    except Exception as e:
        print(f"!!! КРИТИЧЕСКАЯ ОШИБКА ЗАКРЫТИЯ ОРДЕРА: {e} !!!")
        return {"status": "error", "message": str(e)}

class ConsensusDecisionMaker:
    """
    Makes trading decisions based on the consensus of two models.
    """
    def __init__(self, xlstm_pattern_model_path, xlstm_indicator_model_path, xlstm_pattern_scaler_path, xlstm_indicator_scaler_path, sequence_length, pattern_feature_count, indicator_feature_count):
        """
        Initializes the ConsensusDecisionMaker.

        Args:
            xlstm_pattern_model_path (str): Path to the xLSTM pattern model file.
            xlstm_indicator_model_path (str): Path to the xLSTM indicator model file.
            xlstm_pattern_scaler_path (str): Path to the xLSTM pattern scaler file.
            xlstm_indicator_scaler_path (str): Path to the xLSTM indicator scaler file.
            sequence_length (int): The length of the input sequences.
            pattern_feature_count (int): The number of pattern features.
            indicator_feature_count (int): The number of indicator features.
        """
        self.xlstm_pattern_model = XLSTMPatternModel(input_shape=(sequence_length, pattern_feature_count))
        self.xlstm_pattern_model.load_model(xlstm_pattern_model_path, xlstm_pattern_scaler_path)
        
        self.xlstm_indicator_model = XLSTMIndicatorModel(input_shape=(sequence_length, indicator_feature_count))
        self.xlstm_indicator_model.load_model(xlstm_indicator_model_path, xlstm_indicator_scaler_path)

    def get_decision(self, pattern_features, indicator_features, mode='Consensus', confidence_threshold=0.5):
        # verbose=1 - включает отображение "пинга" (время выполнения предсказания)
        # verbose=0 - отключает
        xlstm_pattern_prediction = self.xlstm_pattern_model.predict(pattern_features)
        xlstm_indicator_prediction = self.xlstm_indicator_model.predict(indicator_features)

        # Расширенное логирование с метками
        pred_pattern = xlstm_pattern_prediction[0]
        pred_indicator = xlstm_indicator_prediction[0]
        print(f"DEBUG: xLSTM Pattern Prediction: BUY: {pred_pattern[0]:.4f}, SELL: {pred_pattern[1]:.4f}, HOLD: {pred_pattern[2]:.4f}")
        print(f"DEBUG: xLSTM Indicator Prediction: BUY: {pred_indicator[0]:.4f}, SELL: {pred_indicator[1]:.4f}, HOLD: {pred_indicator[2]:.4f}")

        self.xlstm_pattern_decision_index = np.argmax(xlstm_pattern_prediction, axis=1)[0]
        self.xlstm_indicator_decision_index = np.argmax(xlstm_indicator_prediction, axis=1)[0]
        
        self.xlstm_pattern_decision = ["BUY", "SELL", "HOLD"][self.xlstm_pattern_decision_index]
        self.xlstm_indicator_decision = ["BUY", "SELL", "HOLD"][self.xlstm_indicator_decision_index]

        self.xlstm_pattern_confidence = np.max(xlstm_pattern_prediction)
        self.xlstm_indicator_confidence = np.max(xlstm_indicator_prediction)

        if mode == 'xLSTM_pattern_only':
            return self.xlstm_pattern_decision

        if mode == 'xLSTM_indicator_only':
            return self.xlstm_indicator_decision

        # --- ПРАВИЛЬНАЯ ЛОГИКА КОНСЕНСУСА ---

        # 1. Вычисляем взвешенные средние вероятности
        xlstm_pattern_weight = 0.6
        xlstm_indicator_weight = 0.4

        weighted_prediction = (xlstm_pattern_prediction * xlstm_pattern_weight) + (xlstm_indicator_prediction * xlstm_indicator_weight)

        # 2. Находим класс с максимальной средней вероятностью
        decision_index = np.argmax(weighted_prediction, axis=1)[0]
        self.consensus_confidence = np.max(weighted_prediction)
        potential_decision = ["BUY", "SELL", "HOLD"][decision_index]

        # 3. Логируем средние вероятности
        avg_probs = weighted_prediction[0]
        print(f"DEBUG: Средние вероятности: BUY: {avg_probs[0]:.4f}, SELL: {avg_probs[1]:.4f}, HOLD: {avg_probs[2]:.4f}")

        # 4. Принимаем решение только если уверенность ≥ 50% и решение не HOLD
        if self.consensus_confidence >= confidence_threshold and potential_decision != "HOLD":
            self.consensus_decision = potential_decision
            print(f"DEBUG: Консенсус: {potential_decision} с уверенностью {self.consensus_confidence:.4f}")
            return self.consensus_decision

        # 5. Во всех остальных случаях - HOLD
        self.consensus_decision = "HOLD"
        print(f"DEBUG: Недостаточная уверенность ({self.consensus_confidence:.4f} < {confidence_threshold}) или HOLD решение. Итог: HOLD")
        return "HOLD"
