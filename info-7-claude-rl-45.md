
2. Файл: feature_engineering.py
Здесь мы закомментируем все расчеты индикаторов, кроме close (и volume для is_event, который потом упростим).
2.1. Функция calculate_awesome_oscillator(high, low)
Полностью закомментируйте эту функцию, так как она не будет использоваться.
# 🔥 ЗАКОММЕНТИРОВАНО: Awesome Oscillator не используется
# def calculate_awesome_oscillator(high, low):
#     """Calculates Awesome Oscillator (AO)"""
#     median_price = (high + low) / 2
#     short_sma = talib.SMA(median_price, timeperiod=5)
#     long_sma = talib.SMA(median_price, timeperiod=34)
#     return short_sma - long_sma

2.2. Функция calculate_features(df: pd.DataFrame)
Закомментируйте все расчеты индикаторов, кроме volume (который нужен для is_event, но его мы тоже упростим). ATR_14 также закомментирован.
Найдите этот блок (начинается с try: rsi = talib.RSI(...)):
        # Add indicators one by one with try-except blocks
        try:
            rsi = talib.RSI(close_p, timeperiod=14)
            rsi[np.isinf(rsi)] = np.nan
            df_out['RSI_14'] = pd.Series(rsi, index=df_out.index).ffill().fillna(0)
        except Exception:
            df_out['RSI_14'] = 0
            
        # 🔥 УДАЛЕНО: ATR_14 (пользователь решил его убрать)
        # try:
        #     atr = talib.ATR(high_p, low_p, close_p, timeperiod=14)
        #     atr[np.isinf(atr)] = np.nan
        #     df_out['ATR_14'] = pd.Series(atr, index=df_out.index).ffill().fillna(0)
        # except Exception:
        #     df_out['ATR_14'] = 0
            
        try:
            macd, macdsignal, macdhist = talib.MACD(close_p, fastperiod=12, slowperiod=26, signalperiod=9)
            df_out['MACD_12_26_9'] = pd.Series(macd, index=df_out.index).replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
            df_out['MACD_signal'] = pd.Series(macdsignal, index=df_out.index).replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
            df_out['MACD_hist'] = pd.Series(macdhist, index=df_out.index).replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
        except Exception:
            df_out['MACD_12_26_9'], df_out['MACD_signal'], df_out['MACD_hist'] = 0, 0, 0

        # 🔥 БОЛЛИНДЖЕР ОСТАЕТСЯ ЗАКОММЕНТИРОВАННЫМ
        # try:
        #     upper, middle, lower = talib.BBANDS(close_p, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        #     df_out['BBU_20_2.0'] = pd.Series(upper, index=df_out.index).replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
        #     df_out['BBM_20_2.0'] = pd.Series(middle, index=df_out.index).replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
        #     df_out['BBL_20_2.0'] = pd.Series(lower, index=df_out.index).replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
        # except Exception:
        #     df_out['BBU_20_2.0'], df_out['BBM_20_2.0'], df_out['BBL_20_2.0'] = 0, 0, 0

        try:
            adx = talib.ADX(high_p, low_p, close_p, timeperiod=14)
            adx[np.isinf(adx)] = np.nan
            df_out['ADX_14'] = pd.Series(adx, index=df_out.index).ffill().fillna(0)
        except Exception:
            df_out['ADX_14'] = 0

        try:
            slowk, slowd = talib.STOCH(high_p, low_p, close_p, fastk_period=14, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
            df_out['STOCHk_14_3_3'] = pd.Series(slowk, index=df_out.index).replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
            df_out['STOCHd_14_3_3'] = pd.Series(slowd, index=df_out.index).replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
        except Exception:
            df_out['STOCHk_14_3_3'], df_out['STOCHd_14_3_3'] = 0, 0
            
        # 🔥 НОВЫЙ ИНДИКАТОР: Williams %R (WILLR_14)
        try:
            willr = talib.WILLR(high_p, low_p, close_p, timeperiod=14)
            willr[np.isinf(willr)] = np.nan
            df_out['WILLR_14'] = pd.Series(willr, index=df_out.index).ffill().fillna(0)
        except Exception:
            df_out['WILLR_14'] = 0

        # 🔥 НОВЫЙ ИНДИКАТОР: Awesome Oscillator (AO_5_34)
        try:
            ao = calculate_awesome_oscillator(high_p, low_p) # Используем новую функцию
            ao[np.isinf(ao)] = np.nan
            df_out['AO_5_34'] = pd.Series(ao, index=df_out.index).ffill().fillna(0)
        except Exception:
            df_out['AO_5_34'] = 0

        # 🔥 СОЗДАЕМ is_event С ИНДИКАТОРАМИ (обновляем для AO_5_34)
        required_cols = ['volume', 'AO_5_34', 'RSI_14', 'ADX_14'] # 🔥 ИЗМЕНЕНО: ATR_14 заменен на AO_5_34
        for col in required_cols:
            if col not in df_out.columns:
                df_out[col] = 0 # Заполняем нулями, если вдруг нет

        df_out['is_event'] = (
            (df_out['volume'] > df_out['volume'].rolling(50).quantile(0.9).fillna(0)) | # Объем > 90% квантиля
            (abs(df_out['AO_5_34']) > df_out['AO_5_34'].rolling(50).std().fillna(0) * 1.5) | # 🔥 ИЗМЕНЕНО: AO > 1.5 std
            (abs(df_out['RSI_14'] - 50) > 25) | # RSI выходит из зоны 25-75 (более экстремально)
            (df_out['ADX_14'] > df_out['ADX_14'].shift(5).fillna(0) + 2) # ADX растёт > 2 пункта за 5 баров
        ).astype(int)

Замените его на (ВСЕ ЗАКОММЕНТИРОВАНО, кроме is_event упрощенного):
        # Add indicators one by one with try-except blocks
        # 🔥 ЗАКОММЕНТИРОВАНО: RSI
        # try:
        #     rsi = talib.RSI(close_p, timeperiod=14)
        #     rsi[np.isinf(rsi)] = np.nan
        #     df_out['RSI_14'] = pd.Series(rsi, index=df_out.index).ffill().fillna(0)
        # except Exception:
        #     df_out['RSI_14'] = 0
            
        # 🔥 ЗАКОММЕНТИРОВАНО: ATR_14
        # try:
        #     atr = talib.ATR(high_p, low_p, close_p, timeperiod=14)
        #     atr[np.isinf(atr)] = np.nan
        #     df_out['ATR_14'] = pd.Series(atr, index=df_out.index).ffill().fillna(0)
        # except Exception:
        #     df_out['ATR_14'] = 0
            
        # 🔥 ЗАКОММЕНТИРОВАНО: MACD
        # try:
        #     macd, macdsignal, macdhist = talib.MACD(close_p, fastperiod=12, slowperiod=26, signalperiod=9)
        #     df_out['MACD_12_26_9'] = pd.Series(macd, index=df_out.index).replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
        #     df_out['MACD_signal'] = pd.Series(macdsignal, index=df_out.index).replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
        #     df_out['MACD_hist'] = pd.Series(macdhist, index=df_out.index).replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
        # except Exception:
        #     df_out['MACD_12_26_9'], df_out['MACD_signal'], df_out['MACD_hist'] = 0, 0, 0

        # 🔥 ЗАКОММЕНТИРОВАНО: Боллинджер (BBU, BBM, BBL)
        # try:
        #     upper, middle, lower = talib.BBANDS(close_p, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        #     df_out['BBU_20_2.0'] = pd.Series(upper, index=df_out.index).replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
        #     df_out['BBM_20_2.0'] = pd.Series(middle, index=df_out.index).replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
        #     df_out['BBL_20_2.0'] = pd.Series(lower, index=df_out.index).replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
        # except Exception:
        #     df_out['BBU_20_2.0'], df_out['BBM_20_2.0'], df_out['BBL_20_2.0'] = 0, 0, 0

        # 🔥 ЗАКОММЕНТИРОВАНО: ADX
        # try:
        #     adx = talib.ADX(high_p, low_p, close_p, timeperiod=14)
        #     adx[np.isinf(adx)] = np.nan
        #     df_out['ADX_14'] = pd.Series(adx, index=df_out.index).ffill().fillna(0)
        # except Exception:
        #     df_out['ADX_14'] = 0

        # 🔥 ЗАКОММЕНТИРОВАНО: STOCH
        # try:
        #     slowk, slowd = talib.STOCH(high_p, low_p, close_p, fastk_period=14, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
        #     df_out['STOCHk_14_3_3'] = pd.Series(slowk, index=df_out.index).replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
        #     df_out['STOCHd_14_3_3'] = pd.Series(slowd, index=df_out.index).replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
        # except Exception:
        #     df_out['STOCHk_14_3_3'], df_out['STOCHd_14_3_3'] = 0, 0
            
        # 🔥 ЗАКОММЕНТИРОВАНО: Williams %R (WILLR_14)
        # try:
        #     willr = talib.WILLR(high_p, low_p, close_p, timeperiod=14)
        #     willr[np.isinf(willr)] = np.nan
        #     df_out['WILLR_14'] = pd.Series(willr, index=df_out.index).ffill().fillna(0)
        # except Exception:
        #     df_out['WILLR_14'] = 0

        # 🔥 ЗАКОММЕНТИРОВАНО: Awesome Oscillator (AO_5_34)
        # try:
        #     ao = calculate_awesome_oscillator(high_p, low_p) # Используем новую функцию
        #     ao[np.isinf(ao)] = np.nan
        #     df_out['AO_5_34'] = pd.Series(ao, index=df_out.index).ffill().fillna(0)
        # except Exception:
        #     df_out['AO_5_34'] = 0

        # 🔥 УПРОЩЕНО: is_event будет просто нулями, если нет других осмысленных признаков
        # (или можно удалить, если не хотим, чтобы is_event влиял вообще)
        # Для "close only" теста, is_event не должен влиять.
        df_out['is_event'] = 0 # 🔥 ИЗМЕНЕНО: is_event всегда 0

2.3. Функции извлечения признаков паттернов
Полностью закомментируйте все функции hammer_features, hangingman_features и т.д. вплоть до bullish_belt_hold_features. Они не будут использоваться, так как detect_candlestick_patterns закомментирован.
Найдите этот блок (начинается с def hammer_features(df):):
# --- Feature extraction functions for each pattern ---

def hammer_features(df):
    atr = calculate_atr(df['high'], df['low'], df['close'])
    # ...
    return features

def hangingman_features(df):
    # ...
    return features
# ... и так далее до конца блока

Закомментируйте его полностью:
# --- Feature extraction functions for each pattern ---

# 🔥 ЗАКОММЕНТИРОВАНО: Все функции извлечения признаков паттернов
# def hammer_features(df):
#     atr = calculate_atr(df['high'], df['low'], df['close'])
#     support, resistance = find_support_resistance(df['low'], df['high'])
#     volume_ratio = calculate_volume_ratio(df['volume'])
#     
#     features = pd.DataFrame(index=df.index)
#     features['hammer_f_on_support'] = is_on_level(df['close'], support, atr)
#     features['hammer_f_vol_spike'] = is_volume_spike(volume_ratio)
#     return features

# def hangingman_features(df):
#     atr = calculate_atr(df['high'], df['low'], df['close'])
#     support, resistance = find_support_resistance(df['low'], df['high'])
#     volume_ratio = calculate_volume_ratio(df['volume'])
#     
#     features = pd.DataFrame(index=df.index)
#     features['hangingman_f_on_res'] = is_on_level(df['close'], resistance, atr)
#     features['hangingman_f_vol_spike'] = is_volume_spike(volume_ratio)
#     return features

# ... (и так далее для всех функций hammer_features, hangingman_features, engulfing_features, doji_features, inverted_hammer_features, dragonfly_doji_features, bullish_pin_bar_features, bullish_belt_hold_features, shootingstar_features, bullish_marubozu_features, add_pattern_features)

2.4. Функция prepare_xlstm_rl_features(df: pd.DataFrame)
Обновите список feature_cols, чтобы он содержал только close.
Найдите этот блок:
    feature_cols = [
        # ✅ ВСЕ ТЕХНИЧЕСКИЕ ИНДИКАТОРЫ (БЕЗ БОЛЛИНДЖЕРА И ATR_14)
        'RSI_14', 'MACD_12_26_9', 'MACD_signal', 'MACD_hist',
        'ADX_14', 'STOCHk_14_3_3', 'STOCHd_14_3_3',
        'WILLR_14', # 🔥 НОВЫЙ ИНДИКАТОР
        'AO_5_34',  # 🔥 НОВЫЙ ИНДИКАТОР
        
        # ❌ ВСЕ ПАТТЕРНЫ ЗАКОММЕНТИРОВАНЫ
        # ...
        
        # ✅ ОСТАВЛЯЕМ EVENT SAMPLING
        'is_event'
    ]

Замените его на:
    feature_cols = [
        'close' # 🔥 ИЗМЕНЕНО: Только 'close'
        # ❌ ВСЕ ОСТАЛЬНЫЕ ИНДИКАТОРЫ И ПАТТЕРНЫ ЗАКОММЕНТИРОВАНЫ
        # 'RSI_14', 'MACD_12_26_9', 'MACD_signal', 'MACD_hist',
        # 'ADX_14', 'STOCHk_14_3_3', 'STOCHd_14_3_3',
        # 'WILLR_14',
        # 'AO_5_34',
        # 'is_event' # is_event пока не включаем, так как он не основан на 'close'
    ]

3. Файл: run_live_trading.py
3.1. Глобальная переменная FEATURE_COLUMNS
Обновите список FEATURE_COLUMNS.
Найдите этот блок:
FEATURE_COLUMNS = [
    # ✅ ВСЕ ТЕХНИЧЕСКИЕ ИНДИКАТОРЫ (БЕЗ БОЛЛИНДЖЕРА И ATR_14)
    'RSI_14', 'MACD_12_26_9', 'MACD_signal', 'MACD_hist',
    'ADX_14', 'STOCHk_14_3_3', 'STOCHd_14_3_3',
    'WILLR_14', # 🔥 НОВЫЙ ИНДИКАТОР
    'AO_5_34',  # 🔥 НОВЫЙ ИНДИКАТОР
    
    # ❌ ВСЕ ПАТТЕРНЫ ЗАКОММЕНТИРОВАНЫ
    # ...
    
    # ✅ ОСТАВЛЯЕМ EVENT SAMPLING
    'is_event'
]

Замените его на:
FEATURE_COLUMNS = [
    'close' # 🔥 ИЗМЕНЕНО: Только 'close'
    # ❌ ВСЕ ОСТАЛЬНЫЕ ИНДИКАТОРЫ И ПАТТЕРНЫ ЗАКОММЕНТИРОВАНЫ
    # 'RSI_14', 'MACD_12_26_9', 'MACD_signal', 'MACD_hist',
    # 'ADX_14', 'STOCHk_14_3_3', 'STOCHd_14_3_3',
    # 'WILLR_14',
    # 'AO_5_34',
    # 'is_event'
]

3.2. Функция calculate_dynamic_stops(features_row, position_side, entry_price)
Эта функция использует индикаторы для расчета стопов. Поскольку мы переходим на "close only", эта логика больше не применима. Установим фиксированные стопы для теста.
Найдите этот блок:
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

Замените его на (фиксированные стопы):
def calculate_dynamic_stops(features_row, position_side, entry_price):
    """
    Вычисляет фиксированные стоп-лоссы, так как индикаторы отключены.
    """
    base_sl = -0.5 # 🔥 ИЗМЕНЕНО: Фиксированный Stop Loss (например, -0.5%)
    base_tp = 1.0  # 🔥 ИЗМЕНЕНО: Фиксированный Take Profit (например, 1.0%)
    
    # При отключенных индикаторах используем базовые фиксированные значения
    dynamic_sl = base_sl
    dynamic_tp = base_tp
        
    # Ограничиваем максимальные и минимальные значения
    dynamic_sl = max(dynamic_sl, -3.0)
    dynamic_tp = min(dynamic_tp, 3.0)

    return dynamic_sl, dynamic_tp

4. Файл: trading_env.py
4.1. Функция reset(self, seed=None, options=None)
Обновите список self.feature_columns.
Найдите этот блок:
        self.feature_columns = [
            # ✅ ВСЕ ТЕХНИЧЕСКИЕ ИНДИКАТОРЫ (БЕЗ БОЛЛИНДЖЕРА И ATR_14)
            'RSI_14', 'MACD_12_26_9', 'MACD_signal', 'MACD_hist',
            'ADX_14', 'STOCHk_14_3_3', 'STOCHd_14_3_3',
            'WILLR_14', # 🔥 НОВЫЙ ИНДИКАТОР
            'AO_5_34',  # 🔥 НОВЫЙ ИНДИКАТОР
            
            # ❌ ВСЕ ПАТТЕРНЫ ЗАКОММЕНТИРОВАНЫ
            # ...
            
            # ✅ ОСТАВЛЯЕМ EVENT SAMPLING
            'is_event'
        ]

Замените его на:
        self.feature_columns = [
            'close' # 🔥 ИЗМЕНЕНО: Только 'close'
            # ❌ ВСЕ ОСТАЛЬНЫЕ ИНДИКАТОРЫ И ПАТТЕРНЫ ЗАКОММЕНТИРОВАНЫ
            # 'RSI_14', 'MACD_12_26_9', 'MACD_signal', 'MACD_hist',
            # 'ADX_14', 'STOCHk_14_3_3', 'STOCHd_14_3_3',
            # 'WILLR_14',
            # 'AO_5_34',
            # 'is_event'
        ]

4.2. Функция _calculate_advanced_reward(self, action, pnl_pct, xlstm_prediction)
Удалите всю логику, основанную на индикаторах, для расчета buy_signal_strength, sell_signal_strength, hold_reward и overtrading_penalty.
Найдите этот блок (начинается с current_row = self.df.iloc[self.current_step]):
        current_row = self.df.iloc[self.current_step]
        # Используем индикаторы для определения "явного сигнала"
        buy_signal_strength = (
            (current_row.get('RSI_14', 50) < 30) +
            (current_row.get('ADX_14', 0) > 25) +
            (current_row.get('MACD_hist', 0) > 0.001) +
            (current_row.get('WILLR_14', -50) < -80) + # 🔥 НОВОЕ: WILLR_14 для BUY (сильно перепродано)
            (current_row.get('AO_5_34', 0) > 0) # 🔥 НОВОЕ: AO выше нуля
        )
        sell_signal_strength = (
            (current_row.get('RSI_14', 50) > 70) +
            (current_row.get('ADX_14', 0) > 25) +
            (current_row.get('MACD_hist', 0) < -0.001) +
            (current_row.get('WILLR_14', -50) > -20) + # 🔥 НОВОЕ: WILLR_14 для SELL (сильно перекуплено)
            (current_row.get('AO_5_34', 0) < 0) # 🔥 НОВОЕ: AO ниже нуля
        )

        if action == 2: # HOLD
            # 🔥 ИЗМЕНЕНО: Использование AO_5_34 и ADX_14 для HOLD reward
            ao_value = current_row.get('AO_5_34', 0)
            adx = current_row.get('ADX_14', 0)

            # Если моментум низкий (AO близко к 0) и ADX низкий (флэт)
            if abs(ao_value) < 0.001 and adx < 20: # Пороги нужно будет подобрать
                hold_reward = 0.5
            # Если сильный моментум (большой AO) или сильный тренд (большой ADX)
            elif abs(ao_value) > 0.005 or adx > 30:
                hold_reward = -0.5
            else:
                hold_reward = 0.1
            
            if pnl_pct < 0 and self.steps_in_position > 30:
                hold_penalty = -3
            
            # Добавляем бонус за HOLD, если нет сильных сигналов
            if buy_signal_strength < 1 and sell_signal_strength < 1:
                hold_reward += 1.0
            else:
                hold_reward -= 1.0

        else: # Если действие BUY или SELL (не HOLD)
            # Штраф за overtrading (слишком частые сделки, когда нет явного сигнала)
            # Увеличиваем штраф за слабые BUY-сигналы, если RL предсказывает BUY
            if action == 1 and buy_signal_strength < 2:
                overtrading_penalty = -1.0
            # Увеличиваем штраф за слабые SELL-сигналы, если RL предсказывает SELL
            elif action == 0 and sell_signal_strength < 2:
                overtrading_penalty = -1.0

Замените его на (очень упрощенная логика без индикаторов):
        # 🔥 УПРОЩЕНО: Логика наград без индикаторов
        hold_reward = 0
        overtrading_penalty = 0

        if action == 2: # HOLD
            # Очень простая награда/штраф за HOLD
            hold_reward = 0.1 # Небольшая награда за HOLD
            if pnl_pct < 0 and self.steps_in_position > 30:
                hold_penalty = -3 # Штраф за долгое удержание убыточной позиции
        else: # Если действие BUY или SELL (не HOLD)
            # Нет индикаторных условий для overtrading, просто базовые штрафы
            overtrading_penalty = -0.5 # Небольшой штраф за открытие сделки в отсутствие четких сигналов

5. Файл: train_model.py
Здесь нужно обновить sequence_length, закомментировать весь блок imblearn и сильно упростить логику генерации целевых меток и переклассификации HOLD.
5.1. argparse и sequence_length
Найдите эту строку:
    parser.add_argument('--sequence_length', type=int, default=60, help='Длина последовательности')

Оставьте ее без изменений (уже 60).
5.2. Блок IMBLEARN
Убедитесь, что этот блок полностью закомментирован:
    # 🔥 ЗАКОММЕНТИРОВАНО: Отключаем imblearn
    # try:
    #     from imblearn.over_sampling import SMOTE
    #     # ... (весь код imblearn) ...
    # except ImportError:
    #     print("⚠️ imbalanced-learn не установлен...")
    # except Exception as e:
    #     print(f"❌ Ошибка при oversampling/undersampling: {e}")

5.3. Обновляем логику генерации целевых меток в prepare_xlstm_rl_data
Найдите этот блок:
        # Создаем целевые метки на основе будущих цен + индикаторов
        df['future_return'] = (df['close'].shift(-5) - df['close']) / df['close']
        
        # Увеличиваем пороги для генерации торговых сигналов
        df['base_threshold'] = 0.0015 # 🔥 ИЗМЕНЕНО: С 0.002 до 0.0015 (еще более мягкий порог)
        df['dynamic_threshold'] = np.maximum(
            df['base_threshold'],
            (abs(df['AO_5_34']) / df['close'] * 0.7).fillna(0.0015) # 🔥 ИЗМЕНЕНО: Коэффициент 0.7
        )

        # Классические технические фильтры
        strong_trend = df['ADX_14'] > 15 # 🔥 ИЗМЕНЕНО: С 18 до 15 (еще более мягкий порог)
        
        # Условия для BUY
        rsi_buy_zone = df['RSI_14'] < 50 # 🔥 ИЗМЕНЕНО: С 45 до 50
        macd_buy_signal = (df['MACD_12_26_9'] > df['MACD_signal']) # MACD_hist уже убрали
        willr_buy_signal = df['WILLR_14'] < -60 # 🔥 ИЗМЕНЕНО: С -70 до -60
        ao_buy_signal = df['AO_5_34'] > 0 # AO выше нуля
        
        # Условия для SELL
        rsi_sell_zone = df['RSI_14'] > 50 # 🔥 ИЗМЕНЕНО: С 55 до 50
        macd_sell_signal = (df['MACD_12_26_9'] < df['MACD_signal']) # MACD_hist уже убрали
        willr_sell_signal = df['WILLR_14'] > -40 # 🔥 ИЗМЕНЕНО: С -30 до -40
        ao_sell_signal = df['AO_5_34'] < 0 # AO ниже нуля

        # Условия для BUY/SELL только на основе future_return и классических индикаторов
        buy_condition = (
            (df['future_return'] > df['dynamic_threshold'] * 1.0) &
            (strong_trend & (rsi_buy_zone | macd_buy_signal | willr_buy_signal | ao_buy_signal))
        )

        sell_condition = (
            (df['future_return'] < -df['dynamic_threshold'] * 1.0) &
            (strong_trend & (rsi_sell_zone | macd_sell_signal | willr_sell_signal | ao_sell_signal))
        )

Замените его на (только future_return):
        # 🔥 УПРОЩЕНО: Создаем целевые метки только на основе future_return
        df['future_return'] = (df['close'].shift(-5) - df['close']) / df['close']
        
        # Очень мягкие пороги для BUY/SELL
        buy_threshold = 0.001 # 🔥 ИЗМЕНЕНО: Мягкий порог для BUY (0.1%)
        sell_threshold = -0.001 # 🔥 ИЗМЕНЕНО: Мягкий порог для SELL (-0.1%)

        buy_condition = (df['future_return'] > buy_threshold)
        sell_condition = (df['future_return'] < sell_threshold)
        
        # Устанавливаем метки
        df['target'] = 2  # По умолчанию HOLD
        df.loc[buy_condition, 'target'] = 0  # BUY
        df.loc[sell_condition, 'target'] = 1  # SELL
        
        # 🔥 НОВЫЕ ЛОГИ: Количество сигналов до балансировки
        initial_buy_signals = (df['target'] == 0).sum()
        initial_sell_signals = (df['target'] == 1).sum()
        initial_hold_signals = (df['target'] == 2).sum()
        total_initial_signals = len(df)
        print(f"📊 Исходный баланс классов для {symbol} (до imblearn):")
        print(f"  BUY: {initial_buy_signals} ({initial_buy_signals/total_initial_signals*100:.2f}%)")
        print(f"  SELL: {initial_sell_signals} ({initial_sell_signals/total_initial_signals*100:.2f}%)")
        print(f"  HOLD: {initial_hold_signals} ({initial_hold_signals/total_initial_signals*100:.2f}%)")
        print(f"  Общее количество сигналов: {total_initial_signals}")

        current_buy_count = (df['target'] == 0).sum()
        current_sell_count = (df['target'] == 1).sum()
        current_hold_count = (df['target'] == 2).sum()

        # 🔥 ЗАКОММЕНТИРОВАНО: Блок переклассификации HOLD
        # if current_hold_count > (current_buy_count + current_sell_count) * 3.0:
        #     print(f"⚠️ Сильный дисбаланс классов. Попытка УМНОЙ переклассификации части HOLD-сигналов (с индикаторами).")
        #     hold_indices = df[df['target'] == 2].index
        #     
        #     import random
        #     random.seed(42)
        #     
        #     reclassify_count = int(current_hold_count * 0.10)
        #     if reclassify_count > 0:
        #         reclassify_indices = random.sample(list(hold_indices), min(reclassify_count, len(hold_indices)))
        #         
        #         for idx in reclassify_indices:
        #             if idx < 5: continue
        #             
        #             rsi = df.loc[idx, 'RSI_14']
        #             adx = df.loc[idx, 'ADX_14']
        #             macd_hist = df.loc[idx, 'MACD_hist']
        #             willr = df.loc[idx, 'WILLR_14']
        #             ao = df.loc[idx, 'AO_5_34']
        #             price_change_3_period = df['close'].pct_change(3).loc[idx]
        #
        #             # Условия для переклассификации (с индикаторами) - теперь с AO и WILLR
        #             # 🔥 Условия значительно ослаблены для увеличения количества сигналов
        #             if (rsi < 50 and adx > 15 and macd_hist > 0 and willr < -60 and ao > 0 and price_change_3_period > 0.0015):
        #                 df.loc[idx, 'target'] = 0  # BUY
        #             elif (rsi > 50 and adx > 15 and macd_hist < 0 and willr > -40 and ao < 0 and price_change_3_period < -0.0015):
        #                 df.loc[idx, 'target'] = 1  # SELL
        #             
        #             # 2. Сильный тренд по ADX + движение цены (без других индикаторов для более широкого охвата)
        #             elif (adx > 20 and abs(price_change_3_period) > 0.002):
        #                 df.loc[idx, 'target'] = 0 if price_change_3_period > 0 else 1
        #     
        #     print(f"Баланс классов после УМНОЙ переклассификации (с индикаторами):")
        #     unique, counts = np.unique(df['target'], return_counts=True)
        #     class_names = ['BUY', 'SELL', 'HOLD']
        #     for class_idx, count in zip(unique, counts):
        #         print(f"  {class_names[class_idx]}: {count} ({count/len(df)*100:.1f}%)")
        # else:
        #     print(f"✅ Баланс классов приемлемый, переклассификация HOLD не требуется.")

6. Файл: models/xlstm_rl_model.py
Здесь мы уже упростили архитектуру модели и настроили LR.
6.1. Функция build_model(self)
Найдите эту строку (в конце функции):
        print("✅ Упрощенная xLSTM модель создана!")

Измените ее на:
        print("✅ Упрощенная xLSTM модель (только CLOSE) создана!") # 🔥 ИЗМЕНЕНО: Добавлено уточнение

6.2. Функция train(self, ...)
Удаляем class_weight из xlstm_model.train.
Найдите вызов xlstm_model.train:
        history = self.model.fit(
            X_train_scaled, y_train,
            validation_data=(X_val_scaled, y_val),
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weight,  # <--- Эту строку нужно удалить или закомментировать
            callbacks=callbacks,
            verbose=0,
            shuffle=True
        )

Замените его на (удаляем class_weight):
        history = self.model.fit(
            X_train_scaled, y_train,
            validation_data=(X_val_scaled, y_val),
            epochs=epochs,
            batch_size=batch_size,
            # class_weight=class_weight,  # 🔥 ЗАКОММЕНТИРОВАНО: class_weight больше не нужен
            callbacks=callbacks,
            verbose=0,
            shuffle=True
        )

7. Файл: market_regime_detector.py
Этот файл использует индикаторы для определения режимов. Поскольку мы переходим на "close only", детектор режимов будет бесполезен. Временно отключим его.
7.1. Функция extract_regime_features(self, df)
Закомментируйте все расчеты признаков режима:
    def extract_regime_features(self, df):
        """Извлекает признаки для определения режима рынка"""
        
        # 🔥 ЗАКОММЕНТИРОВАНО: Все признаки режима
        # # Ценовые признаки
        # df['returns'] = df['close'].pct_change()
        # df['volatility'] = df['returns'].rolling(20).std()
        # df['trend_strength'] = df['close'].rolling(20).apply(lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0] if x.iloc[0] != 0 else 0)
        # 
        # # Объемные признаки
        # df['volume_trend'] = df['volume'].rolling(10).mean() / df['volume'].rolling(30).mean()
        # df['volume_volatility'] = df['volume'].rolling(20).std() / df['volume'].rolling(20).mean()
        # 
        # # Технические признаки
        # if 'RSI_14' in df.columns:
        #     df['rsi_regime'] = np.where(df['RSI_14'] > 70, 1, np.where(df['RSI_14'] < 30, -1, 0))
        # else:
        #     df['rsi_regime'] = 0
        # 
        # # 🔥 ЗАКОММЕНТИРОВАНО: bb_position
        # # if 'BBL_20_2.0' in df.columns and 'BBU_20_2.0' in df.columns:
        # #     df['bb_position'] = (df['close'] - df['BBL_20_2.0']) / (df['BBU_20_2.0'] - df['BBL_20_2.0'])
        # # else:
        # #     df['bb_position'] = 0
        # 
        # # 🔥 НОВЫЕ ПРИЗНАКИ РЕЖИМА: AO_5_34 и WILLR_14
        # if 'AO_5_34' in df.columns:
        #     df['ao_regime'] = np.where(df['AO_5_34'] > 0, 1, np.where(df['AO_5_34'] < 0, -1, 0)) # AO > 0 bullish, < 0 bearish
        # else:
        #     df['ao_regime'] = 0
        # 
        # if 'WILLR_14' in df.columns:
        #     df['willr_regime'] = np.where(df['WILLR_14'] < -80, 1, np.where(df['WILLR_14'] > -20, -1, 0)) # WILLR < -80 oversold, > -20 overbought
        # else:
        #     df['willr_regime'] = 0
        # 
        # regime_features = [
        #     'volatility', 'trend_strength', 'volume_trend', 'volume_volatility',
        #     'rsi_regime',
        #     'ao_regime',
        #     'willr_regime'
        # ]

        # 🔥 ВОЗВРАЩАЕМ ПУСТОЙ DF ИЛИ DF ТОЛЬКО С CLOSE, чтобы избежать ошибок
        return df[['close']].copy() # Возвращаем df только с 'close'

7.2. Функции fit и predict_regime
Модифицируйте эти функции, чтобы они возвращали заглушки или использовали минимальные признаки.
Функция fit(self, df):
    def fit(self, df):
        """Обучает детектор на исторических данных"""
        
        # 🔥 ЗАГЛУШКА: Детектор режимов отключен для 'close only' теста
        print("⚠️ Детектор рыночных режимов временно отключен для 'close only' теста.")
        self.is_fitted = True # Считаем его "обученным", чтобы не падать
        return self

Функция predict_regime(self, df):
    def predict_regime(self, df):
        """Предсказывает текущий рыночный режим"""
        
        if not self.is_fitted:
            raise ValueError("Детектор должен быть обучен перед предсказанием")
        
        # 🔥 ЗАГЛУШКА: Всегда возвращаем UNKNOWN
        return 'UNKNOWN', 0.0


8. Файл: visual_graph.py
Здесь нужно обновить, какие признаки используются для предсказания и как они извлекаются.
8.1. Функция plot_predictions(...)
Обновите, как извлекаются признаки.
Найдите эту строку:
    # Подготовка данных для предсказания
    X_predict = []
    # ...
    for i in range(len(symbol_df) - sequence_length):
        X_predict.append(symbol_df.iloc[i:i + sequence_length][feature_cols].values)

Замените ее на:
    # Подготовка данных для предсказания
    X_predict = []
    # ...
    for i in range(len(symbol_df) - sequence_length):
        # 🔥 ИЗМЕНЕНО: Извлекаем только 'close'
        X_predict.append(symbol_df.iloc[i:i + sequence_length][['close']].values)

8.2. Функция main()
Обновите, как feature_cols извлекаются.
Найдите этот блок:
        # 🔥 НОВОЕ: Временно создаем фейковый df для получения feature_cols
        # Это нужно, чтобы получить актуальный список признаков без полной обработки
        temp_df = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
        _, feature_cols = prepare_xlstm_rl_features(temp_df) # Используем фиктивный df для получения feature_cols
        
        # Применяем полную обработку к реальным данным
        processed_df = calculate_features(symbol_df.copy())
        processed_df = detect_candlestick_patterns(processed_df) # Паттерны закомментированы, но вызов нужен
        
        # Убираем NaN, которые могли появиться после расчетов индикаторов
        processed_df.dropna(subset=feature_cols, inplace=True)
        processed_df.reset_index(drop=True, inplace=True)

Замените его на (только close):
        # 🔥 УПРОЩЕНО: Для 'close only' feature_cols всегда будет ['close']
        feature_cols = ['close']
        
        # Применяем минимальную обработку к реальным данным
        processed_df = symbol_df.copy() # 🔥 ИЗМЕНЕНО: Просто копируем, без calculate_features и detect_candlestick_patterns
        
        # Убеждаемся, что есть 'close' и нет NaN
        processed_df.dropna(subset=['close'], inplace=True)
        processed_df.reset_index(drop=True, inplace=True)
