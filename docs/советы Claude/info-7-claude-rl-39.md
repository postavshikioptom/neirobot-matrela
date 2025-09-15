Вот подробная инструкция, как включить все индикаторы и отключить все паттерны, комментируя код, а не удаляя его, и сохраняя ATR_14 как часть индикаторов.

📋 Инструкция: Включаем индикаторы, отключаем паттерны (комментируем)
1. Файл: feature_engineering.py
Здесь мы вернем расчет всех индикаторов и отключим детектирование паттернов.
1.1. Функция calculate_features(df: pd.DataFrame)
Верните расчет всех технических индикаторов, которые вы закомментировали ранее. Убедитесь, что ATR_14 также раскомментирован и остается активным.
Найдите этот блок:
        # 🔥 ВСЕ ИНДИКАТОРЫ ЗАКОММЕНТИРОВАНЫ
        # try:
        #     rsi = talib.RSI(close_p, timeperiod=14)
        #     rsi[np.isinf(rsi)] = np.nan
        #     df_out['RSI_14'] = pd.Series(rsi, index=df_out.index).ffill().fillna(0)
        # except Exception:
        #     df_out['RSI_14'] = 0
            
        # ОСТАВЛЯЕМ ATR_14, он нужен для признаков паттернов
        try:
            atr = talib.ATR(high_p, low_p, close_p, timeperiod=14)
            atr[np.isinf(atr)] = np.nan
            df_out['ATR_14'] = pd.Series(atr, index=df_out.index).ffill().fillna(0)
        except Exception:
            df_out['ATR_14'] = 0

        # И так далее для всех остальных индикаторов...

        # 🔥 СОЗДАЕМ is_event БЕЗ ИНДИКАТОРОВ - на основе только объема и паттернов
        df_out['is_event'] = (
            (df_out['volume'] > df_out['volume'].rolling(50).quantile(0.9).fillna(0))
        ).astype(int)

Замените его на (раскомментируйте все индикаторы, верните is_event к исходному виду):
        # Add indicators one by one with try-except blocks
        try:
            rsi = talib.RSI(close_p, timeperiod=14)
            rsi[np.isinf(rsi)] = np.nan
            df_out['RSI_14'] = pd.Series(rsi, index=df_out.index).ffill().fillna(0)
        except Exception:
            df_out['RSI_14'] = 0
            
        # ОСТАВЛЯЕМ ATR_14, он нужен для признаков паттернов
        try:
            atr = talib.ATR(high_p, low_p, close_p, timeperiod=14)
            atr[np.isinf(atr)] = np.nan
            df_out['ATR_14'] = pd.Series(atr, index=df_out.index).ffill().fillna(0)
        except Exception:
            df_out['ATR_14'] = 0
            
        try:
            macd, macdsignal, macdhist = talib.MACD(close_p, fastperiod=12, slowperiod=26, signalperiod=9)
            df_out['MACD_12_26_9'] = pd.Series(macd, index=df_out.index).replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
            df_out['MACD_signal'] = pd.Series(macdsignal, index=df_out.index).replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
            df_out['MACD_hist'] = pd.Series(macdhist, index=df_out.index).replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
        except Exception:
            df_out['MACD_12_26_9'], df_out['MACD_signal'], df_out['MACD_hist'] = 0, 0, 0

        try:
            upper, middle, lower = talib.BBANDS(close_p, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
            df_out['BBU_20_2.0'] = pd.Series(upper, index=df_out.index).replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
            df_out['BBM_20_2.0'] = pd.Series(middle, index=df_out.index).replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
            df_out['BBL_20_2.0'] = pd.Series(lower, index=df_out.index).replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
        except Exception:
            df_out['BBU_20_2.0'], df_out['BBM_20_2.0'], df_out['BBL_20_2.0'] = 0, 0, 0

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

        # 🔥 СОЗДАЕМ is_event С ИНДИКАТОРАМИ (возвращаем к исходному виду)
        # Убедимся, что все нужные колонки существуют (ATR_14 уже добавлен)
        required_cols = ['volume', 'ATR_14', 'RSI_14', 'ADX_14']
        for col in required_cols:
            if col not in df_out.columns:
                df_out[col] = 0 # Заполняем нулями, если вдруг нет

        df_out['is_event'] = (
            (df_out['volume'] > df_out['volume'].rolling(50).quantile(0.9).fillna(0)) | # Объем > 90% квантиля
            (df_out['ATR_14'] > df_out['ATR_14'].rolling(50).quantile(0.9).fillna(0)) | # ATR > 90% квантиля
            (abs(df_out['RSI_14'] - 50) > 25) | # RSI выходит из зоны 25-75 (более экстремально)
            (df_out['ADX_14'] > df_out['ADX_14'].shift(5).fillna(0) + 2) # ADX растёт > 2 пункта за 5 баров
        ).astype(int)

1.2. Функция detect_candlestick_patterns(df: pd.DataFrame)
Закомментируйте весь код внутри этой функции, чтобы паттерны не детектировались.
Найдите этот блок:
def detect_candlestick_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detects a new set of candlestick patterns and their features.
    The new set includes: Hammer, Engulfing, Doji, Shooting Star, Hanging Man, 3 Black Crows.
    It removes: Morning Star, Evening Star.
    It keeps: 3 White Soldiers as per user request.
    """
    if df.empty:
        return df
        
    ohlc = ['open', 'high', 'low', 'close']
    if not all(col in df.columns for col in ohlc):
        raise ValueError("DataFrame must contain OHLC columns.")
    df[ohlc] = df[ohlc].astype(float)

    open_prices = df['open'].values
    high_prices = df['high'].values
    low_prices = df['low'].values
    close_prices = df['close'].values

    # --- Calculate base patterns ---
    df['CDLHAMMER'] = talib.CDLHAMMER(open_prices, high_prices, low_prices, close_prices)
    df['CDLENGULFING'] = talib.CDLENGULFING(open_prices, high_prices, low_prices, close_prices)
    # ... и так далее для всех CDL паттернов
    
    # --- Calculate features for each pattern ---
    df = add_pattern_features(df)

    pattern_cols = [
        'CDLHAMMER', 'CDLENGULFING', 'CDLDOJI', 'CDLSHOOTINGSTAR',
        'CDLHANGINGMAN', 'CDLMARUBOZU',
        # НОВЫЕ БЫЧЬИ ПАТТЕРНЫ
        'CDLINVERTEDHAMMER', 'CDLDRAGONFLYDOJI', 'CDLBELTHOLD' # ИЗМЕНЕНО: Удален CDLBULLISHKICKING
    ]
    
    # Add new feature columns to the list to ensure they are handled
    feature_cols = [
        'hammer_f', 'hangingman_f', 'engulfing_f', 'doji_f',
        'shootingstar_f', 'bullish_marubozu_f',
        # НОВЫЕ БЫЧЬИ ПАТТЕРНЫ
        'inverted_hammer_f', 'dragonfly_doji_f', 'bullish_pin_bar_f', 'bullish_belt_hold_f' # ИЗМЕНЕНО: Удален bullish_kicker_f
    ]
    
    all_pattern_cols = pattern_cols + feature_cols

    for col in all_pattern_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    return df

Замените его на (полностью закомментируйте тело функции, кроме возврата пустого df):
def detect_candlestick_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detects a new set of candlestick patterns and their features.
    The new set includes: Hammer, Engulfing, Doji, Shooting Star, Hanging Man, 3 Black Crows.
    It removes: Morning Star, Evening Star.
    It keeps: 3 White Soldiers as per user request.
    """
    if df.empty:
        return df
    
    # 🔥 ВЕСЬ КОД ДЛЯ ДЕТЕКТИРОВАНИЯ ПАТТЕРНОВ ЗАКОММЕНТИРОВАН
    # ohlc = ['open', 'high', 'low', 'close']
    # if not all(col in df.columns for col in ohlc):
    #     raise ValueError("DataFrame must contain OHLC columns.")
    # df[ohlc] = df[ohlc].astype(float)

    # open_prices = df['open'].values
    # high_prices = df['high'].values
    # low_prices = df['low'].values
    # close_prices = df['close'].values

    # # --- Calculate base patterns ---
    # df['CDLHAMMER'] = talib.CDLHAMMER(open_prices, high_prices, low_prices, close_prices)
    # df['CDLENGULFING'] = talib.CDLENGULFING(open_prices, high_prices, low_prices, close_prices)
    # df['CDLDOJI'] = talib.CDLDOJI(open_prices, high_prices, low_prices, close_prices)
    # df['CDLSHOOTINGSTAR'] = talib.CDLSHOOTINGSTAR(open_prices, high_prices, low_prices, close_prices)
    # df['CDLHANGINGMAN'] = talib.CDLHANGINGMAN(open_prices, high_prices, low_prices, close_prices)
    # df['CDLMARUBOZU'] = talib.CDLMARUBOZU(open_prices, high_prices, low_prices, close_prices)
    # # НОВЫЕ БЫЧЬИ ПАТТЕРНЫ
    # df['CDLINVERTEDHAMMER'] = talib.CDLINVERTEDHAMMER(open_prices, high_prices, low_prices, close_prices)
    # df['CDLDRAGONFLYDOJI'] = talib.CDLDRAGONFLYDOJI(open_prices, high_prices, low_prices, close_prices)
    # # Для Bullish Pin Bar нет прямого TA-Lib, используем комбинацию или CDLHAMMER
    # df['CDLBELTHOLD'] = talib.CDLBELTHOLD(open_prices, high_prices, low_prices, close_prices)


    # # --- Calculate features for each pattern ---
    # df = add_pattern_features(df)

    # pattern_cols = [
    #     'CDLHAMMER', 'CDLENGULFING', 'CDLDOJI', 'CDLSHOOTINGSTAR',
    #     'CDLHANGINGMAN', 'CDLMARUBOZU',
    #     # НОВЫЕ БЫЧЬИ ПАТТЕРНЫ
    #     'CDLINVERTEDHAMMER', 'CDLDRAGONFLYDOJI', 'CDLBELTHOLD'
    # ]
    
    # # Add new feature columns to the list to ensure they are handled
    # feature_cols = [
    #     'hammer_f', 'hangingman_f', 'engulfing_f', 'doji_f',
    #     'shootingstar_f', 'bullish_marubozu_f',
    #     # НОВЫЕ БЫЧЬИ ПАТТЕРНЫ
    #     'inverted_hammer_f', 'dragonfly_doji_f', 'bullish_pin_bar_f', 'bullish_belt_hold_f'
    # ]
    
    # all_pattern_cols = pattern_cols + feature_cols

    # for col in all_pattern_cols:
    #     if col in df.columns:
    #         df[col] = df[col].fillna(0)

    return df # Возвращаем DataFrame без добавленных паттернов

1.3. Функция prepare_xlstm_rl_features(df: pd.DataFrame)
Обновите список feature_cols, чтобы он включал все индикаторы и исключал все паттерны.
Найдите этот блок:
    feature_cols = [
        # ❌ ОТКЛЮЧЕНЫ ВСЕ ТЕХНИЧЕСКИЕ ИНДИКАТОРЫ
        # 'RSI_14', 'MACD_12_26_9', 'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0',
        # 'ADX_14', 'STOCHk_14_3_3', 'STOCHd_14_3_3', 'ATR_14',
        
        # ✅ ТОЛЬКО БАЗОВЫЕ ПАТТЕРНЫ TA-Lib
        'CDLHAMMER', 'CDLENGULFING', 'CDLDOJI', 'CDLSHOOTINGSTAR',
        'CDLHANGINGMAN', 'CDLMARUBOZU',
        # НОВЫЕ БЫЧЬИ ПАТТЕРНЫ TA-Lib
        'CDLINVERTEDHAMMER', 'CDLDRAGONFLYDOJI', 'CDLBELTHOLD',
        
        # ✅ ТОЛЬКО КОМБИНИРОВАННЫЕ ПРИЗНАКИ ПАТТЕРНОВ
        'hammer_f', 'hangingman_f', 'engulfing_f', 'doji_f',
        'shootingstar_f', 'bullish_marubozu_f',
        # НОВЫЕ КОМБИНИРОВАННЫЕ ПРИЗНАКИ БЫЧЬИХ ПАТТЕРНОВ
        'inverted_hammer_f', 'dragonfly_doji_f', 'bullish_pin_bar_f', 'bullish_belt_hold_f',
        
        # ✅ ОСТАВЛЯЕМ EVENT SAMPLING
        'is_event'
    ]

Замените его на (включаем индикаторы, комментируем паттерны):
    feature_cols = [
        # ✅ ВСЕ ТЕХНИЧЕСКИЕ ИНДИКАТОРЫ
        'RSI_14', 'MACD_12_26_9', 'MACD_signal', 'MACD_hist', 'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0',
        'ADX_14', 'STOCHk_14_3_3', 'STOCHd_14_3_3', 'ATR_14', # ATR_14 теперь как полноценный признак
        
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

2. Файл: run_live_trading.py
Здесь мы обновим глобальный список FEATURE_COLUMNS.
2.1. Глобальная переменная FEATURE_COLUMNS
Найдите этот блок:
# 🔥 НОВЫЕ FEATURE_COLUMNS - ТОЛЬКО ПАТТЕРНЫ
FEATURE_COLUMNS = [
    # ✅ ТОЛЬКО ПАТТЕРНЫ (все индикаторы отключены)
    'CDLHAMMER', 'CDLENGULFING', 'CDLDOJI', 'CDLSHOOTINGSTAR',
    'CDLHANGINGMAN', 'CDLMARUBOZU',
    'CDLINVERTEDHAMMER', 'CDLDRAGONFLYDOJI', 'CDLBELTHOLD',
    'hammer_f', 'hangingman_f', 'engulfing_f', 'doji_f',
    'shootingstar_f', 'bullish_marubozu_f',
    'inverted_hammer_f', 'dragonfly_doji_f', 'bullish_pin_bar_f', 'bullish_belt_hold_f',
    'is_event'
]

Замените его на (включаем индикаторы, комментируем паттерны):
# 🔥 НОВЫЕ FEATURE_COLUMNS - ТОЛЬКО ИНДИКАТОРЫ
FEATURE_COLUMNS = [
    # ✅ ВСЕ ТЕХНИЧЕСКИЕ ИНДИКАТОРЫ
    'RSI_14', 'MACD_12_26_9', 'MACD_signal', 'MACD_hist', 'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0',
    'ADX_14', 'STOCHk_14_3_3', 'STOCHd_14_3_3', 'ATR_14',
    
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

2.2. Функции manage_active_positions и process_new_signal
В этих функциях закомментируйте вызов feature_engineering.detect_candlestick_patterns.
Найдите эти строки:
            features_df = feature_engineering.calculate_features(kline_df.copy())
            features_df = feature_engineering.detect_candlestick_patterns(features_df) # <-- Закомментировать
            # features_df = feature_engineering.calculate_vsa_features(features_df)  # <--- ЗАКОММЕНТИРОВАНО

Замените на:
            features_df = feature_engineering.calculate_features(kline_df.copy())
            # features_df = feature_engineering.detect_candlestick_patterns(features_df) # 🔥 ЗАКОММЕНТИРОВАНО
            # features_df = feature_engineering.calculate_vsa_features(features_df)  # <--- ЗАКОММЕНТИРОВАНО

Сделайте это для обеих функций: manage_active_positions и process_new_signal.
3. Файл: trading_env.py
Здесь мы скорректируем список признаков для RL-среды и логику наград.
3.1. Функция reset(self, seed=None, options=None)
Обновите список self.feature_columns, чтобы он включал все индикаторы и исключал все паттерны.
Найдите этот блок:
        # 🔥 НОВЫЕ FEATURE_COLUMNS - ТОЛЬКО ПАТТЕРНЫ
        self.feature_columns = [
            # ❌ ВСЕ ИНДИКАТОРЫ ОТКЛЮЧЕНЫ
            # 'RSI_14', 'MACD_12_26_9', 'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0',
            # 'ADX_14', 'STOCHk_14_3_3', 'STOCHd_14_3_3', 'ATR_14',
            
            # ✅ ТОЛЬКО ПАТТЕРНЫ
            'CDLHAMMER', 'CDLENGULFING', 'CDLDOJI', 'CDLSHOOTINGSTAR',
            'CDLHANGINGMAN', 'CDLMARUBOZU',
            'CDLINVERTEDHAMMER', 'CDLDRAGONFLYDOJI', 'CDLBELTHOLD',
            'hammer_f', 'hangingman_f', 'engulfing_f', 'doji_f',
            'shootingstar_f', 'bullish_marubozu_f',
            'inverted_hammer_f', 'dragonfly_doji_f', 'bullish_pin_bar_f', 'bullish_belt_hold_f',
            'is_event'
        ]

Замените его на (включаем индикаторы, комментируем паттерны):
        # 🔥 НОВЫЕ FEATURE_COLUMNS - ТОЛЬКО ИНДИКАТОРЫ (для RL среды)
        self.feature_columns = [
            # ✅ ВСЕ ТЕХНИЧЕСКИЕ ИНДИКАТОРЫ
            'RSI_14', 'MACD_12_26_9', 'MACD_signal', 'MACD_hist', 'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0',
            'ADX_14', 'STOCHk_14_3_3', 'STOCHd_14_3_3', 'ATR_14',
            
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

3.2. Функция _calculate_advanced_reward(self, action, pnl_pct, xlstm_prediction)
Верните логику наград к использованию индикаторов, как это было до отключения VSA и паттернов.
Найдите этот блок:
        # 🔥 УБИРАЕМ АНАЛИЗ ИНДИКАТОРОВ ДЛЯ OVERTRADING И HOLD
        hold_reward = 0
        overtrading_penalty = 0

        if action == 2: # HOLD
            # Награда за HOLD, если нет сильных паттернов (или просто небольшая константа)
            current_row = self.df.iloc[self.current_step]
            bullish_pattern_strength = (
                abs(current_row.get('CDLHAMMER', 0)) +
                abs(current_row.get('CDLENGULFING', 0)) +
                abs(current_row.get('CDLINVERTEDHAMMER', 0)) +
                abs(current_row.get('CDLDRAGONFLYDOJI', 0)) +
                abs(current_row.get('CDLBELTHOLD', 0)) +
                current_row.get('hammer_f', 0) +
                current_row.get('inverted_hammer_f', 0) +
                current_row.get('bullish_marubozu_f', 0)
            )
            bearish_pattern_strength = (
                abs(current_row.get('CDLHANGINGMAN', 0)) +
                abs(current_row.get('CDLSHOOTINGSTAR', 0)) +
                (abs(current_row.get('CDLENGULFING', 0)) if current_row.get('CDLENGULFING', 0) < 0 else 0) +
                current_row.get('hangingman_f', 0) +
                current_row.get('shootingstar_f', 0)
            )
            
            if bullish_pattern_strength < 1 and bearish_pattern_strength < 1: # Если нет сильных BUY/SELL паттернов
                hold_reward = 1.0 # Увеличен бонус за правильный HOLD
            else:
                hold_reward = -1.0 # Увеличен штраф за HOLD, если есть сильные сигналы
            
            if pnl_pct < 0 and self.steps_in_position > 30:
                hold_penalty = -3
                
        else: # Если действие BUY или SELL (не HOLD)
            current_row = self.df.iloc[self.current_step]
            # Считаем силу паттернов
            bullish_pattern_strength = (
                abs(current_row.get('CDLHAMMER', 0)) +
                abs(current_row.get('CDLENGULFING', 0)) +
                abs(current_row.get('CDLINVERTEDHAMMER', 0)) +
                abs(current_row.get('CDLDRAGONFLYDOJI', 0)) +
                abs(current_row.get('CDLBELTHOLD', 0)) +
                current_row.get('hammer_f', 0) +
                current_row.get('inverted_hammer_f', 0) +
                current_row.get('bullish_marubozu_f', 0)
            )
            bearish_pattern_strength = (
                abs(current_row.get('CDLHANGINGMAN', 0)) +
                abs(current_row.get('CDLSHOOTINGSTAR', 0)) +
                (abs(current_row.get('CDLENGULFING', 0)) if current_row.get('CDLENGULFING', 0) < 0 else 0) +
                current_row.get('hangingman_f', 0) +
                current_row.get('shootingstar_f', 0)
            )

            # Штраф за overtrading (слишком частые сделки, когда нет явного паттерна)
            if action == 1 and bullish_pattern_strength < 2: # Требуем 2+ сильных паттерна
                overtrading_penalty = -1.0
            elif action == 0 and bearish_pattern_strength < 2: # Требуем 2+ сильных паттерна
                overtrading_penalty = -1.0

Замените его на (возвращаем индикаторы):
        # НОВЫЙ КОД - Корректируем функцию наград для RL (более сбалансированное вознаграждение, с акцентом на HOLD)
        hold_reward = 0
        overtrading_penalty = 0

        if action == 2: # HOLD
            current_row = self.df.iloc[self.current_step]
            volatility = current_row.get('ATR_14', 0) / current_row.get('close', 1)
            adx = current_row.get('ADX_14', 0)

            if volatility < 0.005 and adx < 25:
                hold_reward = 0.5
            elif volatility > 0.01 and adx > 30:
                hold_reward = -0.5
            else:
                hold_reward = 0.1
            
            if pnl_pct < 0 and self.steps_in_position > 30:
                hold_penalty = -3
            
        else: # Если действие BUY или SELL (не HOLD)
            # Штраф за overtrading (слишком частые сделки, когда нет явного сигнала)
            current_row = self.df.iloc[self.current_step]
            # Используем индикаторы для определения "явного сигнала"
            buy_signal_strength = (
                (current_row.get('RSI_14', 50) < 30) +
                (current_row.get('ADX_14', 0) > 25) +
                (current_row.get('MACD_hist', 0) > 0.001)
            )
            sell_signal_strength = (
                (current_row.get('RSI_14', 50) > 70) +
                (current_row.get('ADX_14', 0) > 25) +
                (current_row.get('MACD_hist', 0) < -0.001)
            )

            # Увеличиваем штраф за слабые BUY-сигналы, если RL предсказывает BUY
            if action == 1 and buy_signal_strength < 2: # Требуем 2+ сильных сигнала
                overtrading_penalty = -1.0
            # Увеличиваем штраф за слабые SELL-сигналы, если RL предсказывает SELL
            elif action == 0 and sell_signal_strength < 2: # Требуем 2+ сильных сигнала
                overtrading_penalty = -1.0

        # Добавляем бонус за HOLD, если нет сильных сигналов
        if action == 2: # HOLD
            if buy_signal_strength < 1 and sell_signal_strength < 1: # Если нет сильных BUY/SELL сигналов
                hold_reward += 1.0
            else:
                hold_reward -= 1.0

4. Файл: train_model.py
Это самый важный файл. Мы скорректируем список признаков, логику генерации целевых меток и блок переклассификации HOLD.
4.1. Функция prepare_xlstm_rl_data(data_path, sequence_length=10)
Здесь мы определим, какие признаки будут использоваться для обучения xLSTM модели, и как будут генерироваться целевые метки.
Найдите этот блок:
    feature_cols = [
        # ❌ ВСЕ ИНДИКАТОРЫ ОТКЛЮЧЕНЫ
        # 'RSI_14', 'MACD_12_26_9', 'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0',
        # 'ADX_14', 'STOCHk_14_3_3', 'STOCHd_14_3_3', 'ATR_14',
        
        # ✅ ТОЛЬКО ПАТТЕРНЫ
        'CDLHAMMER', 'CDLENGULFING', 'CDLDOJI', 'CDLSHOOTINGSTAR',
        'CDLHANGINGMAN', 'CDLMARUBOZU',
        'CDLINVERTEDHAMMER', 'CDLDRAGONFLYDOJI', 'CDLBELTHOLD',
        
        # ✅ ТОЛЬКО КОМБИНИРОВАННЫЕ ПРИЗНАКИ ПАТТЕРНОВ
        'hammer_f', 'hangingman_f', 'engulfing_f', 'doji_f',
        'shootingstar_f', 'bullish_marubozu_f',
        'inverted_hammer_f', 'dragonfly_doji_f', 'bullish_pin_bar_f', 'bullish_belt_hold_f',
        
        # ✅ ОСТАВЛЯЕМ EVENT SAMPLING
        'is_event'
    ]

Замените его на (включаем индикаторы, комментируем паттерны):
    feature_cols = [
        # ✅ ВСЕ ТЕХНИЧЕСКИЕ ИНДИКАТОРЫ
        'RSI_14', 'MACD_12_26_9', 'MACD_signal', 'MACD_hist', 'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0',
        'ADX_14', 'STOCHk_14_3_3', 'STOCHd_14_3_3', 'ATR_14', # ATR_14 теперь как полноценный признак
        
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

4.2. Генерация целевых меток BUY/SELL
Верните логику определения buy_condition и sell_condition к использованию индикаторов, как это было до отключения паттернов.
Найдите этот блок:
        # 🔥 НОВЫЕ УСЛОВИЯ БЕЗ ИНДИКАТОРОВ - ТОЛЬКО ПАТТЕРНЫ
        # Создаем целевые метки на основе будущих цен + ПАТТЕРНОВ
        df['future_return'] = (df['close'].shift(-5) - df['close']) / df['close']
        
        # Базовые пороги без адаптации по ATR (так как ATR отключен)
        df['base_threshold'] = 0.012  # Увеличиваем базовый порог
        
        # Классические технические фильтры - ❌ ЗАКОММЕНТИРОВАНЫ
        # strong_trend = df['ADX_14'] > 25
        
        # Условия для BUY/SELL только на основе future_return и паттернов
        # BUY условия - сильные бычьи паттерны
        strong_bullish_patterns = (
            (df['CDLHAMMER'] > 0) |
            (df['CDLENGULFING'] > 0) |
            (df['CDLINVERTEDHAMMER'] > 0) |
            (df['CDLDRAGONFLYDOJI'] > 0) |
            (df['CDLBELTHOLD'] > 0) |
            (df['hammer_f'] >= 2) |
            (df['inverted_hammer_f'] >= 2) |
            (df['bullish_marubozu_f'] >= 2)
        )
        
        # SELL условия - сильные медвежьи паттерны
        strong_bearish_patterns = (
            (df['CDLHANGINGMAN'] > 0) |
            (df['CDLSHOOTINGSTAR'] > 0) |
            (df['CDLENGULFING'] < 0) |  # Медвежье поглощение
            (df['hangingman_f'] >= 2) |
            (df['shootingstar_f'] >= 1) |
            (df['doji_f'] >= 2)  # Doji в зоне сопротивления
        )

        # Более строгие условия для BUY/SELL
        buy_condition = (
            (df['future_return'] > df['base_threshold']) &
            strong_bullish_patterns
        )

        sell_condition = (
            (df['future_return'] < -df['base_threshold']) &
            strong_bearish_patterns
        )

Замените его на (возвращаем индикаторы):
        # Создаем целевые метки на основе будущих цен + индикаторов
        df['future_return'] = (df['close'].shift(-5) - df['close']) / df['close']
        
        # Увеличиваем пороги для генерации торговых сигналов
        df['base_threshold'] = 0.008
        df['dynamic_threshold'] = np.maximum(
            df['base_threshold'],
            (df['ATR_14'] / df['close'] * 1.5).fillna(0.008)
        )

        # Классические технические фильтры
        strong_trend = df['ADX_14'] > 25
        
        # Условия для BUY
        rsi_buy_zone = df['RSI_14'] < 30
        macd_buy_signal = (df['MACD_12_26_9'] > df['MACD_signal']) & \
                          (df['MACD_hist'] > 0.001)
        
        # Условия для SELL
        rsi_sell_zone = df['RSI_14'] > 70
        macd_sell_signal = (df['MACD_12_26_9'] < df['MACD_signal']) & \
                           (df['MACD_hist'] < -0.001)

        # Условия для BUY/SELL только на основе future_return и классических индикаторов
        buy_condition = (
            (df['future_return'] > df['dynamic_threshold'] * 1.0) &
            (strong_trend & rsi_buy_zone & macd_buy_signal)
        )

        sell_condition = (
            (df['future_return'] < -df['dynamic_threshold'] * 1.0) &
            (strong_trend & rsi_sell_zone & macd_sell_signal)
        )

4.3. Блок переклассификации HOLD
Верните этот блок к использованию индикаторов.
Найдите этот блок:
        # 🔥 ПЕРЕКЛАССИФИКАЦИЯ БЕЗ ИНДИКАТОРОВ
        if current_hold_count > (current_buy_count + current_sell_count) * 3.0:
            print(f"⚠️ Переклассификация ТОЛЬКО на основе ПАТТЕРНОВ и движения цены")
            hold_indices = df[df['target'] == 2].index
            
            import random
            random.seed(42)
            
            reclassify_count = int(current_hold_count * 0.10)
            if reclassify_count > 0:
                reclassify_indices = random.sample(list(hold_indices), min(reclassify_count, len(hold_indices)))
                
                for idx in reclassify_indices:
                    if idx < 5: continue
                    
                    # ❌ ЗАКОММЕНТИРОВАНО: НЕ ИСПОЛЬЗУЕМ ИНДИКАТОРЫ ДЛЯ ПЕРЕКЛАССИФИКАЦИИ
                    # rsi = df.loc[idx, 'RSI_14']
                    # adx = df.loc[idx, 'ADX_14']
                    # macd_hist = df.loc[idx, 'MACD_hist']
                    price_change_3_period = df['close'].pct_change(3).loc[idx]
                    price_change_5_period = df['close'].pct_change(5).loc[idx]
                    
                    # Сила бычьих паттернов (используем уже определенные)
                    bullish_strength = (
                        abs(df.loc[idx, 'CDLHAMMER']) +
                        abs(df.loc[idx, 'CDLENGULFING']) +
                        abs(df.loc[idx, 'CDLINVERTEDHAMMER']) +
                        abs(df.loc[idx, 'CDLDRAGONFLYDOJI']) +
                        abs(df.loc[idx, 'CDLBELTHOLD']) +
                        df.loc[idx, 'hammer_f'] +
                        df.loc[idx, 'inverted_hammer_f'] +
                        df.loc[idx, 'bullish_marubozu_f']
                    )
                    
                    # Сила медвежьих паттернов (используем уже определенные)
                    bearish_strength = (
                        abs(df.loc[idx, 'CDLHANGINGMAN']) +
                        abs(df.loc[idx, 'CDLSHOOTINGSTAR']) +
                        (abs(df.loc[idx, 'CDLENGULFING']) if df.loc[idx, 'CDLENGULFING'] < 0 else 0) + # Медвежье поглощение
                        df.loc[idx, 'hangingman_f'] +
                        df.loc[idx, 'shootingstar_f']
                    )
                    
                    # Переклассификация ТОЛЬКО на основе паттернов + движения цены
                    if (bullish_strength >= 2 and price_change_5_period > 0.008):
                        df.loc[idx, 'target'] = 0  # BUY
                    elif (bearish_strength >= 2 and price_change_5_period < -0.008):
                        df.loc[idx, 'target'] = 1  # SELL
                    elif abs(price_change_3_period) > 0.015:  # Сильное движение цены
                        df.loc[idx, 'target'] = 0 if price_change_3_period > 0 else 1

Замените его на (возвращаем индикаторы):
        # НОВЫЙ КОД - Менее агрессивная переклассификация HOLD
        if current_hold_count > (current_buy_count + current_sell_count) * 3.0:
            print(f"⚠️ Сильный дисбаланс классов. Попытка УМНОЙ переклассификации части HOLD-сигналов (с индикаторами).")
            hold_indices = df[df['target'] == 2].index
            
            import random
            random.seed(42)
            
            reclassify_count = int(current_hold_count * 0.10)
            if reclassify_count > 0:
                reclassify_indices = random.sample(list(hold_indices), min(reclassify_count, len(hold_indices)))
                
                for idx in reclassify_indices:
                    if idx < 5: continue
                    
                    rsi = df.loc[idx, 'RSI_14']
                    adx = df.loc[idx, 'ADX_14']
                    macd_hist = df.loc[idx, 'MACD_hist']
                    price_change_3_period = df['close'].pct_change(3).loc[idx]

                    # Условия для переклассификации (с индикаторами)
                    # 1. RSI + ADX + движение цены
                    if (rsi < 30 and adx > 25 and price_change_3_period > 0.005):
                        df.loc[idx, 'target'] = 0  # BUY
                    elif (rsi > 70 and adx > 25 and price_change_3_period < -0.005):
                        df.loc[idx, 'target'] = 1  # SELL
                        
                    # 2. MACD гистограмма + движение цены
                    elif (macd_hist > 0.002 and price_change_3_period > 0.004):
                        df.loc[idx, 'target'] = 0  # BUY
                    elif (macd_hist < -0.002 and price_change_3_period < -0.004):
                        df.loc[idx, 'target'] = 1  # SELL
                        
                    # 3. Сильный тренд по ADX + движение цены
                    elif (adx > 35 and abs(price_change_3_period) > 0.008):
                        df.loc[idx, 'target'] = 0 if price_change_3_period > 0 else 1
            
            print(f"Баланс классов после УМНОЙ переклассификации (с индикаторами):")
            unique, counts = np.unique(df['target'], return_counts=True)
            class_names = ['BUY', 'SELL', 'HOLD']
            for class_idx, count in zip(unique, counts):
                print(f"  {class_names[class_idx]}: {count} ({count/len(df)*100:.1f}%)")
        else:
            print(f"✅ Баланс классов приемлемый, переклассификация HOLD не требуется.")

4.4. Обучение детектора режимов
Вам также нужно убедиться, что детектор режимов в train_model.py (в функции train_xlstm_rl_system) не пытается использовать закомментированные признаки.
Найдите этот блок:
    # После обучения xlstm_model, обучите детектор режимов
    # Возьмите достаточно большой исторический DataFrame для обучения детектора
    # Например, объедините несколько символов или возьмите один большой
    regime_training_df = pd.concat(list(processed_dfs.values())).reset_index(drop=True)
    decision_maker_temp = HybridDecisionMaker(
        xlstm_model_path='models/xlstm_rl_model.keras',
        rl_agent_path=None,  # <--- ИЗМЕНЕНО: Передаем None, так как RL агент еще не обучен
        feature_columns=feature_cols, # <--- feature_cols здесь должен содержать индикаторы
        sequence_length=X.shape[1]
    )
    decision_maker_temp.fit_regime_detector(regime_training_df, xlstm_model, feature_cols)
    decision_maker_temp.regime_detector.save_detector('models/market_regime_detector.pkl')
    print("✅ Детектор режимов сохранен")

Здесь feature_cols уже будет содержать индикаторы, поэтому это должно работать. Однако, нужно убедиться, что и сам файл market_regime_detector.py также обновлен.
5. Файл: market_regime_detector.py
Вам нужно убедиться, что этот файл использует индикаторы и не пытается использовать паттерны или отключенные признаки.
5.1. Функция extract_regime_features(self, df)
Найдите этот блок:
        # Технические признаки
        df['rsi_regime'] = np.where(df['RSI_14'] > 70, 1, np.where(df['RSI_14'] < 30, -1, 0))
        df['bb_position'] = (df['close'] - df['BBL_20_2.0']) / (df['BBU_20_2.0'] - df['BBL_20_2.0'])
        
        regime_features = [
            'volatility', 'trend_strength', 'volume_trend', 'volume_volatility',
            'rsi_regime', 'bb_position' # ИЗМЕНЕНО: Удалено vsa_activity, vsa_direction
        ]

        # Добавляем xLSTM предсказания как фичи режима
        if self.xlstm_model and self.xlstm_feature_columns and len(df) >= self.xlstm_model.input_shape[1]:
            xlstm_preds = []
            sequence_length = self.xlstm_model.input_shape[1]
            for i in range(len(df) - sequence_length + 1):
                sequence_data = df.iloc[i : i + sequence_length][self.xlstm_feature_columns].values
                sequence_reshaped = sequence_data.reshape(1, sequence_length, len(self.xlstm_feature_columns))
                xlstm_preds.append(self.xlstm_model.predict(sequence_reshaped)[0])
            
            # Заполняем NaN в начале, чтобы выровнять длину
            df['xlstm_buy_pred'] = np.nan
            df['xlstm_sell_pred'] = np.nan
            df['xlstm_hold_pred'] = np.nan
            
            # Начинаем заполнять с индекса, где начинаются предсказания
            start_idx = sequence_length - 1
            df.loc[start_idx:, 'xlstm_buy_pred'] = [p[0] for p in xlstm_preds]
            df.loc[start_idx:, 'xlstm_sell_pred'] = [p[1] for p in xlstm_preds]
            df.loc[start_idx:, 'xlstm_hold_pred'] = [p[2] for p in xlstm_preds]

            regime_features.extend(['xlstm_buy_pred', 'xlstm_sell_pred', 'xlstm_hold_pred'])
        
        return df.dropna(subset=regime_features)

Убедитесь, что он содержит все нужные индикаторы и не содержит паттернов или VSA:
    def extract_regime_features(self, df):
        """Извлекает признаки для определения режима рынка"""
        
        # Ценовые признаки
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(20).std()
        df['trend_strength'] = df['close'].rolling(20).apply(lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0] if x.iloc[0] != 0 else 0)
        
        # Объемные признаки
        df['volume_trend'] = df['volume'].rolling(10).mean() / df['volume'].rolling(30).mean()
        df['volume_volatility'] = df['volume'].rolling(20).std() / df['volume'].rolling(20).mean()
        
        # Технические признаки (раскомментированы)
        # Убедитесь, что эти колонки существуют в DF.
        # Если RSI_14, BBL_20_2.0, BBU_20_2.0 не были созданы в feature_engineering,
        # то вам нужно либо создать их здесь, либо заполнить нулями, либо удалить эти строки.
        # В данном случае, мы их раскомментировали в feature_engineering.py, так что они должны быть.
        if 'RSI_14' in df.columns:
            df['rsi_regime'] = np.where(df['RSI_14'] > 70, 1, np.where(df['RSI_14'] < 30, -1, 0))
        else:
            df['rsi_regime'] = 0 # Заполняем нулями, если колонка отсутствует

        if 'BBL_20_2.0' in df.columns and 'BBU_20_2.0' in df.columns:
            df['bb_position'] = (df['close'] - df['BBL_20_2.0']) / (df['BBU_20_2.0'] - df['BBL_20_2.0'])
        else:
            df['bb_position'] = 0 # Заполняем нулями, если колонки отсутствуют
        
        regime_features = [
            'volatility', 'trend_strength', 'volume_trend', 'volume_volatility',
            'rsi_regime', 'bb_position' # Теперь они активны
        ]

        # Добавляем xLSTM предсказания как фичи режима
        # Этот блок остается без изменений, так как xlstm_feature_columns
        # будет содержать индикаторы.
        if self.xlstm_model and self.xlstm_feature_columns and len(df) >= self.xlstm_model.input_shape[1]:
            xlstm_preds = []
            sequence_length = self.xlstm_model.input_shape[1]
            for i in range(len(df) - sequence_length + 1):
                # Убедитесь, что здесь df содержит все необходимые индикаторы
                sequence_data = df.iloc[i : i + sequence_length][self.xlstm_feature_columns].values
                sequence_reshaped = sequence_data.reshape(1, sequence_length, len(self.xlstm_feature_columns))
                xlstm_preds.append(self.xlstm_model.predict(sequence_reshaped)[0])
            
            # Заполняем NaN в начале, чтобы выровнять длину
            df['xlstm_buy_pred'] = np.nan
            df['xlstm_sell_pred'] = np.nan
            df['xlstm_hold_pred'] = np.nan
            
            # Начинаем заполнять с индекса, где начинаются предсказания
            start_idx = sequence_length - 1
            df.loc[start_idx:, 'xlstm_buy_pred'] = [p[0] for p in xlstm_preds]
            df.loc[start_idx:, 'xlstm_sell_pred'] = [p[1] for p in xlstm_preds]
            df.loc[start_idx:, 'xlstm_hold_pred'] = [p[2] for p in xlstm_preds]

            regime_features.extend(['xlstm_buy_pred', 'xlstm_sell_pred', 'xlstm_hold_pred'])
        
        return df.dropna(subset=regime_features)

Аналогично, в функции fit и predict_regime в market_regime_detector.py убедитесь, что features_to_scale и features_to_predict содержат только активные индикаторы и не содержат паттернов.

🚀 Последовательность действий:
Шаг 1: Сделайте бэкапы ваших файлов!
Это критически важно, чтобы вы могли легко вернуться к предыдущей версии.
cp feature_engineering.py feature_engineering_backup_ind.py
cp run_live_trading.py run_live_trading_backup_ind.py
cp trading_env.py trading_env_backup_ind.py
cp train_model.py train_model_backup_ind.py
cp market_regime_detector.py market_regime_detector_backup_ind.py

Шаг 2: Примените все вышеуказанные изменения
Аккуратно пройдитесь по каждому файлу и замените/раскомментируйте/закомментируйте указанные блоки кода.
Шаг 3: Переобучите модель
Запустите скрипт обучения вашей модели. Убедитесь, что он использует измененные файлы.
python train_model.py --data historical_data.csv

Шаг 4: Анализируйте новые логи обучения
Обратите особое внимание на:

"ТОП-10 ВЛИЯТЕЛЬНЫХ ПРИЗНАКОВ": Убедитесь, что там теперь только индикаторы (включая ATR) и is_event.
Распределение классов: Посмотрите, как изменилось распределение BUY/SELL/HOLD.
Метрики (Precision, Recall) по классам BUY/SELL на тестовой выборке: Сравните с предыдущими результатами.
Разницу между loss и val_loss: Насколько сильно модель переобучается.

Этот эксперимент поможет определить, являются ли индикаторы более эффективными предикторами для вашей модели, чем паттерны, и дадут ли они более сбалансированную производительность по классам BUY/SELL.