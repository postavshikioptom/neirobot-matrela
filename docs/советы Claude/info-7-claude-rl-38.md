📋 Подробная инструкция по отключению индикаторов (комментируем, оставляем ATR_14)
1. Файл: feature_engineering.py
В этом файле мы закомментируем расчет всех технических индикаторов, кроме ATR_14, который нужен для внутренних расчетов признаков паттернов. Также скорректируем создание признака is_event.
1.1. Функция calculate_features(df: pd.DataFrame)
Найдите блок, где рассчитываются индикаторы, и закомментируйте все вызовы talib для RSI, MACD, BBANDS, ADX, STOCH. Оставьте вызов talib.ATR и его обработку.
Найдите этот блок:
        # Add indicators one by one with try-except blocks
        try:
            rsi = talib.RSI(close_p, timeperiod=14)
            rsi[np.isinf(rsi)] = np.nan
            df_out['RSI_14'] = pd.Series(rsi, index=df_out.index).ffill().fillna(0)
        except Exception:
            df_out['RSI_14'] = 0
            
        # ДОБАВЬТЕ ЭТОТ БЛОК для ATR_14
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

Замените его на (комментарии и сохранение ATR):
        # Add indicators one by one with try-except blocks
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
            
        # try:
        #     macd, macdsignal, macdhist = talib.MACD(close_p, fastperiod=12, slowperiod=26, signalperiod=9)
        #     df_out['MACD_12_26_9'] = pd.Series(macd, index=df_out.index).replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
        #     df_out['MACD_signal'] = pd.Series(macdsignal, index=df_out.index).replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
        #     df_out['MACD_hist'] = pd.Series(macdhist, index=df_out.index).replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
        # except Exception:
        #     df_out['MACD_12_26_9'], df_out['MACD_signal'], df_out['MACD_hist'] = 0, 0, 0

        # try:
        #     upper, middle, lower = talib.BBANDS(close_p, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        #     df_out['BBU_20_2.0'] = pd.Series(upper, index=df_out.index).replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
        #     df_out['BBM_20_2.0'] = pd.Series(middle, index=df_out.index).replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
        #     df_out['BBL_20_2.0'] = pd.Series(lower, index=df_out.index).replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
        # except Exception:
        #     df_out['BBU_20_2.0'], df_out['BBM_20_2.0'], df_out['BBL_20_2.0'] = 0, 0, 0

        # try:
        #     adx = talib.ADX(high_p, low_p, close_p, timeperiod=14)
        #     adx[np.isinf(adx)] = np.nan
        #     df_out['ADX_14'] = pd.Series(adx, index=df_out.index).ffill().fillna(0)
        # except Exception:
        #     df_out['ADX_14'] = 0

        # try:
        #     slowk, slowd = talib.STOCH(high_p, low_p, close_p, fastk_period=14, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
        #     df_out['STOCHk_14_3_3'] = pd.Series(slowk, index=df_out.index).replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
        #     df_out['STOCHd_14_3_3'] = pd.Series(slowd, index=df_out.index).replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
        # except Exception:
        #     df_out['STOCHk_14_3_3'], df_out['STOCHd_14_3_3'] = 0, 0

1.2. Создание признака is_event
В той же функции calculate_features, найдите блок, где создается is_event. Поскольку мы отключаем индикаторы, is_event должен полагаться только на объем.
Найдите этот блок:
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

Замените его на (только объем):
        # Убедимся, что все нужные колонки существуют (ATR_14 уже добавлен)
        # Поскольку другие индикаторы отключены, is_event будет использовать только объем.
        # required_cols = ['volume', 'ATR_14', 'RSI_14', 'ADX_14'] # ❌ Закомментировано
        required_cols = ['volume'] # ✅ Только объем для is_event
        for col in required_cols:
            if col not in df_out.columns:
                df_out[col] = 0 # Заполняем нулями, если вдруг нет

        df_out['is_event'] = (
            (df_out['volume'] > df_out['volume'].rolling(50).quantile(0.9).fillna(0)) # Объем > 90% квантиля
            # | (df_out['ATR_14'] > df_out['ATR_14'].rolling(50).quantile(0.9).fillna(0)) # ❌ Закомментировано
            # | (abs(df_out['RSI_14'] - 50) > 25) # ❌ Закомментировано
            # | (df_out['ADX_14'] > df_out['ADX_14'].shift(5).fillna(0) + 2) # ❌ Закомментировано
        ).astype(int)

1.3. Функция prepare_xlstm_rl_features(df: pd.DataFrame)
В этой функции мы определяем, какие признаки будут переданы в xLSTM модель. Закомментируйте все технические индикаторы, кроме ATR, который также не будет передаваться в модель напрямую, а только использоваться внутри паттернов.
Найдите этот блок:
    feature_cols = [
        # Технические индикаторы
        'RSI_14', 'MACD_12_26_9', 'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0',
        'ADX_14', 'STOCHk_14_3_3', 'STOCHd_14_3_3', 'ATR_14',
        # Базовые паттерны TA-Lib
        'CDLHAMMER', 'CDLENGULFING', 'CDLDOJI', 'CDLSHOOTINGSTAR',
        'CDLHANGINGMAN', 'CDLMARUBOZU', # CDLMARUBOZU теперь используется для бычьего
        # НОВЫЕ БЫЧЬИ ПАТТЕРНЫ TA-Lib (без CDLBULLISHKICKING)
        'CDLINVERTEDHAMMER', 'CDLDRAGONFLYDOJI', 'CDLBELTHOLD',
        # Комбинированные признаки паттернов
        'hammer_f', 'hangingman_f', 'engulfing_f', 'doji_f',
        'shootingstar_f',
        'bullish_marubozu_f', # ИЗМЕНЕНО: Используем новый комбинированный признак
        # НОВЫЕ КОМБИНИРОВАННЫЕ ПРИЗНАКИ БЫЧЬИХ ПАТТЕРНОВ (без bullish_kicker_f)
        'inverted_hammer_f', 'dragonfly_doji_f', 'bullish_pin_bar_f', 'bullish_belt_hold_f',
        'is_event'
    ]

Замените его на (комментарии):
    feature_cols = [
        # ❌ ОТКЛЮЧЕНЫ ВСЕ ТЕХНИЧЕСКИЕ ИНДИКАТОРЫ (кроме ATR, который используется внутри паттернов, но не как отдельный признак)
        # 'RSI_14', 'MACD_12_26_9', 'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0',
        # 'ADX_14', 'STOCHk_14_3_3', 'STOCHd_14_3_3',
        # 'ATR_14', # ATR_14 рассчитывается, но не передается в модель как отдельный признак
        
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

2. Файл: run_live_trading.py
Здесь мы изменим глобальный список FEATURE_COLUMNS и вернем calculate_dynamic_stops к использованию ATR_14.
2.1. Глобальная переменная FEATURE_COLUMNS
Найдите этот блок:
# 🔥 НОВЫЕ КОЛОНКИ ПРИЗНАКОВ С VSA
FEATURE_COLUMNS = [
    # Технические индикаторы
    'RSI_14', 'MACD_12_26_9', 'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0',
    'ADX_14', 'STOCHk_14_3_3', 'STOCHd_14_3_3',
    'ATR_14', # <--- ДОБАВЛЕНО
    # Паттерны
    'CDLHAMMER', 'CDLENGULFING', 'CDLDOJI', 'CDLSHOOTINGSTAR',
    'CDLHANGINGMAN', 'CDLMARUBOZU',
]

Замените его на (комментарии):
# 🔥 НОВЫЕ FEATURE_COLUMNS - ТОЛЬКО ПАТТЕРНЫ (для HybridDecisionMaker)
FEATURE_COLUMNS = [
    # ❌ ОТКЛЮЧЕНЫ ВСЕ ТЕХНИЧЕСКИЕ ИНДИКАТОРЫ (кроме ATR, который используется для динамических стопов)
    # 'RSI_14', 'MACD_12_26_9', 'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0',
    # 'ADX_14', 'STOCHk_14_3_3', 'STOCHd_14_3_3',
    # 'ATR_14', # ATR_14 используется для динамических стопов, но не подается в модель как отдельный признак
    
    # ✅ ТОЛЬКО БАЗОВЫЕ ПАТТЕРНЫ TA-Lib
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

2.2. Функция calculate_dynamic_stops(features_row, position_side, entry_price)
Эта функция должна использовать ATR_14, поскольку мы его не отключали. Верните ее к исходному виду.
Найдите этот блок (предполагаю, что вы его изменили по прошлой инструкции):
def calculate_dynamic_stops(features_row, position_side, entry_price):
    """
    Динамические стоп-лоссы БЕЗ ATR - используем фиксированные значения
    """
    base_sl = STOP_LOSS_PCT
    base_tp = TAKE_PROFIT_PCT
    
    # Сила паттернов
    pattern_strength = (
        abs(features_row.get('CDLHAMMER', 0)) +
        abs(features_row.get('CDLENGULFING', 0)) +
        features_row.get('hammer_f', 0) +
        features_row.get('engulfing_f', 0)
    )
    
    if pattern_strength > 2:
        dynamic_sl = base_sl * 1.2  # Увеличиваем SL на 20%
        dynamic_tp = base_tp * 1.1  # Увеличиваем TP на 10%
    else:
        dynamic_sl, dynamic_tp = base_sl, base_tp
        
    # Ограничиваем максимальные значения
    dynamic_sl = max(dynamic_sl, -2.5)
    dynamic_tp = min(dynamic_tp, 2.5)

    return dynamic_sl, dynamic_tp

Верните его к исходному виду (с использованием ATR):
def calculate_dynamic_stops(features_row, position_side, entry_price):
    """
    Вычисляет динамические стоп-лоссы на основе волатильности (с ATR)
    """
    base_sl = STOP_LOSS_PCT
    base_tp = TAKE_PROFIT_PCT
    
    # Корректировка на основе волатильности (ATR)
    atr = features_row.get('ATR_14', 0)
    close_price = features_row.get('close', entry_price)
    
    if close_price > 0:
        atr_pct = (atr / close_price) * 100
    else:
        atr_pct = 0

    # Если ATR большой, делаем стопы шире
    if atr_pct > 0.5: # Если ATR > 0.5% от цены
        dynamic_sl = base_sl * (1 + atr_pct) # Увеличиваем SL
        dynamic_tp = base_tp * (1 - atr_pct / 2) # Слегка уменьшаем TP
    else:
        dynamic_sl, dynamic_tp = base_sl, base_tp
        
    # Ограничиваем максимальные и минимальные значения
    dynamic_sl = max(dynamic_sl, -3.0) # Не больше -3%
    dynamic_tp = min(dynamic_tp, 3.0) # Не больше +3%

    return dynamic_sl, dynamic_tp

3. Файл: trading_env.py
Здесь мы скорректируем список признаков для RL-среды и логику наград.
3.1. Функция reset(self, seed=None, options=None)
Найдите этот блок:
        # 🔥 НОВЫЕ FEATURE_COLUMNS - ТОЛЬКО ПАТТЕРНЫ
        self.feature_columns = [
            # ❌ ВСЕ ИНДИКАТОРЫ ОТКЛЮЧЕНЫ
            # 'RSI_14', 'MACD_12_26_9', 'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0',
            # 'ADX_14', 'STOCHk_14_3_3', 'STOCHd_14_3_3', 'ATR_14',
            
            # ✅ ТОЛЬКО ПАТТЕРНЫ
            'CDLHAMMER', 'CDLENGULFING', 'CDLDOJI', 'CDLSHOOTINGSTAR',
            'CDLHANGINGMAN', 'CDLMARUBOZU', # CDLMARUBOZU теперь используется для бычьего
            # НОВЫЕ БЫЧЬИ ПАТТЕРНЫ TA-Lib (без CDLBULLISHKICKING)
            'CDLINVERTEDHAMMER', 'CDLDRAGONFLYDOJI', 'CDLBELTHOLD',
            # Комбинированные признаки паттернов
            'hammer_f', 'hangingman_f', 'engulfing_f', 'doji_f',
            'shootingstar_f', 'bullish_marubozu_f', # ИЗМЕНЕНО: Используем новый комбинированный признак
            # НОВЫЕ КОМБИНИРОВАННЫЕ ПРИЗНАКИ БЫЧЬИХ ПАТТЕРНОВ (без bullish_kicker_f)
            'inverted_hammer_f', 'dragonfly_doji_f', 'bullish_pin_bar_f', 'bullish_belt_hold_f',
            'is_event'
        ]

Замените его на (комментарии):
        # 🔥 НОВЫЕ FEATURE_COLUMNS - ТОЛЬКО ПАТТЕРНЫ (для RL среды)
        self.feature_columns = [
            # ❌ ВСЕ ИНДИКАТОРЫ ОТКЛЮЧЕНЫ
            # 'RSI_14', 'MACD_12_26_9', 'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0',
            # 'ADX_14', 'STOCHk_14_3_3', 'STOCHd_14_3_3',
            # 'ATR_14', # ATR_14 рассчитывается, но не передается в RL среду как отдельный признак
            
            # ✅ ТОЛЬКО БАЗОВЫЕ ПАТТЕРНЫ TA-Lib
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

3.2. Функция _calculate_advanced_reward(self, action, pnl_pct, xlstm_prediction)
Здесь мы полностью уберем логику, основанную на индикаторах, для расчета штрафов за overtrading и наград за HOLD.
Найдите этот блок:
        # 🔥 УБИРАЕМ АНАЛИЗ ИНДИКАТОРОВ ДЛЯ OVERTRADING
        # Вместо индикаторов используем только паттерны
        if action != 2:  # Если не HOLD
            current_row = self.df.iloc[self.current_step]
            
            # Считаем силу паттернов вместо индикаторов
            bullish_pattern_strength = (
                abs(current_row.get('CDLHAMMER', 0)) +
                abs(current_row.get('CDLENGULFING', 0)) +
                current_row.get('hammer_f', 0) +
                current_row.get('bullish_marubozu_f', 0)
            )
            
            bearish_pattern_strength = (
                abs(current_row.get('CDLHANGINGMAN', 0)) +
                abs(current_row.get('CDLSHOOTINGSTAR', 0)) +
                current_row.get('hangingman_f', 0) +
                current_row.get('shootingstar_f', 0)
            )
            
            # Штраф за торговлю без сильных паттернов
            if action == 1 and bullish_pattern_strength < 2:  # BUY без бычьих паттернов
                overtrading_penalty = -1.0
            elif action == 0 and bearish_pattern_strength < 2:  # SELL без медвежьих паттернов
                overtrading_penalty = -1.0
        else: # HOLD
            hold_reward = 0.1 # Небольшая награда за удержание

Замените его на (комментарии и использование паттернов):
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

4. Файл: train_model.py
Это самый важный файл для изменений. Мы скорректируем список признаков, логику генерации целевых меток и, главное, блок переклассификации HOLD.
4.1. Функция prepare_xlstm_rl_data(data_path, sequence_length=10)
Здесь мы определяем, какие признаки будут использоваться для обучения xLSTM модели, и как будут генерироваться целевые метки.
Найдите этот блок:
    feature_cols = [
        # Технические индикаторы
        'RSI_14', 'MACD_12_26_9', 'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0',
        'ADX_14', 'STOCHk_14_3_3', 'STOCHd_14_3_3', 'ATR_14',
        # Базовые паттерны TA-Lib
        'CDLHAMMER', 'CDLENGULFING', 'CDLDOJI', 'CDLSHOOTINGSTAR',
        'CDLHANGINGMAN', 'CDLMARUBOZU', # CDLMARUBOZU теперь используется для бычьего
        # НОВЫЕ БЫЧЬИ ПАТТЕРНЫ TA-Lib (без CDLBULLISHKICKING)
        'CDLINVERTEDHAMMER', 'CDLDRAGONFLYDOJI', 'CDLBELTHOLD',
        # Комбинированные признаки паттернов
        'hammer_f', 'hangingman_f', 'engulfing_f', 'doji_f',
        'shootingstar_f',
        'bullish_marubozu_f', # ИЗМЕНЕНО: Используем новый комбинированный признак
        # НОВЫЕ КОМБИНИРОВАННЫЕ ПРИЗНАКИ БЫЧЬИХ ПАТТЕРНОВ (без bullish_kicker_f)
        'inverted_hammer_f', 'dragonfly_doji_f', 'bullish_pin_bar_f', 'bullish_belt_hold_f',
        'is_event'
    ]

Замените его на (комментарии):
    feature_cols = [
        # ❌ ОТКЛЮЧЕНЫ ВСЕ ТЕХНИЧЕСКИЕ ИНДИКАТОРЫ (кроме ATR, который используется внутри паттернов)
        # 'RSI_14', 'MACD_12_26_9', 'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0',
        # 'ADX_14', 'STOCHk_14_3_3', 'STOCHd_14_3_3',
        # 'ATR_14', # ATR_14 рассчитывается, но не передается в модель как отдельный признак
        
        # ✅ ТОЛЬКО БАЗОВЫЕ ПАТТЕРНЫ TA-Lib
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

4.2. Генерация целевых меток BUY/SELL
Здесь мы полностью уберем зависимость от индикаторов для определения buy_condition и sell_condition.
Найдите этот блок:
        # Классические технические фильтры
        strong_trend = df['ADX_14'] > 25
        
        # Условия для BUY
        rsi_buy_zone = df['RSI_14'] < 30 # ИЗМЕНЕНО: С 38 до 30
        macd_buy_signal = (df['MACD_12_26_9'] > df['MACD_signal']) & \
                          (df['MACD_hist'] > 0.001) # ИЗМЕНЕНО: Убрано .shift(1) для простоты, MACD_hist > 0.001
        
        # Условия для SELL
        rsi_sell_zone = df['RSI_14'] > 70 # ИЗМЕНЕНО: С 62 до 70
        macd_sell_signal = (df['MACD_12_26_9'] < df['MACD_signal']) & \
                           (df['MACD_hist'] < -0.001) # ИЗМЕНЕНО: Убрано .shift(1) для простоты, MACD_hist < -0.001

        # Условия для BUY/SELL только на основе future_return и классических индикаторов/паттернов
        # Делаем условия более строгими для обоих классов (AND), чтобы получить более чистые сигналы
        buy_condition = (
            (df['future_return'] > df['dynamic_threshold'] * 1.0) & # Оставляем порог 1.0
            (strong_trend & rsi_buy_zone & macd_buy_signal) # ИЗМЕНЕНО: Все условия должны быть выполнены для BUY
        )

        sell_condition = (
            (df['future_return'] < -df['dynamic_threshold'] * 1.0) & # Оставляем порог 1.0
            (strong_trend & rsi_sell_zone & macd_sell_signal) # ИЗМЕНЕНО: Все условия должны быть выполнены для SELL
        )

Замените его на (комментарии и только паттерны):
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

4.3. Блок переклассификации HOLD
Это была критическая проблема, которую нужно исправить. Мы полностью закомментируем использование индикаторов в этом блоке.
Найдите этот блок:
                for idx in reclassify_indices:
                    if idx < 5: continue
                    
                    rsi = df.loc[idx, 'RSI_14']
                    adx = df.loc[idx, 'ADX_14']
                    macd_hist = df.loc[idx, 'MACD_hist']
                    price_change_3_period = df['close'].pct_change(3).loc[idx]

                    # Условия для переклассификации (без VSA), делаем их более строгими
                    # 1. RSI + ADX + движение цены
                    if (rsi < 30 and adx > 25 and price_change_3_period > 0.005): # ИЗМЕНЕНО: Более строгие условия
                        df.loc[idx, 'target'] = 0  # BUY
                    elif (rsi > 70 and adx > 25 and price_change_3_period < -0.005): # ИЗМЕНЕНО: Более строгие условия
                        df.loc[idx, 'target'] = 1  # SELL
                        
                    # 2. MACD гистограмма + движение цены
                    elif (macd_hist > 0.002 and price_change_3_period > 0.004): # ИЗМЕНЕНО: Более строгие условия
                        df.loc[idx, 'target'] = 0  # BUY
                    elif (macd_hist < -0.002 and price_change_3_period < -0.004): # ИЗМЕНЕНО: Более строгие условия
                        df.loc[idx, 'target'] = 1  # SELL
                        
                    # 3. Сильный тренд по ADX + движение цены
                    elif (adx > 35 and abs(price_change_3_period) > 0.008): # ИЗМЕНЕНО: Более строгие условия
                        df.loc[idx, 'target'] = 0 if price_change_3_period > 0 else 1

Замените его на (комментарии и использование только паттернов и цены):
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

