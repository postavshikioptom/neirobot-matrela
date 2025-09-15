


🚀 Обновленный план действий:
Шаг 1: Модифицируем train_model.py
1.1. Изменяем целевое соотношение в imblearn
Найдите этот блок:
        # Рассчитываем целевые количества на основе общего числа примеров
        # Целевое соотношение: 20% BUY, 20% SELL, 60% HOLD (более агрессивный oversampling)
        total_samples = len(X)
        target_buy_count = int(total_samples * 0.20)
        target_sell_count = int(total_samples * 0.20)

Замените его на (15% BUY, 15% SELL, 70% HOLD):
        # Рассчитываем целевые количества на основе общего числа примеров
        # Целевое соотношение: 15% BUY, 15% SELL, 70% HOLD (более реалистичный oversampling)
        total_samples = len(X)
        target_buy_count = int(total_samples * 0.15) # 🔥 ИЗМЕНЕНО: с 0.20 до 0.15
        target_sell_count = int(total_samples * 0.15) # 🔥 ИЗМЕНЕНО: с 0.20 до 0.15

1.2. Добавляем больше логов в prepare_xlstm_rl_data
Добавим логи после определения buy_condition, sell_condition и df['target'], чтобы видеть исходное количество сигналов до imblearn.
Найдите этот блок (после определения df['target']):
        # Устанавливаем метки
        df['target'] = 2  # По умолчанию HOLD
        df.loc[buy_condition, 'target'] = 0  # BUY
        df.loc[sell_condition, 'target'] = 1  # SELL

        # ДОБАВЬТЕ: Принудительная балансировка классов (если необходимо)
        # Этот блок можно включать, если после ослабления порогов баланс все еще очень плохой.
        # Он попытается переклассифицировать часть "HOLD" в BUY/SELL на основе других индикаторов.
        # Это может быть "грязным" решением, но иногда необходимо для обучения.
        current_buy_count = (df['target'] == 0).sum()
        current_sell_count = (df['target'] == 1).sum()
        current_hold_count = (df['target'] == 2).sum()

Добавьте логи перед этим блоком:
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

Шаг 2: Модифицируем feature_engineering.py
2.1. Отключаем Боллинджер в calculate_features
Найдите блок с расчетом Боллинджера:
        try:
            upper, middle, lower = talib.BBANDS(close_p, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
            df_out['BBU_20_2.0'] = pd.Series(upper, index=df_out.index).replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
            df_out['BBM_20_2.0'] = pd.Series(middle, index=df_out.index).replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
            df_out['BBL_20_2.0'] = pd.Series(lower, index=df_out.index).replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
        except Exception:
            df_out['BBU_20_2.0'], df_out['BBM_20_2.0'], df_out['BBL_20_2.0'] = 0, 0, 0

Закомментируйте его полностью:
        # 🔥 ЗАКОММЕНТИРОВАНО: Боллинджер (BBU, BBM, BBL)
        # try:
        #     upper, middle, lower = talib.BBANDS(close_p, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        #     df_out['BBU_20_2.0'] = pd.Series(upper, index=df_out.index).replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
        #     df_out['BBM_20_2.0'] = pd.Series(middle, index=df_out.index).replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
        #     df_out['BBL_20_2.0'] = pd.Series(lower, index=df_out.index).replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
        # except Exception:
        #     df_out['BBU_20_2.0'], df_out['BBM_20_2.0'], df_out['BBL_20_2.0'] = 0, 0, 0

2.2. Удаляем Боллинджер из required_cols для is_event
Найдите этот блок:
        # Убедимся, что все нужные колонки существуют (ATR_14 уже добавлен)
        required_cols = ['volume', 'ATR_14', 'RSI_14', 'ADX_14']
        for col in required_cols:
            if col not in df_out.columns:
                df_out[col] = 0 # Заполняем нулями, если вдруг нет

Измените его на (удаляем BBL/BBM/BBU, они не используются):
        # Убедимся, что все нужные колонки существуют (ATR_14 уже добавлен)
        required_cols = ['volume', 'ATR_14', 'RSI_14', 'ADX_14'] # 🔥 BBL/BBM/BBU удалены из списка
        for col in required_cols:
            if col not in df_out.columns:
                df_out[col] = 0 # Заполняем нулями, если вдруг нет

Шаг 3: Модифицируем run_live_trading.py
3.1. Удаляем Боллинджер из FEATURE_COLUMNS
Найдите глобальный список FEATURE_COLUMNS:
FEATURE_COLUMNS = [
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

Измените его на (удаляем BBL/BBM/BBU):
FEATURE_COLUMNS = [
    # ✅ ВСЕ ТЕХНИЧЕСКИЕ ИНДИКАТОРЫ (БЕЗ БОЛЛИНДЖЕРА)
    'RSI_14', 'MACD_12_26_9', 'MACD_signal', 'MACD_hist',
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

Шаг 4: Модифицируем trading_env.py
4.1. Удаляем Боллинджер из self.feature_columns в reset
Найдите список self.feature_columns:
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

Измените его на (удаляем BBL/BBM/BBU):
        # 🔥 НОВЫЕ FEATURE_COLUMNS - ТОЛЬКО ИНДИКАТОРЫ (для RL среды)
        self.feature_columns = [
            # ✅ ВСЕ ТЕХНИЧЕСКИЕ ИНДИКАТОРЫ (БЕЗ БОЛЛИНДЖЕРА)
            'RSI_14', 'MACD_12_26_9', 'MACD_signal', 'MACD_hist',
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

Шаг 5: Модифицируем train_model.py
5.1. Удаляем Боллинджер из feature_cols в prepare_xlstm_rl_data
Найдите список feature_cols:
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

Измените его на (удаляем BBL/BBM/BBU):
    feature_cols = [
        # ✅ ВСЕ ТЕХНИЧЕСКИЕ ИНДИКАТОРЫ (БЕЗ БОЛЛИНДЖЕРА)
        'RSI_14', 'MACD_12_26_9', 'MACD_signal', 'MACD_hist',
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

5.2. Удаляем Боллинджер из features_to_scale в market_regime_detector.py
Найдите список features_to_scale:
        features_to_scale = [
            'volatility', 'trend_strength', 'volume_trend', 'volume_volatility',
            'rsi_regime', 'bb_position' # ИЗМЕНЕНО: Удалено vsa_activity, vsa_direction
        ]

Измените его на (удаляем bb_position):
        features_to_scale = [
            'volatility', 'trend_strength', 'volume_trend', 'volume_volatility',
            'rsi_regime' # 🔥 bb_position удален
        ]

И, конечно, убедитесь, что df['bb_position'] не рассчитывается в extract_regime_features (он был удален в предыдущей инструкции, но стоит перепроверить).
5.3. Обновление feature_cols в DetailedProgressCallback
В train_xlstm_rl_system, в инициализации DetailedProgressCallback, убедитесь, что feature_cols передается обновленный список. Это уже должно быть так, поскольку feature_cols определяется выше.
