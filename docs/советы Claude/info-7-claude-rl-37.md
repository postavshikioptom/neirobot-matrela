Проверил ваш код и вижу серьезную проблему - реализация не соответствует предложенному плану! ❌
🚨 Основные проблемы:
1. В feature_engineering.py
❌ НЕ ВЫПОЛНЕНО: Функция calculate_features() ВСЕ ЕЩЕ ВЫЧИСЛЯЕТ ВСЕ ИНДИКАТОРЫ:
# ЭТИ СТРОКИ ДОЛЖНЫ БЫТЬ ЗАКОММЕНТИРОВАНЫ!
rsi = talib.RSI(close_p, timeperiod=14)  # ❌ ВСЕ ЕЩЕ АКТИВНО
atr = talib.ATR(high_p, low_p, close_p, timeperiod=14)  # ❌ ВСЕ ЕЩЕ АКТИВНО
macd, macdsignal, macdhist = talib.MACD(...)  # ❌ ВСЕ ЕЩЕ АКТИВНО

2. В train_model.py
❌ НЕ ВЫПОЛНЕНО: feature_cols ВСЕ ЕЩЕ СОДЕРЖИТ ВСЕ ИНДИКАТОРЫ:
feature_cols = [
    # ❌ ЭТИ СТРОКИ ДОЛЖНЫ БЫТЬ ЗАКОММЕНТИРОВАНЫ!
    'RSI_14', 'MACD_12_26_9', 'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0',
    'ADX_14', 'STOCHk_14_3_3', 'STOCHd_14_3_3', 'ATR_14',

3. Переклассификация все еще использует индикаторы
❌ НЕ ВЫПОЛНЕНО: В коде все еще используются RSI_14, ADX_14, MACD_hist для переклассификации.
✅ Правильная реализация:
1. Исправьте feature_engineering.py:
def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    ТОЛЬКО ПАТТЕРНЫ - ВСЕ ИНДИКАТОРЫ ОТКЛЮЧЕНЫ
    """
    try:
        if df is None or df.empty:
            return pd.DataFrame()

        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df.dropna(subset=numeric_cols, inplace=True)
        if df.empty:
            return pd.DataFrame()

        # Масштабирование объема (оставляем)
        if 'volume' in df.columns and not df['volume'].empty:
            scaler_volume = MinMaxScaler(feature_range=(0, 100))
            df['volume_scaled'] = scaler_volume.fit_transform(df[['volume']].values)
            df['volume'] = df['volume_scaled']
            df.drop(columns=['volume_scaled'], inplace=True, errors='ignore')
            print("✅ Объем масштабирован, индикаторы ОТКЛЮЧЕНЫ")

        df_out = df.copy()

        # 🔥 ВСЕ ИНДИКАТОРЫ ЗАКОММЕНТИРОВАНЫ
        # try:
        #     rsi = talib.RSI(close_p, timeperiod=14)
        #     df_out['RSI_14'] = pd.Series(rsi, index=df_out.index).ffill().fillna(0)
        # except Exception:
        #     df_out['RSI_14'] = 0
        
        # try:
        #     atr = talib.ATR(high_p, low_p, close_p, timeperiod=14)
        #     df_out['ATR_14'] = pd.Series(atr, index=df_out.index).ffill().fillna(0)
        # except Exception:
        #     df_out['ATR_14'] = 0

        # И так далее для всех остальных индикаторов...

        # 🔥 СОЗДАЕМ is_event БЕЗ ИНДИКАТОРОВ - на основе только объема и паттернов
        df_out['is_event'] = (
            (df_out['volume'] > df_out['volume'].rolling(50).quantile(0.9).fillna(0))
        ).astype(int)

        return df_out

    except Exception as e:
        print(f"FATAL ERROR in calculate_features: {e}")
        return pd.DataFrame()

2. Исправьте train_model.py:
def prepare_xlstm_rl_data(data_path, sequence_length=10):
    """
    ТОЛЬКО ПАТТЕРНЫ - ВСЕ ИНДИКАТОРЫ ОТКЛЮЧЕНЫ  
    """
    print(f"Загрузка данных из {data_path}...")
    full_df = pd.read_csv(data_path)
    
    # 🔥 ТОЛЬКО ПАТТЕРНЫ
    feature_cols = [
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

    # ... остальной код до переклассификации

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
                
                # 🔥 ТОЛЬКО ДВИЖЕНИЕ ЦЕН И ПАТТЕРНЫ
                price_change_3_period = df['close'].pct_change(3).loc[idx]
                price_change_5_period = df['close'].pct_change(5).loc[idx]
                
                # Сила бычьих паттернов
                bullish_strength = (
                    abs(df.loc[idx, 'CDLHAMMER']) +
                    abs(df.loc[idx, 'CDLENGULFING']) +
                    df.loc[idx, 'hammer_f'] +
                    df.loc[idx, 'bullish_marubozu_f']
                )
                
                # Сила медвежьих паттернов  
                bearish_strength = (
                    abs(df.loc[idx, 'CDLHANGINGMAN']) +
                    abs(df.loc[idx, 'CDLSHOOTINGSTAR']) +
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

3. Полностью удалите функции, использующие индикаторы:
# В calculate_dynamic_stops - используйте ТОЛЬКО паттерны:
def calculate_dynamic_stops(features_row, position_side, entry_price):
    """
    БЕЗ ATR - только паттерны
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
        dynamic_sl = base_sl * 1.2
        dynamic_tp = base_tp * 1.1
    else:
        dynamic_sl, dynamic_tp = base_sl, base_tp
        
    return max(dynamic_sl, -2.5), min(dynamic_tp, 2.5)
