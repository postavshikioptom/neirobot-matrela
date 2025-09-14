import pandas as pd
import talib
from functools import lru_cache

import numpy as np
from sklearn.preprocessing import MinMaxScaler # <--- ДОБАВЛЕНО: Импорт MinMaxScaler

# --- Helper functions for pattern features from info-4-patterns.md ---

# 🔥 ЗАКОММЕНТИРОВАНО: ATR функции
# @lru_cache(maxsize=128)
# def cached_calculate_atr(high_tuple, low_tuple, close_tuple, period=14):
#     """Кэшированная версия расчета ATR"""
#     high = np.array(high_tuple)
#     low = np.array(low_tuple)
#     close = np.array(close_tuple)
#     return talib.ATR(high, low, close, timeperiod=period)

# def calculate_atr(high, low, close, period=14):
#     return talib.ATR(high, low, close, timeperiod=period)

def calculate_awesome_oscillator(high, low):
    """Calculates Awesome Oscillator (AO)"""
    median_price = (high + low) / 2
    short_sma = talib.SMA(median_price, timeperiod=5)
    long_sma = talib.SMA(median_price, timeperiod=34)
    return short_sma - long_sma

def calculate_volume_ratio(volume, window=20):
    """Calculates the ratio of the current volume to its moving average."""
    volume_sma = volume.rolling(window=window, min_periods=1).mean()
    # Avoid division by zero
    volume_sma = volume_sma.replace(0, 1)
    return volume / volume_sma

def find_support_resistance(low, high, window=20):
    """Finds support and resistance levels in a given window."""
    support = low.rolling(window=window, min_periods=1).min()
    resistance = high.rolling(window=window, min_periods=1).max()
    return support, resistance

def is_on_level(price, level, atr, threshold=0.3):
    """Checks if the price is close to a given level, based on ATR."""
    return (abs(price - level) < atr * threshold).astype(int)

def is_volume_spike(volume_ratio, threshold=1.5):
    """Checks for a significant volume spike."""
    return (volume_ratio > threshold).astype(int)

# НОВЫЙ КОД - Вспомогательные функции для анализа свечей
def get_body_size(open_p, close_p):
    return abs(close_p - open_p)

def get_total_range(high_p, low_p):
    return high_p - low_p

def get_upper_shadow(open_p, high_p, close_p):
    return high_p - np.maximum(open_p, close_p)

def get_lower_shadow(open_p, low_p, close_p):
    return np.minimum(open_p, close_p) - low_p

def is_small_body(open_p, close_p, high_p, low_p, threshold_factor=0.2):
    body = get_body_size(open_p, close_p)
    total_range = get_total_range(high_p, low_p)
    return (body < total_range * threshold_factor).astype(int)

def is_long_shadow(shadow_size, body_size, threshold_factor=2.0):
    return (shadow_size > body_size * threshold_factor).astype(int)



def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates technical indicators and features for the given DataFrame.
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

        df_out = df.copy()
        
        high_p = df['high'].values
        low_p = df['low'].values
        close_p = df['close'].values

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

        return df_out

    except Exception as e:
        print(f"FATAL ERROR in calculate_features: {e}")
        return pd.DataFrame()

# --- Feature extraction functions for each pattern ---

def hammer_features(df):
    atr = calculate_atr(df['high'], df['low'], df['close'])
    support, resistance = find_support_resistance(df['low'], df['high'])
    volume_ratio = calculate_volume_ratio(df['volume'])
    
    features = pd.DataFrame(index=df.index)
    features['hammer_f_on_support'] = is_on_level(df['close'], support, atr)
    features['hammer_f_vol_spike'] = is_volume_spike(volume_ratio)
    return features

def hangingman_features(df):
    atr = calculate_atr(df['high'], df['low'], df['close'])
    support, resistance = find_support_resistance(df['low'], df['high'])
    volume_ratio = calculate_volume_ratio(df['volume'])
    
    features = pd.DataFrame(index=df.index)
    features['hangingman_f_on_res'] = is_on_level(df['close'], resistance, atr)
    features['hangingman_f_vol_spike'] = is_volume_spike(volume_ratio)
    return features

def engulfing_features(df):
    atr = calculate_atr(df['high'], df['low'], df['close'])
    volume_ratio = calculate_volume_ratio(df['volume'])
    
    features = pd.DataFrame(index=df.index)
    body_size = (df['close'] - df['open']).abs()
    features['engulfing_f_strong'] = (body_size / df['open'] > 0.02).astype(int)
    features['engulfing_f_vol_confirm'] = is_volume_spike(volume_ratio, 1.2)
    return features

def doji_features(df):
    atr = calculate_atr(df['high'], df['low'], df['close'])
    volume_ratio = calculate_volume_ratio(df['volume'])
    
    features = pd.DataFrame(index=df.index)
    features['doji_f_high_vol'] = is_volume_spike(volume_ratio, 1.5)
    features['doji_f_high_atr'] = (atr > atr.rolling(20).mean() * 1.2).astype(int)
    return features

# НОВЫЙ КОД - Функции для извлечения признаков бычьих паттернов
def inverted_hammer_features(df):
    atr = calculate_atr(df['high'], df['low'], df['close'])
    support, _ = find_support_resistance(df['low'], df['high'])
    volume_ratio = calculate_volume_ratio(df['volume'])
    
    features = pd.DataFrame(index=df.index)
    features['ih_f_small_body'] = is_small_body(df['open'], df['close'], df['high'], df['low'])
    features['ih_f_long_upper_shadow'] = is_long_shadow(get_upper_shadow(df['open'], df['high'], df['close']), get_body_size(df['open'], df['close']))
    features['ih_f_on_support'] = is_on_level(df['low'], support, atr, threshold=0.5) # Немного шире для поддержки
    features['ih_f_vol_confirm'] = is_volume_spike(volume_ratio, 1.2)
    return features

def dragonfly_doji_features(df):
    atr = calculate_atr(df['high'], df['low'], df['close'])
    support, _ = find_support_resistance(df['low'], df['high'])
    volume_ratio = calculate_volume_ratio(df['volume'])
    
    features = pd.DataFrame(index=df.index)
    features['dd_f_long_lower_shadow'] = is_long_shadow(get_lower_shadow(df['open'], df['low'], df['close']), get_body_size(df['open'], df['close']), threshold_factor=3.0) # Очень длинная тень
    features['dd_f_on_support'] = is_on_level(df['close'], support, atr, threshold=0.5)
    features['dd_f_vol_confirm'] = is_volume_spike(volume_ratio, 1.5)
    return features

def bullish_pin_bar_features(df):
    atr = calculate_atr(df['high'], df['low'], df['close'])
    support, _ = find_support_resistance(df['low'], df['high'])
    volume_ratio = calculate_volume_ratio(df['volume'])
    
    features = pd.DataFrame(index=df.index)
    features['bpb_f_small_body'] = is_small_body(df['open'], df['close'], df['high'], df['low'])
    features['bpb_f_long_lower_wick'] = is_long_shadow(get_lower_shadow(df['open'], df['low'], df['close']), get_body_size(df['open'], df['close']), threshold_factor=2.5)
    features['bpb_f_on_support'] = is_on_level(df['close'], support, atr, threshold=0.5)
    features['bpb_f_vol_confirm'] = is_volume_spike(volume_ratio, 1.2)
    return features

def bullish_belt_hold_features(df):
    atr = calculate_atr(df['high'], df['low'], df['close'])
    support, _ = find_support_resistance(df['low'], df['high'])
    volume_ratio = calculate_volume_ratio(df['volume'])
    
    features = pd.DataFrame(index=df.index)
    features['bbh_f_long_body'] = (get_body_size(df['open'], df['close']) > atr * 0.8).astype(int) # Длинное тело
    features['bbh_f_open_at_low'] = (abs(df['open'] - df['low']) / get_total_range(df['high'], df['low']) < 0.1).astype(int) # Открытие близко к минимуму
    features['bbh_f_vol_confirm'] = is_volume_spike(volume_ratio, 1.3)
    features['bbh_f_on_support'] = is_on_level(df['close'], support, atr, threshold=0.5)
    return features

def shootingstar_features(df):
    atr = calculate_atr(df['high'], df['low'], df['close'])
    support, resistance = find_support_resistance(df['low'], df['high'])
    
    features = pd.DataFrame(index=df.index)
    features['shootingstar_f_on_res'] = is_on_level(df['close'], resistance, atr)
    return features

def bullish_marubozu_features(df):
    atr = calculate_atr(df['high'], df['low'], df['close'])
    # support, _ = find_support_resistance(df['low'], df['high']) # Можно добавить, если нужно
    volume_ratio = calculate_volume_ratio(df['volume'])
    
    features = pd.DataFrame(index=df.index)
    
    # 1. Сильное бычье тело
    body_size = (df['close'] - df['open']).abs()
    features['bm_f_strong_body'] = (body_size > atr * 0.7).astype(int) # Большое тело (увеличено с 0.5 до 0.7)
    
    # 2. Подтверждение объемом
    features['bm_f_vol_confirm'] = is_volume_spike(volume_ratio, 1.5) # Высокий объем (увеличено с 1.3 до 1.5)
    
    # 3. Открытие > закрытия предыдущей свечи (Kicker-эффект)
    # Проверяем, что предыдущая свеча существует
    prev_close = df['close'].shift(1)
    features['bm_f_gap_up'] = ((df['open'] > prev_close) & (prev_close.notna())).astype(int)
    
    # 4. Бычье направление (закрытие > открытие)
    features['bm_f_bullish_dir'] = (df['close'] > df['open']).astype(int)
    
    return features

def add_pattern_features(df: pd.DataFrame) -> pd.DataFrame:
    """Applies all pattern feature extraction functions to the DataFrame."""
    
    # --- 1. Initialize all potential feature columns with 0 ---
    feature_columns = [
        'hammer_f_on_support', 'hammer_f_vol_spike',
        'hangingman_f_on_res', 'hangingman_f_vol_spike',
        'engulfing_f_strong', 'engulfing_f_vol_confirm',
        'doji_f_high_vol', 'doji_f_high_atr',
        'shootingstar_f_on_res',
        # НОВЫЕ ПРИЗНАКИ БЫЧЬЕГО MARUBOZU
        'bm_f_strong_body', 'bm_f_vol_confirm', 'bm_f_gap_up', 'bm_f_bullish_dir',
        # Оставшиеся бычьи паттерны
        'ih_f_small_body', 'ih_f_long_upper_shadow', 'ih_f_on_support', 'ih_f_vol_confirm',
        'dd_f_long_lower_shadow', 'dd_f_on_support', 'dd_f_vol_confirm',
        'bpb_f_small_body', 'bpb_f_long_lower_wick', 'bpb_f_on_support', 'bpb_f_vol_confirm',
        'bbh_f_long_body', 'bbh_f_open_at_low', 'bbh_f_vol_confirm', 'bbh_f_on_support'
    ]
    for col in feature_columns:
        df[col] = 0

    # --- 2. Calculate and assign features only for detected patterns ---
    hammer_mask = df['CDLHAMMER'] != 0
    if not df[hammer_mask].empty:
        df.loc[hammer_mask, ['hammer_f_on_support', 'hammer_f_vol_spike']] = hammer_features(df[hammer_mask]).values

    hangingman_mask = df['CDLHANGINGMAN'] != 0
    if not df[hangingman_mask].empty:
        df.loc[hangingman_mask, ['hangingman_f_on_res', 'hangingman_f_vol_spike']] = hangingman_features(df[hangingman_mask]).values

    engulfing_mask = df['CDLENGULFING'] != 0
    if not df[engulfing_mask].empty:
        df.loc[engulfing_mask, ['engulfing_f_strong', 'engulfing_f_vol_confirm']] = engulfing_features(df[engulfing_mask]).values

    doji_mask = df['CDLDOJI'] != 0
    if not df[doji_mask].empty:
        df.loc[doji_mask, ['doji_f_high_vol', 'doji_f_high_atr']] = doji_features(df[doji_mask]).values

    shootingstar_mask = df['CDLSHOOTINGSTAR'] != 0
    if not df[shootingstar_mask].empty:
        df.loc[shootingstar_mask, ['shootingstar_f_on_res']] = shootingstar_features(df[shootingstar_mask]).values
        
    # Расчет признаков для бычьего Marubozu (только для бычьих CDLMARUBOZU)
    bullish_marubozu_mask = (df['CDLMARUBOZU'] == 100) # Фильтруем только бычьи Marubozu
    if not df[bullish_marubozu_mask].empty:
        df.loc[bullish_marubozu_mask, ['bm_f_strong_body', 'bm_f_vol_confirm', 'bm_f_gap_up', 'bm_f_bullish_dir']] = bullish_marubozu_features(df[bullish_marubozu_mask]).values

    inverted_hammer_mask = df['CDLINVERTEDHAMMER'] != 0
    if not df[inverted_hammer_mask].empty:
        df.loc[inverted_hammer_mask, ['ih_f_small_body', 'ih_f_long_upper_shadow', 'ih_f_on_support', 'ih_f_vol_confirm']] = inverted_hammer_features(df[inverted_hammer_mask]).values

    dragonfly_doji_mask = df['CDLDRAGONFLYDOJI'] != 0
    if not df[dragonfly_doji_mask].empty:
        df.loc[dragonfly_doji_mask, ['dd_f_long_lower_shadow', 'dd_f_on_support', 'dd_f_vol_confirm']] = dragonfly_doji_features(df[dragonfly_doji_mask]).values

    bullish_pin_bar_mask = (df['CDLDRAGONFLYDOJI'] != 0) | ((df['CDLHAMMER'] != 0) & (df['close'] > df['open']))
    if not df[bullish_pin_bar_mask].empty:
        df.loc[bullish_pin_bar_mask, ['bpb_f_small_body', 'bpb_f_long_lower_wick', 'bpb_f_on_support', 'bpb_f_vol_confirm']] = bullish_pin_bar_features(df[bullish_pin_bar_mask]).values
    
    # bullish_kicker_mask = df['CDLBULLISHKICKING'] != 0 # <--- УДАЛЕНО
    # if not df[bullish_kicker_mask].empty:
    #     df.loc[bullish_kicker_mask, ['bk_f_strong_bullish_body', 'bk_f_vol_confirm']] = bullish_kicker_features(df[bullish_kicker_mask]).values
        
    bullish_belt_hold_mask = df['CDLBELTHOLD'] != 0
    if not df[bullish_belt_hold_mask].empty:
        df.loc[bullish_belt_hold_mask, ['bbh_f_long_body', 'bbh_f_open_at_low', 'bbh_f_vol_confirm', 'bbh_f_on_support']] = bullish_belt_hold_features(df[bullish_belt_hold_mask]).values


    # --- 3. Combine features into final scores ---
    # Now that columns are guaranteed to exist, we can access them directly.
    df['hammer_f'] = (df['hammer_f_on_support'] + df['hammer_f_vol_spike']).astype(int)
    df['hangingman_f'] = (df['hangingman_f_on_res'] + df['hangingman_f_vol_spike']).astype(int)
    df['engulfing_f'] = (df['engulfing_f_strong'] + df['engulfing_f_vol_confirm']).astype(int)
    df['doji_f'] = (df['doji_f_high_vol'] + df['doji_f_high_atr']).astype(int)
    df['shootingstar_f'] = df['shootingstar_f_on_res'].astype(int)
    df['bullish_marubozu_f'] = (df['bm_f_strong_body'] + df['bm_f_vol_confirm'] + df['bm_f_gap_up'] + df['bm_f_bullish_dir']).astype(int)
    df['inverted_hammer_f'] = (df['ih_f_small_body'] + df['ih_f_long_upper_shadow'] + df['ih_f_on_support'] + df['ih_f_vol_confirm']).astype(int)
    df['dragonfly_doji_f'] = (df['dd_f_long_lower_shadow'] + df['dd_f_on_support'] + df['dd_f_vol_confirm']).astype(int)
    df['bullish_pin_bar_f'] = (df['bpb_f_small_body'] + df['bpb_f_long_lower_wick'] + df['bpb_f_on_support'] + df['bpb_f_vol_confirm']).astype(int)
    # df['bullish_kicker_f'] = (df['bk_f_strong_bullish_body'] + df['bk_f_vol_confirm']).astype(int) # <--- УДАЛЕНО
    df['bullish_belt_hold_f'] = (df['bbh_f_long_body'] + df['bbh_f_open_at_low'] + df['bbh_f_vol_confirm'] + df['bbh_f_on_support']).astype(int)

    return df

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

def prepare_price_series(df: pd.DataFrame) -> pd.Series:
    """
    Подготавливает временной ряд цен закрытия для использования в новых моделях.
    
    Args:
        df (pd.DataFrame): Входной DataFrame с колонкой 'close'.
        
    Returns:
        pd.Series: Временной ряд цен закрытия в формате float.
    """
    if 'close' not in df.columns:
        raise ValueError("DataFrame должен содержать колонку 'close'")
    
    return df['close'].astype(float)

def prepare_features_for_models(df: pd.DataFrame) -> dict:
    """
    Подготавливает признаки для использования в новых моделях (Kalman Filter, LSTM, GPR).
    
    Args:
        df (pd.DataFrame): Входной DataFrame с техническими индикаторами.
        
    Returns:
        dict: Словарь с подготовленными данными для моделей.
    """
    # Подготовка временного ряда цен закрытия
    price_series = prepare_price_series(df)
    
    # Возвращаем словарь с подготовленными данными
    return {
        'price_series': price_series
    }


def prepare_xlstm_rl_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepares enhanced features for the unified xLSTM+RL model - INDICATORS ONLY
    """
    df = calculate_features(df)
    df = detect_candlestick_patterns(df)
    
    feature_cols = [
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
    
    return df, feature_cols

if __name__ == '__main__':
    # --- Example Usage and Testing ---
    print("Testing feature engineering module...")
    try:
        # Load sample data
        test_df = pd.read_csv('historical_data.csv')
        print(f"Loaded test data with {len(test_df)} rows.")
        
        # Take data for a single symbol for a clean test
        sample_symbol_df = test_df[test_df['symbol'] == 'BTCUSDT'].copy()
        print(f"Testing with BTCUSDT data ({len(sample_symbol_df)} rows).")

        # Calculate features
        features_df = calculate_features(sample_symbol_df)
        print(f"Features calculated. Resulting data has {len(features_df)} rows.")
        
        # Detect candlestick patterns
        patterns_df = detect_candlestick_patterns(features_df)
        print(f"Patterns detected. Resulting data has {len(patterns_df)} rows.")

        # Save to a test file
        output_file = 'features_test_output.csv'
        patterns_df.to_csv(output_file, index=False)
        
        print(f"Test output saved to {output_file}")
        print("\nFinal columns:")
        print(patterns_df.columns.tolist())

    except FileNotFoundError:
        print("ERROR: Test file 'historical_data.csv' not found.")
    except Exception as e:
        print(f"An error occurred during testing: {e}")

