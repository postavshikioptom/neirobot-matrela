import pandas as pd
import talib
from functools import lru_cache

import numpy as np
from sklearn.preprocessing import MinMaxScaler # <--- ДОБАВЛЕНО: Импорт MinMaxScaler

# --- Helper functions for pattern features from info-4-patterns.md ---

@lru_cache(maxsize=128)
def cached_calculate_atr(high_tuple, low_tuple, close_tuple, period=14):
    """Кэшированная версия расчета ATR"""
    high = np.array(high_tuple)
    low = np.array(low_tuple)
    close = np.array(close_tuple)
    return talib.ATR(high, low, close, timeperiod=period)

def calculate_atr(high, low, close, period=14):
    return talib.ATR(high, low, close, timeperiod=period)

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

def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates a specific set of technical indicators (RSI, MACD, Bollinger Bands, ADX, Stochastic) on a given DataFrame.
    This version is designed to be extremely robust and avoid returning None.
    """
    try:
        if df is None or df.empty:
            return pd.DataFrame()

        # --- Ensure numeric types ---
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Drop rows where essential OHLCV data is missing BEFORE calculations
        df.dropna(subset=numeric_cols, inplace=True)
        if df.empty:
            return pd.DataFrame()

        # =====================================================================
        # НОВЫЙ БЛОК: МАСШТАБИРОВАНИЕ ОБЪЕМА
        # =====================================================================
        if 'volume' in df.columns and not df['volume'].empty:
            # Создаем MinMaxScaler для колонки 'volume' с диапазоном от 0 до 100
            scaler_volume = MinMaxScaler(feature_range=(0, 100))
            
            # Применяем масштабирование. .values.reshape(-1, 1) нужен для работы с одной колонкой.
            # Создаем новую колонку с масштабированным объемом
            df['volume_scaled'] = scaler_volume.fit_transform(df[['volume']].values)
            
            # Заменим оригинальную колонку 'volume' на масштабированную для дальнейших расчетов
            df['volume'] = df['volume_scaled']
            
            # Удалим временную колонку 'volume_scaled'
            df.drop(columns=['volume_scaled'], inplace=True, errors='ignore') # errors='ignore' для безопасности
            print("✅ Объем успешно масштабирован (диапазон 0-100).")
        else:
            print("⚠️ Колонка 'volume' отсутствует или пуста, масштабирование объема пропущено.")
        # =====================================================================
        # КОНЕЦ НОВОГО БЛОКА
        # =====================================================================

        # --- Calculate specified indicators using TA-Lib ---
        # Create a copy to avoid SettingWithCopyWarning
        df_out = df.copy()

        # Use .values to avoid index alignment issues with talib
        open_p = df_out['open'].values.astype(float)
        high_p = df_out['high'].values.astype(float)
        low_p = df_out['low'].values.astype(float)
        close_p = df_out['close'].values.astype(float)

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

        # =====================================================================
        # НОВЫЙ БЛОК: СОЗДАНИЕ ПРИЗНАКА 'is_event' (для Event-Based Sampling)
        # =====================================================================
        # Определяем события, которые потенциально содержат сигналы
        # Используем существующие индикаторы и добавляем RSI/ATR/Volume/ADX изменения
        
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
        # =====================================================================
        # КОНЕЦ НОВОГО БЛОКА 'is_event'
        # =====================================================================

        return df_out

    except Exception as e:
        # If anything at all goes wrong, return an empty dataframe to prevent crashes.
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

def shootingstar_features(df):
    atr = calculate_atr(df['high'], df['low'], df['close'])
    support, resistance = find_support_resistance(df['low'], df['high'])
    
    features = pd.DataFrame(index=df.index)
    features['shootingstar_f_on_res'] = is_on_level(df['close'], resistance, atr)
    return features

def marubozu_features(df):
    """Признаки для паттерна Marubozu"""
    atr = calculate_atr(df['high'], df['low'], df['close'])
    volume_ratio = calculate_volume_ratio(df['volume'])
    
    features = pd.DataFrame(index=df.index)
    # Сила тренда - насколько сильно тело свечи по отношению к ATR
    body_size = (df['close'] - df['open']).abs()
    features['marubozu_f_strong_body'] = (body_size > atr * 0.5).astype(int)
    # Подтверждение объемом
    features['marubozu_f_vol_confirm'] = is_volume_spike(volume_ratio, 1.3)
    # Направление тренда (бычий/медвежий)
    features['marubozu_f_bullish'] = (df['close'] > df['open']).astype(int)
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
        'marubozu_f_strong_body', 'marubozu_f_vol_confirm', 'marubozu_f_bullish'
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
        
    marubozu_mask = df['CDLMARUBOZU'] != 0
    if not df[marubozu_mask].empty:
        df.loc[marubozu_mask, ['marubozu_f_strong_body', 'marubozu_f_vol_confirm', 'marubozu_f_bullish']] = marubozu_features(df[marubozu_mask]).values

    # --- 3. Combine features into final scores ---
    # Now that columns are guaranteed to exist, we can access them directly.
    df['hammer_f'] = (df['hammer_f_on_support'] + df['hammer_f_vol_spike']).astype(int)
    df['hangingman_f'] = (df['hangingman_f_on_res'] + df['hangingman_f_vol_spike']).astype(int)
    df['engulfing_f'] = (df['engulfing_f_strong'] + df['engulfing_f_vol_confirm']).astype(int)
    df['doji_f'] = (df['doji_f_high_vol'] + df['doji_f_high_atr']).astype(int)
    df['shootingstar_f'] = df['shootingstar_f_on_res'].astype(int)
    df['marubozu_f'] = (df['marubozu_f_strong_body'] + df['marubozu_f_vol_confirm'] + df['marubozu_f_bullish']).astype(int)

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
    df['CDLDOJI'] = talib.CDLDOJI(open_prices, high_prices, low_prices, close_prices)
    df['CDLSHOOTINGSTAR'] = talib.CDLSHOOTINGSTAR(open_prices, high_prices, low_prices, close_prices)
    df['CDLHANGINGMAN'] = talib.CDLHANGINGMAN(open_prices, high_prices, low_prices, close_prices)
    df['CDLMARUBOZU'] = talib.CDLMARUBOZU(open_prices, high_prices, low_prices, close_prices)


    # --- Calculate features for each pattern ---
    df = add_pattern_features(df)

    pattern_cols = [
        'CDLHAMMER', 'CDLENGULFING', 'CDLDOJI', 'CDLSHOOTINGSTAR',
        'CDLHANGINGMAN', 'CDLMARUBOZU'  # Заменено CDL3BLACKCROWS на CDLMARUBOZU
    ]
    
    # Add new feature columns to the list to ensure they are handled
    feature_cols = [
        'hammer_f', 'hangingman_f', 'engulfing_f', 'doji_f',
        'shootingstar_f', 'marubozu_f'  # Заменено 3blackcrows_f на marubozu_f
    ]
    
    all_pattern_cols = pattern_cols + feature_cols

    for col in all_pattern_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    return df

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

# === VSA ANALYSIS MODULE ===
def calculate_vsa_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет VSA (Volume Spread Analysis) признаки для анализа умных денег
    """
    df = df.copy()
    
    # Базовые VSA компоненты
    df['spread'] = df['high'] - df['low']
    df['body'] = abs(df['close'] - df['open'])
    df['close_position'] = (df['close'] - df['low']) / df['spread']  # 0=bottom, 1=top
    df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
    df['spread_ratio'] = df['spread'] / df['spread'].rolling(20).mean()
    
    # VSA сигналы для обнаружения умных денег
    
    # 1. No Demand (слабость покупателей)
    df['vsa_no_demand'] = (
        (df['volume_ratio'] < 0.7) &  # низкий объем
        (df['spread_ratio'] < 0.8) &  # узкий спред
        (df['close'] < df['open']) &   # красная свеча
        (df['close_position'] < 0.4)  # закрытие внизу
    ).astype(int)
    
    # 2. No Supply (слабость продавцов)
    df['vsa_no_supply'] = (
        (df['volume_ratio'] < 0.7) &  # низкий объем
        (df['spread_ratio'] < 0.8) &  # узкий спред
        (df['close'] > df['open']) &   # зеленая свеча
        (df['close_position'] > 0.6)  # закрытие вверху
    ).astype(int)
    
    # 3. Stopping Volume (остановочный объем - разворот)
    df['vsa_stopping_volume'] = (
        (df['volume_ratio'] > 2.0) &  # очень высокий объем
        (df['spread_ratio'] > 1.2) &  # широкий спред
        (df['close_position'] > 0.7)  # закрытие вверху после падения
    ).astype(int)
    
    # 4. Climactic Volume (кульминационный объем)
    df['vsa_climactic_volume'] = (
        (df['volume_ratio'] > 3.0) &  # экстремальный объем
        (df['spread_ratio'] > 1.5) &  # очень широкий спред
        (df['close_position'] < 0.3)  # закрытие внизу
    ).astype(int)
    
    # 5. Test (тест - проверка силы/слабости)
    df['vsa_test'] = (
        (df['volume_ratio'] < 0.5) &  # очень низкий объем
        (df['spread_ratio'] < 0.6) &  # узкий спред
        (abs(df['close'] - df['open']) / df['spread'] < 0.3)  # маленькое тело
    ).astype(int)
    
    # 6. Effort vs Result (усилие против результата)
    df['vsa_effort_vs_result'] = (
        (df['volume_ratio'] > 1.8) &  # высокий объем (усилие)
        (df['spread_ratio'] < 0.7) &  # но маленький спред (плохой результат)
        (abs(df['body']) / df['spread'] < 0.4)  # маленькое тело
    ).astype(int)
    
    # Сводный VSA индекс силы/слабости
    df['vsa_strength'] = (
        df['vsa_no_supply'] * 1 +
        df['vsa_stopping_volume'] * 2 +
        df['vsa_test'] * 0.5 -
        df['vsa_no_demand'] * 1 -
        df['vsa_climactic_volume'] * 2 -
        df['vsa_effort_vs_result'] * 1
    )
    
    return df

def prepare_xlstm_rl_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Подготавливает улучшенные признаки для единой xLSTM+RL модели
    """
    df = calculate_features(df)
    df = detect_candlestick_patterns(df)
    df = calculate_vsa_features(df)
    
    xlstm_rl_features = [
        # Технические индикаторы
        'RSI_14', 'MACD_12_26_9', 'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0',
        'ADX_14', 'STOCHk_14_3_3', 'STOCHd_14_3_3',
        # Паттерны
        'CDLHAMMER', 'CDLENGULFING', 'CDLDOJI', 'CDLSHOOTINGSTAR',
        'CDLHANGINGMAN', 'CDLMARUBOZU',
        # VSA сигналы
        'vsa_no_demand', 'vsa_no_supply', 'vsa_stopping_volume', 'vsa_climactic_volume', 'vsa_test', 'vsa_effort_vs_result', 'vsa_strength',
        # Дополнительные рыночные данные
        'volume_ratio', 'spread_ratio', 'close_position'
    ]
    
    return df, xlstm_rl_features

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

def calculate_advanced_vsa_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Расширенные VSA признаки для лучшего качества сигналов
    """
    df = calculate_vsa_features(df)  # Базовые VSA
    
    # Добавляем временные фильтры VSA
    df['vsa_no_demand_filtered'] = (
        (df['vsa_no_demand'] == 1) & 
        (df['vsa_no_demand'].rolling(3).sum() <= 1)  # Не более 1 раза за 3 свечи
    ).astype(int)
    
    df['vsa_stopping_volume_filtered'] = (
        (df['vsa_stopping_volume'] == 1) &
        (df['close'].pct_change() < -0.02)  # Только после падения >2%
    ).astype(int)
    
    # Комбинированные VSA сигналы
    df['vsa_strong_buy'] = (
        (df['vsa_no_supply'] == 1) | 
        (df['vsa_stopping_volume_filtered'] == 1)
    ).astype(int)
    
    df['vsa_strong_sell'] = (
        (df['vsa_no_demand_filtered'] == 1) | 
        (df['vsa_climactic_volume'] == 1)
    ).astype(int)
    
    # VSA momentum
    df['vsa_momentum'] = df['vsa_strength'].rolling(5).mean()
    
    return df
