import pandas as pd
import numpy as np
import argparse
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

from feature_engineering import calculate_features, detect_candlestick_patterns
from models.xlstm_pattern_model import XLSTMPatternModel
from models.xlstm_indicator_model import XLSTMIndicatorModel


def check_for_nan_inf(data, name):
    """Проверяет наличие NaN или Inf значений в данных."""
    if np.any(np.isnan(data)):
        print(f"!!! ВНИМАНИЕ: NaN значения найдены в {name}")
        return True
    if np.any(np.isinf(data)):
        print(f"!!! ВНИМАНИЕ: Inf значения найдены в {name}")
        return True
    return False

def prepare_data_for_training(data_path, sequence_length=10):
    """
    Prepares data for training the models, including feature calculation,
    pattern detection, and creation of sequences for each symbol.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}")

    print(f"Loading data from {data_path}...")
    full_df = pd.read_csv(data_path)
    
    all_X = []
    all_y = []

    # --- Feature columns ---
    # We define them here to ensure consistent indexing later
    pattern_cols = [
        'CDLHAMMER', 'CDLENGULFING', 'CDLDOJI', 'CDLSHOOTINGSTAR',
        'CDLHANGINGMAN', 'CDLMARUBOZU',  # Заменено CDL3BLACKCROWS
        'hammer_f', 'hangingman_f', 'engulfing_f', 'doji_f',
        'shootingstar_f', 'marubozu_f',  # Заменено 3blackcrows_f
        # ДОБАВИТЬ новые признаки:
        'hammer_f_on_support', 'hammer_f_vol_spike',
        'hangingman_f_on_res', 'hangingman_f_vol_spike',
        'engulfing_f_strong', 'engulfing_f_vol_confirm',
        'doji_f_high_vol', 'doji_f_high_atr',
        'shootingstar_f_on_res',
        'marubozu_f_strong_body', 'marubozu_f_vol_confirm', 'marubozu_f_bullish'
    ]
    indicator_cols = ['RSI_14', 'MACD_12_26_9', 'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0', 'ADX_14', 'STOCHk_14_3_3', 'STOCHd_14_3_3']
    feature_cols = ['open', 'high', 'low', 'close', 'volume'] + indicator_cols + pattern_cols

    symbols = full_df['symbol'].unique()
    print(f"Found {len(symbols)} symbols. Processing each one...")

    for symbol in symbols:
        df = full_df[full_df['symbol'] == symbol].copy()
        print(f"\nProcessing symbol: {symbol}, initial rows: {len(df)}")
        
        if len(df) < sequence_length + 20: # Need enough data for indicators and sequences
            print(f"Skipping symbol {symbol}: not enough data ({len(df)} rows)")
            continue

        # --- Feature Engineering ---
        df = calculate_features(df)
        print(f"Rows after calculate_features: {len(df)}")
        df = detect_candlestick_patterns(df)
        print(f"Rows after detect_candlestick_patterns: {len(df)}")
        
        # --- Create labels (target variable) ---
        df['target'] = 0
        df.loc[df['close'].shift(-5) > df['close'], 'target'] = 1  # Buy
        df.loc[df['close'].shift(-5) < df['close'], 'target'] = 2  # Sell
        
        # --- Clean up and select columns ---
        print(f"Rows before final processing: {len(df)}")
        
        # --- Clean up and select columns ---
        # A more robust cleaning process
        initial_row_count = len(df)
        
        # Check for NaNs created by indicators
        nan_check_cols = indicator_cols
        nan_info_after_features = df[nan_check_cols].isnull().sum()
        if nan_info_after_features.sum() > 0:
            print(f"NaNs created by feature engineering:\n{nan_info_after_features[nan_info_after_features > 0]}")
            
        # Drop rows that have NaN in critical columns after feature calculation
        df.dropna(subset=indicator_cols, inplace=True)
        
        if len(df) < initial_row_count:
            print(f"Dropped {initial_row_count - len(df)} rows due to NaNs in indicator columns.")

        if df.empty:
            print(f"Skipping symbol {symbol}: not enough data after cleaning NaNs from indicators.")
            continue

        
        # Ensure all required feature columns exist, fill missing with 0
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0
        
        df = df[feature_cols + ['target']]

        # --- Prepare sequences for this symbol ---
        if len(df) > sequence_length:
            for i in range(len(df) - sequence_length):
                all_X.append(df.iloc[i:i + sequence_length][feature_cols].values)
                all_y.append(df.iloc[i + sequence_length]['target'])

    if not all_X:
        print("No data available for training after processing all symbols.")
        return None, None, None

    print(f"Total sequences created: {len(all_X)}")
    
    X = np.array(all_X, dtype=np.float32)
    y = to_categorical(np.array(all_y), num_classes=3)

    if check_for_nan_inf(X, "X (все данные)"):
        print("Попытка исправить NaN/Inf значения в X...")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    if check_for_nan_inf(y, "y (метки)"):
        print("Попытка исправить NaN/Inf значения в y...")
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

    # --- Split data for each model ---
    # Create a temporary dataframe to get column indices easily
    temp_df_for_indices = pd.DataFrame(columns=feature_cols)
    pattern_indices = [temp_df_for_indices.columns.get_loc(c) for c in pattern_cols]
    indicator_indices = [temp_df_for_indices.columns.get_loc(c) for c in indicator_cols]

    X_patterns = X[:, :, pattern_indices]
    X_indicators = X[:, :, indicator_indices]

    return X_patterns, X_indicators, y

def train_xlstm_pattern_model(X_patterns, y):
    """
    Trains the xLSTM model for pattern recognition.
    """
    print("\n--- Training xLSTM Pattern Model ---")
    if X_patterns is None or len(X_patterns) == 0:
        print("No pattern data to train xLSTM model.")
        return

    X_train, X_test, y_train, y_test = train_test_split(X_patterns, y, test_size=0.2, random_state=42)
    
    print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples.")
    model = XLSTMPatternModel(input_shape=(X_train.shape[1], X_train.shape[2]))
    model.train(X_train, y_train)
    model.save_model()
    
    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"xLSTM Pattern Model Accuracy: {accuracy * 100:.2f}%")

def train_xlstm_indicator_model(X_indicators, y):
    """
    Trains the xLSTM model for indicator analysis.
    """
    print("\n--- Training xLSTM Indicator Model ---")
    if X_indicators is None or len(X_indicators) == 0:
        print("No indicator data to train xLSTM model.")
        return

    X_train, X_test, y_train, y_test = train_test_split(X_indicators, y, test_size=0.2, random_state=42)
    
    print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples.")

    if check_for_nan_inf(X_train, "X_train (индикаторы)"):
        print("Исправляем NaN/Inf значения в обучающих данных (индикаторы)...")
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    if check_for_nan_inf(y_train, "y_train (индикаторы)"):
        print("Исправляем NaN/Inf значения в метках (индикаторы)...")
        y_train = np.nan_to_num(y_train, nan=0.0, posinf=0.0, neginf=0.0)

    model = XLSTMIndicatorModel(input_shape=(X_train.shape[1], X_train.shape[2]))
    model.train(X_train, y_train)
    model.save_model()
    
    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"xLSTM Indicator Model Accuracy: {accuracy * 100:.2f}%")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script for training new trading models.')
    parser.add_argument('--data', type=str, default='historical_data.csv', help='Path to the historical data file.')
    parser.add_argument('--model', type=str, default='all', help='Which model to train: "xlstm_pattern", "xlstm_indicator", or "all".')
    args = parser.parse_args()

    try:
        X_patterns, X_indicators, y = prepare_data_for_training(args.data)

        if X_patterns is not None and y is not None:
            if args.model in ['xlstm_pattern', 'all']:
                train_xlstm_pattern_model(X_patterns, y)
            
            if args.model in ['xlstm_indicator', 'all']:
                train_xlstm_indicator_model(X_indicators, y)
        else:
            print("Training skipped due to lack of data.")

    except Exception as e:
        print(f"An error occurred during training: {e}")
        import traceback
        traceback.print_exc()
