import pandas as pd
import numpy as np
import argparse
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Новые импорты
from feature_engineering import calculate_features, detect_candlestick_patterns, calculate_vsa_features
from models.xlstm_rl_model import XLSTMRLModel
from rl_agent import IntelligentRLAgent
from trading_env import TradingEnvRL

def prepare_xlstm_rl_data(data_path, sequence_length=10):
    """
    Подготавливает данные для единой xLSTM+RL системы
    """
    print(f"Загрузка данных из {data_path}...")
    full_df = pd.read_csv(data_path)
    
    # Объединенные признаки для новой архитектуры
    feature_cols = [
        # Технические индикаторы
        'RSI_14', 'MACD_12_26_9', 'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0', 
        'ADX_14', 'STOCHk_14_3_3', 'STOCHd_14_3_3',
        # Паттерны
        'CDLHAMMER', 'CDLENGULFING', 'CDLDOJI', 'CDLSHOOTINGSTAR', 
        'CDLHANGINGMAN', 'CDLMARUBOZU',
        # VSA признаки (новые!)
        'vsa_no_demand', 'vsa_no_supply', 'vsa_stopping_volume', 
        'vsa_climactic_volume', 'vsa_test', 'vsa_effort_vs_result', 'vsa_strength',
        # Дополнительные рыночные данные
        'volume_ratio', 'spread_ratio', 'close_position'
    ]
    
    all_X = []
    all_y = []
    processed_dfs = {}  # Сохраняем обработанные данные для RL
    
    symbols = full_df['symbol'].unique()
    print(f"Найдено {len(symbols)} символов. Обрабатываем каждый...")
    
    for symbol in symbols:
        df = full_df[full_df['symbol'] == symbol].copy()
        print(f"\nОбработка символа: {symbol}, строк: {len(df)}")
        
        if len(df) < sequence_length + 50:  # Нужно достаточно данных
            continue
            
        # === НОВАЯ ОБРАБОТКА С VSA ===
        df = calculate_features(df)
        df = detect_candlestick_patterns(df)
        df = calculate_vsa_features(df)  # Добавляем VSA!
        
        # Создаем целевые метки на основе будущих цен + VSA подтверждения
        df['future_return'] = (df['close'].shift(-5) - df['close']) / df['close']
        df['target'] = 2  # По умолчанию HOLD
        
        # BUY: положительная доходность + VSA подтверждение покупки
        buy_condition = (
            (df['future_return'] > 0.01) &  # >1% роста
            ((df['vsa_no_supply'] == 1) | (df['vsa_stopping_volume'] == 1) | (df['vsa_strength'] > 1))
        )
        df.loc[buy_condition, 'target'] = 0
        
        # SELL: отрицательная доходность + VSA подтверждение продажи
        sell_condition = (
            (df['future_return'] < -0.01) &  # >1% падения
            ((df['vsa_no_demand'] == 1) | (df['vsa_climactic_volume'] == 1) | (df['vsa_strength'] < -1))
        )
        df.loc[sell_condition, 'target'] = 1
        
        # Убираем NaN и обеспечиваем наличие всех признаков
        df.dropna(subset=['future_return'], inplace=True)
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0
        
        df = df[feature_cols + ['target', 'close', 'volume']].copy()
        
        # Сохраняем обработанные данные для RL
        processed_dfs[symbol] = df
        
        # Создаем последовательности для xLSTM
        if len(df) > sequence_length:
            for i in range(len(df) - sequence_length):
                all_X.append(df.iloc[i:i + sequence_length][feature_cols].values)
                all_y.append(df.iloc[i + sequence_length]['target'])
    
    if not all_X:
        raise ValueError("Нет данных для обучения после обработки всех символов")
        
    print(f"Создано последовательностей: {len(all_X)}")
    
    X = np.array(all_X, dtype=np.float32)
    y = to_categorical(np.array(all_y), num_classes=3)
    
    # Исправляем NaN/Inf значения
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    
    return X, y, processed_dfs, feature_cols

def train_xlstm_rl_system(X, y, processed_dfs, feature_cols):
    """
    Обучает единую систему xLSTM + RL
    """
    print("\n=== ЭТАП 1: ОБУЧЕНИЕ xLSTM МОДЕЛИ ===")
    
    # Разделяем данные
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    print(f"Обучающая выборка: {len(X_train)}")
    print(f"Валидационная выборка: {len(X_val)}")
    print(f"Тестовая выборка: {len(X_test)}")
    
    # Создаем и обучаем xLSTM модель
    xlstm_model = XLSTMRLModel(
        input_shape=(X_train.shape[1], X_train.shape[2]),
        memory_units=128,
        attention_units=64
    )
    
    # Обучение с ранним остановом
    history = xlstm_model.train(
        X_train, y_train,
        X_val, y_val,
        epochs=100,
        batch_size=32
    )
    
    # Оценка xLSTM
    loss, accuracy, precision, recall = xlstm_model.model.evaluate(X_test, y_test, verbose=0)
    print(f"xLSTM Точность: {accuracy * 100:.2f}%")
    
    # Сохраняем xLSTM модель
    xlstm_model.save_model()
    
    print("\n=== ЭТАП 2: ОБУЧЕНИЕ RL АГЕНТА ===")
    
    # Выбираем данные для RL обучения (используем несколько символов)
    rl_symbols = list(processed_dfs.keys())[:3]  # Берем первые 3 символа
    
    rl_agent = None # Initialize rl_agent
    for i, symbol in enumerate(rl_symbols):
        df = processed_dfs[symbol]
        print(f"\nОбучение RL на символе {symbol} ({i+1}/{len(rl_symbols)})")
        
        # Разделяем на train/eval для RL
        split_idx = int(len(df) * 0.8)
        train_df = df.iloc[:split_idx].copy()
        eval_df = df.iloc[split_idx:].copy()
        
        # Создаем RL агента
        rl_agent = IntelligentRLAgent(algorithm='PPO')
        
        # Создаем среды
        vec_env = rl_agent.create_training_environment(train_df, xlstm_model)
        rl_agent.create_evaluation_environment(eval_df, xlstm_model)
        
        # Строим и обучаем агента
        rl_agent.build_agent(vec_env)
        rl_agent.train_with_callbacks(
            total_timesteps=50000,
            eval_freq=2000
        )
        
        # Сохраняем лучшего агента
        rl_agent.save_agent(f'models/rl_agent_{symbol}')
    
    print("\n=== ОБУЧЕНИЕ ЗАВЕРШЕНО ===")
    return xlstm_model, rl_agent

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Обучение новой системы xLSTM + VSA + RL')
    parser.add_argument('--data', type=str, default='historical_data.csv', help='Путь к данным')
    parser.add_argument('--sequence_length', type=int, default=10, help='Длина последовательности')
    args = parser.parse_args()
    
    try:
        # Подготавливаем данные
        X, y, processed_dfs, feature_cols = prepare_xlstm_rl_data(args.data, args.sequence_length)
        
        # Обучаем систему
        xlstm_model, rl_agent = train_xlstm_rl_system(X, y, processed_dfs, feature_cols)
        
        print("✅ Новая система xLSTM + VSA + RL успешно обучена!")
        
    except Exception as e:
        print(f"❌ Ошибка при обучении: {e}")
        import traceback
        traceback.print_exc()
