# visual_graph.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import pickle
import argparse
import os

# 🔥 НОВОЕ: Импортируем CustomFocalLoss и XLSTMLayer для загрузки модели
from train_model import CustomFocalLoss
from models.xlstm_memory_cell import XLSTMLayer
from models.xlstm_rl_model import XLSTMRLModel # Для доступа к структуре модели

# Убедитесь, что config.py доступен
try:
    import config
except ImportError:
    print("⚠️ config.py не найден. Использую значения по умолчанию.")
    class Config:
        SEQUENCE_LENGTH = 60
        # Добавьте другие необходимые константы, если они используются в других местах
    config = Config

def plot_predictions(symbol_df, model, scaler, feature_cols, sequence_length):
    """
    Строит график цены с наложенными предсказаниями BUY/SELL/HOLD.
    """
    if symbol_df.empty:
        print(f"Нет данных для символа {symbol_df['symbol'].iloc[0]}")
        return

    # Подготовка данных для предсказания
    X_predict = []
    # Убедимся, что DataFrame имеет достаточно строк для формирования последовательностей
    if len(symbol_df) < sequence_length:
        print(f"Недостаточно данных ({len(symbol_df)} строк) для формирования последовательностей длиной {sequence_length}.")
        return

    for i in range(len(symbol_df) - sequence_length):
        # 🔥 ИЗМЕНЕНО: Извлекаем только 'close'
        X_predict.append(symbol_df.iloc[i:i + sequence_length][['close']].values)
    
    if not X_predict:
        print(f"Недостаточно данных для формирования последовательностей для предсказания.")
        return

    X_predict = np.array(X_predict, dtype=np.float32)
    X_predict = np.nan_to_num(X_predict, nan=0.0, posinf=0.0, neginf=0.0)

    # Масштабирование данных для предсказания
    X_predict_reshaped = X_predict.reshape(-1, X_predict.shape[-1])
    X_predict_scaled = scaler.transform(X_predict_reshaped).reshape(X_predict.shape)

    # Получение предсказаний
    raw_predictions = model.predict(X_predict_scaled, verbose=0)
    predicted_classes = np.argmax(raw_predictions, axis=1)

    # Создаем DataFrame для удобства визуализации
    # Индексы предсказаний соответствуют последней свече в последовательности
    prediction_df = pd.DataFrame({
        'predicted_class': predicted_classes,
        'buy_prob': raw_predictions[:, 0],
        'sell_prob': raw_predictions[:, 1],
        'hold_prob': raw_predictions[:, 2]
    }, index=symbol_df.index[sequence_length:]) # Индексы должны соответствовать предсказанным свечам

    # Объединяем с исходным DataFrame
    plot_df = symbol_df.loc[prediction_df.index].copy()
    plot_df = pd.concat([plot_df, prediction_df], axis=1)
    
    # Визуализация
    plt.figure(figsize=(18, 10))
    
    # График цены
    plt.plot(plot_df.index, plot_df['close'], label='Цена закрытия', color='blue', alpha=0.7)

    # Отмечаем предсказания BUY
    buy_signals = plot_df[plot_df['predicted_class'] == 0]
    plt.scatter(buy_signals.index, buy_signals['close'], marker='^', color='green', s=100, label='BUY (Предсказание)', alpha=0.8)

    # Отмечаем предсказания SELL
    sell_signals = plot_df[plot_df['predicted_class'] == 1]
    plt.scatter(sell_signals.index, sell_signals['close'], marker='v', color='red', s=100, label='SELL (Предсказание)', alpha=0.8)

    # Отмечаем реальные BUY/SELL сигналы (если они есть в данных)
    # (Предполагаем, что 'target' колонка была создана в feature_engineering)
    if 'target' in plot_df.columns:
        real_buy_signals = plot_df[plot_df['target'] == 0]
        plt.scatter(real_buy_signals.index, real_buy_signals['close'], marker='^', color='lime', s=200, facecolors='none', edgecolors='lime', label='BUY (Реальный)', alpha=0.8)
        
        real_sell_signals = plot_df[plot_df['target'] == 1]
        plt.scatter(real_sell_signals.index, real_sell_signals['close'], marker='v', color='darkred', s=200, facecolors='none', edgecolors='darkred', label='SELL (Реальный)', alpha=0.8)


    plt.title(f'Прогноз xLSTM модели для {symbol_df["symbol"].iloc[0]}')
    plt.xlabel('Индекс свечи')
    plt.ylabel('Цена')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'predictions_graph_{symbol_df["symbol"].iloc[0]}.png')
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Визуализация предсказаний xLSTM модели.')
    parser.add_argument('--data', type=str, default='historical_data.csv', help='Путь к историческим данным.')
    parser.add_argument('--model_path', type=str, default='models/xlstm_rl_model.keras', help='Путь к сохраненной модели xLSTM.')
    parser.add_argument('--scaler_path', type=str, default='models/xlstm_rl_scaler.pkl', help='Путь к сохраненному скейлеру.')
    parser.add_argument('--symbol', type=str, default='BTCUSDT', help='Символ для визуализации (например, BTCUSDT).')
    args = parser.parse_args()

    # Загрузка модели и скейлера
    try:
        # 🔥 НОВОЕ: Передаем custom_objects для загрузки кастомных слоев и функции потерь
        model = tf.keras.models.load_model(args.model_path, custom_objects={'XLSTMLayer': XLSTMLayer, 'CustomFocalLoss': CustomFocalLoss})
        with open(args.scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        print(f"✅ Модель и скейлер успешно загружены.")
    except Exception as e:
        print(f"❌ Ошибка загрузки модели или скейлера: {e}")
        print("Убедитесь, что модель обучена и файлы существуют.")
        return

    # Загрузка и подготовка данных
    try:
        full_df = pd.read_csv(args.data)
        symbol_df = full_df[full_df['symbol'] == args.symbol].copy()
        
        if symbol_df.empty:
            print(f"Символ {args.symbol} не найден в данных.")
            return

        # Применяем те же функции Feature Engineering, что и при обучении
        from feature_engineering import calculate_features, detect_candlestick_patterns, prepare_xlstm_rl_features
        
        # 🔥 УПРОЩЕНО: Для 'close only' feature_cols всегда будет ['close']
        feature_cols = ['close']
        
        # Применяем минимальную обработку к реальным данным
        processed_df = symbol_df.copy() # 🔥 ИЗМЕНЕНО: Просто копируем, без calculate_features и detect_candlestick_patterns
        
        # Убеждаемся, что есть 'close' и нет NaN
        processed_df.dropna(subset=['close'], inplace=True)
        processed_df.reset_index(drop=True, inplace=True)

        print(f"✅ Данные для {args.symbol} подготовлены. Строк: {len(processed_df)}")

        # Визуализация
        plot_predictions(processed_df, model, scaler, feature_cols, config.SEQUENCE_LENGTH)

    except FileNotFoundError:
        print(f"❌ Ошибка: Файл данных '{args.data}' не найден.")
    except Exception as e:
        print(f"❌ Произошла ошибка при подготовке данных или визуализации: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()