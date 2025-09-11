import pandas as pd
import numpy as np
import argparse
import os
import gc
import pickle  # ДОБАВЬТЕ эту строку
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

import tensorflow as tf

def configure_gpu_memory():
    """Настройка GPU для стабильной работы"""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Включаем рост памяти + ограничиваем максимум
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                # Ограничиваем память до 80% от доступной
                tf.config.experimental.set_virtual_device_configuration(
                    gpu,
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*10)]  # 10GB max
                )
            print(f"✅ GPU память настроена для {len(gpus)} устройств")
        except RuntimeError as e:
            print(f"⚠️ Ошибка настройки GPU: {e}")
    else:
        print("⚠️ GPU не найден, будет использоваться CPU")

configure_gpu_memory()

# ✅ НАСТРОЙКИ ДЛЯ СОВМЕСТИМОСТИ С РАЗНЫМИ СРЕДАМИ
# Отключаем XLA если возникают проблемы (можно включить обратно)
# os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'

# Новые импорты
from feature_engineering import calculate_features, detect_candlestick_patterns, calculate_vsa_features
from models.xlstm_rl_model import XLSTMRLModel
from rl_agent import IntelligentRLAgent
from trading_env import TradingEnvRL
from hybrid_decision_maker import HybridDecisionMaker

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
        # df['target'] = 2  # По умолчанию HOLD - эту строку мы теперь устанавливаем ниже

        # BUY: положительная доходность + VSA подтверждение покупки (СНИЖЕНЫ ПОРОГИ)
        buy_condition = (
            (df['future_return'] > 0.003) &  # СНИЖЕНО с 0.01 до 0.003 (0.3% роста)
            ((df['vsa_no_supply'] == 1) | (df['vsa_stopping_volume'] == 1) | (df['vsa_strength'] > 0.5)) # СНИЖЕНО с 1 до 0.5
        )
        
        # SELL: отрицательная доходность + VSA подтверждение продажи (СНИЖЕНЫ ПОРОГИ)
        sell_condition = (
            (df['future_return'] < -0.003) &  # СНИЖЕНО с -0.01 до -0.003 (-0.3% падения)
            ((df['vsa_no_demand'] == 1) | (df['vsa_climactic_volume'] == 1) | (df['vsa_strength'] < -0.5)) # СНИЖЕНО с -1 до -0.5
        )
        
        # Сначала устанавливаем все в HOLD, затем переписываем
        df['target'] = 2  # По умолчанию HOLD
        df.loc[buy_condition, 'target'] = 0 # BUY
        df.loc[sell_condition, 'target'] = 1 # SELL

        # ДОБАВЬТЕ: Принудительная балансировка классов (если необходимо)
        # Этот блок можно включать, если после ослабления порогов баланс все еще очень плохой.
        # Он попытается переклассифицировать часть "HOLD" в BUY/SELL на основе других индикаторов.
        # Это может быть "грязным" решением, но иногда необходимо для обучения.
        current_buy_count = (df['target'] == 0).sum()
        current_sell_count = (df['target'] == 1).sum()
        current_hold_count = (df['target'] == 2).sum()

        if current_hold_count > (current_buy_count + current_sell_count) * 2: # Если HOLD в 2+ раза больше
            print(f"⚠️ Сильный дисбаланс классов. Попытка переклассификации части HOLD-сигналов.")
            hold_indices = df[df['target'] == 2].index
            
            import random
            random.seed(42) # Для воспроизводимости
            
            # Переклассифицируем 15% HOLD в BUY/SELL на основе RSI, ADX
            reclassify_count = int(current_hold_count * 0.15)
            if reclassify_count > 0:
                reclassify_indices = random.sample(list(hold_indices), min(reclassify_count, len(hold_indices)))
                
                for idx in reclassify_indices:
                    # Простая логика: если RSI < 35 и ADX растет -> BUY
                    if df.loc[idx, 'RSI_14'] < 35 and df.loc[idx, 'ADX_14'] > df.loc[idx-1, 'ADX_14']:
                        df.loc[idx, 'target'] = 0  # BUY
                    # Если RSI > 65 и ADX растет -> SELL
                    elif df.loc[idx, 'RSI_14'] > 65 and df.loc[idx, 'ADX_14'] > df.loc[idx-1, 'ADX_14']:
                        df.loc[idx, 'target'] = 1  # SELL
            
            print(f"Баланс классов после переклассификации:")
            unique, counts = np.unique(df['target'], return_counts=True)
            class_names = ['BUY', 'SELL', 'HOLD']
            for class_idx, count in zip(unique, counts):
                print(f"  {class_names[class_idx]}: {count} ({count/len(df)*100:.1f}%)")
        
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
    
    # ДОБАВЬТЕ: Проверка баланса классов
    print(f"Баланс классов:")
    unique, counts = np.unique(np.argmax(y, axis=1), return_counts=True)
    class_names = ['BUY', 'SELL', 'HOLD']
    for class_idx, count in zip(unique, counts):
        print(f"  {class_names[class_idx]}: {count} ({count/len(y)*100:.1f}%)")
    
    return X, y, processed_dfs, feature_cols

def train_xlstm_rl_system(X, y, processed_dfs, feature_cols):
    """
    Обучает единую систему xLSTM + RL
    """
    print("\n=== ЭТАП 1: ОБУЧЕНИЕ xLSTM МОДЕЛИ ===")
    
    # ДОБАВЬТЕ: Создание директории для моделей
    os.makedirs('models', exist_ok=True)
    
    # Разделяем данные
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    # ДОБАВЬТЕ: Принудительная очистка памяти
    gc.collect()
    tf.keras.backend.clear_session()
    
    print(f"Обучающая выборка: {len(X_train)}")
    print(f"Валидационная выборка: {len(X_val)}")
    print(f"Тестовая выборка: {len(X_test)}")
    
    # В функции train_xlstm_rl_system(), после создания xlstm_model добавьте:
    print(f"Форма данных для обучения: X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"Количество признаков: {len(feature_cols)}")

    # Проверяем наличие NaN/Inf в данных перед обучением
    if np.any(np.isnan(X_train)) or np.any(np.isinf(X_train)):
        print("⚠️ Обнаружены NaN/Inf в тренировочных данных!")
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
        print("✅ NaN/Inf исправлены")
        
    # Проверяем, есть ли сохраненная модель
    checkpoint_path = 'models/xlstm_checkpoint_latest.keras'
    scaler_path = 'models/xlstm_rl_scaler.pkl'

    if os.path.exists(checkpoint_path):
        print("Найдена сохраненная модель, загружаем...")
        try:
            xlstm_model = XLSTMRLModel(
                input_shape=(X_train.shape[1], X_train.shape[2]),
                memory_units=128,
                attention_units=64
            )
            # ИСПРАВЛЕНО: Загружаем модель и scaler
            xlstm_model.model = tf.keras.models.load_model(checkpoint_path)
            
            # Загружаем scaler если он существует
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    xlstm_model.scaler = pickle.load(f)
                xlstm_model.is_trained = True
                print("✅ Модель и scaler загружены, продолжаем обучение")
            else:
                print("⚠️ Scaler не найден, будет создан новый")
                
        except Exception as e:
            print(f"⚠️ Не удалось загрузить модель: {e}, начинаем заново")
            xlstm_model = XLSTMRLModel(
                input_shape=(X_train.shape[1], X_train.shape[2]),
                memory_units=128,
                attention_units=64
            )
    else:
        xlstm_model = XLSTMRLModel(
            input_shape=(X_train.shape[1], X_train.shape[2]),
            memory_units=128,
            attention_units=64
        )

    # ДОБАВЬТЕ: Кастомные колбэки для мониторинга и очистки памяти
    class MemoryCleanupCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if epoch % 10 == 0:  # Каждые 10 эпох
                gc.collect()
                # ИСПРАВЛЕНО: НЕ очищаем сессию во время обучения
                # tf.keras.backend.clear_session()  # Это может сломать обучение!
                print(f"Эпоха {epoch}: Память очищена")
    
    class DetailedProgressCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            try:
                # Добавляем больше метрик для мониторинга
                lr = self.model.optimizer.learning_rate.numpy()
                print(f"Эпоха {epoch+1}/100 - loss: {logs['loss']:.4f} - val_loss: {logs['val_loss']:.4f} - lr: {lr:.2e}")
                
                # Проверяем на переобучение
                if logs['val_loss'] > logs['loss'] * 2:
                    print("⚠️ Возможное переобучение!")
            except Exception as e:
                # Fallback если не удается получить learning rate
                print(f"Эпоха {epoch+1}/100 - loss: {logs['loss']:.4f} - val_loss: {logs['val_loss']:.4f}")
            
    # Обучение с улучшенными колбэками
    history = xlstm_model.train(
        X_train, y_train,
        X_val, y_val,
        epochs=100,
        batch_size=16,  # УМЕНЬШИЛИ batch_size для стабильности
        custom_callbacks=[
            MemoryCleanupCallback(),
            DetailedProgressCallback(),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=15,
                min_lr=1e-7,
                verbose=0 # <-- ИЗМЕНЕНО: 0 для отключения логирования изменений LR
            )
        ]
    )

    # После завершения обучения xLSTM, добавьте:
    print(f"\n📊 Статистика обучения xLSTM:")
    print(f"Финальная loss: {history.history['loss'][-1]:.4f}")
    print(f"Финальная val_loss: {history.history['val_loss'][-1]:.4f}")
    print(f"Лучшая val_loss: {min(history.history['val_loss']):.4f}")
    print(f"Количество эпох: {len(history.history['loss'])}")
    
    # Оценка xLSTM
    try:
        X_test_scaled = xlstm_model.scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
        loss, accuracy, precision, recall = xlstm_model.model.evaluate(X_test_scaled, y_test, verbose=0)
        print(f"xLSTM Точность: {accuracy * 100:.2f}%")
        print(f"xLSTM Precision: {precision * 100:.2f}%")
        print(f"xLSTM Recall: {recall * 100:.2f}%")
    except Exception as e:
        print(f"⚠️ Ошибка при оценке модели: {e}")
    
    # Сохраняем xLSTM модель
    xlstm_model.save_model()

    # После обучения xlstm_model, обучите детектор режимов
    # Возьмите достаточно большой исторический DataFrame для обучения детектора
    # Например, объедините несколько символов или возьмите один большой
    regime_training_df = pd.concat(list(processed_dfs.values())).reset_index(drop=True)
    decision_maker_temp = HybridDecisionMaker(
        xlstm_model_path='models/xlstm_rl_model.keras',
        rl_agent_path='models/rl_agent_BTCUSDT', # Временно, он не будет использоваться для принятия решений
        feature_columns=feature_cols,
        sequence_length=X.shape[1] # Передаем sequence_length
    )
    decision_maker_temp.fit_regime_detector(regime_training_df, xlstm_model, feature_cols)
    decision_maker_temp.regime_detector.save_detector('models/market_regime_detector.pkl')
    print("✅ Детектор режимов сохранен")
    
    print("\n=== ЭТАП 2: ОБУЧЕНИЕ RL АГЕНТА ===")
    
    # ДОБАВЬТЕ: Ограничиваем количество символов для стабильности
    rl_symbols = list(processed_dfs.keys())[:2]  # Только 2 символа вместо 3
    
    rl_agent = None
    for i, symbol in enumerate(rl_symbols):
        df = processed_dfs[symbol]
        print(f"\nОбучение RL на символе {symbol} ({i+1}/{len(rl_symbols)})")
        
        # ДОБАВЬТЕ: Очистка памяти перед каждым RL агентом
        gc.collect()
        
        # Разделяем данные
        split_idx = int(len(df) * 0.8)
        train_df = df.iloc[:split_idx].copy()
        eval_df = df.iloc[split_idx:].copy()
        
        if len(train_df) < 100 or len(eval_df) < 50:
            print(f"⚠️ Недостаточно данных для {symbol}, пропускаем")
            continue
            
        # Создаем RL агента
        rl_agent = IntelligentRLAgent(algorithm='PPO')
        
        try:
            # Создаем среды
            vec_env = rl_agent.create_training_environment(train_df, xlstm_model)
            rl_agent.create_evaluation_environment(eval_df, xlstm_model)
            
            # Строим и обучаем агента
            rl_agent.build_agent(vec_env)
            
            # ДОБАВЬТЕ: Обучение меньшими порциями с сохранениями
            for step in range(0, 50000, 10000):  # По 10k шагов
                print(f"RL обучение: шаги {step}-{min(step+10000, 50000)}")
                rl_agent.train_with_callbacks(
                    total_timesteps=10000,
                    eval_freq=2000
                )
                # Сохраняем промежуточные результаты
                rl_agent.save_agent(f'models/rl_agent_{symbol}_step_{step}')
                gc.collect()  # Очищаем память
            
            # Финальное сохранение
            rl_agent.save_agent(f'models/rl_agent_{symbol}')
            
        except Exception as e:
            print(f"⚠️ Ошибка при обучении RL для {symbol}: {e}")
            continue
    
    print("\n=== ОБУЧЕНИЕ ЗАВЕРШЕНО ===")
    
    print("\n🔍 Проверка сохраненных файлов:")
    saved_files = [
        'models/xlstm_rl_model.keras',
        'models/xlstm_rl_scaler.pkl',
        'models/market_regime_detector.pkl'
    ]

    for file_path in saved_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path) / (1024*1024)  # MB
            print(f"✅ {file_path} ({size:.1f} MB)")
        else:
            print(f"❌ {file_path} не найден")

    # Проверяем RL агентов
    for symbol in rl_symbols:
        rl_path = f'models/rl_agent_{symbol}'
        if os.path.exists(rl_path + '.zip'):
            size = os.path.getsize(rl_path + '.zip') / (1024*1024)
            print(f"✅ {rl_path}.zip ({size:.1f} MB)")
            
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
