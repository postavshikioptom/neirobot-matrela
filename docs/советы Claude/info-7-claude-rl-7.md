

Почему GPU завис на 43/100:
Скорее всего проблемы с памятью или бесконечным циклом в RL обучении. В коде есть потенциальные проблемы:
# Этот цикл может зависнуть если данных мало
for i, symbol in enumerate(rl_symbols):
    # ...
    rl_agent.train_with_callbacks(
        total_timesteps=50000,  # Много шагов
        eval_freq=2000
    )

Рекомендации для TPU:

Уменьшите RL часть:

total_timesteps=10000  # Вместо 50000


Увеличьте batch_size для xLSTM:

batch_size=64  # Вместо 32


Добавьте мониторинг:

print(f"Обучение RL агента: {i+1}/{len(rl_symbols)}, timesteps: {timesteps}/50000")

Вот улучшения для предотвращения зависания без потери качества:
1. Улучшенное управление памятью GPU
train_model.py

import gc
import tensorflow as tf

# В начале скрипта добавьте:
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

configure_gpu_memory()

2. Добавьте периодическую очистку памяти
def train_xlstm_rl_system(X, y, processed_dfs, feature_cols):
    print("\n=== ЭТАП 1: ОБУЧЕНИЕ xLSTM МОДЕЛИ ===")
    
    # Разделяем данные
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    # ДОБАВЬТЕ: Принудительная очистка памяти
    gc.collect()
    tf.keras.backend.clear_session()
    
    print(f"Обучающая выборка: {len(X_train)}")
    print(f"Валидационная выборка: {len(X_val)}")
    print(f"Тестовая выборка: {len(X_test)}")
    
    # Создаем модель
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
                tf.keras.backend.clear_session()
                print(f"Эпоха {epoch}: Память очищена")
    
    class DetailedProgressCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
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
                verbose=1
            )
        ]
    )

3. Улучшите XLSTMRLModel класс
В models/xlstm_rl_model.py добавьте:
def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32, custom_callbacks=None):
    """Обучение с улучшенной стабильностью"""
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=25,  # Увеличили терпение
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'models/xlstm_checkpoint_epoch_{epoch:02d}.keras',
            monitor='val_loss',
            save_best_only=False,  # Сохраняем каждые несколько эпох
            save_freq=10 * len(X_train) // batch_size,  # Каждые 10 эпох
            verbose=1
        ),
        tf.keras.callbacks.CSVLogger('training_log.csv', append=True)
    ]
    
    if custom_callbacks:
        callbacks.extend(custom_callbacks)
    
    # ДОБАВЬТЕ: Градиентное обрезание для стабильности
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.001,
        clipnorm=1.0  # Обрезание градиентов
    )
    self.model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    
    # Обучение с validation_split=0 (используем переданные данные)
    history = self.model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
        shuffle=True
    )
    
    return history

4. Добавьте проверочные точки и восстановление
def train_xlstm_rl_system(X, y, processed_dfs, feature_cols):
    # В начале функции добавьте:
    checkpoint_path = 'models/xlstm_checkpoint_latest.keras'
    
    # Проверяем, есть ли сохраненная модель
    if os.path.exists(checkpoint_path):
        print("Найдена сохраненная модель, загружаем...")
        try:
            xlstm_model = XLSTMRLModel(
                input_shape=(X_train.shape[1], X_train.shape[2]),
                memory_units=128,
                attention_units=64
            )
            xlstm_model.model = tf.keras.models.load_model(checkpoint_path)
            print("✅ Модель загружена, продолжаем обучение")
        except:
            print("⚠️ Не удалось загрузить модель, начинаем заново")
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

5. Улучшите RL часть
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

Эти изменения должны значительно повысить стабильность без потери качества обучения.