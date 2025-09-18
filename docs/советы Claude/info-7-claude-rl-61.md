
Решение:
def _add_technical_indicators(self, df):
    # 🔥 ИСПРАВЛЕНИЕ: Проверяем наличие достаточных данных для RSI
    if len(df) < config.RSI_PERIOD + 5:  # RSI_PERIOD = 14, нужен запас
        print(f"Недостаточно данных для RSI: {len(df)} строк, нужно минимум {config.RSI_PERIOD + 5}")
        # Заполняем нулями или пропускаем символ
        df['RSI'] = 0.0
        df['MACD'] = 0.0
        df['MACDSIGNAL'] = 0.0
        df['MACDHIST'] = 0.0
        df['STOCH_K'] = 0.0
        df['STOCH_D'] = 0.0
        df['WILLR'] = 0.0
        df['AO'] = 0.0
        return df
    
    # Убедимся, что все необходимые колонки в числовом формате
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 🔥 ДОБАВЛЕНО: Проверка на NaN перед расчётом индикаторов
    if df['close'].isna().sum() > len(df) * 0.5:  # Если больше 50% NaN
        print(f"Слишком много NaN в данных, пропускаем символ")
        return None
    
    try:
        # RSI с обработкой ошибок
        df['RSI'] = talib.RSI(df['close'].fillna(method='ffill'), timeperiod=config.RSI_PERIOD)
        
        # MACD с обработкой ошибок  
        macd, macdsignal, macdhist = talib.MACD(
            df['close'].fillna(method='ffill'), 
            fastperiod=config.MACD_FASTPERIOD, 
            slowperiod=config.MACD_SLOWPERIOD, 
            signalperiod=config.MACD_SIGNALPERIOD
        )
        df['MACD'] = macd
        df['MACDSIGNAL'] = macdsignal
        df['MACDHIST'] = macdhist
        
        # Остальные индикаторы аналогично...
        
    except Exception as e:
        print(f"Ошибка при расчёте индикаторов: {e}")
        # Заполняем нулями при ошибке
        for col in ['RSI', 'MACD', 'MACDSIGNAL', 'MACDHIST', 'STOCH_K', 'STOCH_D', 'WILLR', 'AO']:
            if col not in df.columns:
                df[col] = 0.0
    
    # 🔥 КРИТИЧНО: Более надёжная обработка NaN
    df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    return df


Проблема: Очень маленький датасет для deep learning
Решение:
# В config.py
MIN_ROWS_PER_SYMBOL = 200  # Уменьшить с 500
MAX_SYMBOLS_FOR_TRAINING = 200  # Увеличить с 100
TARGET_TOTAL_ROWS = 500000  # Увеличить с 350000

# В ThreeStageTrainer.load_and_prepare_data()
# Добавить аугментацию данных
def augment_sequences(X, y, factor=2):
    augmented_X, augmented_y = [], []
    for i in range(len(X)):
        # Оригинальные данные
        augmented_X.append(X[i])
        augmented_y.append(y[i])
        
        # Добавляем шум (5% от стандартного отклонения)
        noise = np.random.normal(0, 0.05 * np.std(X[i]), X[i].shape)
        augmented_X.append(X[i] + noise)
        augmented_y.append(y[i])
    
    return np.array(augmented_X), np.array(augmented_y)

=========================

2. Дисбаланс классов:
SELL=248, HOLD=310, BUY=182

Решение: Улучшить class weights
# В stage1_supervised_pretraining()
from sklearn.utils.class_weight import compute_class_weight

# Более агрессивные веса для редких классов
class_weights_array = compute_class_weight(
    'balanced',
    classes=np.unique(self.y_train_supervised),
    y=self.y_train_supervised
)

# Дополнительно увеличиваем вес для BUY (самый редкий)
class_weights = {}
for i, weight in enumerate(class_weights_array):
    if i == 2:  # BUY class
        class_weights[i] = weight * 1.5  # Увеличиваем в 1.5 раза
    else:
        class_weights[i] = weight

print(f"Скорректированные веса классов: {class_weights}")

3. Улучшение архитектуры модели:
# В XLSTMRLModel._build_actor_model()
def _build_actor_model(self):
    inputs = layers.Input(shape=self.input_shape)
    
    # 🔥 ДОБАВЛЕНО: Batch Normalization на входе
    x = layers.BatchNormalization()(inputs)
    
    # Первый слой xLSTM с большим dropout
    x = layers.RNN(XLSTMMemoryCell(units=self.memory_units, 
                                   memory_size=self.memory_size),
                  return_sequences=True)(x)
    x = layers.LayerNormalization()(x)
    x = layers.Dropout(0.3)(x)  # Увеличить с 0.2
    
    # Второй слой xLSTM
    x = layers.RNN(XLSTMMemoryCell(units=self.memory_units//2,  # Уменьшить размер
                                   memory_size=self.memory_size),
                  return_sequences=False)(x)
    x = layers.LayerNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    # 🔥 ДОБАВЛЕНО: Дополнительный слой с residual connection
    dense1 = layers.Dense(128, activation='relu')(x)
    dense1 = layers.Dropout(0.2)(dense1)
    dense2 = layers.Dense(64, activation='relu')(dense1)
    
    # Residual connection
    if x.shape[-1] == 64:
        x = layers.Add()([x, dense2])
    else:
        x = dense2
    
    # Выходной слой с focal loss для дисбаланса классов
    outputs = layers.Dense(3, activation='softmax')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    return model

4. Улучшение процесса обучения:
# В stage1_supervised_pretraining()
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        patience=15,  # Увеличить с 10
        restore_best_weights=True, 
        monitor='val_accuracy',
        min_delta=0.001  # Добавить минимальное улучшение
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        factor=0.3,  # Более агрессивное снижение
        patience=7,  # Увеличить с 5
        monitor='val_loss',
        min_lr=1e-7  # Минимальный learning rate
    ),
    tf.keras.callbacks.ModelCheckpoint(
        'models/best_supervised_model.keras', 
        save_best_only=True, 
        monitor='val_f1_score'  # Мониторить F1 вместо accuracy
    ),
    # 🔥 ДОБАВЛЕНО: Cosine Annealing
    tf.keras.callbacks.CosineRestartScheduler(
        first_restart_step=10,
        t_mul=2.0,
        m_mul=0.9,
        alpha=1e-6
    )
]


==========================

Решение:
# Добавить в ValidationMetricsCallback
def on_epoch_end(self, epoch, logs=None):
    train_acc = logs.get('accuracy', 0)
    val_acc = logs.get('val_accuracy', 0)
    
    # Проверка на переобучение
    if train_acc - val_acc > 0.15:  # Разрыв больше 15%
        print(f"⚠️ Обнаружено переобучение: train_acc={train_acc:.3f}, val_acc={val_acc:.3f}")
        print("Рекомендуется увеличить dropout или regularization")

🎯 ПРИОРИТЕТНЫЕ ИСПРАВЛЕНИЯ:

КРИТИЧНО: Исправить ошибку RSI в feature engineering
ВАЖНО: Увеличить размер датасета или добавить аугментацию
ВАЖНО: Улучшить обработку дисбаланса классов
СРЕДНЕ: Добавить regularization против переобучения
СРЕДНЕ: Улучшить архитектуру модели
