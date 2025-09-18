Исправление в models/xlstm_rl_model.py:
# 🔥 ЗАМЕНИТЬ существующую f1_score функцию на:
import tensorflow_addons as tfa

def compile_for_supervised_learning(self):
    """Компилирует модель для этапа 1: Supervised Learning"""
    self.actor_model.compile(
        optimizer=self.supervised_optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy', tfa.metrics.F1Score(num_classes=3, average='macro')]  # 🔥 ПРАВИЛЬНАЯ F1 для multiclass
    )
    print("✅ Модель скомпилирована для supervised learning")

# Альтернативно, если tensorflow_addons недоступен:
class MulticlassF1Score(tf.keras.metrics.Metric):
    def __init__(self, num_classes=3, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.f1_scores = []
        for i in range(num_classes):
            self.f1_scores.append(tf.keras.metrics.F1Score(threshold=0.5))
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis=-1)
        for i in range(self.num_classes):
            y_true_binary = tf.cast(tf.equal(y_true, i), tf.float32)
            y_pred_binary = tf.cast(tf.equal(y_pred, i), tf.float32)
            self.f1_scores[i].update_state(y_true_binary, y_pred_binary, sample_weight)
    
    def result(self):
        return tf.reduce_mean([f1.result() for f1 in self.f1_scores])
    
    def reset_state(self):
        for f1 in self.f1_scores:
            f1.reset_state()

2. Residual Connection - ЛОГИЧЕСКАЯ ОШИБКА ✅
Согласен! Размерность после RNN может быть не 64.
Исправление в models/xlstm_rl_model.py:
def _build_actor_model(self):
    # ... предыдущий код ...
    
    # 🔥 ИСПРАВЛЕНИЕ: Дополнительный слой с правильным residual connection
    dense1 = layers.Dense(128, activation='relu')(x)
    dense1 = layers.Dropout(0.2)(dense1)
    dense2 = layers.Dense(64, activation='relu')(dense1)
    
    # 🔥 ИСПРАВЛЕНО: Приводим x к размеру 64 перед Add
    x_resized = layers.Dense(64)(x)  # Приводим к нужному размеру
    x = layers.Add()([x_resized, dense2])  # Теперь размеры совпадают
    
    # Выходной слой
    outputs = layers.Dense(3, activation='softmax')(x)

3. Deprecated fillna method ✅
Полностью согласен! method='ffill' deprecated в pandas 2.0+
Исправление в feature_engineering.py:
def _add_technical_indicators(self, df):
    try:
        # 🔥 ИСПРАВЛЕНО: Заменяем deprecated method
        df['RSI'] = talib.RSI(df['close'].ffill(), timeperiod=config.RSI_PERIOD)  # 🔥 ИЗМЕНЕНО
        
        macd, macdsignal, macdhist = talib.MACD(
            df['close'].ffill(),  # 🔥 ИЗМЕНЕНО
            fastperiod=config.MACD_FASTPERIOD, 
            slowperiod=config.MACD_SLOWPERIOD, 
            signalperiod=config.MACD_SIGNALPERIOD
        )
        # ... остальные индикаторы аналогично ...
        
    except Exception as e:
        # ... обработка ошибок ...
    
    # 🔥 ИСПРАВЛЕНО: Более надёжная обработка NaN
    df = df.ffill().bfill().fillna(0)  # 🔥 ИЗМЕНЕНО
    
    return df

4. Memory Leak в predict методах ✅
Согласен! Нужна периодическая очистка памяти.
Исправление в models/xlstm_rl_model.py и других файлах:
class XLSTMRLModel:
    def __init__(self, input_shape, memory_size=64, memory_units=128):
        # ... существующий код ...
        self.prediction_count = 0  # 🔥 ДОБАВЛЕНО: Счётчик предсказаний
    
    def predict_action(self, state):
        """Предсказывает действие на основе состояния"""
        if len(state.shape) == 2:
            state = np.expand_dims(state, axis=0)
        
        if state.dtype != np.float32:
            state = state.astype(np.float32)
        
        action_probs = self.actor_model.predict(state, verbose=0)[0]
        
        # 🔥 ДОБАВЛЕНО: Периодическая очистка памяти
        self.prediction_count += 1
        if self.prediction_count % 100 == 0:  # Каждые 100 предсказаний
            tf.keras.backend.clear_session()
            print(f"Очистка памяти после {self.prediction_count} предсказаний")
        
        return action_probs

5. Дублирование return в stage3_rl_finetuning ✅
Согласен! Есть дублированный return.
Исправление в train_model.py:
def stage3_rl_finetuning(self):
    # ... весь код метода ...
    
    print("=== РЕЗУЛЬТАТЫ RL FINE-TUNING ===")
    print(f"Лучшая прибыль на валидации: {best_val_profit:.2f}%")
    print(f"Средняя награда за эпизод: {np.mean(rl_metrics['episode_rewards']):.4f}")
    print(f"Средняя прибыль за эпизод: {np.mean(rl_metrics['episode_profits']):.2f}%")
    
    self._plot_rl_metrics(rl_metrics)
    
    return rl_metrics  # 🔥 УДАЛИТЬ дублированный return
    # return rl_metrics  # 🔥 УДАЛЕНО: Дублированный return

6. Отсутствие проверки XLSTMMemoryCell ✅
Согласен! Нужна безопасная проверка импорта.
Исправление в models/xlstm_rl_model.py:
# 🔥 ДОБАВЛЕНО: Безопасный импорт
try:
    from models.xlstm_memory_cell import XLSTMMemoryCell
except ImportError as e:
    print(f"❌ Ошибка импорта XLSTMMemoryCell: {e}")
    print("Убедитесь, что файл models/xlstm_memory_cell.py существует")
    raise ImportError("XLSTMMemoryCell не найден")

# В методе load():
def load(self, path='models', stage=""):
    # ... код загрузки ...
    
    # 🔥 ИСПРАВЛЕНО: Безопасная загрузка с проверкой
    try:
        if os.path.exists(actor_path) and os.path.exists(critic_path):
            self.actor_model = tf.keras.models.load_model(
                actor_path, 
                custom_objects={'XLSTMMemoryCell': XLSTMMemoryCell, 'f1_score': MulticlassF1Score}
            )
            # ... остальной код ...
        else:
            print(f"Не удалось найти сохраненные модели для этапа: {stage}")
    except Exception as e:
        print(f"❌ Ошибка при загрузке моделей: {e}")
        return False

7. Неправильная индексация в _create_sequences ✅
Согласен! Хардкод индекса 3 небезопасен.
Исправление в feature_engineering.py:
def _create_sequences(self, data):
    """Создает последовательности для обучения"""
    X = []
    y_close = []
    
    # 🔥 ИСПРАВЛЕНО: Безопасное определение индекса 'close'
    try:
        close_index = self.base_features.index('close')
    except ValueError:
        print("❌ Колонка 'close' не найдена в base_features")
        close_index = 3  # Fallback к индексу 3
    
    for i in range(len(data) - self.sequence_length):
        X.append(data[i:i+self.sequence_length])
        # 🔥 ИСПРАВЛЕНО: Использование динамического индекса
        y_close.append(data[i+self.sequence_length, close_index])
    
    return np.array(X), np.array(y_close)

🎯 ДОПОЛНИТЕЛЬНЫЕ ИСПРАВЛЕНИЯ:
В config.py добавить:
# 🔥 ДОБАВЛЕНО: Параметры для предотвращения memory leak
MEMORY_CLEANUP_FREQUENCY = 100  # Очистка каждые 100 предсказаний
MAX_PREDICTION_BATCH_SIZE = 32  # Максимальный размер батча

В validation_metrics_callback.py добавить проверку переобучения:
def on_epoch_end(self, epoch, logs=None):
    if (epoch + 1) % 5 == 0:
        # ... существующий код ...
        
        # 🔥 ДОБАВЛЕНО: Проверка на переобучение
        train_acc = logs.get('accuracy', 0) if logs else 0
        val_acc = logs.get('val_accuracy', 0) if logs else 0
        
        if train_acc - val_acc > 0.15:  # Разрыв больше 15%
            print(f"⚠️ Обнаружено переобучение: train_acc={train_acc:.3f}, val_acc={val_acc:.3f}")
            print("Рекомендуется увеличить dropout или regularization")
