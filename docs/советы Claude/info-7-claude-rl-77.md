
Нам нужно сделать адаптивный порог менее агрессивным в сторону снижения.
Изменения в feature_engineering.py (метод calculate_adaptive_threshold)
def calculate_adaptive_threshold(self, df, base_threshold=0.005):
    """
    Рассчитывает адаптивный порог на основе волатильности рынка
    ... (существующий код) ...
    
    # Нормализуем ATR относительно текущей цены
    last_price = df['close'].iloc[-1]
    if last_price > 0:
        normalized_atr = atr / last_price
    else:
        normalized_atr = 0.001
    
    # 🔥 ИСПРАВЛЕНО: Более консервативная настройка множителя для низковолатильных активов
    # Уменьшаем множитель, чтобы не делать порог слишком чувствительным
    atr_multiplier = config.ADAPTIVE_THRESHOLD_MULTIPLIER
    # Если normalized_atr очень низкий, мы можем немного увеличить множитель,
    # но не так агрессивно, как раньше, чтобы не получить слишком много BUY/SELL
    if normalized_atr < 0.0005:  # Если ATR меньше 0.05%
        atr_multiplier = 0.9   # Немного увеличиваем множитель
    elif normalized_atr < 0.001: # Если ATR меньше 0.1%
        atr_multiplier = 0.8
        
    # Рассчитываем адаптивный порог на основе волатильности
    # Минимальный порог 0.05% (0.0005), максимальный 1% (0.01)
    adaptive_threshold = max(0.0005, min(0.01, normalized_atr * atr_multiplier))
    
    # 🔥 ИСПРАВЛЕНО: Убираем принудительное снижение порога,
    # так как теперь мы хотим его немного увеличить для HOLD
    # recommended_threshold_from_log = self._get_recommended_threshold_from_data(df, future_window=config.FUTURE_WINDOW)
    # if recommended_threshold_from_log is not None and recommended_threshold_from_log < adaptive_threshold * 0.5:
    #     print(f"Принудительно снижаем адаптивный порог до рекомендованного: {recommended_threshold_from_log:.6f}")
    #     adaptive_threshold = recommended_threshold_from_log
        
    print(f"[ADAPTIVE] Base threshold: {base_threshold:.6f}, ATR: {normalized_atr:.6f}, "
          f"Adaptive threshold: {adaptive_threshold:.6f}")
    
    return adaptive_threshold

Изменения в config.py:
# Было:
# ADAPTIVE_THRESHOLD_MIN = 0.0001  # Минимальный порог (0.01%)
# ADAPTIVE_THRESHOLD_MAX = 0.02    # Максимальный порог (2%)

# Измените на:
ADAPTIVE_THRESHOLD_MIN = 0.0005  # Минимальный порог (0.05%) - немного повышаем
ADAPTIVE_THRESHOLD_MAX = 0.01    # Максимальный порог (1%) - немного понижаем
ADAPTIVE_THRESHOLD_MULTIPLIER = 0.7 # Вернуть к стандартному значению или немного уменьшить

Исправление ошибки AttributeError: 'str' object has no attribute 'name'
Эта ошибка происходит в CosineDecayCallback, когда он пытается получить доступ к self.model.optimizer.learning_rate. Проблема в том, что self.model.optimizer в этот момент может быть строкой (названием оптимизатора), а не объектом оптимизатора.
Изменения в файле train_model.py (в классе CosineDecayCallback)
class CosineDecayCallback(tf.keras.callbacks.Callback):
    """Кастомный Cosine Decay callback для TensorFlow 2.19.0"""
    def __init__(self, initial_learning_rate, decay_steps, alpha=0.0):
        super().__init__()
        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.alpha = alpha
    
    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.decay_steps:
            cosine_decay = 0.5 * (1 + math.cos(math.pi * epoch / self.decay_steps))
            decayed_learning_rate = (self.initial_learning_rate - self.alpha) * cosine_decay + self.alpha
            
            # 🔥 ИСПРАВЛЕНО: Проверяем, что self.model.optimizer является объектом оптимизатора
            if hasattr(self.model, 'optimizer') and hasattr(self.model.optimizer, 'learning_rate'):
                tf.keras.backend.set_value(self.model.optimizer.learning_rate, decayed_learning_rate)
            else:
                print(f"⚠️ Ошибка: Оптимизатор модели не найден или не имеет атрибута 'learning_rate' на эпохе {epoch}")
                # Если оптимизатор не найден, можно попробовать получить его из logs или пропустить
                # (в зависимости от того, как инициализируется self.model.optimizer)


===========
Вот как нужно изменить код в файле train_model.py (в классе CosineDecayCallback):
class CosineDecayCallback(tf.keras.callbacks.Callback):
    """Кастомный Cosine Decay callback для TensorFlow 2.19.0"""
    def __init__(self, initial_learning_rate, decay_steps, alpha=0.0):
        super().__init__()
        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.alpha = alpha
        # 🔥 ДОБАВЛЕНО: Сохраняем ссылку на оптимизатор после компиляции
        self.optimizer_ref = None 
    
    def on_train_begin(self, logs=None):
        # 🔥 ДОБАВЛЕНО: Получаем ссылку на оптимизатор после начала обучения
        # self.model - это скомпилированная Keras модель (actor_model в данном случае)
        if hasattr(self.model, 'optimizer') and hasattr(self.model.optimizer, 'learning_rate'):
            self.optimizer_ref = self.model.optimizer
        else:
            print("⚠️ Ошибка: Оптимизатор модели не найден или не имеет атрибута 'learning_rate' в on_train_begin.")

    def on_epoch_begin(self, epoch, logs=None):
        if self.optimizer_ref is None:
            print(f"⚠️ Ошибка: Оптимизатор не инициализирован для CosineDecayCallback на эпохе {epoch}.")
            return

        if epoch < self.decay_steps:
            cosine_decay = 0.5 * (1 + math.cos(math.pi * epoch / self.decay_steps))
            decayed_learning_rate = (self.initial_learning_rate - self.alpha) * cosine_decay + self.alpha
            
            tf.keras.backend.set_value(self.optimizer_ref.learning_rate, decayed_learning_rate)
            # 🔥 ДОБАВЛЕНО: Логируем текущую скорость обучения
            print(f"Epoch {epoch+1}: Установлена скорость обучения: {tf.keras.backend.get_value(self.optimizer_ref.learning_rate):.6f}")

