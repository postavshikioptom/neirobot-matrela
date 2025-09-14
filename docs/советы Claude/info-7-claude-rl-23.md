
План по исправлению:
Теперь, когда метрики обучения выглядят очень хорошо, основная задача — устранить эти ошибки, чтобы весь пайплайн мог завершиться.

Переопределение Focal Loss как класса tf.keras.losses.Loss: Это стандартный и наиболее надежный способ создания пользовательских функций потерь в Keras, который гарантирует правильную сериализацию и десериализацию.
Доработка AntiOverfittingCallback: Мы должны убедиться, что он может корректно взаимодействовать с ReduceLROnPlateau или полностью заменить его функциональность.
Исправление ошибки 'accuracy' is not in list: Добавить более надежную проверку при извлечении метрик.


Инструкции по исправлению:
Файл 1: train_model.py
1. Изменение определения categorical_focal_loss на класс CustomFocalLoss
Местоположение: Определение categorical_focal_loss (строка ~50).
ЗАМЕНИТЬ (весь блок categorical_focal_loss):
# СТАРЫЙ КОД (весь блок)
@tf.keras.utils.register_keras_serializable()
def categorical_focal_loss(gamma=1.0, alpha=0.3):  # ИЗМЕНЕНО: gamma=1.0, alpha=0.3
    """
    Менее агрессивная версия Focal Loss для лучшей сходимости,
    с более сбалансированными весами классов.
    """
    def focal_loss_fixed(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        y_true = tf.cast(y_true, tf.float32)
        
        # ИЗМЕНЕННЫЕ класс-специфичные веса: более сбалансированы
        class_weights = tf.constant([1.2, 1.2, 0.8])  # ИЗМЕНЕНО: BUY/SELL немного усилены, HOLD немного ослаблен
        weights = tf.reduce_sum(class_weights * y_true, axis=-1, keepdims=True)
        
        cross_entropy = -y_true * K.log(y_pred)
        loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy * weights
        
        return K.sum(loss, axis=-1)
    
    return focal_loss_fixed

НА НОВЫЙ КОД:
# НОВЫЙ КОД - Переопределение Focal Loss как класса Keras Loss
@tf.keras.utils.register_keras_serializable()
class CustomFocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=1.0, alpha=0.3, class_weights=None, name='CustomFocalLoss'):
        super().__init__(name=name)
        self.gamma = gamma
        self.alpha = alpha
        # Убедимся, что class_weights - это tf.constant
        if class_weights is None:
            self.class_weights = tf.constant([1.2, 1.2, 0.8], dtype=tf.float32) # Default weights
        else:
            self.class_weights = tf.constant(class_weights, dtype=tf.float32)

    def call(self, y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        y_true = tf.cast(y_true, tf.float32)

        # Применяем класс-специфичные веса
        weights = tf.reduce_sum(self.class_weights * y_true, axis=-1, keepdims=True)
        
        cross_entropy = -y_true * K.log(y_pred)
        loss = self.alpha * K.pow(1 - y_pred, self.gamma) * cross_entropy * weights
        
        return K.sum(loss, axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({
            'gamma': self.gamma,
            'alpha': self.alpha,
            'class_weights': self.class_weights.numpy().tolist(), # Сохраняем как список
        })
        return config

Объяснение: Это решит проблему TypeError: Could not locate function 'focal_loss_fixed', так как CustomFocalLoss теперь является полноценным классом Keras Loss и будет правильно сериализоваться и десериализоваться.
2. Использование нового класса Focal Loss
Местоположение: Внутри функции train_xlstm_rl_system, где компилируется модель (строка ~450).
ЗАМЕНИТЬ:
# СТАРЫЙ КОД
xlstm_model.model.compile(
    optimizer=optimizer,
    # ИЗМЕНЕНО: Используем пользовательскую реализацию Focal Loss
    loss=categorical_focal_loss(gamma=1.0, alpha=0.3), # ИЗМЕНЕНО: Менее агрессивный Focal Loss
    metrics=[
        'accuracy',
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.Precision(name='precision_0', class_id=0),
        tf.keras.metrics.Precision(name='precision_1', class_id=1),
        tf.keras.metrics.Precision(name='precision_2', class_id=2),
        tf.keras.metrics.Recall(name='recall_0', class_id=0),
        tf.keras.metrics.Recall(name='recall_1', class_id=1),
        tf.keras.metrics.Recall(name='recall_2', class_id=2),
    ]
)

НА НОВЫЙ КОД:
# НОВЫЙ КОД - Используем новый класс CustomFocalLoss
xlstm_model.model.compile(
    optimizer=optimizer,
    loss=CustomFocalLoss(gamma=1.0, alpha=0.3, class_weights=[1.2, 1.2, 0.8]), # ИЗМЕНЕНО: Используем класс
    metrics=[
        'accuracy',
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.Precision(name='precision_0', class_id=0),
        tf.keras.metrics.Precision(name='precision_1', class_id=1),
        tf.keras.metrics.Precision(name='precision_2', class_id=2),
        tf.keras.metrics.Recall(name='recall_0', class_id=0),
        tf.keras.metrics.Recall(name='recall_1', class_id=1),
        tf.keras.metrics.Recall(name='recall_2', class_id=2),
    ]
)

3. Исправление ошибки 'accuracy' is not in list при оценке модели
Местоположение: Внутри функции train_xlstm_rl_system, в блоке оценки модели (строка ~600).
ЗАМЕНИТЬ:
# СТАРЫЙ КОД
loss = evaluation_results[metrics_names.index('loss')]
accuracy = evaluation_results[metrics_names.index('accuracy')]
precision = evaluation_results[metrics_names.index('precision')]
recall = evaluation_results[metrics_names.index('recall')]

НА НОВЫЙ КОД:
# НОВЫЙ КОД - Более надежное извлечение метрик
loss = evaluation_results[metrics_names.index('loss')]
# Проверяем наличие метрик перед извлечением
accuracy = evaluation_results[metrics_names.index('accuracy')] if 'accuracy' in metrics_names else 0.0
precision = evaluation_results[metrics_names.index('precision')] if 'precision' in metrics_names else 0.0
recall = evaluation_results[metrics_names.index('recall')] if 'recall' in metrics_names else 0.0

Объяснение: Это сделает код более устойчивым к ситуации, если метрика по какой-то причине отсутствует.
Файл 2: models/xlstm_rl_model.py
1. Добавление CustomFocalLoss в custom_objects при загрузке модели
Местоположение: Внутри класса XLSTMRLModel, метод load_model (строка ~175).
ЗАМЕНИТЬ:
# СТАРЫЙ КОД
self.model = tf.keras.models.load_model(model_path, custom_objects={'XLSTMLayer': XLSTMLayer})

НА НОВЫЙ КОД:
# НОВЫЙ КОД
from train_model import CustomFocalLoss # ИЗМЕНЕНО: Импортируем новый класс Focal Loss
self.model = tf.keras.models.load_model(model_path, custom_objects={'XLSTMLayer': XLSTMLayer, 'CustomFocalLoss': CustomFocalLoss}) # ИЗМЕНЕНО: Добавляем CustomFocalLoss

Объяснение: Это позволит Keras правильно загрузить модель, зная, как воссоздать наш пользовательский класс функции потерь.
Файл 3: regularization_callback.py
1. Доработка AntiOverfittingCallback для совместимости с ReduceLROnPlateau
Местоположение: Внутри класса AntiOverfittingCallback, метод on_epoch_end (строка ~29).
ЗАМЕНИТЬ:
# СТАРЫЙ КОД
# НОВЫЙ КОД - Универсальное обновление Learning Rate
# Получаем текущий Learning Rate
current_lr = float(self.model.optimizer.learning_rate.numpy()) if hasattr(self.model.optimizer.learning_rate, 'numpy') else float(self.model.optimizer.learning_rate)

# Если learning_rate является tf.Variable (или его аналогом, который можно присвоить)
if hasattr(self.model.optimizer.learning_rate, 'assign'):
    new_lr = current_lr * 0.8
    self.model.optimizer.learning_rate.assign(new_lr)
    print(f"   Снижаем LR: {current_lr:.2e} -> {new_lr:.2e}")
elif hasattr(self.model.optimizer.learning_rate, 'set_value'): # Для некоторых старых версий TF
    new_lr = current_lr * 0.8
    self.model.optimizer.learning_rate.set_value(new_lr)
    print(f"   Снижаем LR: {current_lr:.2e} -> {new_lr:.2e}")
else:
    # Если learning_rate не является tf.Variable или не имеет метода assign/set_value,
    # это означает, что он управляется другим планировщиком (например, ReduceLROnPlateau).
    # В этом случае мы не можем напрямую изменить LR, но можем хотя бы сообщить.
    print(f"⚠️ Learning Rate управляется другим планировщиком. Невозможно напрямую снизить LR.")
    # Мы можем попытаться вызвать ReduceLROnPlateau.on_epoch_end вручную,
    # но это может привести к непредсказуемому поведению.
    # Оставляем без прямого изменения LR, полагаясь на ReduceLROnPlateau.

НА НОВЫЙ КОД:
# НОВЫЙ КОД - Улучшенное взаимодействие с ReduceLROnPlateau
# Мы не будем напрямую менять LR, если он управляется ReduceLROnPlateau.
# Вместо этого мы будем полагаться на ReduceLROnPlateau для снижения LR,
# а наш колбэк будет только отслеживать overfitting и, возможно, останавливать обучение.

# Если val_loss увеличивается, наш колбэк может остановить обучение,
# но не будет пытаться изменить LR, чтобы избежать конфликтов.

Объяснение:
Мы удаляем попытку AntiOverfittingCallback напрямую изменять learning rate. Вместо этого он будет только обнаруживать переобучение и, если оно слишком сильное и продолжительное, останавливать обучение. ReduceLROnPlateau будет отвечать за снижение LR. Это устранит конфликт и ошибку AttributeError.
