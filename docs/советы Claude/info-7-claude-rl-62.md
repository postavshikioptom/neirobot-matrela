🔥 ТОЧНЫЕ ИЗМЕНЕНИЯ В КОДЕ:
1. В файле xlstm_rl_model.py:
# ДОБАВИТЬ в начало файла после импортов:
import tensorflow.keras.backend as K

def f1_score(y_true, y_pred):
    """Кастомная F1-метрика для TensorFlow"""
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    
    precision_val = precision(y_true, y_pred)
    recall_val = recall(y_true, y_pred)
    return 2*((precision_val*recall_val)/(precision_val+recall_val+K.epsilon()))

# ИЗМЕНИТЬ метод compile_for_supervised_learning():
def compile_for_supervised_learning(self):
    """Компилирует модель для этапа 1: Supervised Learning"""
    self.actor_model.compile(
        optimizer=self.supervised_optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy', f1_score]  # 🔥 ИЗМЕНЕНО: добавлена f1_score
    )
    print("✅ Модель скомпилирована для supervised learning")

2. В файле train_model.py:
# ДОБАВИТЬ в начало файла после импортов:
import math

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
            tf.keras.backend.set_value(self.model.optimizer.learning_rate, decayed_learning_rate)

# ЗАМЕНИТЬ callbacks в методе stage1_supervised_pretraining():
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        patience=15,
        restore_best_weights=True, 
        monitor='val_accuracy',
        min_delta=0.001
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        factor=0.3,
        patience=7,
        monitor='val_loss',
        min_lr=1e-7
    ),
    tf.keras.callbacks.ModelCheckpoint(
        'models/best_supervised_model.keras', 
        save_best_only=True, 
        monitor='val_f1_score'  # 🔥 ИЗМЕНЕНО: мониторим F1-score
    ),
    # 🔥 ЗАМЕНЕНО: вместо CosineRestartScheduler
    CosineDecayCallback(
        initial_learning_rate=0.001,
        decay_steps=config.SUPERVISED_EPOCHS,
        alpha=1e-6
    ),
    ValidationMetricsCallback(self.X_val_supervised, self.y_val_supervised)
]

ВСЁ! Больше никаких изменений не требуется.