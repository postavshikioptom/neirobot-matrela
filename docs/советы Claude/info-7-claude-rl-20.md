import tensorflow as tf
import numpy as np

class AntiOverfittingCallback(tf.keras.callbacks.Callback):
    """
    Колбэк для борьбы с переобучением
    """
    def __init__(self, patience=5, min_improvement=0.01):
        super().__init__()
        self.patience = patience
        self.min_improvement = min_improvement
        self.wait = 0
        self.best_val_loss = np.inf
        
    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs.get('val_loss')
        train_loss = logs.get('loss')
        
        # Проверяем на переобучение
        overfitting_ratio = val_loss / train_loss if train_loss > 0 else 1.0
        
        if overfitting_ratio > 1.3:  # Если val_loss > train_loss * 1.3
            print(f"\n⚠️ Обнаружено переобучение на эпохе {epoch+1}!")
            print(f"   Val_loss/


Файл 7: Новый файл regularization_callback.py
Создать новый файл:
import tensorflow as tf
import numpy as np

class AntiOverfittingCallback(tf.keras.callbacks.Callback):
    """
    Колбэк для борьбы с переобучением
    """
    def __init__(self, patience=5, min_improvement=0.01):
        super().__init__()
        self.patience = patience
        self.min_improvement = min_improvement
        self.wait = 0
        self.best_val_loss = np.inf
        
    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs.get('val_loss')
        train_loss = logs.get('loss')
        
        # Проверяем на переобучение
        overfitting_ratio = val_loss / train_loss if train_loss > 0 else 1.0
        
        if overfitting_ratio > 1.3:  # Если val_loss > train_loss * 1.3
            print(f"\n⚠️ Обнаружено переобучение на эпохе {epoch+1}!")
            print(f"   Val_loss/Train_loss = {overfitting_ratio:.2f}")
            
            # Уменьшаем learning rate
            current_lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
            new_lr = current_lr * 0.8
            tf.keras.backend.set_value(self.model.optimizer.learning_rate, new_lr)
            print(f"   Снижаем LR: {current_lr:.2e} -> {new_lr:.2e}")
        
        # Отслеживаем улучшение val_loss
        if val_loss < self.best_val_loss - self.min_improvement:
            self.best_val_loss = val_loss
            self.wait = 0
        else:
            self.wait += 1
            
        if self.wait >= self.patience and epoch > 20:  # Минимум 20 эпох
            print(f"\n🛑 Останавливаем обучение: нет улучшений {self.patience} эпох")
            self.model.stop_training = True

Файл 8: train_model.py
Местоположение: Добавить импорт и использование нового колбэка (строка ~20)
ДОБАВИТЬ В ИМПОРТЫ:
# ДОБАВИТЬ В ИМПОРТЫ
from regularization_callback import AntiOverfittingCallback

В функции train_xlstm_rl_system, в блоке callbacks (строка ~400):
ЗАМЕНИТЬ:
# СТАРЫЙ КОД
callbacks = [
    tf.keras.callbacks.EarlyStopping(...),
    MemoryCleanupCallback(),
    DetailedProgressCallback(),
    tf.keras.callbacks.ReduceLROnPlateau(...)
]

НА НОВЫЙ КОД:
# НОВЫЙ КОД С ДОПОЛНИТЕЛЬНЫМИ КОЛБЭКАМИ
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=15,
        restore_best_weights=True,
        verbose=1,
        min_delta=0.001
    ),
    AntiOverfittingCallback(patience=8, min_improvement=0.005),  # НОВЫЙ КОЛБЭК
    MemoryCleanupCallback(),
    DetailedProgressCallback(),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.7,
        patience=8,
        min_lr=1e-6,
        verbose=1
    )
]

Файл 9: models/xlstm_rl_model.py
Местоположение: Метод build_model, добавление Dropout (строка ~80)
НАЙТИ блок:
# Классификационные слои
dense1 = Dense(64, activation='relu', kernel_regularizer=l2(0.001), name='dense_1')(xlstm_final)
dropout1 = Dropout(0.4)(dense1)

dense2 = Dense(32, activation='relu', kernel_regularizer=l2(0.001), name='dense_2')(dropout1)
dropout2 = Dropout(0.3)(dense2)

ЗАМЕНИТЬ НА:
# УСИЛЕННАЯ РЕГУЛЯРИЗАЦИЯ
dense1 = Dense(64, activation='relu', kernel_regularizer=l2(0.002), name='dense_1')(xlstm_final)  # Увеличиваем L2
dropout1 = Dropout(0.5)(dense1)  # Увеличиваем dropout с 0.4 до 0.5

dense2 = Dense(32, activation='relu', kernel_regularizer=l2(0.002), name='dense_2')(dropout1)  # Увеличиваем L2
dropout2 = Dropout(0.4)(dense2)  # Увеличиваем dropout с 0.3 до 0.4

# ДОБАВЛЯЕМ ДОПОЛНИТЕЛЬНЫЙ СЛОЙ ДЛЯ ЛУЧШЕЙ РЕГУЛЯРИЗАЦИИ
dense3 = Dense(16, activation='relu', kernel_regularizer=l2(0.001), name='dense_3')(dropout2)
dropout3 = Dropout(0.3)(dense3)

И ИЗМЕНИТЬ выходной слой:
# СТАРЫЙ КОД
outputs = Dense(3, activation='softmax', name='output_layer')(dropout2)

НА:
# НОВЫЙ КОД
outputs = Dense(3, activation='softmax', name='output_layer')(dropout3)  # Используем dropout3

Файл 10: train_model.py
Местоположение: Настройка оптимизатора (строка ~450)
ЗАМЕНИТЬ:
# СТАРЫЙ КОД
optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.001,
    clipnorm=1.0
)

НА НОВЫЙ КОД:
# НОВЫЙ КОД - БОЛЕЕ КОНСЕРВАТИВНЫЕ НАСТРОЙКИ
optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.0005,  # Уменьшаем LR с 0.001 до 0.0005
    clipnorm=0.5,          # Уменьшаем clipnorm с 1.0 до 0.5
    weight_decay=0.0001    # Добавляем weight decay
)

Файл 11: train_model.py
Местоположение: Настройка batch_size (строка ~480)
ЗАМЕНИТЬ:
# СТАРЫЙ КОД
history = xlstm_model.train(
    X_train_to_model, y_train,
    X_val_to_model, y_val,
    epochs=100,
    batch_size=16,  # Старый размер
    class_weight=class_weight_dict,
    custom_callbacks=callbacks
)

НА НОВЫЙ КОД:
# НОВЫЙ КОД - УВЕЛИЧИВАЕМ BATCH SIZE ДЛЯ СТАБИЛЬНОСТИ
history = xlstm_model.train(
    X_train_to_model, y_train,
    X_val_to_model, y_val,
    epochs=80,      # Уменьшаем количество эпох с 100 до 80
    batch_size=32,  # Увеличиваем batch_size с 16 до 32
    class_weight=class_weight_dict,
    custom_callbacks=callbacks
)

Файл 12: Новый файл validation_metrics.py
Создать новый файл для лучшего мониторинга:
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

class ValidationMetricsCallback(tf.keras.callbacks.Callback):
    """
    Детальный мониторинг метрик валидации
    """
    def __init__(self, X_val, y_val, class_names=['BUY', 'SELL', 'HOLD']):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val
        self.class_names = class_names
        
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 5 == 0:  # Каждые 5 эпох
            y_pred = self.model.predict(self.X_val, verbose=0)
            y_pred_classes = np.argmax(y_pred, axis=1)
            y_true_classes = np.argmax(self.y_val, axis=1)
            
            print(f"\n📊 Детальные метрики на эпохе {epoch+1}:")
            
            # Confusion Matrix
            cm = confusion_matrix(y_true_classes, y_pred_classes)
            print("Confusion Matrix:")
            print("     BUY  SELL HOLD")
            for i, row in enumerate(cm):
                print(f"{self.class_names[i]:4s} {row}")
            
            # Classification Report
            report = classification_report(
                y_true_classes, y_pred_classes, 
                target_names=self.class_names,
                output_dict=True
            )
            
            print(f"Macro Avg F1-Score: {report['macro avg']['f1-score']:.3f}")
            print(f"Weighted Avg F1-Score: {report['weighted avg']['f1-score']:.3f}")
            
            # Проверка на дисбаланс предсказаний
            pred_distribution = np.bincount(y_pred_classes) / len(y_pred_classes)
            print(f"Распределение предсказаний: BUY={pred_distribution[0]:.1%}, SELL={pred_distribution[1]:.1%}, HOLD={pred_distribution[2]:.1%}")

Файл 13: train_model.py
Добавить новый колбэк в список (после импорта ValidationMetricsCallback):
# ДОБАВИТЬ В ИМПОРТЫ
from validation_metrics import ValidationMetricsCallback

# В блоке callbacks добавить:
callbacks = [
    # ... существующие колбэки ...
    ValidationMetricsCallback(X_val_to_model, y_val),  # НОВЫЙ КОЛБЭК
]

Резюме изменений:

Увеличили пороги для BUY/SELL - теперь нужны более сильные сигналы
Исправили веса классов - уменьшили веса BUY/SELL, увеличили вес HOLD
Улучшили регуляризацию - больше dropout, L2, меньше LR
Добавили защиту от переобучения - новые колбэки
Исправили распределение классов - больше HOLD, меньше торговых сигналов
Добавили детальный мониторинг - лучше видим что происходит

Эти изменения должны решить основную проблему: модель будет чаще предсказывать HOLD (увеличится recall для HOLD) и будет менее агрессивной с торговыми сигналами, что улучшит общую точность.