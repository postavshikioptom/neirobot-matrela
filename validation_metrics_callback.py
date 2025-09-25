import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import numpy as np
import gc
import threading
import queue
import time
# import logging # 🔥 УДАЛЕНО: Импорт logging

# 🔥 УДАЛЕНО: Инициализация логгера
# logger = logging.getLogger('validation_callback')

class ValidationMetricsCallback(tf.keras.callbacks.Callback):
    """
    Детальный мониторинг метрик валидации для Supervised Pre-training (Этап 1)
    """
    def __init__(self, X_val, y_val, class_names=['SELL', 'HOLD', 'BUY'], max_val_samples=5000):
        super().__init__()
        # 🔥 ДОБАВЛЕНО: Ограничение размера валидационных данных
        if len(X_val) > max_val_samples:
            indices = np.random.choice(len(X_val), max_val_samples, replace=False)
            self.X_val = X_val[indices]
            self.y_val = y_val[indices]
            print(f"Ограничены валидационные данные: {len(X_val)} -> {max_val_samples}")
        else:
            self.X_val = X_val
            self.y_val = y_val
        
        self.class_names = class_names
        # 🔥 ДОБАВЛЕНО: Определяем все возможные метки (0, 1, 2)
        self.all_labels = [0, 1, 2]
        # 🔥 ДОБАВЛЕНО: Асинхронная валидация
        self.validation_queue = queue.Queue()
        self.validation_thread = None
        self.validation_in_progress = False 
        
    def on_epoch_end(self, epoch, logs=None):
        # 🔥 ИЗМЕНЕНО: Проверяем каждую эпоху для быстрого реагирования на дисбаланс
        if (epoch + 1) % 1 == 0:  # Каждую эпоху вместо каждой 5-й
            # 🔥 ИСПРАВЛЕНО: Асинхронная валидация
            if not self.validation_in_progress:
                self.validation_in_progress = True
                self.validation_thread = threading.Thread(
                    target=self._async_validation, 
                    args=(epoch, logs)
                )
                self.validation_thread.daemon = True
                self.validation_thread.start()
            else:
                print(f"⚠️ Пропускаем валидацию на эпохе {epoch+1} - предыдущая еще выполняется")
    
    def _async_validation(self, epoch, logs):
        """🔥 ДОБАВЛЕНО: Асинхронная валидация"""
        try:
            print(f"\n📊 Детальные метрики на эпохе {epoch+1}:")
            
            # Используем небольшие батчи для предсказаний
            batch_size = 256
            y_pred_classes_list = []
            
            for i in range(0, len(self.X_val), batch_size):
                batch_X = self.X_val[i:i+batch_size]
                batch_pred_probs = self.model.predict(batch_X, verbose=0)
                batch_pred_classes = np.argmax(batch_pred_probs, axis=1)
                y_pred_classes_list.extend(batch_pred_classes)
                
                # Принудительная очистка каждые несколько батчей
                if i % (batch_size * 5) == 0:
                    gc.collect()
            
            y_pred_classes = np.array(y_pred_classes_list)
            
            if self.y_val.ndim > 1 and self.y_val.shape[1] > 1:
                y_true_classes = np.argmax(self.y_val, axis=1)
            else:
                y_true_classes = self.y_val
            
            # Confusion Matrix
            cm = confusion_matrix(y_true_classes, y_pred_classes, labels=self.all_labels)
            print("Confusion Matrix:")
            
            header = "     " + " ".join([f"{name:4s}" for name in self.class_names])
            print(header)
            for i, row in enumerate(cm):
                row_str = " ".join([f"{val:4d}" for val in row])
                print(f"{self.class_names[i]:4s} {row_str}")
            
            # Classification Report
            report_dict = classification_report(
                y_true_classes, y_pred_classes, 
                target_names=self.class_names,
                labels=self.all_labels,
                output_dict=True,
                zero_division=0
            )
            
            print(f"Macro Avg F1-Score: {report_dict['macro avg']['f1-score']:.3f}")
            print(f"Weighted Avg F1-Score: {report_dict['weighted avg']['f1-score']:.3f}")
            
            # Распределение предсказаний
            pred_distribution = np.bincount(y_pred_classes, minlength=len(self.class_names)) / len(y_pred_classes)
            pred_dist_str = ", ".join([f"{name}={dist:.1%}" for name, dist in zip(self.class_names, pred_distribution)])
            print(f"Распределение предсказаний: {pred_dist_str}")
            
            # 🔥 ДОБАВЛЕНО: Вычисление F1-метрики через sklearn
            f1_macro = f1_score(y_true_classes, y_pred_classes, average='macro', zero_division=0)
            f1_weighted = f1_score(y_true_classes, y_pred_classes, average='weighted', zero_division=0)
            print(f"F1 Macro: {f1_macro:.3f}, F1 Weighted: {f1_weighted:.3f}")
            
            # 🔥 ДОБАВЛЕНО: Проверка на переобучение
            train_acc = logs.get('accuracy', 0) if logs else 0
            val_acc = logs.get('val_accuracy', 0) if logs else 0
            
            if train_acc - val_acc > 0.15:
                print(f"⚠️ Обнаружено переобучение: train_acc={train_acc:.3f}, val_acc={val_acc:.3f}")
                print("Рекомендуется увеличить dropout или regularization")
            
            # Очистка памяти
            del y_pred_classes_list, y_pred_classes, y_true_classes, cm, report_dict
            gc.collect()
            
        except Exception as e:
            print(f"❌ Ошибка в асинхронной валидации: {e}")
        finally:
            self.validation_in_progress = False
    
    def on_train_end(self, logs=None):
        """🔥 ДОБАВЛЕНО: Ожидание завершения валидации"""
        if self.validation_thread and self.validation_thread.is_alive():
            print("Ожидание завершения валидации...")
            self.validation_thread.join(timeout=30)  # Максимум 30 секунд ожидания