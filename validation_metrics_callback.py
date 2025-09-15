import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
# import logging # 🔥 УДАЛЕНО: Импорт logging

# 🔥 УДАЛЕНО: Инициализация логгера
# logger = logging.getLogger('validation_callback')

class ValidationMetricsCallback(tf.keras.callbacks.Callback):
    """
    Детальный мониторинг метрик валидации для Supervised Pre-training (Этап 1)
    """
    def __init__(self, X_val, y_val, class_names=['SELL', 'HOLD', 'BUY']):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val
        self.class_names = class_names
        
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 5 == 0:  # Каждые 5 эпох
            # 🔥 ИЗМЕНЕНО: logger.info -> print
            print(f"\n📊 Детальные метрики на эпохе {epoch+1}:")
            
            y_pred_probs = self.model.predict(self.X_val, verbose=0)
            y_pred_classes = np.argmax(y_pred_probs, axis=1)
            
            # Если y_val уже one-hot, преобразуем
            if self.y_val.ndim > 1 and self.y_val.shape[1] > 1:
                y_true_classes = np.argmax(self.y_val, axis=1)
            else:
                y_true_classes = self.y_val
            
            # Confusion Matrix
            cm = confusion_matrix(y_true_classes, y_pred_classes)
            # 🔥 ИЗМЕНЕНО: logger.info -> print
            print("Confusion Matrix:")
            
            # Форматируем вывод матрицы
            header = "     " + " ".join([f"{name:4s}" for name in self.class_names])
            # 🔥 ИЗМЕНЕНО: logger.info -> print
            print(header)
            for i, row in enumerate(cm):
                row_str = " ".join([f"{val:4d}" for val in row])
                # 🔥 ИЗМЕНЕНО: logger.info -> print
                print(f"{self.class_names[i]:4s} {row_str}")
            
            # Classification Report
            report_dict = classification_report(
                y_true_classes, y_pred_classes, 
                target_names=self.class_names,
                output_dict=True,
                zero_division=0 # Избегаем предупреждений при нулевом делении
            )
            
            # 🔥 ИЗМЕНЕНО: logger.info -> print
            print(f"Macro Avg F1-Score: {report_dict['macro avg']['f1-score']:.3f}")
            # 🔥 ИЗМЕНЕНО: logger.info -> print
            print(f"Weighted Avg F1-Score: {report_dict['weighted avg']['f1-score']:.3f}")
            
            # Распределение предсказаний
            pred_distribution = np.bincount(y_pred_classes, minlength=len(self.class_names)) / len(y_pred_classes)
            pred_dist_str = ", ".join([f"{name}={dist:.1%}" for name, dist in zip(self.class_names, pred_distribution)])
            # 🔥 ИЗМЕНЕНО: logger.info -> print
            print(f"Распределение предсказаний: {pred_dist_str}")