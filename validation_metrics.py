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
            pred_distribution = np.bincount(y_pred_classes, minlength=len(self.class_names)) / len(y_pred_classes)
            print(f"Распределение предсказаний: BUY={pred_distribution[0]:.1%}, SELL={pred_distribution[1]:.1%}, HOLD={pred_distribution[2]:.1%}")