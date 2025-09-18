Как исправить:
Нужно явно указать labels (список всех возможных классов) в вызове classification_report и confusion_matrix. Это заставит sklearn учитывать все 3 класса, даже если некоторые из них отсутствуют в текущей выборке.
Файл, который нужно изменить: validation_metrics_callback.py
Вот исправленный код для validation_metrics_callback.py:
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
        # 🔥 ДОБАВЛЕНО: Определяем все возможные метки (0, 1, 2)
        self.all_labels = [0, 1, 2] 
        
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 5 == 0:  # Каждые 5 эпох
            # 🔥 ИЗМЕНЕНО: logger.info -> print
            print(f"\n📊 Детальные метрики на эпохе {epoch+1}:")
            
            y_pred_probs = self.model.predict(self.X_val, verbose=0)
            y_pred_classes = np.argmax(y_pred_probs, axis=1)
            
            if self.y_val.ndim > 1 and self.y_val.shape[1] > 1:
                y_true_classes = np.argmax(self.y_val, axis=1)
            else:
                y_true_classes = self.y_val
            
            # Confusion Matrix
            # 🔥 ИЗМЕНЕНО: Добавлен параметр 'labels=self.all_labels'
            cm = confusion_matrix(y_true_classes, y_pred_classes, labels=self.all_labels) 
            # 🔥 ИЗМЕНЕНО: logger.info -> print
            print("Confusion Matrix:")
            
            header = "     " + " ".join([f"{name:4s}" for name in self.class_names])
            # 🔥 ИЗМЕНЕНО: logger.info -> print
            print(header)
            for i, row in enumerate(cm):
                row_str = " ".join([f"{val:4d}" for val in row])
                # 🔥 ИЗМЕНЕНО: logger.info -> print
                print(f"{self.class_names[i]:4s} {row_str}")
            
            # Classification Report
            # 🔥 ИЗМЕНЕНО: Добавлен параметр 'labels=self.all_labels'
            report_dict = classification_report(
                y_true_classes, y_pred_classes, 
                target_names=self.class_names,
                labels=self.all_labels, # 🔥 ДОБАВЛЕНО
                output_dict=True,
                zero_division=0 
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



Файл, который также нужно изменить: train_model.py
Хотя ошибка возникла в коллбэке, аналогичная проблема может возникнуть и в финальной оценке в train_model.py. Нужно применить то же исправление.
Вот исправленный фрагмент кода для train_model.py (только те части, которые меняются):
# ... (остальной код) ...

    def stage1_supervised_pretraining(self):
        """ЭТАП 1: Supervised Pre-training"""
        print("=== ЭТАП 1: SUPERVISED PRE-TRAINING ===")
        
        self.model.compile_for_supervised_learning()
        
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                patience=10, restore_best_weights=True, monitor='val_accuracy'
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                factor=0.5, patience=5, monitor='val_loss'
            ),
            tf.keras.callbacks.ModelCheckpoint(
                'models/best_supervised_model.keras', 
                save_best_only=True, monitor='val_accuracy'
            ),
            ValidationMetricsCallback(self.X_val, self.y_val)
        ]
        
        print(f"Начинаем supervised обучение на {config.SUPERVISED_EPOCHS} эпох...")
        
        history = self.model.actor_model.fit(
            self.X_train, self.y_train,
            validation_data=(self.X_val, self.y_val),
            epochs=config.SUPERVISED_EPOCHS,
            batch_size=config.SUPERVISED_BATCH_SIZE,
            callbacks=callbacks,
            verbose=1
        )
        
        print("=== РЕЗУЛЬТАТЫ SUPERVISED ОБУЧЕНИЯ ===")
        
        y_pred_probs = self.model.actor_model.predict(self.X_test, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)
        
        accuracy = accuracy_score(self.y_test, y_pred)
        print(f"Точность на тестовой выборке: {accuracy:.4f}")
        
        class_names = ['SELL', 'HOLD', 'BUY']
        # 🔥 ИЗМЕНЕНО: Добавлен параметр 'labels=[0, 1, 2]'
        report = classification_report(self.y_test, y_pred, target_names=class_names, labels=[0, 1, 2], zero_division=0) 
        print(f"Классификационный отчет:\n{report}")
        
        # 🔥 ИЗМЕНЕНО: Добавлен параметр 'labels=[0, 1, 2]'
        cm = confusion_matrix(self.y_test, y_pred, labels=[0, 1, 2])
        print(f"Матрица путаницы:\n{cm}")
        
        pred_dist = np.bincount(y_pred, minlength=3)
        total_pred = len(y_pred)
        print(f"Распределение предсказаний:")
        print(f"SELL: {pred_dist[0]} ({pred_dist[0]/total_pred:.2%})")
        print(f"HOLD: {pred_dist[1]} ({pred_dist[1]/total_pred:.2%})")
        print(f"BUY: {pred_dist[2]} ({pred_dist[2]/total_pred:.2%})")
        
        self.model.save(stage="_supervised")
        self.model.is_supervised_trained = True
        
        self._plot_training_history(history, "supervised")
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm,
            'history': history.history
        }

# ... (остальной код) ...

Объяснение исправлений:

labels=self.all_labels (в коллбэке) / labels=[0, 1, 2] (в train_model.py): Мы явно указываем sklearn, что у нас есть 3 класса (с метками 0, 1, 2), даже если в текущей выборке отсутствуют примеры какого-либо класса. Это позволяет classification_report и confusion_matrix корректно строить отчеты, включая строки и столбцы для отсутствующих классов (которые будут иметь нули).

Эти изменения должны полностью устранить ValueError и обеспечить корректное отображение метрик для всех 3 классов.