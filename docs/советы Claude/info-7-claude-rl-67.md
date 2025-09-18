
Удалить класс MulticlassF1Score полностью


Файл: models/xlstm_rl_model.py
Удалить блок
class MulticlassF1Score(...): ...
(весь класс)


Убрать любые ссылки на этот класс при сохранении/загрузке модели


Файл: models/xlstm_rl_model.py — метод load(...)
В вызове tf.keras.models.load_model() оставить только:
custom_objects={'XLSTMMemoryCell': XLSTMMemoryCell}
(убрать упоминания MulticlassF1Score / f1_score / tfa)


Убедиться, что модель не компилируется с кастомной TF-F1 метрикой


Файл: models/xlstm_rl_model.py — метод compile_for_supervised_learning()
Должно быть (без F1 метрики):

metrics = [
    'accuracy',
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.Recall(name='recall')
]
self.actor_model.compile(
    optimizer=self.supervised_optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=metrics
)


Вычислять multiclass F1 в ValidationMetricsCallback через sklearn


Файл: validation_metrics_callback.py (или ваш callback)
В on_epoch_end (или в асинхронной валидации) добавить/обновить:

from sklearn.metrics import f1_score

# после получения y_true_classes и y_pred_classes:
f1_macro = f1_score(y_true_classes, y_pred_classes, average='macro', zero_division=0)
f1_weighted = f1_score(y_true_classes, y_pred_classes, average='weighted', zero_division=0)
print(f"F1 Macro: {f1_macro:.3f}, F1 Weighted: {f1_weighted:.3f}")


Дополнительно: исправить опечатки в predict_count инкрементации


В models/xlstm_rl_model.py:
заменить
self.prediction_count + = 1
на
self.prediction_count += 1
(в обоих predict_action и predict_value)
