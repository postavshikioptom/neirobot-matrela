
Удалить импорт tensorflow_addons
Файл: models/xlstm_rl_model.py — вверху (удалить строку)

import tensorflow_addons as tfa


Заменить compile_for_supervised_learning — убрать tfa.metrics.F1Score из metrics
Замените текущую реализацию на это (оставляем accuracy, precision, recall; F1 считаем в ValidationMetricsCallback):

def compile_for_supervised_learning(self):
    """Компилирует модель для этапа 1: Supervised Learning"""
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
    print("✅ Модель скомпилирована для supervised learning")


Удалить или изменить custom_objects при загрузке модели
Файл: models/xlstm_rl_model.py — в методе load(...) уберите упоминание f1_score / tfa метрики из custom_objects. Например:

self.actor_model = tf.keras.models.load_model(
    actor_path,
    custom_objects={'XLSTMMemoryCell': XLSTMMemoryCell}  # убрали f1_score / tfa
)


Использовать ValidationMetricsCallback для вычисления и логирования multiclass F1
Вы уже имеете validation_metrics_callback.py — убедитесь, что он вызывается в training (он есть) и что он считает macro/weighted F1 по sklearn.classification_report. Если нет — замените/добавьте внутри callback вызов:

from sklearn.metrics import f1_score

# в on_epoch_end или в асинхронной валидации:
f1_macro = f1_score(y_true_classes, y_pred_classes, average='macro', zero_division=0)
f1_weighted = f1_score(y_true_classes, y_pred_classes, average='weighted', zero_division=0)
print(f"F1 Macro: {f1_macro:.3f}, F1 Weighted: {f1_weighted:.3f}")

(У вас уже есть classification_report в callback — просто дополните вывод F1 из report_dict или используйте sklearn.f1_score.)


Опционально: если хотите оставить метрику F1 в model.metrics (не только callback), можно реализовать кастомный tf.keras.metrics.Metric. Но это сложнее и не нужно — callback + sklearn на валидации достаточно и надёжно в Kaggle.


Удалить зависимость из requirements / инструкции



Уберите tensorflow_addons из requirements.txt / setup.
В README/скриптах объясните, что multiclass F1 вычисляется в ValidationMetricsCallback.

Короткая сводка изменений (файлы и места):

models/xlstm_rl_model.py

удалить import tensorflow_addons
заменить compile_for_supervised_learning как выше
убрать f1_score из custom_objects в load()


validation_metrics_callback.py (если нужно) — добавить f1_score вычисление через sklearn и вывод

После этих правок запуск в Kaggle должен продолжиться без ModuleNotFoundError для tensorflow_addons.