
План по исправлению:
Основное внимание уделим новой критической ошибке с CustomFocalLoss.
Файл 1: train_model.py
1. Исправление TypeError: CustomFocalLoss.__init__() got an unexpected keyword argument 'reduction'
Причина: Keras автоматически добавляет reduction при сохранении tf.keras.losses.Loss объектов, но наш __init__ его не принимает.
Решение: Добавить **kwargs в __init__ метод CustomFocalLoss и передать их в super().__init__().
Местоположение: Внутри класса CustomFocalLoss, метод __init__ (строка ~50).
ЗАМЕНИТЬ:
# СТАРЫЙ КОД
class CustomFocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=1.0, alpha=0.3, class_weights=None, name='CustomFocalLoss'):
        super().__init__(name=name)
        self.gamma = gamma
        self.alpha = alpha
        # ... остальной код

НА НОВЫЙ КОД:
# НОВЫЙ КОД - Добавляем **kwargs для совместимости с Keras
class CustomFocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=1.0, alpha=0.3, class_weights=None, name='CustomFocalLoss', **kwargs): # ИЗМЕНЕНО: Добавлен **kwargs
        super().__init__(name=name, **kwargs) # ИЗМЕНЕНО: Передаем **kwargs в super()
        self.gamma = gamma
        self.alpha = alpha
        # ... остальной код

Объяснение: Это позволит CustomFocalLoss корректно обрабатывать аргумент reduction (и любые другие, которые Keras может добавить при сериализации), перенаправляя их в базовый класс tf.keras.losses.Loss.
Файл 2: train_model.py
2. Исправление xLSTM Точность: 0.00% после оценки
Причина: Эта ошибка, вероятно, связана с тем, что после EarlyStopping и restore_best_weights=True, модель восстанавливает веса, а затем, возможно, есть какой-то внутренний сброс состояния или проблема с метриками, когда model.evaluate вызывается после этого. Или же model.evaluate по какой-то причине не возвращает метрику accuracy для всех классов, а наш код не полностью это обрабатывает.
Решение: Убедиться, что model.evaluate возвращает метрики как словарь, и использовать этот словарь для более надежного извлечения.
Местоположение: Внутри функции train_xlstm_rl_system, в блоке оценки модели (строка ~600).
ЗАМЕНИТЬ:
# СТАРЫЙ КОД
evaluation_results = xlstm_model.model.evaluate(X_test_scaled, y_test, verbose=0)
metrics_names = xlstm_model.model.metrics_names

loss = evaluation_results[metrics_names.index('loss')] if 'loss' in metrics_names else 0.0
# ... (остальной код извлечения метрик)

НА НОВЫЙ КОД:
# НОВЫЙ КОД - Использование model.evaluate с return_dict=True
evaluation_results_dict = xlstm_model.model.evaluate(X_test_scaled, y_test, verbose=0, return_dict=True) # ИЗМЕНЕНО: return_dict=True

loss = evaluation_results_dict.get('loss', 0.0)
accuracy = evaluation_results_dict.get('accuracy', 0.0)
precision = evaluation_results_dict.get('precision', 0.0)
recall = evaluation_results_dict.get('recall', 0.0)

print(f"xLSTM Loss: {loss:.4f}")
print(f"xLSTM Точность: {accuracy * 100:.2f}%")
print(f"xLSTM Precision: {precision * 100:.2f}%")
print(f"xLSTM Recall: {recall * 100:.2f}%")

# Выводим метрики по классам, если они есть
for i, class_name in enumerate(['BUY', 'SELL', 'HOLD']):
    prec_i = evaluation_results_dict.get(f'precision_{i}', 0.0)
    rec_i = evaluation_results_dict.get(f'recall_{i}', 0.0)
    print(f"  Class {i} ({class_name}): Prec={prec_i:.2f}, Rec={rec_i:.2f}")

Объяснение: Использование return_dict=True делает извлечение метрик более явным и устойчивым, поскольку мы напрямую обращаемся к элементам словаря по имени, а не полагаемся на индексы.

Дополнительные действия (очень важно):
Перед следующим запуском обучения удалите все сохраненные файлы модели и скейлера:

models/xlstm_rl_model.keras
models/xlstm_rl_scaler.pkl
Также, если ты обучаешь RL-агента, то и его файлы: models/rl_agent_ТВОЙСИМВОЛ.zip и любые промежуточные сохранения агента.
И файл детектора режимов: models/market_regime_detector.pkl
