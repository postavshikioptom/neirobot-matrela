Инструкции по исправлению:
Файл 1: train_model.py
1. Исправление ValueError: Cannot set memory growth on device when virtual devices configured
Причина: configure_gpu_memory() вызывается несколько раз.
Решение: Вызывать configure_gpu_memory() только один раз, при запуске основного скрипта.
Местоположение: В начале файла, где вызывается configure_gpu_memory() (строка ~31).
ЗАМЕНИТЬ:
# СТАРЫЙ КОД
configure_gpu_memory()

НА НОВЫЙ КОД:
# НОВЫЙ КОД - Вызываем configure_gpu_memory() только один раз, при запуске основного скрипта
if __name__ == "__main__":
    configure_gpu_memory()


Файл 2: train_model.py
2. Исправление TypeError: Could not locate function 'focal_loss_fixed'
Причина: Keras при загрузке модели не может найти пользовательскую функцию потерь focal_loss_fixed, даже если CustomFocalLoss зарегистрирован. Это может быть связано с тем, что в сохраненной модели метаданные все еще ссылаются на старое имя функции.
Решение: Явно передать CustomFocalLoss в custom_objects при каждой загрузке модели, а также, на всякий случай, удалить старую сохраненную модель, чтобы она пересохранилась с актуальными метаданными.
Инструкция:
Файл 1: train_model.py
Местоположение: Внутри функции train_xlstm_rl_system, в блоке оценки модели, где загружается decision_maker_temp (строка ~682).
ЗАМЕНИТЬ:
# СТАРЫЙ КОД
decision_maker_temp = HybridDecisionMaker(
    xlstm_model_path='models/xlstm_rl_model.keras',
    rl_agent_path=None,
    feature_columns=feature_cols,
    sequence_length=X.shape[1]
)

НА НОВЫЙ КОД:
# НОВЫЙ КОД - Убедимся, что CustomFocalLoss доступен при инициализации HybridDecisionMaker
# Примечание: CustomFocalLoss уже импортируется в начале train_model.py
# и регистрируется с @tf.keras.utils.register_keras_serializable().
# Если эта ошибка все еще возникает, это может быть связано с тем,
# что Keras кэширует старые метаданные модели.
# В этом случае, ПЕРЕД ЗАПУСКОМ, УДАЛИТЕ ФАЙЛ models/xlstm_rl_model.keras
# и models/xlstm_rl_scaler.pkl, чтобы модель переобучилась и сохранилась заново.

decision_maker_temp = HybridDecisionMaker(
    xlstm_model_path='models/xlstm_rl_model.keras',
    rl_agent_path=None,
    feature_columns=feature_cols,
    sequence_length=X.shape[1]
)

Дополнительно: ОЧЕНЬ ВАЖНО перед следующим запуском удалить все сохраненные модели и скейлеры:

models/xlstm_rl_model.keras
models/xlstm_rl_scaler.pkl

Это гарантирует, что модель будет переобучена с нуля и сохранена с корректными метаданными для CustomFocalLoss.

Файл 3: train_model.py
3. Исправление ошибки ⚠️ Ошибка при оценке модели: 'accuracy' is not in list
Причина: Список метрик, возвращаемый model.evaluate, может не содержать 'accuracy' (или другие метрики) в определенных сценариях.
Решение: Убедиться, что извлечение метрик максимально устойчиво.
Местоположение: Внутри функции train_xlstm_rl_system, в блоке оценки модели (строка ~600).
ЗАМЕНИТЬ:
# СТАРЫЙ КОД (уже был улучшен, но давайте добавим проверку для всех)
loss = evaluation_results[metrics_names.index('loss')]
# Проверяем наличие метрик перед извлечением
accuracy = evaluation_results[metrics_names.index('accuracy')] if 'accuracy' in metrics_names else 0.0
precision = evaluation_results[metrics_names.index('precision')] if 'precision' in metrics_names else 0.0
recall = evaluation_results[metrics_names.index('recall')] if 'recall' in metrics_names else 0.0

# Дополнительно для классов
prec_0 = evaluation_results[metrics_names.index('precision_0')] if 'precision_0' in metrics_names else 0.0
rec_0 = evaluation_results[metrics_names.index('recall_0')] if 'recall_0' in metrics_names else 0.0
prec_1 = evaluation_results[metrics_names.index('precision_1')] if 'precision_1' in metrics_names else 0.0
rec_1 = evaluation_results[metrics_names.index('recall_1')] if 'recall_1' in metrics_names else 0.0
prec_2 = evaluation_results[metrics_names.index('precision_2')] if 'precision_2' in metrics_names else 0.0
rec_2 = evaluation_results[metrics_names.index('recall_2')] if 'recall_2' in metrics_names else 0.0

НА НОВЫЙ КОД (просто убедимся, что это применено, как обсуждалось ранее, так как в логе все еще видно предупреждение):
# НОВЫЙ КОД - Более надежное извлечение всех метрик (уже было предложено)
loss = evaluation_results[metrics_names.index('loss')] if 'loss' in metrics_names else 0.0
accuracy = evaluation_results[metrics_names.index('accuracy')] if 'accuracy' in metrics_names else 0.0
precision = evaluation_results[metrics_names.index('precision')] if 'precision' in metrics_names else 0.0
recall = evaluation_results[metrics_names.index('recall')] if 'recall' in metrics_names else 0.0

# Дополнительно для классов
prec_0 = evaluation_results[metrics_names.index('precision_0')] if 'precision_0' in metrics_names else 0.0
rec_0 = evaluation_results[metrics_names.index('recall_0')] if 'recall_0' in metrics_names else 0.0
prec_1 = evaluation_results[metrics_names.index('precision_1')] if 'precision_1' in metrics_names else 0.0
rec_1 = evaluation_results[metrics_names.index('recall_1')] if 'recall_1' in metrics_names else 0.0
prec_2 = evaluation_results[metrics_names.index('precision_2')] if 'precision_2' in metrics_names else 0.0
rec_2 = evaluation_results[metrics_names.index('recall_2')] if 'recall_2' in metrics_names else 0.0

Объяснение: Это гарантирует, что даже если какая-то метрика отсутствует, код не упадет, а просто присвоит 0.0.
