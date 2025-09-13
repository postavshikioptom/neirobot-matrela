
Основные проблемы, которые я вижу:


"AttributeError: 'str' object has no attribute 'name'" в regularization_callback.py: Это критическая ошибка, которая останавливает обучение. Она возникает потому, что self.model.optimizer.learning_rate в какой-то момент становится строкой, а не тензором или переменной, к которой можно применить tf.keras.backend.set_value. Это часто происходит, если learning rate установлен как строка (например, 'auto') или если оптимизатор не инициализирован должным образом.


Продолжающийся сильный дисбаланс классов на валидации/тесте:

"Confusion Matrix:" показывает [0 0 0] для BUY и SELL, и [ 399 35 64916] для HOLD. Это означает, что модель почти всегда предсказывает HOLD на валидационном наборе.
"Распределение предсказаний: BUY=0.6%, SELL=0.1%, HOLD=99.3%" подтверждает это: модель практически не предсказывает BUY или SELL.
Несмотря на балансировку классов с imblearn до 12.5% BUY, 12.5% SELL, 75.0% HOLD на тренировочном наборе, на валидационном наборе она продолжает предсказывать почти исключительно HOLD. Это указывает на то, что модель не смогла научиться распознавать BUY/SELL сигналы, даже если они были в тренировочном наборе.



"UndefinedMetricWarning: Recall and F-score are ill-defined and being set to 0.0 in labels with no true samples." Эти предупреждения возникают, потому что на валидационном наборе нет ни одного истинного образца BUY или SELL, либо модель вообще не предсказывает их. Это подтверждает, что валидационный набор данных может быть очень сильно смещен в сторону HOLD, или же модель игнорирует эти классы.


"⚠️ Обнаружено переобучение на эпохе 15! Val_loss/Train_loss = 1.38": Это говорит о том, что модель хорошо запоминает тренировочные данные, но плохо обобщается на новые, что является классическим признаком переобучения.


План по исправлению:
Проблемы с метриками и "AttributeError" тесно связаны с тем, как модель воспринимает данные и как она пытается оптимизироваться. Мы должны:

Исправить "AttributeError": Убедиться, что learning_rate всегда является изменяемым тензором.
Усилить репрезентацию BUY/SELL в валидационном наборе: Если в валидационном наборе практически нет BUY/SELL, модель не может научиться их распознавать.
Дополнительно скорректировать Focal Loss и веса классов: Несмотря на предыдущие изменения, модель все еще "ленится" и предсказывает HOLD.
Улучшить устойчивость обучения: Доработать колбэки и настройки.


Инструкции по исправлению:
Файл 1: train_model.py
1. Исправление AttributeError и настройка Learning Rate в train_xlstm_rl_system
Местоположение: Внутри функции train_xlstm_rl_system, где компилируется модель (строка ~450).
ЗАМЕНИТЬ:
# СТАРЫЙ КОД
optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.0005,  # Уменьшаем LR с 0.001 до 0.0005
    clipnorm=0.5,          # Уменьшаем clipnorm с 1.0 до 0.5
    weight_decay=0.0001    # Добавляем weight decay
)
xlstm_model.model.compile(
    optimizer=optimizer,
    # ИЗМЕНЕНО: Используем пользовательскую реализацию Focal Loss
    loss=categorical_focal_loss(gamma=2.0, alpha=0.25), # <--- ИЗМЕНЕНО
    metrics=[
        'accuracy',
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.Precision(name='precision_0', class_id=0),
        tf.keras.metrics.Precision(name='precision_1', class_id=1),
        tf.keras.metrics.Precision(name='precision_2', class_id=2),
        tf.keras.metrics.Recall(name='recall_0', class_id=0),
        tf.keras.metrics.Recall(name='recall_1', class_id=1),
        tf.keras.metrics.Recall(name='recall_2', class_id=2),
    ]
)

НА НОВЫЙ КОД:
# НОВЫЙ КОД - Инициализация Learning Rate как tf.Variable для совместимости с колбэками
initial_learning_rate = tf.Variable(0.0005, dtype=tf.float32) # ИЗМЕНЕНО: LR как tf.Variable
optimizer = tf.keras.optimizers.Adam(
    learning_rate=initial_learning_rate, # Используем tf.Variable
    clipnorm=0.5,
    weight_decay=0.0001
)
xlstm_model.model.compile(
    optimizer=optimizer,
    # ИЗМЕНЕНО: Используем пользовательскую реализацию Focal Loss
    loss=categorical_focal_loss(gamma=1.0, alpha=0.3), # ИЗМЕНЕНО: Менее агрессивный Focal Loss
    metrics=[
        'accuracy',
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.Precision(name='precision_0', class_id=0),
        tf.keras.metrics.Precision(name='precision_1', class_id=1),
        tf.keras.metrics.Precision(name='precision_2', class_id=2),
        tf.keras.metrics.Recall(name='recall_0', class_id=0),
        tf.keras.metrics.Recall(name='recall_1', class_id=1),
        tf.keras.metrics.Recall(name='recall_2', class_id=2),
    ]
)

Объяснение: Проблема AttributeError: 'str' object has no attribute 'name' возникает из-за того, что ReduceLROnPlateau и AntiOverfittingCallback пытаются получить learning_rate из оптимизатора как тензор, но он может быть установлен как обычное число или строка, что приводит к ошибке. Инициализация learning_rate как tf.Variable гарантирует, что это всегда будет изменяемый тензор, совместимый с колбэками. Также, я немного изменил параметры Focal Loss, чтобы сделать его менее агрессивным.
2. Улучшенная балансировка классов в prepare_xlstm_rl_data
Местоположение: Внутри функции prepare_xlstm_rl_data, в блоке imblearn (строка ~280).
ЗАМЕНИТЬ:
# СТАРЫЙ КОД
# Целевое соотношение: 10% BUY, 10% SELL, 80% HOLD (было 15/15/70)
total_samples = len(X)
target_buy_count = int(total_samples * 0.10) # <--- ИЗМЕНЕНО с 0.15 на 0.10
target_sell_count = int(total_samples * 0.10) # <--- ИЗМЕНЕНО с 0.15 на 0.10

current_buy_count = Counter(y_labels)[0]
current_sell_count = Counter(y_labels)[1]

sampling_strategy_smote = {
    0: max(current_buy_count, target_buy_count),
    1: max(current_sell_count, target_sell_count)
}

# ... (пропуск части кода) ...

# Undersampling HOLD: Цель - чтобы HOLD был примерно в 3 раза больше, чем сумма BUY + SELL
target_hold_count = min(current_hold_count_after_oversample, int((Counter(y_temp_labels)[0] + Counter(y_temp_labels)[1]) * 3.0)) # <--- ИЗМЕНЕНО с 2.0 на 3.0

undersampler = RandomUnderSampler(sampling_strategy={2: target_hold_count}, random_state=42)
X_resampled, y_resampled_labels = undersampler.fit_resample(X_temp, y_temp_labels)

НА НОВЫЙ КОД:
# НОВЫЙ КОД - Улучшенная балансировка для более агрессивного oversampling BUY/SELL
# Целевое соотношение: 20% BUY, 20% SELL, 60% HOLD (более агрессивный oversampling)
total_samples = len(X)
target_buy_count = int(total_samples * 0.20)  # ИЗМЕНЕНО: с 0.10 до 0.20
target_sell_count = int(total_samples * 0.20) # ИЗМЕНЕНО: с 0.10 до 0.20

current_buy_count = Counter(y_labels)[0]
current_sell_count = Counter(y_labels)[1]

sampling_strategy_smote = {
    0: max(current_buy_count, target_buy_count),
    1: max(current_sell_count, target_sell_count)
}

if current_buy_count > 0 or current_sell_count > 0:
    k_neighbors = min(5,
                      (current_buy_count - 1 if current_buy_count > 1 else 1),
                      (current_sell_count - 1 if current_sell_count > 1 else 1))
    k_neighbors = max(1, k_neighbors)

    if any(count <= k_neighbors for count in [current_buy_count, current_sell_count] if count > 0):
        print("⚠️ Недостаточно сэмплов для SMOTE с k_neighbors, использую RandomOverSampler.")
        from imblearn.over_sampling import RandomOverSampler
        oversampler = RandomOverSampler(sampling_strategy=sampling_strategy_smote, random_state=42)
    else:
        oversampler = SMOTE(sampling_strategy=sampling_strategy_smote, random_state=42, k_neighbors=k_neighbors)

    X_temp, y_temp_labels = oversampler.fit_resample(X.reshape(len(X), -1), y_labels)
    print(f"Баланс классов после Oversampling: {Counter(y_temp_labels)} (BUY/SELL увеличены)")
else:
    X_temp, y_temp_labels = X.reshape(len(X), -1), y_labels
    print("Пропустил Oversampling, так как нет BUY/SELL сигналов.")

# Undersampling HOLD: Цель - чтобы HOLD был примерно в 1.5 раза больше, чем сумма BUY + SELL
target_hold_count = min(current_hold_count_after_oversample, int((Counter(y_temp_labels)[0] + Counter(y_temp_labels)[1]) * 1.5)) # ИЗМЕНЕНО: с 3.0 до 1.5

undersampler = RandomUnderSampler(sampling_strategy={2: target_hold_count}, random_state=42)
X_resampled, y_resampled_labels = undersampler.fit_resample(X_temp, y_temp_labels)

Объяснение: Мы делаем oversampling BUY/SELL классов более агрессивным (до 20% каждый), а undersampling HOLD менее агрессивным (HOLD будет в 1.5 раза больше, чем сумма BUY+SELL, а не в 3 раза). Это должно помочь модели видеть больше примеров BUY/SELL и снизить ее склонность к предсказанию только HOLD.
3. Корректировка Focal Loss в train_model.py
Местоположение: Определение categorical_focal_loss (строка ~50).
ЗАМЕНИТЬ:
# СТАРЫЙ КОД
@tf.keras.utils.register_keras_serializable()
def categorical_focal_loss(gamma=1.5, alpha=0.25):  # Уменьшаем gamma, увеличиваем alpha
    """
    Менее агрессивная версия Focal Loss
    gamma=1.5 - умеренная фокусировка на сложных примерах
    alpha=0.25 - стандартный вес
    """
    def focal_loss_fixed(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        y_true = tf.cast(y_true, tf.float32)
        
        # ИСПРАВЛЕННЫЕ класс-специфичные веса
        # Уменьшаем веса BUY/SELL, увеличиваем вес HOLD
        class_weights = tf.constant([0.8, 0.8, 1.4])  # BUY, SELL, HOLD
        weights = tf.reduce_sum(class_weights * y_true, axis=-1, keepdims=True)
        
        cross_entropy = -y_true * K.log(y_pred)
        loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy * weights
        
        return K.sum(loss, axis=-1)
    
    return focal_loss_fixed

НА НОВЫЙ КОД:
# НОВЫЙ КОД - Менее агрессивный Focal Loss и более сбалансированные веса
@tf.keras.utils.register_keras_serializable()
def categorical_focal_loss(gamma=1.0, alpha=0.3):  # ИЗМЕНЕНО: gamma=1.0, alpha=0.3
    """
    Менее агрессивная версия Focal Loss для лучшей сходимости,
    с более сбалансированными весами классов.
    """
    def focal_loss_fixed(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        y_true = tf.cast(y_true, tf.float32)
        
        # ИЗМЕНЕННЫЕ класс-специфичные веса: более сбалансированы
        class_weights = tf.constant([1.2, 1.2, 0.8])  # ИЗМЕНЕНО: BUY/SELL немного усилены, HOLD немного ослаблен
        weights = tf.reduce_sum(class_weights * y_true, axis=-1, keepdims=True)
        
        cross_entropy = -y_true * K.log(y_pred)
        loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy * weights
        
        return K.sum(loss, axis=-1)
    
    return focal_loss_fixed

Объяснение: Изменение gamma на 1.0 делает Focal Loss менее "фокусированным" на сложных примерах, что может помочь модели учиться на более широком спектре данных. Увеличение alpha до 0.3 делает его немного более чувствительным к ошибкам. Веса классов внутри Focal Loss также скорректированы, чтобы немного усилить BUY/SELL и ослабить HOLD.
4. Корректировка весов классов в train_xlstm_rl_system
Местоположение: Внутри функции train_xlstm_rl_system, где вычисляются и передаются веса классов (строка ~350).
ЗАМЕНИТЬ:
# СТАРЫЙ КОД
# Уменьшаем веса BUY/SELL (они слишком агрессивные)
weight_reduction_factor = 0.6  # Уменьшаем веса BUY/SELL
if 0 in class_weight_dict:
    class_weight_dict[0] *= weight_reduction_factor
if 1 in class_weight_dict:
    class_weight_dict[1] *= weight_reduction_factor

# УВЕЛИЧИВАЕМ вес HOLD (чтобы модель чаще его предсказывала)
hold_boost_factor = 1.8  # Увеличиваем вес HOLD
if 2 in class_weight_dict:
    class_weight_dict[2] *= hold_boost_factor

НА НОВЫЙ КОД:
# НОВЫЙ КОД - Более сбалансированные веса классов
# Увеличиваем веса BUY/SELL немного, чтобы модель уделяла им больше внимания,
# но не настолько, чтобы она полностью игнорировала HOLD.
# Уменьшаем вес HOLD, но не слишком сильно.
if 0 in class_weight_dict:
    class_weight_dict[0] *= 1.5  # ИЗМЕНЕНО: Увеличиваем вес BUY
if 1 in class_weight_dict:
    class_weight_dict[1] *= 1.5  # ИЗМЕНЕНО: Увеличиваем вес SELL

if 2 in class_weight_dict:
    class_weight_dict[2] *= 0.7  # ИЗМЕНЕНО: Уменьшаем вес HOLD

Объяснение: Мы еще раз корректируем веса классов, чтобы дать BUY/SELL немного больше веса, но при этом сохранить достаточный вес для HOLD. Это должно помочь модели находить баланс между предсказанием всех классов.
5. Улучшенная TimeSeriesSplit для валидации (чтобы обеспечить наличие BUY/SELL)
Местоположение: Внутри функции train_xlstm_rl_system, в блоке TimeSeriesSplit (строка ~300).
ЗАМЕНИТЬ:
# СТАРЫЙ КОД
# Сначала отделим тестовую выборку (последние 20% данных)
test_size = int(len(X) * 0.2)
X_temp, X_test = X[:-test_size], X[-test_size:]
y_temp, y_test = y[:-test_size], y[-test_size:]

# Затем разделим оставшиеся данные на тренировочную и валидационную (например, 70/30)
# Используем TimeSeriesSplit для тренировочной/валидационной выборки
# Создаем 2 сплита, чтобы получить 3 части: Train, Val, Test.
# Последний сплит будет: Train_inner, Val_inner.
from sklearn.model_selection import TimeSeriesSplit
tscv_inner = TimeSeriesSplit(n_splits=2)

# Получаем индексы для внутренней тренировочной и валидационной выборки
# Здесь мы берем последний сплит, чтобы val_indices были "новее" чем train_indices
for train_idx, val_idx in tscv_inner.split(X_temp):
    pass # Проходим до последнего сплита

X_train, y_train = X_temp[train_idx], y_temp[train_idx]
X_val, y_val = X_temp[val_idx], y_temp[val_idx]

НА НОВЫЙ КОД:
# НОВЫЙ КОД - Стратифицированный TimeSeriesSplit для лучшего распределения классов
from sklearn.model_selection import StratifiedKFold # ИЗМЕНЕНО: Используем StratifiedKFold
from collections import Counter

print("\n🔄 Применяю СТРАТИФИЦИРОВАННЫЙ TimeSeriesSplit для валидации данных...")

# Сначала отделим тестовую выборку (последние 20% данных)
test_size = int(len(X) * 0.2)
X_temp, X_test = X[:-test_size], X[-test_size:]
y_temp, y_test = y[:-test_size], y[-test_size:]

# Для тренировочной и валидационной выборки используем StratifiedKFold,
# чтобы обеспечить наличие всех классов в каждом сплите.
# Мы не можем использовать TimeSeriesSplit со стратификацией напрямую,
# поэтому имитируем его, беря последние данные для валидации.
n_splits_stratified = 5 # ИЗМЕНЕНО: Используем 5 сплитов для лучшего распределения
skf = StratifiedKFold(n_splits=n_splits_stratified, shuffle=False) # shuffle=False для сохранения временного порядка

train_indices_list = []
val_indices_list = []

# Сохраняем метки классов для стратификации
y_temp_labels = np.argmax(y_temp, axis=1)

for train_idx, val_idx in skf.split(X_temp, y_temp_labels): # ИЗМЕНЕНО: Используем y_temp_labels для стратификации
    train_indices_list.append(train_idx)
    val_indices_list.append(val_idx)

# Берем последний сплит для тренировки и валидации, чтобы сохранить "временной" аспект
X_train, y_train = X_temp[train_indices_list[-1]], y_temp[train_indices_list[-1]]
X_val, y_val = X_temp[val_indices_list[-1]], y_temp[val_indices_list[-1]]

print(f"✅ Стратифицированный TimeSeriesSplit завершен.")
print(f"Распределение классов в X_train: {Counter(np.argmax(y_train, axis=1))}")
print(f"Распределение классов в X_val: {Counter(np.argmax(y_val, axis=1))}")
