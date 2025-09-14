Хорошо, я вижу логи. Это очень интересная и необычная ситуация.
Вы правы, accuracy, val_accuracy, precision, val_precision, recall, val_recall все равны 1.00 (или 100%), а loss и val_loss довольно низкие (около 0.43 и 0.29 соответственно) и почти не меняются. Это не является нормальным поведением для обучения модели классификации с тремя классами, особенно на финансовых данных.
🚨 Корневая проблема: Модель предсказывает только один класс (или слишком мало разнообразия)
Когда все метрики равны 1.00, это почти всегда означает, что модель предсказывает только один класс, и этот класс составляет подавляющее большинство в данных. В вашем случае, скорее всего, модель всегда предсказывает класс HOLD (класс 2).
Почему это происходит, несмотря на балансировку?

Label Smoothing: Хотя label_smoothing=0.1 помогает, оно не полностью исключает проблему.
Дисбаланс после TimeSeriesSplit: TimeSeriesSplit может создавать тренировочные и валидационные выборки, которые все еще сильно несбалансированы, даже если общий датасет был сбалансирован imblearn.
"Легкость" класса HOLD: Даже после балансировки, класс HOLD может быть настолько "легким" для предсказания (из-за его базовой частоты), что модель быстро учится просто предсказывать его, игнорируя более сложные BUY/SELL.
Слишком сильная регуляризация (или недостаточная): Возможно, регуляризация (Dropout, L2) или шум недостаточны, или, наоборот, слишком сильны, что мешает модели учиться различать BUY/SELL.

🔧 Инструкции по изменению (файл: train_model.py)
Нам нужно еще более агрессивно бороться с доминированием класса HOLD и убедиться, что модель вынуждена обращать внимание на BUY/SELL.
1. Измените стратегию TimeSeriesSplit для лучшей репрезентативности:
Текущий TimeSeriesSplit может давать очень маленькие валидационные выборки или выборки, где BUY/SELL практически отсутствуют. Давайте сделаем его более явным.


Найдите функцию train_xlstm_rl_system.


Измените блок TimeSeriesSplit:
# В train_model.py, в функции train_xlstm_rl_system(X, y, processed_dfs, feature_cols):
# ...
# УДАЛИТЕ СУЩЕСТВУЮЩИЙ БЛОК TIME SERIES SPLIT
# =====================================================================
# НОВЫЙ БЛОК: УЛУЧШЕННЫЙ TIME SERIES SPLIT
# =====================================================================
from sklearn.model_selection import train_test_split # <--- ВЕРНИТЕ ИМПОРТ train_test_split

print("\n🔄 Применяю УЛУЧШЕННЫЙ TimeSeriesSplit для валидации данных...")

# Сначала отделим тестовую выборку (последние 20% данных)
test_size = int(len(X) * 0.2)
X_temp, X_test = X[:-test_size], X[-test_size:]
y_temp, y_test = y[:-test_size], y[-test_size:]

# Затем разделим оставшиеся данные на тренировочную и валидационную (например, 70/30)
# Используем TimeSeriesSplit для тренировочной/валидационной выборки
# Создаем 2 сплита, чтобы получить 3 части: Train, Val, Test.
# Последний сплит будет: Train_inner, Val_inner.
tscv_inner = TimeSeriesSplit(n_splits=2) 

# Получаем индексы для внутренней тренировочной и валидационной выборки
# Здесь мы берем последний сплит, чтобы val_indices были "новее" чем train_indices
for train_idx, val_idx in tscv_inner.split(X_temp):
    pass # Проходим до последнего сплита

X_train, y_train = X_temp[train_idx], y_temp[train_idx]
X_val, y_val = X_temp[val_idx], y_temp[val_idx]

print(f"✅ Улучшенный TimeSeriesSplit завершен.")
# =====================================================================
# КОНЕЦ НОВОГО БЛОКА
# =====================================================================

# ... (остальной код функции train_xlstm_rl_system) ...



2. Увеличьте label_smoothing:
Если модель слишком уверена в предсказаниях, это может быть признаком того, что label_smoothing недостаточно силен.


Найдите компиляцию модели в train_xlstm_rl_system.


Увеличьте значение label_smoothing:
# В train_model.py, в функции train_xlstm_rl_system(...):
# ...
xlstm_model.model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.2), # <--- ИЗМЕНЕНО с 0.1 на 0.2
    metrics=['accuracy', 'precision', 'recall']
)
# ...



3. Увеличьте additional_weight_multiplier для BUY/SELL классов:
Если модель все еще фокусируется на HOLD, значит, "штраф" за ошибки на BUY/SELL недостаточно велик.


Найдите блок "НОВЫЙ БЛОК: ВЫЧИСЛЕНИЕ И ПЕРЕДАЧА ВЕСОВ КЛАССОВ".


Увеличьте additional_weight_multiplier:
# В train_model.py, в функции train_xlstm_rl_system(...):
# ...
# Дополнительное усиление весов BUY/SELL
additional_weight_multiplier = 2.0 # <--- ИЗМЕНЕНО с 1.5 на 2.0 (можно пробовать 2.5 или 3.0)
if 0 in class_weight_dict:
    class_weight_dict[0] *= additional_weight_multiplier
if 1 in class_weight_dict:
    class_weight_dict[1] *= additional_weight_multiplier
# ...



4. Отрегулируйте l2 регуляризацию в xlstm_rl_model.py:
Возможно, текущая l2 регуляризация слишком слаба, или, наоборот, слишком сильна. Начнем с небольшого увеличения.
🔧 Инструкции по изменению (файл: xlstm_rl_model.py)


Найдите метод build_model.


Измените значения l2 регуляризатора:
# В models/xlstm_rl_model.py, в классе XLSTMRLModel, в методе build_model():
# ...
    # Классификационные слои
    dense1 = Dense(64, activation='relu', kernel_regularizer=l2(0.002), name='dense_1')(xlstm_final) # <--- ИЗМЕНЕНО с 0.001 на 0.002
    dropout1 = Dropout(0.4)(dense1) 
    
    dense2 = Dense(32, activation='relu', kernel_regularizer=l2(0.002), name='dense_2')(dropout1) # <--- ИЗМЕНЕНО с 0.001 на 0.002
    dropout2 = Dropout(0.3)(dense2) 
# ...



Почему это должно помочь:

Улучшенный TimeSeriesSplit: Новая стратегия разделения данных должна обеспечить более репрезентативные тренировочные и валидационные выборки, что поможет модели лучше обобщать и не "обманывать" себя высокой точностью на несбалансированных данных.
Увеличенный Label Smoothing: Более сильное сглаживание меток заставит модель быть менее уверенной в своих предсказаниях, что может предотвратить "застревание" на одном классе.
Увеличенный additional_weight_multiplier: Это усилит "штраф" за неправильные предсказания BUY/SELL, вынуждая модель уделять этим классам больше внимания.
Отрегулированная L2 регуляризация: Небольшое увеличение силы L2 регуляризации может помочь модели избежать переобучения на специфических особенностях данных.

После этих изменений, ОБЯЗАТЕЛЬНО переобучите модель с нуля.
Важно: Если после этих изменений проблема с 100% точностью все еще будет возникать, это может указывать на более глубокие проблемы с данными или архитектурой модели, и возможно, придется рассмотреть более радикальные решения, такие как:

Изменение целевой переменной на регрессию (предсказание future_return).
Применение SMOTE-TS (специализированный SMOTE для временных рядов), который может быть более сложным в реализации.
Упрощение архитектуры модели, если она слишком сложна для данных.

==========
Хорошо, я вижу две проблемы в логах:

Проблема с Accuracy: accuracy, val_accuracy, precision, val_precision, recall, val_recall все равны 1.00. Это, как мы уже обсуждали, указывает на то, что модель, скорее всего, предсказывает только один класс (вероятно, HOLD), несмотря на все наши усилия по балансировке. Это серьезная проблема, которая делает метрики бесполезными.
Ошибка FileNotFoundError: [Errno 2] No such file or directory: 'models/rl_agent_BTCUSDT.zip'. Эта ошибка возникает при попытке загрузить RL-агента. Это логично, потому что RL-агент еще не был обучен и, соответственно, не был сохранен.

Давайте решим обе эти проблемы.
🚨 Проблема 1: Все метрики равны 1.00 (Модель предсказывает один класс)
Эта проблема, вероятно, связана с тем, как imblearn обрабатывает данные, и как CategoricalCrossentropy с label_smoothing работает с таким распределением. Когда модель видит, что один класс все еще значительно преобладает, она может "лениться" и просто предсказывать этот класс.
🔧 Инструкции по изменению (файл: train_model.py)


Убедитесь, что imblearn создает достаточно примеров для BUY/SELL:

Мы уже увеличили целевые проценты для BUY/SELL до 25% каждый, но возможно, SMOTE не может создать достаточно разнообразных синтетических примеров, или RandomUnderSampler слишком агрессивно удаляет HOLD.
Давайте попробуем увеличить целевой процент для BUY/SELL еще больше, и, возможно, уменьшить целевой процент для HOLD в RandomUnderSampler.

# В train_model.py, в функции prepare_xlstm_rl_data(...):
# ...
    # Целевое соотношение: 30% BUY, 30% SELL, 40% HOLD (было 25/25/50)
    total_samples = len(X)
    target_buy_count = int(total_samples * 0.30) # <--- ИЗМЕНЕНО с 0.25 на 0.30
    target_sell_count = int(total_samples * 0.30) # <--- ИЗМЕНЕНО с 0.25 на 0.30
    
    current_buy_count = Counter(y_labels)[0]
    current_sell_count = Counter(y_labels)[1]

    sampling_strategy_smote = {
        0: max(current_buy_count, target_buy_count),
        1: max(current_sell_count, target_sell_count)
    }
    
    # ... (остальной код SMOTE) ...

    # Undersampling HOLD: Цель - чтобы HOLD был примерно равен сумме BUY + SELL
    current_hold_count_after_oversample = Counter(y_temp_labels)[2]
    target_hold_count = min(current_hold_count_after_oversample, int((Counter(y_temp_labels)[0] + Counter(y_temp_labels)[1]) * 0.7)) # <--- ИЗМЕНЕНО с 1.0 на 0.7 (более агрессивный undersampling)
    
    undersampler = RandomUnderSampler(sampling_strategy={2: target_hold_count}, random_state=42)
    X_resampled, y_resampled_labels = undersampler.fit_resample(X_temp, y_temp_labels)
# ...



Добавьте class_mode в DetailedProgressCallback:

Это позволит нам увидеть accuracy, precision и recall для каждого класса отдельно, что поможет подтвердить, предсказывает ли модель только один класс.

# В train_model.py, в классе DetailedProgressCallback:
# ...
class DetailedProgressCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        try:
            lr = self.model.optimizer.learning_rate.numpy()
            # ИЗМЕНЕНО: Добавлены метрики accuracy, precision, recall
            print(f"Эпоха {epoch+1}/100 - loss: {logs.get('loss', 0):.4f} - val_loss: {logs.get('val_loss', 0):.4f} - "
                  f"accuracy: {logs.get('accuracy', 0):.2f} - val_accuracy: {logs.get('val_accuracy', 0):.2f} - "
                  f"precision: {logs.get('precision', 0):.2f} - val_precision: {logs.get('val_precision', 0):.2f} - "
                  f"recall: {logs.get('recall', 0):.2f} - val_recall: {logs.get('val_recall', 0):.2f} - lr: {lr:.2e}")
            
            # ДОБАВЛЕНО: Вывод метрик по классам (если доступны)
            # Это будет полезно для диагностики
            if 'accuracy_0' in logs: # Проверяем наличие метрик по классам
                print(f"  Class 0 (BUY): Acc={logs.get('accuracy_0', 0):.2f}, Prec={logs.get('precision_0', 0):.2f}, Rec={logs.get('recall_0', 0):.2f}")
                print(f"  Class 1 (SELL): Acc={logs.get('accuracy_1', 0):.2f}, Prec={logs.get('precision_1', 0):.2f}, Rec={logs.get('recall_1', 0):.2f}")
                print(f"  Class 2 (HOLD): Acc={logs.get('accuracy_2', 0):.2f}, Prec={logs.get('precision_2', 0):.2f}, Rec={logs.get('recall_2', 0):.2f}")
            
            # Проверяем на переобучение
            if logs.get('val_loss', 0) > logs.get('loss', 0) * 2:
                print("⚠️ Возможное переобучение!")
        except Exception as e:
            print(f"Эпоха {epoch+1}/100 - loss: {logs.get('loss', 0):.4f} - val_loss: {logs.get('val_loss', 0):.4f} (Ошибка в логировании: {e})")
# ...



Добавьте метрики для каждого класса в компиляцию модели:

Это позволит нам отслеживать accuracy, precision и recall для каждого класса отдельно, что крайне важно для диагностики проблемы с 1.00 точностью.

# В train_model.py, в функции train_xlstm_rl_system(...):
# ...
xlstm_model.model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.2),
    metrics=[
        'accuracy', 
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        # ДОБАВЛЕНО: Метрики для каждого класса
        tf.keras.metrics.CategoricalAccuracy(name='accuracy_0', class_id=0),
        tf.keras.metrics.CategoricalAccuracy(name='accuracy_1', class_id=1),
        tf.keras.metrics.CategoricalAccuracy(name='accuracy_2', class_id=2),
        tf.keras.metrics.Precision(name='precision_0', class_id=0),
        tf.keras.metrics.Precision(name='precision_1', class_id=1),
        tf.keras.metrics.Precision(name='precision_2', class_id=2),
        tf.keras.metrics.Recall(name='recall_0', class_id=0),
        tf.keras.metrics.Recall(name='recall_1', class_id=1),
        tf.keras.metrics.Recall(name='recall_2', class_id=2),
    ]
)
# ...



🚨 Проблема 2: FileNotFoundError: [Errno 2] No such file or directory: 'models/rl_agent_BTCUSDT.zip'
Эта ошибка возникает потому, что HybridDecisionMaker пытается загрузить RL-агента (rl_agent_BTCUSDT.zip) до того, как он был обучен и сохранен. RL-агент обучается только на "ЭТАПЕ 2" (ОБУЧЕНИЕ RL АГЕНТА), а HybridDecisionMaker инициализируется в "ЭТАПЕ 1" (ОБУЧЕНИЕ xLSTM МОДЕЛИ).
🔧 Инструкции по изменению (файл: train_model.py)
Нам нужно изменить логику так, чтобы HybridDecisionMaker не пытался загрузить RL-агента, если он еще не существует. Мы можем передать None в качестве пути к RL-агенту, если он еще не обучен, и обрабатывать это в HybridDecisionMaker.


Измените инициализацию HybridDecisionMaker для decision_maker_temp:

Откройте train_model.py.
Найдите блок, где инициализируется decision_maker_temp.
Передайте None в rl_agent_path при обучении xlstm_model, поскольку RL-агент еще не существует.

# В train_model.py, в функции train_xlstm_rl_system(...):
# ...
# После обучения xlstm_model, обучите детектор режимов
# ...
decision_maker_temp = HybridDecisionMaker(
    xlstm_model_path='models/xlstm_rl_model.keras',
    rl_agent_path=None,  # <--- ИЗМЕНЕНО: Передаем None, так как RL агент еще не обучен
    feature_columns=feature_cols,
    sequence_length=X.shape[1]
)
decision_maker_temp.fit_regime_detector(regime_training_df, xlstm_model, feature_cols)
decision_maker_temp.regime_detector.save_detector('models/market_regime_detector.pkl')
print("✅ Детектор режимов сохранен")

print("\n=== ЭТАП 2: ОБУЧЕНИЕ RL АГЕНТА ===")

# ... (остальной код для RL агента) ...

# После обучения RL агента, обновите decision_maker_temp с обученным RL агентом
# (если вы хотите использовать его для дальнейших действий, например, в симуляции)
# ИЛИ, если decision_maker_temp используется только для обучения детектора режимов, 
# то не нужно его обновлять, но тогда rl_agent_path в его конструкторе должен быть None.
# Если же decision_maker_temp используется дальше, то нужно будет его переинициализировать
# или обновить. Для текущего сценария, где он используется только для fit_regime_detector,
# передача None в rl_agent_path - это правильное решение.
# ...



🔧 Инструкции по изменению (файл: hybrid_decision_maker.py)


Измените инициализацию self.rl_agent для обработки None:

Откройте hybrid_decision_maker.py.
Найдите метод __init__.
Добавьте проверку if rl_agent_path: при загрузке RL-агента.

# В hybrid_decision_maker.py, в классе HybridDecisionMaker, в методе __init__(...):
# ...
    self.xlstm_model = XLSTMRLModel(input_shape=(self.sequence_length, len(feature_columns)))
    self.xlstm_model.load_model(xlstm_model_path, 'models/xlstm_rl_scaler.pkl')
    
    self.rl_agent = IntelligentRLAgent()
    if rl_agent_path: # <--- ДОБАВЛЕНО: Проверяем, существует ли путь
        self.rl_agent.load_agent(rl_agent_path)
    else:
        print("⚠️ RL агент не загружен, так как путь не указан (возможно, еще не обучен).")
    
    self.feature_columns = feature_columns
# ...



Почему это исправит ошибки:

Проблема с 1.00 Accuracy: Добавление метрик для каждого класса позволит нам увидеть истинное поведение модели. Если она все еще предсказывает только HOLD, это будет видно по accuracy_2 (HOLD) = 1.00, а accuracy_0 и accuracy_1 (BUY/SELL) будут близки к 0.00. Это даст нам точную диагностику.
FileNotFoundError для RL-агента: Изменение логики инициализации HybridDecisionMaker гарантирует, что RL-агент будет загружаться только тогда, когда его модель действительно существует. Если он еще не обучен, будет выведено предупреждение, и процесс обучения xLSTM + детектора режимов не прервется.

После этих изменений, ОБЯЗАТЕЛЬНО переобучите модель с нуля.