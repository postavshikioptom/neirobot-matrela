Хорошо, я внимательно изучил новые логи. Твои наблюдения снова очень точны!
Модель стала предсказывать BUY лучше, но SELL полностью провалился, а HOLD все еще плохо распознается. Это классический пример "перетягивания одеяла" между классами, что очень характерно для несбалансированных данных в трейдинге.
Анализ текущих логов (до 22-й эпохи):
Что стало лучше:


Recall для BUY класса:

Class 0 (BUY): Prec=0.71, Rec=0.93 (на эпохе 21). Это очень высокий Recall для BUY! Модель теперь хорошо определяет истинные BUY-сигналы.
Распределение предсказаний: BUY=32.2% на эпохе 21, что значительно выше, чем 0.6-23.8% в предыдущих версиях.



Общая производительность на валидации:

val_accuracy достигает 0.82 (эпоха 19).
Macro Avg F1-Score достигает 0.776 (эпоха 21).
Вывод: На валидационном наборе модель теперь работает значительно лучше, чем в предыдущих итерациях.



Проблемы, которые остаются / Ухудшения:


Катастрофический провал SELL класса:

Class 1 (SELL): Prec=0.31, Rec=0.03 (эпоха 1), затем Prec=0.00, Rec=0.00 (эпоха 2-5).
Распределение предсказаний: SELL=0.0% на эпохе 1, затем 25.4-30.5% (но с очень низким Recall).
Вывод: Мы слишком сильно подавили SELL. Модель практически не предсказывает его, даже когда он должен быть.



Низкий Recall для HOLD класса:

Class 2 (HOLD): Prec=0.90, Rec=0.55 (на эпохе 21). Recall для HOLD все еще низкий. Несмотря на то, что HOLD является мажоритарным классом, модель его плохо распознает.



Переобучение (Overfitting):

Сообщения ⚠️ Обнаружено переобучение появляются регулярно.
EarlyStopping срабатывает на 22-й эпохе, восстанавливая веса с 18-й эпохи. Это означает, что модель все еще быстро переобучается.



Катастрофический провал на тестовом наборе:

xLSTM Loss: 0.5917 (очень высокий).
xLSTM Точность: 22.10%, Precision: 18.31%, Recall: 16.22%.
Class 0 (BUY): Prec=0.00, Rec=0.00, Class 1 (SELL): Prec=0.00, Rec=0.00, Class 2 (HOLD): Prec=1.00, Rec=0.16.
Вывод: Модель по-прежнему полностью теряет способность к обобщению на тестовом наборе. Это указывает на то, что, хотя на валидации она выглядит хорошо, на совершенно новых данных она не работает.



Твои замечания о признаках с минусом для BUY:

CDLDOJI: -0.2682 (для BUY)
CDLDOJI: 0.3258 (для HOLD)
Это означает, что когда модель предсказывает BUY, CDLDOJI в среднем имеет отрицательное значение, а когда предсказывает HOLD, CDLDOJI имеет положительное значение. Это контринтуитивно, так как доджи обычно ассоциируется с неопределенностью. Это подтверждает, что модель, возможно, использует признаки не так, как мы ожидаем, или что сами метки BUY/SELL/HOLD в данных все еще не идеальны.

План по исправлению (с учетом твоих предложений и анализа):
Мы должны сбалансировать веса классов, улучшить качество SELL и HOLD сигналов, а также продолжить борьбу с переобучением и, главное, с плохой генерализацией на тестовом наборе.
Шаг 1: Корректировка весов классов (сбалансировать BUY, SELL, HOLD)
Твоё предложение "поставить весы все же больше в BUY и HOLD а меньше в SELL" — это правильное направление. Мы попробуем сделать BUY и HOLD более привлекательными, а SELL — менее.
Шаг 2: Улучшение качества SELL и HOLD сигналов (в логике генерации меток)
Мы сделаем условия для sell_condition более строгими, чтобы модель училась на более "чистых" медвежьих сигналах, и ослабим buy_condition. Также, нужно улучшить генерацию HOLD-сигналов.
Шаг 3: Дальнейшая борьба с переобучением и улучшение генерализации
Катастрофический провал на тестовом наборе указывает на очень сильное переобучение или на то, что тестовый набор сильно отличается от тренировочного/валидационного.

Инструкции по реализации:
Файл 1: train_model.py
1. Корректировка весов классов (сбалансировать BUY, SELL, HOLD):
Местоположение: В функции train_xlstm_rl_system, в блоке вычисления class_weight_dict (строка ~350).
ЗАМЕНИТЬ:
# СТАРЫЙ КОД (более сбалансированные веса)
    if 0 in class_weight_dict:
        class_weight_dict[0] *= 2.0  # ИЗМЕНЕНО: Увеличиваем вес BUY (с 2.5 до 2.0 - немного меньше агрессии)
    if 1 in class_weight_dict:
        class_weight_dict[1] *= 1.2  # ИЗМЕНЕНО: Увеличиваем вес SELL (с 0.8 до 1.2 - чтобы не был совсем 0)
    
    if 2 in class_weight_dict:
        class_weight_dict[2] *= 1.0  # ИЗМЕНЕНО: Устанавливаем вес HOLD на 1.0 (с 1.2 - для баланса)

НА НОВЫЙ КОД (усилить BUY и HOLD, но немного больше SELL):
# НОВЫЙ КОД - Корректируем веса классов (усилить BUY и HOLD, но немного больше SELL)
    if 0 in class_weight_dict:
        class_weight_dict[0] *= 1.8  # ИЗМЕНЕНО: Немного уменьшаем BUY (с 2.0 до 1.8)
    if 1 in class_weight_dict:
        class_weight_dict[1] *= 1.8  # ИЗМЕНЕНО: Увеличиваем SELL (с 1.2 до 1.8 - равный BUY)
    
    if 2 in class_weight_dict:
        class_weight_dict[2] *= 1.5  # ИЗМЕНЕНО: Значительно увеличиваем HOLD (с 1.0 до 1.5)

Объяснение: Мы делаем BUY и SELL равными, но немного менее агрессивными, чем в предыдущей итерации. При этом значительно увеличиваем вес HOLD, чтобы модель чаще его предсказывала и улучшила Recall.
2. Улучшение качества SELL и HOLD сигналов в логике генерации меток:
Мы сделаем условия для sell_condition более строгими, чтобы модель училась на более "чистых" медвежьих сигналах, и ослабим buy_condition. Также, нужно улучшить генерацию HOLD-сигналов.
Местоположение: В функции prepare_xlstm_rl_data, в блоке "НОВЫЙ КОД - Условия для BUY/SELL только на основе future_return и классических индикаторов/паттернов (без VSA)" (строка ~150-200).
ЗАМЕНИТЬ:
# СТАРЫЙ КОД
        # Классические технические фильтры
        strong_trend = df['ADX_14'] > 25
        
        # Более строгие условия для BUY
        rsi_buy_zone = df['RSI_14'] < 35 # ИЗМЕНЕНО: С 40 до 35 (более экстремально)
        macd_buy_signal = (df['MACD_12_26_9'] > df['MACD_signal']) & \
                          (df['MACD_12_26_9'].shift(1) <= df['MACD_signal'].shift(1)) & \
                          (df['MACD_hist'] > 0.001) # ИЗМЕНЕНО: MACD_hist должен быть положительным
        
        # Менее строгие условия для SELL
        rsi_sell_zone = df['RSI_14'] > 65 # ИЗМЕНЕНО: С 60 до 65 (более экстремально)
        macd_sell_signal = (df['MACD_12_26_9'] < df['MACD_signal']) & \
                           (df['MACD_12_26_9'].shift(1) >= df['MACD_signal'].shift(1)) & \
                           (df['MACD_hist'] < -0.001) # ИЗМЕНЕНО: MACD_hist должен быть отрицательным

        # Условия для BUY/SELL только на основе future_return и классических индикаторов/паттернов
        buy_condition = (
            (df['future_return'] > df['dynamic_threshold'] * 1.2) & # ИЗМЕНЕНО: Более высокий порог для BUY
            (strong_trend & rsi_buy_zone & macd_buy_signal) # ИЗМЕНЕНО: Все условия должны быть выполнены для BUY
        )

        sell_condition = (
            (df['future_return'] < -df['dynamic_threshold'] * 0.8) & # ИЗМЕНЕНО: Менее строгий порог для SELL
            (strong_trend | rsi_sell_zone | macd_sell_signal) # ИЗМЕНЕНО: Одно из условий для SELL
        )

НА НОВЫЙ КОД (более сбалансированные условия для BUY/SELL/HOLD):
# НОВЫЙ КОД - Условия для BUY/SELL (более сбалансированные)
        # Увеличиваем пороги для генерации торговых сигналов
        df['base_threshold'] = 0.008
        df['dynamic_threshold'] = np.maximum(
            df['base_threshold'],
            (df['ATR_14'] / df['close'] * 1.5).fillna(0.008)
        )

        # Классические технические фильтры
        strong_trend = df['ADX_14'] > 25
        
        # Более строгие условия для BUY (как раньше, но немного ослабляем)
        rsi_buy_zone = df['RSI_14'] < 38 # ИЗМЕНЕНО: С 35 до 38 (чуть менее экстремально)
        macd_buy_signal = (df['MACD_12_26_9'] > df['MACD_signal']) & \
                          (df['MACD_12_26_9'].shift(1) <= df['MACD_signal'].shift(1)) & \
                          (df['MACD_hist'] > 0.0005) # ИЗМЕНЕНО: С 0.001 до 0.0005 (чуть менее строгий)
        
        # Более строгие условия для SELL (усиливаем)
        rsi_sell_zone = df['RSI_14'] > 62 # ИЗМЕНЕНО: С 65 до 62 (чуть менее экстремально)
        macd_sell_signal = (df['MACD_12_26_9'] < df['MACD_signal']) & \
                           (df['MACD_12_26_9'].shift(1) >= df['MACD_signal'].shift(1)) & \
                           (df['MACD_hist'] < -0.0005) # ИЗМЕНЕНО: С -0.001 до -0.0005 (чуть менее строгий)

        # Условия для BUY/SELL только на основе future_return и классических индикаторов/паттернов
        buy_condition = (
            (df['future_return'] > df['dynamic_threshold'] * 1.0) & # ИЗМЕНЕНО: С 1.2 до 1.0 (менее строгий порог для BUY)
            (strong_trend & rsi_buy_zone & macd_buy_signal) # Оставляем AND для качества BUY
        )

        sell_condition = (
            (df['future_return'] < -df['dynamic_threshold'] * 1.0) & # ИЗМЕНЕНО: С 0.8 до 1.0 (чуть более строгий порог для SELL)
            (strong_trend & rsi_sell_zone & macd_sell_signal) # ИЗМЕНЕНО: Делаем AND для качества SELL
        )

Объяснение:

Для BUY: Мы немного ослабляем условия, чтобы модель не была слишком избирательна, но сохраняем AND для качества.
Для SELL: Мы значительно усиливаем условия, делая их AND, чтобы модель училась на более чистых SELL-сигналах.
Для HOLD: В блоке переклассификации HOLD мы не будем его трогать, так как уже увеличили его вес.

3. Корректировка AntiOverfittingCallback patience:
EarlyStopping сработал на 22-й эпохе, восстанавливая веса с 18-й. Это означает, что patience в AntiOverfittingCallback (12 эпох) может быть слишком большим, если ReduceLROnPlateau срабатывает раньше.
Местоположение: В списке callbacks в функции train_xlstm_rl_system (строка ~400).
ЗАМЕНИТЬ:
# СТАРЫЙ КОД
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            mode='min',
            patience=15,
            restore_best_weights=True,
            verbose=1,
            min_delta=0.001
        ),
        AntiOverfittingCallback(patience=12, min_improvement=0.005),  # ИЗМЕНЕНО: patience до 12
        MemoryCleanupCallback(),
        DetailedProgressCallback(X_val_to_model, feature_cols), # ИЗМЕНЕНО: Передаем X_val_to_model и feature_cols
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.7,
            patience=8,
            min_lr=1e-6,
            verbose=1
        ),

НА НОВЫЙ КОД (уменьшаем patience в AntiOverfittingCallback):
# НОВЫЙ КОД - Уменьшаем patience в AntiOverfittingCallback
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            mode='min',
            patience=15,
            restore_best_weights=True,
            verbose=1,
            min_delta=0.001
        ),
        AntiOverfittingCallback(patience=8, min_improvement=0.005),  # ИЗМЕНЕНО: patience до 8
        MemoryCleanupCallback(),
        DetailedProgressCallback(X_val_to_model, feature_cols),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.7,
            patience=8,
            min_lr=1e-6,
            verbose=1
        ),

Объяснение: Уменьшаем patience для AntiOverfittingCallback с 12 до 8. Это сделает его более агрессивным в остановке обучения при переобучении, что может помочь, если ReduceLROnPlateau не успевает сработать.

Файл 4: trading_env.py
1. Корректировка _calculate_advanced_reward для борьбы с дисбалансом:
Мы должны убедиться, что функция наград для RL-агента также отражает наши новые приоритеты (сбалансировать BUY, SELL, HOLD).
Местоположение: Внутри метода _calculate_advanced_reward (строка ~130).
ЗАМЕНИТЬ:
# СТАРЫЙ КОД
            buy_signal_strength = (
                (current_row.get('RSI_14', 50) < 35) + # ИЗМЕНЕНО: С 40 до 35
                (current_row.get('ADX_14', 0) > 25) + # ИЗМЕНЕНО: С 20 до 25
                (current_row.get('MACD_hist', 0) > 0.002) # ИЗМЕНЕНО: С 0.001 до 0.002
            )
            sell_signal_strength = (
                (current_row.get('RSI_14', 50) > 65) + # ИЗМЕНЕНО: С 60 до 65
                (current_row.get('ADX_14', 0) > 25) + # ИЗМЕНЕНО: С 20 до 25
                (current_row.get('MACD_hist', 0) < -0.002) # ИЗМЕНЕНО: С -0.001 до -0.002
            )

            # Увеличиваем штраф за слабые BUY-сигналы, если RL предсказывает BUY
            if action == 1 and buy_signal_strength < 2: # ИЗМЕНЕНО: Требуем 2+ сильных сигнала
                overtrading_penalty = -3.0 # ИЗМЕНЕНО: Увеличен штраф (с -2.0 до -3.0)
            # Уменьшаем штраф за слабые SELL-сигналы, если RL предсказывает SELL
            elif action == 0 and sell_signal_strength < 1: # ИЗМЕНЕНО: Требуем 1+ сильный сигнал
                overtrading_penalty = -0.2 # ИЗМЕНЕНО: Уменьшен штраф (с -0.5 до -0.2)

НА НОВЫЙ КОД (более сбалансированное вознаграждение):
# НОВЫЙ КОД - Корректируем функцию наград для RL (более сбалансированное вознаграждение)
            buy_signal_strength = (
                (current_row.get('RSI_14', 50) < 38) + # ИЗМЕНЕНО: С 35 до 38
                (current_row.get('ADX_14', 0) > 25) +
                (current_row.get('MACD_hist', 0) > 0.001) # ИЗМЕНЕНО: С 0.002 до 0.001
            )
            sell_signal_strength = (
                (current_row.get('RSI_14', 50) > 62) + # ИЗМЕНЕНО: С 65 до 62
                (current_row.get('ADX_14', 0) > 25) +
                (current_row.get('MACD_hist', 0) < -0.001) # ИЗМЕНЕНО: С -0.002 до -0.001
            )

            # Увеличиваем штраф за слабые BUY-сигналы, если RL предсказывает BUY
            if action == 1 and buy_signal_strength < 2: # Требуем 2+ сильных сигнала
                overtrading_penalty = -2.0 # ИЗМЕНЕНО: С -3.0 до -2.0 (чуть менее агрессивно)
            # Увеличиваем штраф за слабые SELL-сигналы, если RL предсказывает SELL
            elif action == 0 and sell_signal_strength < 2: # ИЗМЕНЕНО: Требуем 2+ сильных сигнала
                overtrading_penalty = -2.0 # ИЗМЕНЕНО: С -0.2 до -2.0 (более агрессивно)

            # Добавляем бонус за HOLD, если нет сильных сигналов
            if action == 2: # HOLD
                if buy_signal_strength < 1 and sell_signal_strength < 1: # Если нет сильных BUY/SELL сигналов
                    hold_reward += 0.5 # Бонус за правильный HOLD
                else:
                    hold_reward -= 0.5 # Штраф за HOLD, если есть сильные сигналы

Объяснение: Мы делаем условия для "явного сигнала" в RL-среде более сбалансированными для BUY и SELL (требуем 2+ сильных сигнала для обоих). Штрафы за слабые сигналы теперь одинаковы. Дополнительно, мы даем бонус за HOLD, если нет сильных BUY/SELL сигналов, и штрафуем за HOLD, если они есть.



=========

Инструкции по реализации:
Файл 1: train_model.py
1. Корректировка весов классов (более сбалансированные):
Местоположение: В функции train_xlstm_rl_system, в блоке вычисления class_weight_dict (строка ~350).
ЗАМЕНИТЬ:
# СТАРЫЙ КОД (усилить BUY и HOLD, но немного больше SELL)
    if 0 in class_weight_dict:
        class_weight_dict[0] *= 1.8  # ИЗМЕНЕНО: Немного уменьшаем BUY (с 2.0 до 1.8)
    if 1 in class_weight_dict:
        class_weight_dict[1] *= 1.8  # ИЗМЕНЕНО: Увеличиваем SELL (с 1.2 до 1.8 - равный BUY)
    
    if 2 in class_weight_dict:
        class_weight_dict[2] *= 1.5  # ИЗМЕНЕНО: Значительно увеличиваем HOLD (с 1.0 до 1.5)

НА НОВЫЙ КОД (более сбалансированные, с акцентом на HOLD):
# НОВЫЙ КОД - Корректируем веса классов (более сбалансированные, с акцентом на HOLD)
    if 0 in class_weight_dict:
        class_weight_dict[0] *= 1.5  # ИЗМЕНЕНО: Уменьшаем BUY (с 1.8 до 1.5)
    if 1 in class_weight_dict:
        class_weight_dict[1] *= 1.5  # ИЗМЕНЕНО: Устанавливаем SELL на 1.5 (равный BUY)
    
    if 2 in class_weight_dict:
        class_weight_dict[2] *= 2.0  # ИЗМЕНЕНО: Значительно увеличиваем HOLD (с 1.5 до 2.0)

Объяснение: Мы уменьшаем агрессию для BUY и SELL, делая их равными (1.5), и еще сильнее увеличиваем вес HOLD (2.0). Это должно заставить модель чаще предсказывать HOLD и быть более осторожной с BUY/SELL.
2. Улучшение логики генерации меток (BUY, SELL, HOLD):
Мы сделаем условия для BUY и SELL более симметричными и логичными, чтобы избежать контринтуитивных связей с признаками.
Местоположение: В функции prepare_xlstm_rl_data, в блоке "НОВЫЙ КОД - Условия для BUY/SELL только на основе future_return и классических индикаторов/паттернов (без VSA)" (строка ~150-200).
ЗАМЕНИТЬ:
# СТАРЫЙ КОД
        # Классические технические фильтры
        strong_trend = df['ADX_14'] > 25
        
        # Более строгие условия для BUY
        rsi_buy_zone = df['RSI_14'] < 35 # ИЗМЕНЕНО: С 40 до 35 (более экстремально)
        macd_buy_signal = (df['MACD_12_26_9'] > df['MACD_signal']) & \
                          (df['MACD_12_26_9'].shift(1) <= df['MACD_signal'].shift(1)) & \
                          (df['MACD_hist'] > 0.001) # ИЗМЕНЕНО: MACD_hist должен быть положительным
        
        # Менее строгие условия для SELL
        rsi_sell_zone = df['RSI_14'] > 65 # ИЗМЕНЕНО: С 60 до 65 (более экстремально)
        macd_sell_signal = (df['MACD_12_26_9'] < df['MACD_signal']) & \
                           (df['MACD_12_26_9'].shift(1) >= df['MACD_signal'].shift(1)) & \
                           (df['MACD_hist'] < -0.001) # ИЗМЕНЕНО: MACD_hist должен быть отрицательным

        # Условия для BUY/SELL только на основе future_return и классических индикаторов/паттернов
        buy_condition = (
            (df['future_return'] > df['dynamic_threshold'] * 1.2) & # ИЗМЕНЕНО: Более высокий порог для BUY
            (strong_trend & rsi_buy_zone & macd_buy_signal) # ИЗМЕНЕНО: Все условия должны быть выполнены для BUY
        )

        sell_condition = (
            (df['future_return'] < -df['dynamic_threshold'] * 0.8) & # ИЗМЕНЕНО: Менее строгий порог для SELL
            (strong_trend | rsi_sell_zone | macd_sell_signal) # ИЗМЕНЕНО: Одно из условий для SELL
        )

НА НОВЫЙ КОД (более симметричные и логичные условия):
# НОВЫЙ КОД - Условия для BUY/SELL (более симметричные и логичные)
        # Увеличиваем пороги для генерации торговых сигналов
        df['base_threshold'] = 0.008
        df['dynamic_threshold'] = np.maximum(
            df['base_threshold'],
            (df['ATR_14'] / df['close'] * 1.5).fillna(0.008)
        )

        # Классические технические фильтры
        strong_trend = df['ADX_14'] > 25
        
        # Условия для BUY
        rsi_buy_zone = df['RSI_14'] < 30 # ИЗМЕНЕНО: С 38 до 30 (более экстремально для BUY)
        macd_buy_signal = (df['MACD_12_26_9'] > df['MACD_signal']) & \
                          (df['MACD_hist'] > 0.001) # ИЗМЕНЕНО: Убрано .shift(1) для простоты, MACD_hist > 0.001
        
        # Условия для SELL
        rsi_sell_zone = df['RSI_14'] > 70 # ИЗМЕНЕНО: С 62 до 70 (более экстремально для SELL)
        macd_sell_signal = (df['MACD_12_26_9'] < df['MACD_signal']) & \
                           (df['MACD_hist'] < -0.001) # ИЗМЕНЕНО: Убрано .shift(1) для простоты, MACD_hist < -0.001

        # Условия для BUY/SELL только на основе future_return и классических индикаторов/паттернов
        # Делаем условия более строгими для обоих классов (AND), чтобы получить более чистые сигналы
        buy_condition = (
            (df['future_return'] > df['dynamic_threshold'] * 1.0) & # Оставляем порог 1.0
            (strong_trend & rsi_buy_zone & macd_buy_signal) # ИЗМЕНЕНО: Все условия должны быть выполнены для BUY
        )

        sell_condition = (
            (df['future_return'] < -df['dynamic_threshold'] * 1.0) & # Оставляем порог 1.0
            (strong_trend & rsi_sell_zone & macd_sell_signal) # ИЗМЕНЕНО: Все условия должны быть выполнены для SELL
        )

Объяснение:

Мы делаем условия для BUY и SELL более экстремальными по RSI и MACD, чтобы генерировать только сильные, чистые сигналы.
Теперь оба buy_condition и sell_condition используют AND для всех фильтров. Это уменьшит количество генерируемых BUY/SELL сигналов, но повысит их качество, что должно помочь модели лучше их изучить.
Это также должно помочь с проблемой "минусов в признаках", так как мы теперь ищем более однозначные сигналы.

3. Корректировка логики переклассификации HOLD:
Поскольку мы значительно увеличили вес HOLD, мы должны быть осторожны, чтобы не переусердствовать с его подавлением.
Местоположение: В функции prepare_xlstm_rl_data, в блоке "НОВЫЙ КОД - Условия для переклассификации HOLD без VSA" (строка ~220-280).
ЗАМЕНИТЬ:
# СТАРЫЙ КОД
        # Если HOLD все еще сильно преобладает, пытаемся переклассифицировать
        if current_hold_count > (current_buy_count + current_sell_count) * 2.5:
            print(f"⚠️ Сильный дисбаланс классов. Попытка УМНОЙ переклассификации части HOLD-сигналов (без VSA).")
            hold_indices = df[df['target'] == 2].index
            
            import random
            random.seed(42)
            
            reclassify_count = int(current_hold_count * 0.20)
            if reclassify_count > 0:
                reclassify_indices = random.sample(list(hold_indices), min(reclassify_count, len(hold_indices)))
                
                for idx in reclassify_indices:
                    if idx < 5: continue
                    
                    rsi = df.loc[idx, 'RSI_14']
                    adx = df.loc[idx, 'ADX_14']
                    macd_hist = df.loc[idx, 'MACD_hist']
                    price_change_3_period = df['close'].pct_change(3).loc[idx]

                    # Условия для переклассификации (без VSA)
                    # 1. RSI + ADX + движение цены
                    if (rsi < 40 and adx > 20 and price_change_3_period > 0.003): # ИЗМЕНЕНО: Убрано volume_ratio
                        df.loc[idx, 'target'] = 0  # BUY
                    elif (rsi > 60 and adx > 20 and price_change_3_period < -0.003): # ИЗМЕНЕНО: Убрано volume_ratio
                        df.loc[idx, 'target'] = 1  # SELL
                        
                    # 2. MACD гистограмма + движение цены
                    elif (macd_hist > 0.001 and price_change_3_period > 0.002): # ИЗМЕНЕНО: Убрано volume_ratio
                        df.loc[idx, 'target'] = 0  # BUY
                    elif (macd_hist < -0.001 and price_change_3_period < -0.002): # ИЗМЕНЕНО: Убрано volume_ratio
                        df.loc[idx, 'target'] = 1  # SELL
                        
                    # 3. Сильный тренд по ADX + движение цены
                    elif (adx > 30 and abs(price_change_3_period) > 0.005):
                        df.loc[idx, 'target'] = 0 if price_change_3_period > 0 else 1
            
            print(f"Баланс классов после УМНОЙ переклассификации (без VSA):")
            unique, counts = np.unique(df['target'], return_counts=True)
            class_names = ['BUY', 'SELL', 'HOLD']
            for class_idx, count in zip(unique, counts):
                print(f"  {class_names[class_idx]}: {count} ({count/len(df)*100:.1f}%)")
        else:
            print(f"✅ Баланс классов приемлемый, переклассификация HOLD не требуется.")

НА НОВЫЙ КОД (менее агрессивная переклассификация HOLD):
# НОВЫЙ КОД - Менее агрессивная переклассификация HOLD
        # Если HOLD все еще сильно преобладает, НО теперь с более высоким весом,
        # мы можем быть менее агрессивны в переклассификации.
        # Уменьшаем порог для переклассификации HOLD.
        if current_hold_count > (current_buy_count + current_sell_count) * 3.0: # ИЗМЕНЕНО: С 2.5 до 3.0
            print(f"⚠️ Сильный дисбаланс классов. Попытка УМНОЙ переклассификации части HOLD-сигналов (без VSA).")
            hold_indices = df[df['target'] == 2].index
            
            import random
            random.seed(42)
            
            reclassify_count = int(current_hold_count * 0.10) # ИЗМЕНЕНО: С 0.20 до 0.10 (меньше переклассификации)
            if reclassify_count > 0:
                reclassify_indices = random.sample(list(hold_indices), min(reclassify_count, len(hold_indices)))
                
                for idx in reclassify_indices:
                    if idx < 5: continue
                    
                    rsi = df.loc[idx, 'RSI_14']
                    adx = df.loc[idx, 'ADX_14']
                    macd_hist = df.loc[idx, 'MACD_hist']
                    price_change_3_period = df['close'].pct_change(3).loc[idx]

                    # Условия для переклассификации (без VSA), делаем их более строгими
                    # 1. RSI + ADX + движение цены
                    if (rsi < 30 and adx > 25 and price_change_3_period > 0.005): # ИЗМЕНЕНО: Более строгие условия
                        df.loc[idx, 'target'] = 0  # BUY
                    elif (rsi > 70 and adx > 25 and price_change_3_period < -0.005): # ИЗМЕНЕНО: Более строгие условия
                        df.loc[idx, 'target'] = 1  # SELL
                        
                    # 2. MACD гистограмма + движение цены
                    elif (macd_hist > 0.002 and price_change_3_period > 0.004): # ИЗМЕНЕНО: Более строгие условия
                        df.loc[idx, 'target'] = 0  # BUY
                    elif (macd_hist < -0.002 and price_change_3_period < -0.004): # ИЗМЕНЕНО: Более строгие условия
                        df.loc[idx, 'target'] = 1  # SELL
                        
                    # 3. Сильный тренд по ADX + движение цены
                    elif (adx > 35 and abs(price_change_3_period) > 0.008): # ИЗМЕНЕНО: Более строгие условия
                        df.loc[idx, 'target'] = 0 if price_change_3_period > 0 else 1
            
            print(f"Баланс классов после УМНОЙ переклассификации (без VSA):")
            unique, counts = np.unique(df['target'], return_counts=True)
            class_names = ['BUY', 'SELL', 'HOLD']
            for class_idx, count in zip(unique, counts):
                print(f"  {class_names[class_idx]}: {count} ({count/len(df)*100:.1f}%)")
        else:
            print(f"✅ Баланс классов приемлемый, переклассификация HOLD не требуется.")

Объяснение: Мы уменьшаем процент переклассификации HOLD и делаем условия для переклассификации более строгими. Это должно позволить большему количеству HOLD-сигналов оставаться HOLD.

Файл 4: trading_env.py
1. Корректировка _calculate_advanced_reward для борьбы с дисбалансом:
Мы должны убедиться, что функция наград для RL-агента также отражает наши новые приоритеты (сбалансировать BUY, SELL, HOLD).
Местоположение: Внутри метода _calculate_advanced_reward (строка ~130).
ЗАМЕНИТЬ:
# СТАРЫЙ КОД
            buy_signal_strength = (
                (current_row.get('RSI_14', 50) < 38) + # ИЗМЕНЕНО: С 35 до 38
                (current_row.get('ADX_14', 0) > 25) +
                (current_row.get('MACD_hist', 0) > 0.001) # ИЗМЕНЕНО: С 0.002 до 0.001
            )
            sell_signal_strength = (
                (current_row.get('RSI_14', 50) > 62) + # ИЗМЕНЕНО: С 65 до 62
                (current_row.get('ADX_14', 0) > 25) +
                (current_row.get('MACD_hist', 0) < -0.001) # ИЗМЕНЕНО: С -0.002 до -0.001
            )

            # Увеличиваем штраф за слабые BUY-сигналы, если RL предсказывает BUY
            if action == 1 and buy_signal_strength < 2: # Требуем 2+ сильных сигнала
                overtrading_penalty = -2.0 # ИЗМЕНЕНО: С -3.0 до -2.0 (чуть менее агрессивно)
            # Увеличиваем штраф за слабые SELL-сигналы, если RL предсказывает SELL
            elif action == 0 and sell_signal_strength < 2: # ИЗМЕНЕНО: Требуем 2+ сильных сигнала
                overtrading_penalty = -2.0 # ИЗМЕНЕНО: С -0.2 до -2.0 (более агрессивно)

            # Добавляем бонус за HOLD, если нет сильных сигналов
            if action == 2: # HOLD
                if buy_signal_strength < 1 and sell_signal_strength < 1: # Если нет сильных BUY/SELL сигналов
                    hold_reward += 0.5 # Бонус за правильный HOLD
                else:
                    hold_reward -= 0.5 # Штраф за HOLD, если есть сильные сигналы

НА НОВЫЙ КОД (более сбалансированное вознаграждение, с акцентом на HOLD):
# НОВЫЙ КОД - Корректируем функцию наград для RL (более сбалансированное вознаграждение, с акцентом на HOLD)
            buy_signal_strength = (
                (current_row.get('RSI_14', 50) < 30) + # ИЗМЕНЕНО: С 38 до 30
                (current_row.get('ADX_14', 0) > 25) +
                (current_row.get('MACD_hist', 0) > 0.001)
            )
            sell_signal_strength = (
                (current_row.get('RSI_14', 50) > 70) + # ИЗМЕНЕНО: С 62 до 70
                (current_row.get('ADX_14', 0) > 25) +
                (current_row.get('MACD_hist', 0) < -0.001)
            )

            # Увеличиваем штраф за слабые BUY-сигналы, если RL предсказывает BUY
            if action == 1 and buy_signal_strength < 2: # Требуем 2+ сильных сигнала
                overtrading_penalty = -1.0 # ИЗМЕНЕНО: С -2.0 до -1.0 (менее агрессивно)
            # Увеличиваем штраф за слабые SELL-сигналы, если RL предсказывает SELL
            elif action == 0 and sell_signal_strength < 2: # Требуем 2+ сильных сигнала
                overtrading_penalty = -1.0 # ИЗМЕНЕНО: С -2.0 до -1.0 (менее агрессивно)

            # Добавляем бонус за HOLD, если нет сильных сигналов
            if action == 2: # HOLD
                if buy_signal_strength < 1 and sell_signal_strength < 1: # Если нет сильных BUY/SELL сигналов
                    hold_reward += 1.0 # ИЗМЕНЕНО: Увеличен бонус за правильный HOLD (с 0.5 до 1.0)
                else:
                    hold_reward -= 1.0 # ИЗМЕНЕНО: Увеличен штраф за HOLD, если есть сильные сигналы (с -0.5 до -1.0)
