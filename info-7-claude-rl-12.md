🚨 Главная проблема: Баланс классов все еще сильно смещен к HOLD (89-98%)
Ваши текущие результаты показывают, что даже "умная" переклассификация и ослабленные пороги не дают достаточного количества BUY/SELL сигналов. Это означает, что сами условия для определения этих сигналов все еще слишком строги, или что базовая динамика рынка не генерирует достаточно "кандидатов" для переклассификации.
Ключевым моментом здесь будет максимально агрессивная, но осмысленная генерация BUY/SELL меток до применения imblearn. imblearn будет работать лучше, если у него будет больше "исходных" примеров для oversampling'а.

🔧 Инструкции по изменению (файл: train_model.py)
1. Еще более агрессивное ослабление порогов и VSA-скора:
Мы пойдем на более рискованные, но необходимые шаги, чтобы увеличить количество BUY/SELL.


Найдите функцию prepare_xlstm_rl_data.


В блоке "НОВЫЙ БЛОК: ДИНАМИЧЕСКИЕ ПОРОГИ И ВЗВЕШЕННЫЙ VSA-СКОР" измените:
# В train_model.py, в функции prepare_xlstm_rl_data(...):
# ...
        # 1. Динамический порог future_return на основе ATR
        # Еще больше ослабляем порог, чтобы захватить больше движений
        df['dynamic_future_threshold'] = df['ATR_14'] / df['close'] * 0.8 # СНИЖЕНО с 1.5 до 0.8
        df['dynamic_future_threshold'] = df['dynamic_future_threshold'].replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0005) # Минимальный порог 0.05% (было 0.1%)
        
        # 2. Взвешенное сочетание VSA-сигналов для более гибких условий
        # Еще больше ослабляем пороги для VSA-сисилы, чтобы больше сигналов проходило
        df['vsa_buy_score'] = (
            0.3 * (df['vsa_no_supply'] == 1) +
            0.3 * (df['vsa_stopping_volume'] == 1) +
            0.4 * (df['vsa_strength'] > 0.1) # СНИЖЕНО с 0.25 до 0.1
        )
        df['vsa_sell_score'] = (
            0.3 * (df['vsa_no_demand'] == 1) +
            0.3 * (df['vsa_climactic_volume'] == 1) +
            0.4 * (df['vsa_strength'] < -0.1) # СНИЖЕНО с -0.25 до -0.1
        )

        # BUY: положительная доходность (динамический порог) + взвешенный VSA-скор
        buy_condition = (
            (df['future_return'] > df['dynamic_future_threshold']) &
            (df['vsa_buy_score'] > 0.2) # СНИЖЕНО с 0.3 до 0.2
        )
        
        # SELL: отрицательная доходность (динамический порог) + взвешенный VSA-скор
        sell_condition = (
            (df['future_return'] < -df['dynamic_future_threshold']) &
            (df['vsa_sell_score'] > 0.2) # СНИЖЕНО с 0.3 до 0.2
        )
# ...



2. Максимально агрессивная, но умная переклассификация HOLD-сигналов:
Мы увеличим процент переклассификации и сделаем условия еще более мягкими.


В блоке "НОВЫЙ БЛОК: УЛУЧШЕННАЯ ПЕРЕКЛАССИФИКАЦИЯ HOLD-СИГНАЛОВ" измените:
# В train_model.py, в функции prepare_xlstm_rl_data(...):
# ...
        # Переклассифицируем 40% HOLD (было 30%)
        reclassify_count = int(current_hold_count * 0.40) # УВЕЛИЧЕНО до 40%
        if reclassify_count > 0:
            reclassify_indices = random.sample(list(hold_indices), min(reclassify_count, len(hold_indices)))
            
            for idx in reclassify_indices:
                if idx < 1: continue 
                
                rsi = df.loc[idx, 'RSI_14']
                adx = df.loc[idx, 'ADX_14']
                adx_prev = df.loc[idx-1, 'ADX_14']
                price_change_5_period = df['close'].pct_change(5).loc[idx]
                atr_ratio = df.loc[idx, 'ATR_14'] / df.loc[idx, 'close']
                
                # Условия для переклассификации (ЕЩЕ БОЛЕЕ МЯГКИЕ)
                # 1. Слабый RSI + растущий ADX + небольшое движение → BUY
                if (rsi < 45 and adx > adx_prev and abs(price_change_5_period) > 0.0005): # RSI < 45 (было 40), price_change_5_period > 0.0005 (было 0.001)
                    df.loc[idx, 'target'] = 0  # BUY

                # 2. RSI > 55 + растущий ADX + небольшое движение → SELL
                elif (rsi > 55 and adx > adx_prev and abs(price_change_5_period) > 0.0005): # RSI > 55 (было 60), price_change_5_period < -0.0005 (было -0.001)
                    df.loc[idx, 'target'] = 1  # SELL

                # 3. Подтверждение по объему (слабый объем, но есть движение)
                elif (df['volume'].loc[idx] > df['volume'].rolling(20).quantile(0.6).loc[idx] and # Квантиль 0.6 (было 0.7)
                    ((price_change_5_period > 0.0005 and rsi < 50) or (price_change_5_period < -0.0005 and rsi > 50))): # RSI < 50 / > 50
                    df.loc[idx, 'target'] = 0 if price_change_5_period > 0 else 1

                # 4. Смена тренда: ADX растет, RSI отходит от 50
                elif (abs(rsi - 50) > 3 and adx > adx_prev and abs(adx - adx_prev) > 0.3): # abs(rsi-50) > 3 (было 5), abs(adx-adx_prev) > 0.3 (было 0.5)
                    df.loc[idx, 'target'] = 0 if rsi < 50 else 1 # Сделаем более агрессивным
# ...



3. Более агрессивная стратегия imblearn:
Мы изменим целевые проценты для SMOTE/Undersampler, чтобы добиться более сбалансированного результата.


В блоке "НОВЫЙ БЛОК: ИСПОЛЬЗОВАНИЕ IMBLEARN ДЛЯ БАЛАНСИРОВКИ КЛАССОВ" измените:
# В train_model.py, в функции prepare_xlstm_rl_data(...):
# ...
        print(f"Баланс классов ДО imblearn: {Counter(y_labels)}")

        # Целевое соотношение: 25% BUY, 25% SELL, 50% HOLD (было 20/20/60)
        total_samples = len(X)
        target_buy_count = int(total_samples * 0.25) # Цель 25% BUY
        target_sell_count = int(total_samples * 0.25) # Цель 25% SELL
        
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
            # Убедимся, что k_neighbors не меньше 1
            k_neighbors = max(1, k_neighbors)

            if any(count <= k_neighbors for count in [current_buy_count, current_sell_count] if count > 0):
                print("⚠️ Недостаточно сэмплов для SMOTE с k_neighbors, использую RandomOverSampler.")
                from imblearn.over_sampling import RandomOverSampler
                oversampler = RandomOverSampler(sampling_strategy=sampling_strategy_smote, random_state=42)
            else:
                oversampler = SMOTE(sampling_strategy=sampling_strategy_smote, random_state=42, k_neighbors=k_neighbors)

            X_temp, y_temp_labels = oversampler.fit_resample(X.reshape(len(X), -1), y_labels)
            print(f"Баланс классов после Oversampling: {Counter(y_temp_labels)}")
        else:
            X_temp, y_temp_labels = X.reshape(len(X), -1), y_labels
            print("Пропустил Oversampling, так как нет BUY/SELL сигналов.")

        # Undersampling HOLD: Цель - чтобы HOLD был примерно равен сумме BUY + SELL
        current_hold_count_after_oversample = Counter(y_temp_labels)[2]
        target_hold_count = min(current_hold_count_after_oversample, int((Counter(y_temp_labels)[0] + Counter(y_temp_labels)[1]) * 1.0)) # СНИЖЕНО с 1.5 до 1.0
        
        undersampler = RandomUnderSampler(sampling_strategy={2: target_hold_count}, random_state=42)
        X_resampled, y_resampled_labels = undersampler.fit_resample(X_temp, y_temp_labels)

        # ...



4. Добавление ATR в FEATURE_COLUMNS (файл: train_model.py):
Чтобы ATR_14 использовался моделью, его нужно добавить в список признаков.


Найдите список feature_cols в prepare_xlstm_rl_data.


Добавьте 'ATR_14' в этот список:
# В train_model.py, в функции prepare_xlstm_rl_data(...):
# ...
feature_cols = [
    # Технические индикаторы
    'RSI_14', 'MACD_12_26_9', 'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0', 
    'ADX_14', 'STOCHk_14_3_3', 'STOCHd_14_3_3',
    'ATR_14', # <--- ДОБАВЛЕНО
    # Паттерны
    'CDLHAMMER', 'CDLENGULFING', 'CDLDOJI', 'CDLSHOOTINGSTAR', 
    'CDLHANGINGMAN', 'CDLMARUBOZU',
# ...



5. Добавление ATR в TradingEnvRL (файл: trading_env.py):
Если ATR_14 является признаком для xLSTM, то TradingEnvRL также должен его использовать.


Найдите список self.feature_columns в методе reset класса TradingEnvRL.


Добавьте 'ATR_14' в этот список:
# В trading_env.py, в классе TradingEnvRL, в методе reset(...):
# ...
    self.feature_columns = [
        'RSI_14', 'MACD_12_26_9', 'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0', 
        'ADX_14', 'STOCHk_14_3_3', 'STOCHd_14_3_3',
        'ATR_14', # <--- ДОБАВЛЕНО
        'CDLHAMMER', 'CDLENGULFING', 'CDLDOJI', 'CDLSHOOTINGSTAR', 
        'CDLHANGINGMAN', 'CDLMARUBOZU',
# ...



6. Обновление FEATURE_COLUMNS в run_live_trading.py и advanced_simulation_engine.py:
Если ATR_14 используется, он также должен быть в списке FEATURE_COLUMNS для консистентности.


В run_live_trading.py и advanced_simulation_engine.py найдите список FEATURE_COLUMNS.


Добавьте 'ATR_14' в этот список:
# В run_live_trading.py и advanced_simulation_engine.py, в константе FEATURE_COLUMNS:
FEATURE_COLUMNS = [
    # Технические индикаторы
    'RSI_14', 'MACD_12_26_9', 'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0', 
    'ADX_14', 'STOCHk_14_3_3', 'STOCHd_14_3_3',
    'ATR_14', # <--- ДОБАВЛЕНО
# ...




Почему эти изменения должны помочь:

Максимальная генерация "кандидатов": Более агрессивные пороги для future_return и vsa_strength (вместе с динамическим порогом на основе ATR) создадут гораздо больше начальных BUY/SELL меток.
Улучшенная "умная" переклассификация: Увеличенный процент переклассификации (40%) в сочетании с еще более мягкими условиями (RSI < 45, > 55; volume quantile 0.6; ADX change 0.3) позволит "вытащить" еще больше потенциально значимых сигналов из класса HOLD.
Мощная балансировка imblearn: С увеличенным количеством "исходных" BUY/SELL примеров, imblearn сможет более эффективно создавать синтетические примеры и снижать количество HOLD, достигая целевых 25%/25%/50%.
ATR как признак: Включение ATR_14 в модель позволит ей учитывать волатильность напрямую при принятии решений.

Это довольно агрессивные изменения для балансировки, но они необходимы, когда данные настолько несбалансированы. После их применения, баланс классов должен значительно улучшиться.

Важные замечания:

После этих изменений, ОБЯЗАТЕЛЬНО переобучите модель с нуля.
Мониторьте производительность: Поскольку мы генерируем больше BUY/SELL сигналов, это может увеличить количество сделок в симуляции и, возможно, в реальной торговле.
Качество vs. Количество: Цель - не просто получить 25/25/50, а получить качественные сигналы в этих классах. Балансировка помогает модели учиться, но финальное качество будет зависеть от того, насколько хорошо эти сгенерированные сигналы отражают реальные возможности.

Теперь эти изменения должны дать вам гораздо более сбалансированное распределение классов.


========
Хорошо, я внимательно прочитал предложения от другой AI-модели и сравнил их с нашим текущим кодом и предыдущими рекомендациями. Я согласен, что проблема с балансом классов глубока, и нам нужны более системные решения, которые не просто "вытаскивают" сигналы случайно, а фокусируются на "смысловых сигналах в контексте".
Я выбрал три улучшения, которые хорошо вписываются в нашу текущую архитектуру, не требуют создания принципиально новых индикаторов (кроме комбинации существующих) и не противоречат предыдущим изменениям:

Семплирование по "событиям" (Event-Based Sampling): Мы сфокусируем обучение на более "интересных" барах, где есть потенциальные сигналы.
Усиление взвешивания классов: Дополнительно увеличим "важность" классов BUY/SELL для xLSTM.
Бонус за исследование в RL: Поощрим RL-агента за попытки принимать решения BUY/SELL, даже если они редки.

Вот как это реализовать в коде:

🚀 Новые инструкции по улучшению баланса BUY/SELL/HOLD
1. Файл: feature_engineering.py
Мы добавим новый признак is_event, который будет отмечать бары с высокой активностью (события).


Найдите функцию calculate_features.


В конце этой функции, перед return df_out, добавьте расчет is_event:
# В feature_engineering.py, в функции calculate_features(df: pd.DataFrame):
# ... (весь существующий код расчета индикаторов) ...

    # =====================================================================
    # НОВЫЙ БЛОК: СОЗДАНИЕ ПРИЗНАКА 'is_event' (для Event-Based Sampling)
    # =====================================================================
    # Определяем события, которые потенциально содержат сигналы
    # Используем существующие индикаторы и добавляем RSI/ATR/Volume/ADX изменения
    
    # Убедимся, что все нужные колонки существуют (ATR_14 уже добавлен)
    required_cols = ['volume', 'ATR_14', 'RSI_14', 'ADX_14']
    for col in required_cols:
        if col not in df_out.columns:
            df_out[col] = 0 # Заполняем нулями, если вдруг нет

    df_out['is_event'] = (
        (df_out['volume'] > df_out['volume'].rolling(50).quantile(0.9).fillna(0)) | # Объем > 90% квантиля
        (df_out['ATR_14'] > df_out['ATR_14'].rolling(50).quantile(0.9).fillna(0)) | # ATR > 90% квантиля
        (abs(df_out['RSI_14'] - 50) > 25) | # RSI выходит из зоны 25-75 (более экстремально)
        (df_out['ADX_14'] > df_out['ADX_14'].shift(5).fillna(0) + 2) # ADX растёт > 2 пункта за 5 баров
    ).astype(int)
    # =====================================================================
    # КОНЕЦ НОВОГО БЛОКА 'is_event'
    # =====================================================================

    return df_out

# ... (остальной код файла) ...



Важно: Убедитесь, что ATR_14 уже добавлен в feature_engineering.py как было предложено ранее.


2. Файл: train_model.py
Здесь мы используем is_event для фильтрации данных перед обучением, а также усилим взвешивание классов.


Найдите функцию prepare_xlstm_rl_data.


Внутри цикла for symbol in symbols:, после всех вызовов calculate_features, detect_candlestick_patterns, calculate_vsa_features, но перед созданием целевых меток, добавьте фильтрацию по is_event.


Добавьте 'is_event' в список feature_cols.


Усильте взвешивание классов (class_weight) для BUY/SELL.


Измените логику imblearn для oversampling'а только на "событийных" барах, если это возможно, или просто используйте его на отфильтрованных данных.
# В train_model.py, в функции prepare_xlstm_rl_data(data_path, sequence_length=10):
# ...
feature_cols = [
    # Технические индикаторы
    'RSI_14', 'MACD_12_26_9', 'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0', 
    'ADX_14', 'STOCHk_14_3_3', 'STOCHd_14_3_3',
    'ATR_14', # Уже добавлено
    # Паттерны
    'CDLHAMMER', 'CDLENGULFING', 'CDLDOJI', 'CDLSHOOTINGSTAR', 
    'CDLHANGINGMAN', 'CDLMARUBOZU',
    # VSA признаки (новые!)
    'vsa_no_demand', 'vsa_no_supply', 'vsa_stopping_volume', 
    'vsa_climactic_volume', 'vsa_test', 'vsa_effort_vs_result', 'vsa_strength',
    # Дополнительные рыночные данные
    'volume_ratio', 'spread_ratio', 'close_position',
    'is_event' # <--- ДОБАВЛЕНО: Новый признак
]
# ...
for symbol in symbols:
    df = full_df[full_df['symbol'] == symbol].copy()
    # ... (существующий код для calculate_features, detect_candlestick_patterns, calculate_vsa_features) ...

    # =====================================================================
    # НОВЫЙ БЛОК: ФИЛЬТРАЦИЯ ПО 'is_event' (Event-Based Sampling)
    # =====================================================================
    initial_rows = len(df)
    df_event_filtered = df[df['is_event'] == 1].copy()
    
    if len(df_event_filtered) < sequence_length + 50:
        print(f"⚠️ Для символа {symbol} недостаточно событий ({len(df_event_filtered)}), использую все данные.")
        df_processed = df.copy() # Если событий мало, используем все данные
    else:
        print(f"✅ Для символа {symbol} отфильтровано {len(df_event_filtered)} событий из {initial_rows} баров.")
        df_processed = df_event_filtered.copy()
    
    # Сброс индекса после фильтрации, чтобы избежать проблем
    df_processed.reset_index(drop=True, inplace=True)

    if len(df_processed) < sequence_length + 50: # Проверяем после фильтрации
        continue # Пропускаем символ, если данных все еще недостаточно
    
    df = df_processed # Теперь работаем с отфильтрованным DataFrame
    # =====================================================================
    # КОНЕЦ НОВОГО БЛОКА
    # =====================================================================

    # ... (остальной код функции prepare_xlstm_rl_data, включая создание buy_condition/sell_condition, переклассификацию, и блок imblearn) ...



Усиление взвешивания классов:

Найдите функцию train_xlstm_rl_system.
Внутри неё, перед вызовом xlstm_model.train, измените расчет class_weight_dict:

# В train_model.py, в функции train_xlstm_rl_system(...):
# ...
    # ДОБАВЬТЕ: Вычисление весов классов для борьбы с дисбалансом
    y_integers = np.argmax(y_train, axis=1) # Преобразуем one-hot в целые числа
    class_weights_array = compute_class_weight(
        'balanced',
        classes=np.unique(y_integers),
        y=y_integers
    )
    class_weight_dict = {i: class_weights_array[i] for i in range(len(class_weights_array))}

    # =====================================================================
    # НОВЫЙ БЛОК: ДОПОЛНИТЕЛЬНОЕ УСИЛЕНИЕ ВЗВЕШИВАНИЯ КЛАССОВ BUY/SELL
    # =====================================================================
    # Умножим веса BUY и SELL на дополнительный коэффициент
    # Это заставит модель еще больше "страдать" от ошибок на этих классах
    additional_weight_multiplier = 1.5 # Можно экспериментировать: 1.2, 1.5, 2.0
    if 0 in class_weight_dict: # BUY
        class_weight_dict[0] *= additional_weight_multiplier
    if 1 in class_weight_dict: # SELL
        class_weight_dict[1] *= additional_weight_multiplier
    # =====================================================================
    # КОНЕЦ НОВОГО БЛОКА
    # =====================================================================

    print(f"📊 Веса классов для обучения: {class_weight_dict}")

    # ... (остальной код функции train_xlstm_rl_system) ...



3. Файл: trading_env.py
Мы добавим бонус за исследование в систему наград RL-агента.


Найдите функцию _calculate_advanced_reward.


В конце этой функции, перед return total_reward, добавьте бонус за исследование:
# В trading_env.py, в функции _calculate_advanced_reward(...):
# ... (весь существующий код расчета base_reward, vsa_bonus, vsa_penalty и т.д.) ...

    # =====================================================================
    # НОВЫЙ БЛОК: БОНУС ЗА ИССЛЕДОВАНИЕ (Exploration Bonus)
    # =====================================================================
    exploration_bonus = 0
    # Небольшой бонус за попытку BUY/SELL, чтобы агент не застрял в HOLD
    if action in [0, 1]: # Если действие - BUY или SELL
        # Бонус уменьшается по мере увеличения числа попыток этого действия
        # Используем небольшой постоянный бонус, чтобы стимулировать попытки
        exploration_bonus = 0.5 # Небольшой, но постоянный бонус
    
    # Можно сделать более сложный бонус, зависящий от редкости действия
    # Например, если бы мы отслеживали, сколько раз каждое действие было выбрано
    # Но для простоты начнем с постоянного бонуса
    # =====================================================================
    # КОНЕЦ НОВОГО БЛОКА
    # =====================================================================

    total_reward = base_reward + vsa_bonus + vsa_penalty + speed_bonus + hold_penalty + exploration_bonus # <--- ДОБАВЛЕНО: exploration_bonus
    
    return total_reward




Почему эти изменения должны помочь:

Event-Based Sampling: Модель будет обучаться на меньшем, но более информативном наборе данных, который содержит "события". Это снижает шум от "скучных" HOLD-периодов и позволяет модели лучше фокусироваться на паттернах, предшествующих значимым движениям. Это должно улучшить качество сигналов BUY/SELL.
Усиленное взвешивание классов: Дополнительное увеличение весов для BUY/SELL заставит xLSTM модель уделять этим классам еще больше внимания во время обучения, даже если imblearn не смог идеально сбалансировать данные.
RL Exploration Bonus: Этот бонус напрямую стимулирует RL-агента пробовать действия BUY/SELL, даже если они кажутся рискованными или редкими. Это поможет агенту быстрее находить прибыльные стратегии для этих действий, а не просто застревать в HOLD.

После применения этих изменений, запустите обучение снова. Мы ожидаем увидеть значительное улучшение баланса классов, так как мы атакуем проблему с нескольких сторон: более "умная" разметка, фильтрация данных и усиление сигналов на уровне обучения.