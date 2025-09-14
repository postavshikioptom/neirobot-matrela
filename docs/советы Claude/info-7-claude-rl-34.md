
Инструкции по реализации:
Файл 1: train_model.py
1. Корректировка весов классов (сбалансировать BUY, SELL, HOLD):
Местоположение: В функции train_xlstm_rl_system, в блоке вычисления class_weight_dict (строка ~350).
ЗАМЕНИТЬ:
# СТАРЫЙ КОД (усилить BUY и HOLD, ослабить SELL)
    if 0 in class_weight_dict:
        class_weight_dict[0] *= 2.5  # ИЗМЕНЕНО: Значительно увеличиваем вес BUY (с 1.5 до 2.5)
    if 1 in class_weight_dict:
        class_weight_dict[1] *= 0.8  # ИЗМЕНЕНО: Значительно уменьшаем вес SELL (с 1.5 до 0.8)
    
    if 2 in class_weight_dict:
        class_weight_dict[2] *= 1.2  # ИЗМЕНЕНО: Увеличиваем вес HOLD (с 0.7 до 1.2)

НА НОВЫЙ КОД (более сбалансированные веса):
# НОВЫЙ КОД - Корректируем веса классов (более сбалансированные)
    if 0 in class_weight_dict:
        class_weight_dict[0] *= 2.0  # ИЗМЕНЕНО: Увеличиваем вес BUY (с 2.5 до 2.0 - немного меньше агрессии)
    if 1 in class_weight_dict:
        class_weight_dict[1] *= 1.2  # ИЗМЕНЕНО: Увеличиваем вес SELL (с 0.8 до 1.2 - чтобы не был совсем 0)
    
    if 2 in class_weight_dict:
        class_weight_dict[2] *= 1.0  # ИЗМЕНЕНО: Устанавливаем вес HOLD на 1.0 (с 1.2 - для баланса)

Объяснение: Мы пытаемся найти "золотую середину". BUY все еще имеет высокий вес, чтобы модель уделяла ему внимание. SELL получает немного больший вес, чтобы модель не игнорировала его полностью. HOLD получает нейтральный вес, чтобы модель не была слишком агрессивной в его подавлении.
2. Улучшение качества BUY-сигналов в логике генерации меток:
Мы сделаем условия для buy_condition более строгими, чтобы модель училась на более "чистых" бычьих сигналах, и ослабим sell_condition.
Местоположение: В функции prepare_xlstm_rl_data, в блоке "НОВЫЙ КОД - Условия для BUY/SELL только на основе future_return и классических индикаторов/паттернов (без VSA)" (строка ~150-200).
ЗАМЕНИТЬ:
# СТАРЫЙ КОД
        # Классические технические фильтры
        strong_trend = df['ADX_14'] > 25
        # high_volume = df['volume_ratio'] > 1.5 # <--- УДАЛЕНО: volume_ratio больше нет, так как calculate_vsa_features отключен
        
        rsi_buy_zone = df['RSI_14'] < 40
        rsi_sell_zone = df['RSI_14'] > 60
        
        macd_buy_signal = (df['MACD_12_26_9'] > df['MACD_signal']) & (df['MACD_12_26_9'].shift(1) <= df['MACD_signal'].shift(1))
        macd_sell_signal = (df['MACD_12_26_9'] < df['MACD_signal']) & (df['MACD_12_26_9'].shift(1) >= df['MACD_signal'].shift(1))

        # Условия для BUY/SELL только на основе future_return и классических индикаторов/паттернов
        buy_condition = (
            (df['future_return'] > df['dynamic_threshold']) &
            (strong_trend | rsi_buy_zone | macd_buy_signal) # ИЗМЕНЕНО: Убрано high_volume
        )

        sell_condition = (
            (df['future_return'] < -df['dynamic_threshold']) &
            (strong_trend | rsi_sell_zone | macd_sell_signal) # ИЗМЕНЕНО: Убрано high_volume
        )

НА НОВЫЙ КОД (более строгие BUY, менее строгие SELL):
# НОВЫЙ КОД - Условия для BUY/SELL (более строгие BUY, менее строгие SELL)
        # Увеличиваем пороги для генерации торговых сигналов
        df['base_threshold'] = 0.008
        df['dynamic_threshold'] = np.maximum(
            df['base_threshold'],
            (df['ATR_14'] / df['close'] * 1.5).fillna(0.008)
        )

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

Объяснение:

Для BUY: Мы делаем условия гораздо более строгими (dynamic_threshold * 1.2 и strong_trend & rsi_buy_zone & macd_buy_signal). Это должно привести к меньшему количеству BUY-сигналов, но более высокому их качеству, что, в свою очередь, должно помочь модели лучше их изучить.
Для SELL: Мы делаем условия менее строгими (dynamic_threshold * 0.8 и strong_trend | rsi_sell_zone | macd_sell_signal). Это должно увеличить количество SELL-сигналов в тренировочном наборе, чтобы модель не игнорировала этот класс.

Файл 3: trading_env.py
1. Корректировка _calculate_advanced_reward для борьбы с дисбалансом:
Мы должны убедиться, что функция наград для RL-агента также отражает наши новые приоритеты (усилить BUY, ослабить SELL).
Местоположение: Внутри метода _calculate_advanced_reward (строка ~130).
ЗАМЕНИТЬ:
# СТАРЫЙ КОД
            buy_signal_strength = (
                (current_row.get('RSI_14', 50) < 40) +
                (current_row.get('ADX_14', 0) > 20) +
                (current_row.get('MACD_hist', 0) > 0.001) # ИЗМЕНЕНО: MACD_hist должен быть значительно положительным
            )
            sell_signal_strength = (
                (current_row.get('RSI_14', 50) > 60) +
                (current_row.get('ADX_14', 0) > 20) +
                (current_row.get('MACD_hist', 0) < -0.001) # ИЗМЕНЕНО: MACD_hist должен быть значительно отрицательным
            )

            # Увеличиваем штраф за слабые BUY-сигналы, если RL предсказывает BUY
            if action == 1 and buy_signal_strength < 1:
                overtrading_penalty = -2.0 # ИЗМЕНЕНО: Увеличен штраф (с -1.0 до -2.0)
            # Уменьшаем штраф за слабые SELL-сигналы, если RL предсказывает SELL
            elif action == 0 and sell_signal_strength < 1:
                overtrading_penalty = -0.5 # ИЗМЕНЕНО: Уменьшен штраф (с -1.0 до -0.5)

НА НОВЫЙ КОД (более агрессивное усиление BUY, ослабление SELL):
# НОВЫЙ КОД - Корректируем функцию наград для RL (более агрессивное усиление BUY, ослабление SELL)
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

Объяснение: Мы делаем условия для "явного сигнала" в RL-среде более строгими для BUY (требуем 2+ сильных сигнала) и менее строгими для SELL (требуем 1+ сильный сигнал), а также корректируем соответствующие штрафы. Это должно еще сильнее подтолкнуть RL-агента к поиску BUY-сигналов и уменьшить его склонность к SELL.

