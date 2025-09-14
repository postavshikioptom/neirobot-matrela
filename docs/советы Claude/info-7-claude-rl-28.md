Хорошо, Максимально подробный план по полному удалению всех упоминаний VSA из вашего кода. Это устранит все KeyError и позволит модели работать только на индикаторах и паттернах.
Важно: Перед началом работы обязательно сделайте резервную копию всех файлов!

Файл 1: feature_engineering.py
Этот файл содержит определение VSA-признаков. Мы полностью удалим этот модуль.


Полное удаление функции calculate_vsa_features и всех вспомогательных функций VSA:

Удалить весь блок кода, начинающийся с # === VSA ANALYSIS MODULE === и заканчивающийся функцией calculate_advanced_vsa_features.
Удалить также функцию calculate_advanced_vsa_features в конце файла.



Удаление VSA-признаков из prepare_xlstm_rl_features:

Местоположение: Функция prepare_xlstm_rl_features, список xlstm_rl_features (строка ~224).
ЗАМЕНИТЬ: (удалить эти строки)
# СТАРЫЙ КОД (удалить эти строки)
        # VSA сигналы
        'vsa_no_demand', 'vsa_no_supply', 'vsa_stopping_volume',
        'vsa_climactic_volume', 'vsa_test', 'vsa_effort_vs_result', 'vsa_strength',
        # Дополнительные рыночные данные
        'volume_ratio', 'spread_ratio', 'close_position'


НА НОВЫЙ КОД: (просто удали эти строки, чтобы их не было в списке)



Удаление импорта calculate_vsa_features:

Местоположение: В начале файла, в импортах.
Удалить строку:
# СТАРЫЙ КОД
from .xlstm_memory_cell import XLSTMLayer  # Импортируем настоящий xLSTM
from sklearn.utils.class_weight import compute_class_weight


НА НОВЫЙ КОД:
# НОВЫЙ КОД
# (Ничего не менять, убедиться, что нет импорта calculate_vsa_features)






Файл 2: train_model.py
Этот файл готовит данные и обучает модель.


Удаление VSA-признаков из feature_cols:

Местоположение: Функция prepare_xlstm_rl_data, список feature_cols (строка ~70).
ЗАМЕНИТЬ: (удалить эти строки)
# СТАРЫЙ КОД (удалить эти строки)
        # VSA признаки (новые!)
        'vsa_no_demand', 'vsa_no_supply', 'vsa_stopping_volume',
        'vsa_climactic_volume', 'vsa_test', 'vsa_effort_vs_result', 'vsa_strength',
        # Дополнительные рыночные данные
        'volume_ratio', 'spread_ratio', 'close_position',


НА НОВЫЙ КОД: (просто удали эти строки, чтобы их не было в списке)



Удаление VSA-зависимой логики из генерации целевых меток:

Местоположение: В функции prepare_xlstm_rl_data, в блоке, где определяются buy_condition и sell_condition (строка ~150-200).
ЗАМЕНИТЬ ВЕСЬ ЭТОТ БЛОК:
# СТАРЫЙ КОД (весь блок "НОВЫЙ КОД - Условия для BUY/SELL только на основе future_return и классических индикаторов/паттернов")
        # Увеличиваем пороги для генерации торговых сигналов
        df['base_threshold'] = 0.008  # Минимальный порог 0.8%
        df['dynamic_threshold'] = np.maximum(
            df['base_threshold'],
            (df['ATR_14'] / df['close'] * 1.5).fillna(0.008) # ИЗМЕНЕНО: Увеличиваем множитель ATR
        )

        # Классические технические фильтры
        strong_trend = df['ADX_14'] > 25
        high_volume = df['volume_ratio'] > 1.5 # <--- ЭТОТ ПРИЗНАК БОЛЬШЕ НЕТ
        # RSI экстремумы, но менее строгие, так как нет VSA для подтверждения
        rsi_buy_zone = df['RSI_14'] < 40 # ИЗМЕНЕНО: менее экстремально
        rsi_sell_zone = df['RSI_14'] > 60 # ИЗМЕНЕНО: менее экстремально
        
        # MACD пересечения
        macd_buy_signal = (df['MACD_12_26_9'] > df['MACD_signal']) & (df['MACD_12_26_9'].shift(1) <= df['MACD_signal'].shift(1))
        macd_sell_signal = (df['MACD_12_26_9'] < df['MACD_signal']) & (df['MACD_12_26_9'].shift(1) >= df['MACD_signal'].shift(1))

        # Условия для BUY/SELL только на основе future_return и классических индикаторов/паттернов
        buy_condition = (
            (df['future_return'] > df['dynamic_threshold']) &
            (strong_trend | high_volume | rsi_buy_zone | macd_buy_signal) # ИЗМЕНЕНО: Использование классических индикаторов
        )

        sell_condition = (
            (df['future_return'] < -df['dynamic_threshold']) &
            (strong_trend | high_volume | rsi_sell_zone | macd_sell_signal) # ИЗМЕНЕНО: Использование классических индикаторов
        )


НА НОВЫЙ КОД:
# НОВЫЙ КОД - Условия для BUY/SELL только на основе future_return и классических индикаторов/паттернов (без VSA)
        # Увеличиваем пороги для генерации торговых сигналов
        df['base_threshold'] = 0.008  # Минимальный порог 0.8%
        df['dynamic_threshold'] = np.maximum(
            df['base_threshold'],
            (df['ATR_14'] / df['close'] * 1.5).fillna(0.008)
        )

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





Удаление VSA-зависимой логики из переклассификации HOLD:

Местоположение: В функции prepare_xlstm_rl_data, в блоке, где происходит переклассификация HOLD (строка ~220-280).
ЗАМЕНИТЬ ВЕСЬ ЭТОТ БЛОК:
# СТАРЫЙ КОД (весь блок "НОВЫЙ КОД - Условия для переклассификации HOLD без VSA")
        # Если HOLD все еще сильно преобладает, пытаемся переклассифицировать
        if current_hold_count > (current_buy_count + current_sell_count) * 2.5: # ИЗМЕНЕНО: Более агрессивный порог
            print(f"⚠️ Сильный дисбаланс классов. Попытка УМНОЙ переклассификации части HOLD-сигналов (без VSA).")
            hold_indices = df[df['target'] == 2].index
            
            import random
            random.seed(42)
            
            reclassify_count = int(current_hold_count * 0.20) # Переклассифицируем 20% HOLD
            if reclassify_count > 0:
                reclassify_indices = random.sample(list(hold_indices), min(reclassify_count, len(hold_indices)))
                
                for idx in reclassify_indices:
                    if idx < 5: continue # Нужна история для расчетов
                    
                    rsi = df.loc[idx, 'RSI_14']
                    adx = df.loc[idx, 'ADX_14']
                    macd_hist = df.loc[idx, 'MACD_hist']
                    price_change_3_period = df['close'].pct_change(3).loc[idx] # 3-периодное изменение
                    volume_ratio = df.loc[idx, 'volume_ratio'] # <--- ЭТОТ ПРИЗНАК БОЛЬШЕ НЕТ

                    # Условия для переклассификации (без VSA)
                    # 1. RSI + ADX + объем
                    if (rsi < 40 and adx > 20 and volume_ratio > 1.2 and price_change_3_period > 0.003): # ИЗМЕНЕНО: Более строгие условия
                        df.loc[idx, 'target'] = 0  # BUY
                    elif (rsi > 60 and adx > 20 and volume_ratio > 1.2 and price_change_3_period < -0.003): # ИЗМЕНЕНО: Более строгие условия
                        df.loc[idx, 'target'] = 1  # SELL
                        
                    # 2. MACD гистограмма + объем
                    elif (macd_hist > 0.001 and volume_ratio > 1.3 and price_change_3_period > 0.002): # ИЗМЕНЕНО: Более строгие условия
                        df.loc[idx, 'target'] = 0  # BUY
                    elif (macd_hist < -0.001 and volume_ratio > 1.3 and price_change_3_period < -0.002): # ИЗМЕНЕНО: Более строгие условия
                        df.loc[idx, 'target'] = 1  # SELL
                        
                    # 3. Сильный тренд по ADX + движение цены
                    elif (adx > 30 and abs(price_change_3_period) > 0.005): # ИЗМЕНЕНО: Более строгие условия
                        df.loc[idx, 'target'] = 0 if price_change_3_period > 0 else 1
            
            print(f"Баланс классов после УМНОЙ переклассификации (без VSA):")
            unique, counts = np.unique(df['target'], return_counts=True)
            class_names = ['BUY', 'SELL', 'HOLD']
            for class_idx, count in zip(unique, counts):
                print(f"  {class_names[class_idx]}: {count} ({count/len(df)*100:.1f}%)")
        else:
            print(f"✅ Баланс классов приемлемый, переклассификация HOLD не требуется.")


НА НОВЫЙ КОД:
# НОВЫЙ КОД - Условия для переклассификации HOLD без VSA
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






Файл 3: hybrid_decision_maker.py
Этот файл содержит логику принятия гибридных решений.


Удаление VSA-признаков из feature_columns:

Местоположение: В конструкторе __init__, где определяется self.feature_columns.
Удалить строки:
# СТАРЫЙ КОД (удалить эти строки)
        # VSA признаки (новые!)
        'vsa_no_demand', 'vsa_no_supply', 'vsa_stopping_volume',
        'vsa_climactic_volume', 'vsa_test', 'vsa_effort_vs_result', 'vsa_strength',
        # Дополнительные рыночные данные
        'volume_ratio', 'spread_ratio', 'close_position',


НА НОВЫЙ КОД: (просто удали эти строки, чтобы их не было в списке)



Удаление VSA-зависимой логики из _analyze_vsa_context:

Местоположение: Метод _analyze_vsa_context.
ЗАМЕНИТЬ ВЕСЬ ЭТОТ МЕТОД:
# СТАРЫЙ КОД (весь метод _analyze_vsa_context)
    def _analyze_vsa_context(self, row):
        """Анализирует VSA контекст для улучшения решений"""
        vsa_signals = {
            'bullish_strength': 0,
            'bearish_strength': 0,
            'uncertainty': 0,
            'volume_confirmation': False
        }
        
        # Бычьи VSA сигналы
        if row['vsa_no_supply'] == 1:
            vsa_signals['bullish_strength'] += 2
        if row['vsa_stopping_volume'] == 1:
            vsa_signals['bullish_strength'] += 3
        if row['vsa_strength'] > 1:
            vsa_signals['bullish_strength'] += 1
            
        # Медвежьи VSA сигналы  
        if row['vsa_no_demand'] == 1:
            vsa_signals['bearish_strength'] += 2
        if row['vsa_climactic_volume'] == 1:
            vsa_signals['bearish_strength'] += 3
        if row['vsa_strength'] < -1:
            vsa_signals['bearish_strength'] += 1
            
        # Неопределенность
        if row['vsa_test'] == 1:
            vsa_signals['uncertainty'] += 2
        if row['vsa_effort_vs_result'] == 1:
            vsa_signals['uncertainty'] += 1
            
        # Подтверждение объемом
        if row['volume_ratio'] > 1.5:
            vsa_signals['volume_confirmation'] = True
            
        return vsa_signals


НА НОВЫЙ КОД (пустой заглушка):
# НОВЫЙ КОД - Заглушка для _analyze_vsa_context
    def _analyze_vsa_context(self, row):
        """VSA отключен, возвращаем пустые сигналы."""
        return {
            'bullish_strength': 0,
            'bearish_strength': 0,
            'uncertainty': 0,
            'volume_confirmation': False
        }





Удаление VSA-зависимой логики из _create_rl_observation:

Местоположение: Метод _create_rl_observation.
ЗАМЕНИТЬ:
# СТАРЫЙ КОД
    def _create_rl_observation(self, xlstm_prediction, latest_row):
        """Создает наблюдение для RL агента"""
        vsa_features = np.array([
            latest_row['vsa_no_demand'],
            latest_row['vsa_no_supply'], 
            latest_row['vsa_stopping_volume'],
            latest_row['vsa_climactic_volume'],
            latest_row['vsa_test'],
            latest_row['vsa_effort_vs_result'],
            latest_row['vsa_strength']
        ])
        
        portfolio_state = np.array([
            self.current_balance / 10000,  # Нормализованный баланс
            self.current_position,  # -1, 0, 1
            0,  # Нереализованный PnL (упрощенно)
            self.steps_in_position / 100.0
        ])
        
        return np.concatenate([xlstm_prediction, vsa_features, portfolio_state])


НА НОВЫЙ КОД:
# НОВЫЙ КОД - _create_rl_observation без VSA
    def _create_rl_observation(self, xlstm_prediction, latest_row):
        """Создает наблюдение для RL агента (без VSA)"""
        # VSA признаки удалены, поэтому размер наблюдения уменьшится.
        # Убедитесь, что TradingEnvRL также отражает это изменение.
        
        portfolio_state = np.array([
            self.current_balance / 10000,  # Нормализованный баланс
            self.current_position,  # -1, 0, 1
            0,  # Нереализованный PnL (упрощенно)
            self.steps_in_position / 100.0
        ])
        
        return np.concatenate([xlstm_prediction, portfolio_state]) # ИЗМЕНЕНО: Без vsa_features





Удаление VSA-зависимой логики из _make_final_decision:

Местоположение: Метод _make_final_decision, в блоке if xlstm_decision == 'BUY': и elif xlstm_decision == 'SELL':, а также в блоке if vsa_signals['uncertainty'] >= 3:.
ЗАМЕНИТЬ:
# СТАРЫЙ КОД
        # При разногласиях используем VSA для принятия решения
        if xlstm_decision == 'BUY':
            if vsa_signals['bullish_strength'] >= 2 and vsa_signals['volume_confirmation']:
                print("VSA подтверждает покупку")
                return 'BUY'
            elif vsa_signals['bearish_strength'] >= 2:
                print("VSA противоречит покупке")
                return 'HOLD'
                
        elif xlstm_decision == 'SELL':
            if vsa_signals['bearish_strength'] >= 2 and vsa_signals['volume_confirmation']:
                print("VSA подтверждает продажу")
                return 'SELL'
            elif vsa_signals['bullish_strength'] >= 2:
                print("VSA противоречит продаже")
                return 'HOLD'
        
        # Если слишком много неопределенности, держим HOLD
        if vsa_signals['uncertainty'] >= 3:
            print("Высокая неопределенность VSA, решение: HOLD")
            return 'HOLD'
        
        # По умолчанию возвращаем RL решение
        print(f"Финальное решение по RL: {rl_decision}")
        return rl_decision


НА НОВЫЙ КОД:
# НОВЫЙ КОД - _make_final_decision без VSA
        # Если xLSTM и RL согласны, принимаем решение
        if xlstm_decision == rl_decision:
            print(f"xLSTM и RL согласны: {xlstm_decision}")
            return xlstm_decision
        
        # Если не согласны, и xLSTM уверенность высокая, доверяем xLSTM
        if xlstm_conf >= threshold + 0.1: # ИЗМЕНЕНО: Добавляем небольшой запас
            print(f"xLSTM уверенность ({xlstm_conf:.3f}) выше RL, решение: {xlstm_decision}")
            return xlstm_decision
        
        # В противном случае, по умолчанию возвращаем RL решение (или HOLD, если RL не уверен)
        print(f"xLSTM и RL не согласны, доверяем RL: {rl_decision}")
        return rl_decision





Удаление VSA-зависимых данных из get_decision_explanation:

Местоположение: Метод get_decision_explanation, в блоке VSA сигналы:.
ЗАМЕНИТЬ:
# СТАРЫЙ КОД
        VSA сигналы:
        - Бычья сила: {last_decision['vsa_signals']['bullish_strength']}
        - Медвежья сила: {last_decision['vsa_signals']['bearish_strength']}
        - Неопределенность: {last_decision['vsa_signals']['uncertainty']}
        - Подтверждение объемом: {last_decision['vsa_signals']['volume_confirmation']}


НА НОВЫЙ КОД:
# НОВЫЙ КОД - Объяснение без VSA
        # VSA сигналы отключены
        - VSA сигналы: Отключены






Файл 4: market_regime_detector.py
Этот файл определяет рыночные режимы.


Удаление VSA-режимных признаков из extract_regime_features:

Местоположение: Функция extract_regime_features, где определяются vsa_activity и vsa_direction.
Удалить эти строки:
# СТАРЫЙ КОД (удалить эти строки)
        # VSA режимные признаки
        df['vsa_activity'] = df['vsa_strength'].rolling(10).std()
        df['vsa_direction'] = df['vsa_strength'].rolling(10).mean()


НА НОВЫЙ КОД: (просто удали эти строки)



Удаление VSA-признаков из списка regime_features:

Местоположение: В функции extract_regime_features, список regime_features.
ЗАМЕНИТЬ:
# СТАРЫЙ КОД
        regime_features = [
            'volatility', 'trend_strength', 'volume_trend', 'volume_volatility',
            'rsi_regime', 'bb_position', 'vsa_activity', 'vsa_direction'
        ]


НА НОВЫЙ КОД:
# НОВЫЙ КОД - regime_features без VSA
        regime_features = [
            'volatility', 'trend_strength', 'volume_trend', 'volume_volatility',
            'rsi_regime', 'bb_position' # ИЗМЕНЕНО: Удалено vsa_activity, vsa_direction
        ]





Удаление VSA-признаков из списка features_to_scale в fit:

Местоположение: Функция fit, список features_to_scale.
ЗАМЕНИТЬ:
# СТАРЫЙ КОД
        features_to_scale = [
            'volatility', 'trend_strength', 'volume_trend', 'volume_volatility',
            'rsi_regime', 'bb_position', 'vsa_activity', 'vsa_direction'
        ]


НА НОВЫЙ КОД:
# НОВЫЙ КОД - features_to_scale без VSA
        features_to_scale = [
            'volatility', 'trend_strength', 'volume_trend', 'volume_volatility',
            'rsi_regime', 'bb_position' # ИЗМЕНЕНО: Удалено vsa_activity, vsa_direction
        ]





Удаление VSA-признаков из списка features_to_predict в predict_regime:

Местоположение: Функция predict_regime, список features_to_predict.
ЗАМЕНИТЬ:
# СТАРЫЙ КОД
        features_to_predict = [
            'volatility', 'trend_strength', 'volume_trend', 'volume_volatility',
            'rsi_regime', 'bb_position', 'vsa_activity', 'vsa_direction'
        ]


НА НОВЫЙ КОД:
# НОВЫЙ КОД - features_to_predict без VSA
        features_to_predict = [
            'volatility', 'trend_strength', 'volume_trend', 'volume_volatility',
            'rsi_regime', 'bb_position' # ИЗМЕНЕНО: Удалено vsa_activity, vsa_direction
        ]






**Файл 5: Основной скрипт (run_trading_loop() или ваш главный файл) **
Этот файл управляет торговым циклом.


Удаление VSA-признаков из FEATURE_COLUMNS:

Местоположение: Глобальная константа FEATURE_COLUMNS (строка ~30).
ЗАМЕНИТЬ: (удалить эти строки)
# СТАРЫЙ КОД (удалить эти строки)
        # VSA признаки
        'vsa_no_demand', 'vsa_no_supply', 'vsa_stopping_volume',
        'vsa_climactic_volume', 'vsa_test', 'vsa_effort_vs_result', 'vsa_strength',
        # Дополнительные рыночные данные
        'volume_ratio', 'spread_ratio', 'close_position'


НА НОВЫЙ КОД: (просто удали эти строки, чтобы их не было в списке)



Удаление VSA-зависимой логики из manage_active_positions:

Местоположение: Функция manage_active_positions, блок "НОВАЯ ОБРАБОТКА С VSA" (строка ~90).
ЗАМЕНИТЬ:
# СТАРЫЙ КОД
        features_df = feature_engineering.calculate_features(kline_df.copy())
        features_df = feature_engineering.detect_candlestick_patterns(features_df)
        features_df = feature_engineering.calculate_vsa_features(features_df)  # Добавляем VSA!


НА НОВЫЙ КОД:
# НОВЫЙ КОД - Обработка без VSA
        features_df = feature_engineering.calculate_features(kline_df.copy())
        features_df = feature_engineering.detect_candlestick_patterns(features_df)
        # features_df = feature_engineering.calculate_vsa_features(features_df)  # <--- ЗАКОММЕНТИРОВАНО


Местоположение: Функция manage_active_positions, блок "VSA сигнал на закрытие" (строка ~120).
ЗАМЕНИТЬ:
# СТАРЫЙ КОД
        # 3. VSA сигнал на закрытие (новая логика!)
        elif should_close_by_vsa(features_df.iloc[-1], pos['side']):
            should_close = True
            close_reason = "VSA_SIGNAL"


НА НОВЫЙ КОД:
# НОВЫЙ КОД - Удаляем VSA сигнал на закрытие
        # 3. VSA сигнал на закрытие (отключен)
        # elif should_close_by_vsa(features_df.iloc[-1], pos['side']): # <--- ЗАКОММЕНТИРОВАНО
        #     should_close = True
        #     close_reason = "VSA_SIGNAL"





Удаление функции should_close_by_vsa:

Местоположение: Функция should_close_by_vsa (строка ~160).
Удалить весь этот метод.



Удаление VSA-зависимой логики из process_new_signal:

Местоположение: Функция process_new_signal, блок "ПОЛНАЯ ОБРАБОТКА С VSA" (строка ~210).
ЗАМЕНИТЬ:
# СТАРЫЙ КОД
        features_df = feature_engineering.calculate_features(kline_df.copy())
        features_df = feature_engineering.detect_candlestick_patterns(features_df)
        features_df = feature_engineering.calculate_vsa_features(features_df)


НА НОВЫЙ КОД:
# НОВЫЙ КОД - Обработка без VSA
        features_df = feature_engineering.calculate_features(kline_df.copy())
        features_df = feature_engineering.detect_candlestick_patterns(features_df)
        # features_df = feature_engineering.calculate_vsa_features(features_df) # <--- ЗАКОММЕНТИРОВАНО


Местоположение: Функция process_new_signal, блок "Дополнительная проверка VSA подтверждения" (строка ~230).
ЗАМЕНИТЬ:
# СТАРЫЙ КОД
    if validate_decision_with_vsa(features_df.iloc[-1], decision):
        open_result = trade_manager.open_market_position(session, decision, symbol)
        
        if open_result.get('status') == 'SUCCESS':
            performance_monitor.log_trade_opened(symbol, decision, vsa_confirmed=True)
            # Логируем с полной информацией
            notification_system.send_trade_alert(symbol, "OPEN", open_result['price'], reason=f"VSA_CONFIRMED_{decision}")
            trade_logger.log_enhanced_trade_with_quality_metrics(symbol, 'OPEN', open_result, None, 0,
                             decision_maker, features_df.iloc[-1], f"VSA_CONFIRMED_{decision}")
            
            # Сохраняем позицию
            active_positions = load_active_positions()
            active_positions[symbol] = {
                'side': decision,
                'entry_price': open_result['price'],
                'quantity': open_result['quantity'],
                'timestamp': time.time(),
                'duration': 0,
                'vsa_entry_strength': features_df.iloc[-1]['vsa_strength']  # Сохраняем VSA силу входа
            }
            save_active_positions(active_positions)
            
            opened_trades_counter += 1
            print(f"✅ Сделка #{opened_trades_counter}/{OPEN_TRADE_LIMIT} открыта с VSA подтверждением.")
            
            if opened_trades_counter >= OPEN_TRADE_LIMIT:
                print("!!! ДОСТИГНУТ ЛИМИТ ОТКРЫТЫХ СДЕЛОК !!!")
                set_trader_status('MANAGING_ONLY')
    else:
        print(f"❌ VSA не подтверждает решение {decision} для {symbol}")


НА НОВЫЙ КОД:
# НОВЫЙ КОД - Открытие сделки без VSA подтверждения
    # Дополнительная проверка VSA подтверждения (отключена)
    # if validate_decision_with_vsa(features_df.iloc[-1], decision): # <--- ЗАКОММЕНТИРОВАНО
    open_result = trade_manager.open_market_position(session, decision, symbol)
    
    if open_result.get('status') == 'SUCCESS':
        performance_monitor.log_trade_opened(symbol, decision, vsa_confirmed=False) # ИЗМЕНЕНО: vsa_confirmed=False
        # Логируем с полной информацией
        notification_system.send_trade_alert(symbol, "OPEN", open_result['price'], reason=f"MODEL_DECISION_{decision}") # ИЗМЕНЕНО: Причина
        trade_logger.log_enhanced_trade_with_quality_metrics(symbol, 'OPEN', open_result, None, 0,
                         decision_maker, features_df.iloc[-1], f"MODEL_DECISION_{decision}") # ИЗМЕНЕНО: Причина
        
        # Сохраняем позицию
        active_positions = load_active_positions()
        active_positions[symbol] = {
            'side': decision,
            'entry_price': open_result['price'],
            'quantity': open_result['quantity'],
            'timestamp': time.time(),
            'duration': 0,
            # 'vsa_entry_strength': features_df.iloc[-1]['vsa_strength']  # <--- УДАЛЕНО: VSA сила входа
        }
        save_active_positions(active_positions)
        
        opened_trades_counter += 1
        print(f"✅ Сделка #{opened_trades_counter}/{OPEN_TRADE_LIMIT} открыта на основе решения модели.") # ИЗМЕНЕНО: Сообщение
        
        if opened_trades_counter >= OPEN_TRADE_LIMIT:
            print("!!! ДОСТИГНУТ ЛИМИТ ОТКРЫТЫХ СДЕЛОК !!!")
            set_trader_status('MANAGING_ONLY')
    # else: # <--- УДАЛЕНО: Блок else для VSA подтверждения
    #     print(f"❌ VSA не подтверждает решение {decision} для {symbol}")





Удаление функции validate_decision_with_vsa:

Местоположение: Функция validate_decision_with_vsa (строка ~270).
Удалить весь этот метод.



Удаление VSA-зависимой логики из calculate_dynamic_stops:

Местоположение: Функция calculate_dynamic_stops (строка ~370).
ЗАМЕНИТЬ ВЕСЬ ЭТОТ МЕТОД:
# СТАРЫЙ КОД (весь метод calculate_dynamic_stops)
def calculate_dynamic_stops(features_row, position_side, entry_price):
    """
    Вычисляет динамические стоп-лоссы на основе VSA и волатильности
    """
    base_sl = STOP_LOSS_PCT  # -1.0%
    base_tp = TAKE_PROFIT_PCT  # 1.5%
    
    # Корректировка на основе VSA силы
    vsa_strength = features_row.get('vsa_strength', 0)
    volume_ratio = features_row.get('volume_ratio', 1)
    
    if position_side == 'BUY':
        # Для лонгов: сильные бычьи VSA = более широкие стопы (больше веры в движение)
        if vsa_strength > 2 and volume_ratio > 1.5:
            dynamic_sl = base_sl * 0.7  # Уменьшаем SL до -0.7%
            dynamic_tp = base_tp * 1.3  # Увеличиваем TP до 1.95%
        elif vsa_strength < -1:  # Слабые сигналы = тайтовые стопы
            dynamic_sl = base_sl * 1.5  # Увеличиваем SL до -1.5%
            dynamic_tp = base_tp * 0.8  # Уменьшаем TP до 1.2%
        else:
            dynamic_sl, dynamic_tp = base_sl, base_tp
            
    else:  # SELL
        if vsa_strength < -2 and volume_ratio > 1.5:
            dynamic_sl = base_sl * 0.7  # Более широкие стопы для сильных медвежьих сигналов
            dynamic_tp = base_tp * 1.3
        elif vsa_strength > 1:
            dynamic_sl = base_sl * 1.5  # Тайтовые стопы при слабых сигналах
            dynamic_tp = base_tp * 0.8
        else:
            dynamic_sl, dynamic_tp = base_sl, base_tp
    
    return dynamic_sl, dynamic_tp


НА НОВЫЙ КОД (только базовые стопы):
# НОВЫЙ КОД - Динамические стопы без VSA
def calculate_dynamic_stops(features_row, position_side, entry_price):
    """
    Вычисляет динамические стоп-лоссы на основе волатильности (без VSA)
    """
    base_sl = STOP_LOSS_PCT  # -1.0%
    base_tp = TAKE_PROFIT_PCT  # 1.5%
    
    # Можно добавить адаптацию на основе ATR или ADX, если нужно.
    # Например, если ATR высокий, можно немного расширить стопы.
    # Для простоты пока оставим базовые.
    
    return base_sl, base_tp






Файл 6: trade_logger.py
Этот файл логирует сделки.


Удаление VSA-признаков из FIELDNAMES:

Местоположение: Глобальная константа FIELDNAMES (строка ~15).
ЗАМЕНИТЬ: (удалить эти строки)
# СТАРЫЙ КОД (удалить эти строки)
    # --- VSA сигналы ---
    'vsa_no_demand', 'vsa_no_supply', 'vsa_stopping_volume',
    'vsa_climactic_volume', 'vsa_test', 'vsa_effort_vs_result', 'vsa_strength',


НА НОВЫЙ КОД: (просто удали эти строки, чтобы их не было в списке)



Удаление VSA-зависимой логики из calculate_signal_quality:

Местоположение: Функция calculate_signal_quality.
ЗАМЕНИТЬ ВЕСЬ ЭТОТ МЕТОД:
# СТАРЫЙ КОД (весь метод calculate_signal_quality)
def calculate_signal_quality(log_data):
    """
    Вычисляет метрики качества торгового сигнала
    """
    quality_metrics = {}
    
    # VSA качество (0-100)
    vsa_signals_count = sum([
        log_data.get('vsa_no_demand', 0),
        log_data.get('vsa_no_supply', 0),
        log_data.get('vsa_stopping_volume', 0),
        log_data.get('vsa_climactic_volume', 0)
    ])
    vsa_strength = abs(log_data.get('vsa_strength', 0))
    quality_metrics['vsa_quality'] = min(100, (vsa_signals_count * 25) + (vsa_strength * 10))
    
    # xLSTM уверенность качество
    xlstm_confidence = log_data.get('xlstm_confidence', 0)
    quality_metrics['xlstm_quality'] = xlstm_confidence * 100
    
    # Согласованность моделей
    xlstm_decision = log_data.get('final_decision', 'HOLD')
    rl_decision = log_data.get('rl_decision', 'HOLD')
    quality_metrics['model_consensus'] = 100 if xlstm_decision == rl_decision else 50
    
    # Общее качество сигнала
    quality_metrics['overall_signal_quality'] = (
        quality_metrics['vsa_quality'] * 0.4 +
        quality_metrics['xlstm_quality'] * 0.4 +
        quality_metrics['model_consensus'] * 0.2
    )
    
    return quality_metrics


НА НОВЫЙ КОД (без VSA):
# НОВЫЙ КОД - calculate_signal_quality без VSA
def calculate_signal_quality(log_data):
    """
    Вычисляет метрики качества торгового сигнала (без VSA)
    """
    quality_metrics = {}
    
    # xLSTM уверенность качество
    xlstm_confidence = log_data.get('xlstm_confidence', 0)
    quality_metrics['xlstm_quality'] = xlstm_confidence * 100
    
    # Согласованность моделей
    xlstm_decision = log_data.get('final_decision', 'HOLD')
    rl_decision = log_data.get('rl_decision', 'HOLD')
    quality_metrics['model_consensus'] = 100 if xlstm_decision == rl_decision else 50
    
    # Общее качество сигнала (без VSA)
    quality_metrics['overall_signal_quality'] = (
        quality_metrics['xlstm_quality'] * 0.6 + # ИЗМЕНЕНО: Увеличиваем вес xLSTM
        quality_metrics['model_consensus'] * 0.4 # ИЗМЕНЕНО: Увеличиваем вес консенсуса
    )
    
    return quality_metrics






Файл 7: rl_agent.py
Этот файл содержит RL-агента.


Удаление VSA-признаков из observation_space:

Местоположение: В конструкторе __init__, где определяется self.observation_space.
ЗАМЕНИТЬ:
# СТАРЫЙ КОД
        # Пространство наблюдений: xLSTM выход + VSA + портфель
        # xLSTM выход (3) + VSA признаки (7) + портфель (4) = 14 признаков
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32
        )


НА НОВЫЙ КОД:
# НОВЫЙ КОД - observation_space без VSA
        # Пространство наблюдений: xLSTM выход + портфель
        # xLSTM выход (3) + портфель (4) = 7 признаков
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32 # ИЗМЕНЕНО: Размер 7
        )





Удаление VSA-зависимой логики из _get_vsa_features:

Местоположение: Метод _get_vsa_features.
ЗАМЕНИТЬ ВЕСЬ ЭТОТ МЕТОД:
# СТАРЫЙ КОД (весь метод _get_vsa_features)
    def _get_vsa_features(self):
        """Получает текущие VSA признаки"""
        if self.current_step >= len(self.df):
            return np.zeros(7)
            
        current_row = self.df.iloc[self.current_step]
        return np.array([
            current_row['vsa_no_demand'],
            current_row['vsa_no_supply'], 
            current_row['vsa_stopping_volume'],
            current_row['vsa_climactic_volume'],
            current_row['vsa_test'],
            current_row['vsa_effort_vs_result'],
            current_row['vsa_strength']
        ])


НА НОВЫЙ КОД (пустой заглушка):
# НОВЫЙ КОД - Заглушка для _get_vsa_features
    def _get_vsa_features(self):
        """VSA отключен, возвращаем пустой массив"""
        return np.zeros(0) # ИЗМЕНЕНО: Возвращаем пустой массив (или массив нулей, если нужен определенный размер)





Удаление VSA-зависимой логики из _get_observation:

Местоположение: Метод _get_observation.
ЗАМЕНИТЬ:
# СТАРЫЙ КОД
        xlstm_pred = self._get_xlstm_prediction()  # 3 элемента
        vsa_features = self._get_vsa_features()    # 7 элементов  
        portfolio_state = self._get_portfolio_state()  # 4 элемента
        
        return np.concatenate([xlstm_pred, vsa_features, portfolio_state]).astype(np.float32)


НА НОВЫЙ КОД:
# НОВЫЙ КОД - _get_observation без VSA
        xlstm_pred = self._get_xlstm_prediction()  # 3 элемента
        # vsa_features = self._get_vsa_features()    # <--- УДАЛЕНО
        portfolio_state = self._get_portfolio_state()  # 4 элемента
        
        return np.concatenate([xlstm_pred, portfolio_state]).astype(np.float32) # ИЗМЕНЕНО: Без vsa_features





Удаление VSA-признаков из self.feature_columns в reset:

Местоположение: Метод reset, список self.feature_columns.
ЗАМЕНИТЬ: (удалить эти строки)
# СТАРЫЙ КОД (удалить эти строки)
        'vsa_no_demand', 'vsa_no_supply', 'vsa_stopping_volume',
        'vsa_climactic_volume', 'vsa_test', 'vsa_effort_vs_result', 'vsa_strength'


НА НОВЫЙ КОД: (просто удали эти строки, чтобы их не было в списке)



Удаление VSA-зависимой логики из _calculate_advanced_reward:

Местоположение: Метод _calculate_advanced_reward.
ЗАМЕНИТЬ ВЕСЬ ЭТОТ МЕТОД:
# СТАРЫЙ КОД (весь метод _calculate_advanced_reward)
    def _calculate_advanced_reward(self, action, pnl_pct, vsa_features, xlstm_prediction):
        """
        Расширенная система наград с учетом качества сигналов
        """
        base_reward = pnl_pct if pnl_pct != 0 else 0
        
        # Бонусы за качественные VSA сигналы (ОСЛАБЛЕНЫ ПОРОГИ)
        vsa_bonus = 0
        if action in [0, 1]: # SELL или BUY
            # BUY (действие 1): если есть no_supply (vsa_features[1]) или stopping_volume (vsa_features[2])
            if action == 1 and (vsa_features[1] > 0 or vsa_features[2] > 0 or vsa_features[6] > 0.2): # Добавлено: vsa_strength > 0.2
                vsa_bonus = 2 # СНИЖЕНО с 3 до 2, чтобы не перевешивать PnL
            # SELL (действие 0): если есть no_demand (vsa_features[0]) или climactic_volume (vsa_features[3])
            elif action == 0 and (vsa_features[0] > 0 or vsa_features[3] > 0 or vsa_features[6] < -0.2): # Добавлено: vsa_strength < -0.2
                vsa_bonus = 2 # СНИЖЕНО с 3 до 2

        # Штраф за противоречащие VSA сигналы (ОСЛАБЛЕНЫ ПОРОГИ)
        vsa_penalty = 0
        if action == 1 and (vsa_features[0] > 0 or vsa_features[3] > 0 or vsa_features[6] < -0.5): # Усилен порог для penalization
            vsa_penalty = -3 # СНИЖЕНО с -5 до -3
        elif action == 0 and (vsa_features[1] > 0 or vsa_features[2] > 0 or vsa_features[6] > 0.5): # Усилен порог для penalization
            vsa_penalty = -3 # СНИЖЕНО с -5 до -3

        # Бонус за скорость закрытия прибыльных позиций
        speed_bonus = 0
        if pnl_pct > 0 and self.steps_in_position < 20:
            speed_bonus = 2

        # Штраф за долгое удержание убыточных позиций
        hold_penalty = 0
        if pnl_pct < 0 and self.steps_in_position > 30:
            hold_penalty = -3

        # Бонус за уверенность xLSTM
        xlstm_conf = np.max(xlstm_prediction)
        if xlstm_conf > 0.7:
            base_reward += xlstm_conf * 2

        # Штраф за противоречие xLSTM
        predicted_action_idx = np.argmax(xlstm_prediction)
        xlstm_to_rl_map = {0: 1, 1: 0, 2: 2}  # xLSTM_BUY->RL_BUY, xLSTM_SELL->RL_SELL
        
        if action != 2 and action != xlstm_to_rl_map.get(predicted_action_idx):
            base_reward -= 1

        # Штраф за отклонение от баланса (риск-менеджмент)
        if self.balance < self.initial_balance * 0.9:
            base_reward -= 5

        # НОВЫЙ БЛОК: СКОРРЕКТИРОВАННЫЙ БОНУС ЗА ИССЛЕДОВАНИЕ И ЭНТРОПИЮ
        exploration_bonus = 0
        # Меньший, но все еще стимулирующий бонус
        if action in [0, 1]: # Если действие - BUY или SELL
            exploration_bonus = 0.2 # <--- ИЗМЕНЕНО с 0.5 на 0.2
        
        entropy_bonus = 0
        # Ослабляем бонус за энтропию
        entropy = -np.sum(xlstm_prediction * np.log(xlstm_prediction + 1e-10))
        normalized_entropy = entropy / np.log(len(xlstm_prediction))
        entropy_bonus = normalized_entropy * 0.2 # <--- ИЗМЕНЕНО с 0.5 на 0.2

        # НОВЫЙ БЛОК: ЯВНОЕ ВОЗНАГРАЖДЕНИЕ ЗА HOLD И ШТРАФ ЗА OVERTRADING
        hold_reward = 0
        overtrading_penalty = 0

        # Если действие HOLD
        if action == 2: # HOLD
            # Вознаграждаем за HOLD, если рынок действительно находится в консолидации
            # (например, низкая волатильность, нет сильного тренда)
            current_row = self.df.iloc[self.current_step]
            volatility = current_row.get('ATR_14', 0) / current_row.get('close', 1) # Нормализованная волатильность
            adx = current_row.get('ADX_14', 0)

            if volatility < 0.005 and adx < 25: # Низкая волатильность и слабый тренд
                hold_reward = 0.5 # Небольшой бонус за правильный HOLD
            elif volatility > 0.01 and adx > 30: # Высокая волатильность и сильный тренд - HOLD менее желателен
                hold_reward = -0.5 # Небольшой штраф за HOLD в тренде
            else:
                hold_reward = 0.1 # Небольшой нейтральный бонус за HOLD
            
            # Штраф за слишком долгое удержание позиции (если она убыточна)
            if pnl_pct < 0 and self.steps_in_position > 30:
                hold_penalty = -3 # Уже есть, но убедимся, что он применяется к HOLD
            
        else: # Если действие BUY или SELL (не HOLD)
            # Штраф за overtrading (слишком частые сделки, когда нет явного сигнала)
            # Используем VSA-скор для определения "явного сигнала"
            current_row = self.df.iloc[self.current_step]
            vsa_buy_score = (0.3 * (current_row.get('vsa_no_supply', 0) == 1) + 0.3 * (current_row.get('vsa_stopping_volume', 0) == 1) + 0.4 * (current_row.get('vsa_strength', 0) > 0.1))
            vsa_sell_score = (0.3 * (current_row.get('vsa_no_demand', 0) == 1) + 0.3 * (current_row.get('vsa_climactic_volume', 0) == 1) + 0.4 * (current_row.get('vsa_strength', 0) < -0.1))

            if action == 1 and vsa_buy_score < 0.4: # Если BUY, но VSA-скор низкий
                overtrading_penalty = -1.0
            elif action == 0 and vsa_sell_score < 0.4: # Если SELL, но VSA-скор низкий
                overtrading_penalty = -1.0

        total_reward = base_reward + vsa_bonus + vsa_penalty + speed_bonus + hold_penalty + exploration_bonus + entropy_bonus + hold_reward + overtrading_penalty # <--- ДОБАВЛЕНО: hold_reward, overtrading_penalty
        
        return total_reward


НА НОВЫЙ КОД (без VSA):
# НОВЫЙ КОД - _calculate_advanced_reward без VSA
    def _calculate_advanced_reward(self, action, pnl_pct, xlstm_prediction): # ИЗМЕНЕНО: Удален vsa_features
        """
        Расширенная система наград с учетом качества сигналов (без VSA)
        """
        base_reward = pnl_pct if pnl_pct != 0 else 0
        
        # Бонус за скорость закрытия прибыльных позиций
        speed_bonus = 0
        if pnl_pct > 0 and self.steps_in_position < 20:
            speed_bonus = 2

        # Штраф за долгое удержание убыточных позиций
        hold_penalty = 0
        if pnl_pct < 0 and self.steps_in_position > 30:
            hold_penalty = -3

        # Бонус за уверенность xLSTM
        xlstm_conf = np.max(xlstm_prediction)
        if xlstm_conf > 0.7:
            base_reward += xlstm_conf * 2

        # Штраф за противоречие xLSTM
        predicted_action_idx = np.argmax(xlstm_prediction)
        xlstm_to_rl_map = {0: 1, 1: 0, 2: 2}  # xLSTM_BUY->RL_BUY, xLSTM_SELL->RL_SELL
        
        if action != 2 and action != xlstm_to_rl_map.get(predicted_action_idx):
            base_reward -= 1

        # Штраф за отклонение от баланса (риск-менеджмент)
        if self.balance < self.initial_balance * 0.9:
            base_reward -= 5

        # СКОРРЕКТИРОВАННЫЙ БОНУС ЗА ИССЛЕДОВАНИЕ И ЭНТРОПИЮ
        exploration_bonus = 0
        if action in [0, 1]:
            exploration_bonus = 0.2
        
        entropy_bonus = 0
        entropy = -np.sum(xlstm_prediction * np.log(xlstm_prediction + 1e-10))
        normalized_entropy = entropy / np.log(len(xlstm_prediction))
        entropy_bonus = normalized_entropy * 0.2

        # ЯВНОЕ ВОЗНАГРАЖДЕНИЕ ЗА HOLD И ШТРАФ ЗА OVERTRADING
        hold_reward = 0
        overtrading_penalty = 0

        if action == 2: # HOLD
            current_row = self.df.iloc[self.current_step]
            volatility = current_row.get('ATR_14', 0) / current_row.get('close', 1)
            adx = current_row.get('ADX_14', 0)

            if volatility < 0.005 and adx < 25:
                hold_reward = 0.5
            elif volatility > 0.01 and adx > 30:
                hold_reward = -0.5
            else:
                hold_reward = 0.1
            
            if pnl_pct < 0 and self.steps_in_position > 30:
                hold_penalty = -3
            
        else: # Если действие BUY или SELL (не HOLD)
            # Штраф за overtrading (слишком частые сделки, когда нет явного сигнала)
            current_row = self.df.iloc[self.current_step]
            # Без VSA, используем индикаторы для определения "явного сигнала"
            buy_signal_strength = (
                (current_row.get('RSI_14', 50) < 40) +
                (current_row.get('ADX_14', 0) > 20) +
                (current_row.get('MACD_hist', 0) > 0)
            )
            sell_signal_strength = (
                (current_row.get('RSI_14', 50) > 60) +
                (current_row.get('ADX_14', 0) > 20) +
                (current_row.get('MACD_hist', 0) < 0)
            )

            if action == 1 and buy_signal_strength < 1: # Если BUY, но слабые индикаторы
                overtrading_penalty = -1.0
            elif action == 0 and sell_signal_strength < 1: # Если SELL, но слабые индикаторы
                overtrading_penalty = -1.0

        total_reward = base_reward + speed_bonus + hold_penalty + exploration_bonus + entropy_bonus + hold_reward + overtrading_penalty # ИЗМЕНЕНО: Удалены vsa_bonus, vsa_penalty
        
        return total_reward






Файл 8: parameter_optimizer.py
Этот файл оптимизирует параметры бота.


Удаление vsa_weight из current_params:

Местоположение: Конструктор __init__, словарь self.current_params.
ЗАМЕНИТЬ:
# СТАРЫЙ КОД
        'confidence_threshold': 0.65,
        'take_profit_pct': 1.5,
        'stop_loss_pct': -1.0,
        'vsa_weight': 0.4, # <--- УДАЛИТЬ ЭТУ СТРОКУ
        'xlstm_weight': 0.6


НА НОВЫЙ КОД:
# НОВЫЙ КОД
        'confidence_threshold': 0.65,
        'take_profit_pct': 1.5,
        'stop_loss_pct': -1.0,
        'xlstm_weight': 0.6 # ИЗМЕНЕНО: Удален vsa_weight





Удаление vsa_weight из bounds:

Местоположение: Метод optimize_parameters, список bounds.
ЗАМЕНИТЬ:
# СТАРЫЙ КОД
        bounds = [
            (0.5, 0.9),   # confidence_threshold
            (0.8, 3.0),   # take_profit_pct
            (-3.0, -0.5), # stop_loss_pct
            (0.2, 0.8),   # vsa_weight # <--- УДАЛИТЬ ЭТУ СТРОКУ
            (0.2, 0.8)    # xlstm_weight
        ]


НА НОВЫЙ КОД:
# НОВЫЙ КОД
        bounds = [
            (0.5, 0.9),   # confidence_threshold
            (0.8, 3.0),   # take_profit_pct
            (-3.0, -0.5), # stop_loss_pct
            (0.2, 0.8)    # xlstm_weight # ИЗМЕНЕНО: Удален vsa_weight
        ]





Удаление vsa_weight из x0:

Местоположение: Метод optimize_parameters, список x0.
ЗАМЕНИТЬ:
# СТАРЫЙ КОД
        x0 = [
            self.current_params['confidence_threshold'],
            self.current_params['take_profit_pct'],
            abs(self.current_params['stop_loss_pct']),  # Делаем положительным для оптимизации
            self.current_params['vsa_weight'], # <--- УДАЛИТЬ ЭТУ СТРОКУ
            self.current_params['xlstm_weight']
        ]


НА НОВЫЙ КОД:
# НОВЫЙ КОД
        x0 = [
            self.current_params['confidence_threshold'],
            self.current_params['take_profit_pct'],
            abs(self.current_params['stop_loss_pct']),  # Делаем положительным для оптимизации
            self.current_params['xlstm_weight'] # ИЗМЕНЕНО: Удален vsa_weight
        ]





Удаление vsa_weight из обновления self.current_params:

Местоположение: Метод optimize_parameters, обновление self.current_params.
ЗАМЕНИТЬ:
# СТАРЫЙ КОД
        self.current_params = {
            'confidence_threshold': result.x[0],
            'take_profit_pct': result.x[1],
            'stop_loss_pct': -result.x[2],  # Возвращаем отрицательное значение
            'vsa_weight': result.x[3], # <--- УДАЛИТЬ ЭТУ СТРОКУ
            'xlstm_weight': result.x[4]
        }


НА НОВЫЙ КОД:
# НОВЫЙ КОД
        self.current_params = {
            'confidence_threshold': result.x[0],
            'take_profit_pct': result.x[1],
            'stop_loss_pct': -result.x[2],  # Возвращаем отрицательное значение
            'xlstm_weight': result.x[3] # ИЗМЕНЕНО: Удален vsa_weight (индекс тоже меняется!)
        }






Файл 9: performance_monitor.py
Этот файл мониторит производительность.


Удаление vsa_confirmed_trades из reset_daily_stats:

Местоположение: Метод reset_daily_stats, словарь self.daily_stats.
ЗАМЕНИТЬ:
# СТАРЫЙ КОД
        'winning_trades': 0,
        'losing_trades': 0,
        'vsa_confirmed_trades': 0, # <--- УДАЛИТЬ ЭТУ СТРОКУ
        'model_accuracy': [],


НА НОВЫЙ КОД:
# НОВЫЙ КОД
        'winning_trades': 0,
        'losing_trades': 0,
        'model_accuracy': [], # ИЗМЕНЕНО: Удален vsa_confirmed_trades





Удаление vsa_confirmed_trades из log_trade_opened:

Местоположение: Метод log_trade_opened, параметр vsa_confirmed и его использование.
ЗАМЕНИТЬ:
# СТАРЫЙ КОД
    def log_trade_opened(self, symbol, decision, vsa_confirmed=False):
        """Логирует открытие сделки"""
        self.daily_stats['trades_opened'] += 1
        if vsa_confirmed:
            self.daily_stats['vsa_confirmed_trades'] += 1
        
        self.save_stats()


НА НОВЫЙ КОД:
# НОВЫЙ КОД
    def log_trade_opened(self, symbol, decision): # ИЗМЕНЕНО: Удален vsa_confirmed
        """Логирует открытие сделки (без VSA подтверждения)"""
        self.daily_stats['trades_opened'] += 1
        # if vsa_confirmed: # <--- УДАЛЕНО
        #     self.daily_stats['vsa_confirmed_trades'] += 1 # <--- УДАЛЕНО
        
        self.save_stats()





Удаление vsa_confirmed_trades из print_current_stats:

Местоположение: Метод print_current_stats, строка вывода.
ЗАМЕНИТЬ:
# СТАРЫЙ КОД
        print(f"🎯 Win Rate: {win_rate:.1f}%")
        print(f"✅ VSA подтвержденных: {stats['vsa_confirmed_trades']}") # <--- УДАЛИТЬ ЭТУ СТРОКУ


НА НОВЫЙ КОД:
# НОВЫЙ КОД
        print(f"🎯 Win Rate: {win_rate:.1f}%")
        # print(f"✅ VSA подтвержденных: {stats['vsa_confirmed_trades']}") # <--- УДАЛЕНО






Файл 10: advanced_simulation_engine.py
Этот файл отвечает за симуляцию.


Удаление VSA-признаков из feature_columns:

Местоположение: Конструктор __init__, список self.feature_columns (строка ~20).
ЗАМЕНИТЬ: (удалить эти строки)
# СТАРЫЙ КОД (удалить эти строки)
        'vsa_no_demand', 'vsa_no_supply', 'vsa_stopping_volume',
        'vsa_climactic_volume', 'vsa_test', 'vsa_effort_vs_result', 'vsa_strength',
        'volume_ratio', 'spread_ratio', 'close_position'


НА НОВЫЙ КОД: (просто удали эти строки, чтобы их не было в списке)



Удаление VSA-зависимой логики из prepare_symbol_data:

Местоположение: Функция prepare_symbol_data, блок "Полная обработка с VSA" (строка ~55).
ЗАМЕНИТЬ:
# СТАРЫЙ КОД
        df_symbol = calculate_features(df_symbol)
        df_symbol = detect_candlestick_patterns(df_symbol)
        df_symbol = calculate_vsa_features(df_symbol)


НА НОВЫЙ КОД:
# НОВЫЙ КОД - Обработка без VSA
        df_symbol = calculate_features(df_symbol)
        df_symbol = detect_candlestick_patterns(df_symbol)
        # df_symbol = calculate_vsa_features(df_symbol) # <--- ЗАКОММЕНТИРОВАНО





Удаление VSA-зависимой стратегии из run_comprehensive_simulation:

Местоположение: Функция run_comprehensive_simulation, словарь strategies (строка ~90).
ЗАМЕНИТЬ:
# СТАРЫЙ КОД
        'hybrid_aggressive': {'confidence_threshold': 0.4, 'tp': 1.0, 'sl': -1.5},
        'vsa_only': {'vsa_only': True, 'tp': 1.2, 'sl': -1.2}, # <--- УДАЛИТЬ ЭТУ СТРОКУ
    }


НА НОВЫЙ КОД:
# НОВЫЙ КОД
        'hybrid_aggressive': {'confidence_threshold': 0.4, 'tp': 1.0, 'sl': -1.5},
    } # ИЗМЕНЕНО: Удалена vsa_only стратегия





Удаление VSA-зависимой логики из simulate_strategy:

Местоположение: Функция simulate_strategy, блок if vsa_only: (строка ~140).
ЗАМЕНИТЬ:
# СТАРЫЙ КОД
        if vsa_only:
            decision = self.get_vsa_only_decision(df.iloc[i])
        else:
            try:
                decision = self.decision_maker.get_decision(sequence_df, confidence_threshold)
            except:
                decision = 'HOLD'


НА НОВЫЙ КОД:
# НОВЫЙ КОД - Получение решения без VSA
        # if vsa_only: # <--- УДАЛЕНО
        #     decision = self.get_vsa_only_decision(df.iloc[i]) # <--- УДАЛЕНО
        # else: # <--- УДАЛЕНО
        try:
            decision = self.decision_maker.get_decision(sequence_df, confidence_threshold)
        except:
            decision = 'HOLD'





Удаление функции get_vsa_only_decision:

Местоположение: Функция get_vsa_only_decision (строка ~220).
Удалить весь этот метод.



Удаление VSA-зависимых строк из отчета generate_comprehensive_report:

Местоположение: Функция generate_comprehensive_report, строка print("КОМПЛЕКСНЫЙ ОТЧЕТ ПО СИМУЛЯЦИЯМ xLSTM + VSA + RL").
ЗАМЕНИТЬ:
# СТАРЫЙ КОД
print("КОМПЛЕКСНЫЙ ОТЧЕТ ПО СИМУЛЯЦИЯМ xLSTM + VSA + RL")


НА НОВЫЙ КОД:
# НОВЫЙ КОД
print("КОМПЛЕКСНЫЙ ОТЧЕТ ПО СИМУЛЯЦИЯМ xLSTM + RL") # ИЗМЕНЕНО: Удалено VSA






Файл 11: models/xlstm_rl_model.py (Проверка, что все VSA-зависимые части удалены)

В этом файле напрямую не было VSA-признаков, только логика регуляризации и архитектура. Убедитесь, что все изменения, предложенные в предыдущих шагах, были применены.


После выполнения всех этих шагов (и, как всегда, удаления старых сохраненных моделей и скейлеров), ваш код будет полностью очищен от VSA-логики.
Затем можно будет запустить обучение и оценить производительность модели xLSTM + RL только на индикаторах и паттернах. Это позволит нам изолировать проблему и понять, был ли VSA действительно источником проблем с генерализацией.