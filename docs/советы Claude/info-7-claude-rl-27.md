План по исправлению:

Удалить VSA-зависимую логику из генерации целевых меток: Поскольку мы временно отключаем VSA-признаки, мы должны также убрать их использование из условий, которые определяют, является ли свеча BUY, SELL или HOLD. Мы будем полагаться только на future_return и классические индикаторы/паттерны.


Инструкции по реализации:
Файл 1: train_model.py
1. Удаление VSA-зависимой логики из генерации целевых меток
Местоположение: В функции prepare_xlstm_rl_data, в блоке "НОВЫЙ КОД - БОЛЕЕ СТРОГИЕ УСЛОВИЯ ДЛЯ BUY/SELL" (строка ~150-200).
ЗАМЕНИТЬ ВЕСЬ ЭТОТ БЛОК:
# СТАРЫЙ КОД (весь блок "НОВЫЙ КОД - БОЛЕЕ СТРОГИЕ УСЛОВИЯ ДЛЯ BUY/SELL")
        # Увеличиваем пороги для генерации торговых сигналов
        df['base_threshold'] = 0.008  # Увеличиваем с 0.0005 до 0.008 (0.8%)
        df['dynamic_threshold'] = np.maximum(
            df['base_threshold'],
            (df['ATR_14'] / df['close'] * 1.2).fillna(0.008)  # Увеличиваем множитель
        )

        # Более строгие VSA условия
        df['vsa_buy_strength'] = (
            0.5 * (df['vsa_no_supply'] == 1) +
            0.5 * (df['vsa_stopping_volume'] == 1) +
            0.3 * np.clip(df['vsa_strength'] / 2.0, 0, 1)  # Более строгая нормализация
        )

        df['vsa_sell_strength'] = (
            0.5 * (df['vsa_no_demand'] == 1) +
            0.5 * (df['vsa_climactic_volume'] == 1) +
            0.3 * np.clip(-df['vsa_strength'] / 2.0, 0, 1)
        )

        # Дополнительные технические фильтры
        strong_trend = df['ADX_14'] > 25
        high_volume = df['volume_ratio'] > 1.5
        rsi_extreme_buy = df['RSI_14'] < 30
        rsi_extreme_sell = df['RSI_14'] > 70

        # БОЛЕЕ СТРОГИЕ условия для BUY/SELL
        buy_condition = (
            (df['future_return'] > df['dynamic_threshold']) &
            (df['vsa_buy_strength'] > 0.6) &  # Увеличиваем порог с 0.2 до 0.6
            (strong_trend | high_volume | rsi_extreme_buy)  # Дополнительное подтверждение
        )

        sell_condition = (
            (df['future_return'] < -df['dynamic_threshold']) &
            (df['vsa_sell_strength'] > 0.6) &  # Увеличиваем порог с 0.2 до 0.6
            (strong_trend | high_volume | rsi_extreme_sell)  # Дополнительное подтверждение
        )

НА НОВЫЙ КОД:
# НОВЫЙ КОД - Условия для BUY/SELL только на основе future_return и классических индикаторов/паттернов
        # Увеличиваем пороги для генерации торговых сигналов
        df['base_threshold'] = 0.008  # Минимальный порог 0.8%
        df['dynamic_threshold'] = np.maximum(
            df['base_threshold'],
            (df['ATR_14'] / df['close'] * 1.5).fillna(0.008) # ИЗМЕНЕНО: Увеличиваем множитель ATR
        )

        # Классические технические фильтры
        strong_trend = df['ADX_14'] > 25
        high_volume = df['volume_ratio'] > 1.5
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

Объяснение: Мы полностью убрали ссылки на VSA-признаки (vsa_buy_strength, vsa_sell_strength, vsa_strength) из логики определения целевых меток. Теперь модель будет использовать только future_return, ATR_14, ADX_14, volume_ratio, RSI_14 и MACD для генерации меток BUY/SELL/HOLD. Я также немного скорректировал пороги RSI и добавил сигналы MACD, чтобы условия были более осмысленными без VSA.
Файл 2: train_model.py
2. Удаление VSA-зависимой логики из переклассификации HOLD
Местоположение: В функции prepare_xlstm_rl_data, в блоке "НОВЫЙ КОД - УМЕНЬШАЕМ ПЕРЕКЛАССИФИКАЦИЮ" (строка ~220).
ЗАМЕНИТЬ ВЕСЬ ЭТОТ БЛОК:
# СТАРЫЙ КОД (весь блок "НОВЫЙ КОД - УМЕНЬШАЕМ ПЕРЕКЛАССИФИКАЦИЮ")
        if current_hold_count < (current_buy_count + current_sell_count) * 2.0:  # Если HOLD < 66%
            print(f"⚠️ Слишком мало HOLD сигналов. ДОБАВЛЯЕМ HOLD вместо переклассификации.")
            
            # Вместо переклассификации HOLD в BUY/SELL, делаем обратное
            # Переклассифицируем часть слабых BUY/SELL в HOLD
            
            weak_buy_indices = df[
                (df['target'] == 0) &
                (df['vsa_buy_strength'] < 0.4) &  # Слабые VSA сигналы
                (df['RSI_14'] > 35) & (df['RSI_14'] < 65)  # RSI в нейтральной зоне
            ].index
            
            weak_sell_indices = df[
                (df['target'] == 1) &
                (df['vsa_sell_strength'] < 0.4) &  # Слабые VSA сигналы
                (df['RSI_14'] > 35) & (df['RSI_14'] < 65)  # RSI в нейтральной зоне
            ].index
            
            # Переклассифицируем 30% слабых сигналов в HOLD
            import random
            random.seed(42)
            
            if len(weak_buy_indices) > 0:
                reclassify_buy = random.sample(
                    list(weak_buy_indices),
                    min(int(len(weak_buy_indices) * 0.3), len(weak_buy_indices))
                )
                df.loc[reclassify_buy, 'target'] = 2  # Переводим в HOLD
            
            if len(weak_sell_indices) > 0:
                reclassify_sell = random.sample(
                    list(weak_sell_indices),
                    min(int(len(weak_sell_indices) * 0.3), len(weak_sell_indices))
                )
                df.loc[reclassify_sell, 'target'] = 2  # Переводим в HOLD

        else:
            print(f"✅ Баланс классов приемлемый, переклассификация не нужна.")

НА НОВЫЙ КОД:
# НОВЫЙ КОД - Условия для переклассификации HOLD без VSA
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
                    volume_ratio = df.loc[idx, 'volume_ratio']

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

Объяснение: Мы полностью переписали логику переклассификации HOLD, чтобы она не зависела от VSA-признаков. Теперь она использует только классические индикаторы (RSI, ADX, MACD, volume_ratio) и price_change_3_period (которые будут присутствовать в DataFrame). Я также немного скорректировал пороги, чтобы они были более агрессивными в поиске BUY/SELL сигналов среди HOLD.
