

2. Файл: feature_engineering.py
Здесь уже все изменения (AO, WILLR, нет ATR/BBANDS, паттерны закомментированы) реализованы верно. Дополнительных изменений в этом файле не требуется.
3. Файл: run_live_trading.py
Здесь уже все изменения (AO, WILLR, нет ATR/BBANDS, паттерны закомментированы) реализованы верно. Дополнительных изменений в этом файле не требуется.
4. Файл: trading_env.py
Здесь необходимо обновить sequence_length в нескольких местах.
4.1. Функция _get_xlstm_prediction(self)
Найдите эти строки:
        if self.current_step < 10:  # Нужно минимум 10 свечей для последовательности
            return np.array([0.33, 0.33, 0.34])  # Равномерное распределение
        
        # Берем последние 10 свечей для xLSTM
        sequence_data = self.df.iloc[self.current_step-10:self.current_step]

Замените их на (обновляем до 30):
        if self.current_step < self.sequence_length:  # 🔥 ИЗМЕНЕНО: Используем self.sequence_length
            return np.array([0.33, 0.33, 0.34])
        
        # Берем последние SEQUENCE_LENGTH свечей для xLSTM
        sequence_data = self.df.iloc[self.current_step-self.sequence_length:self.current_step] # 🔥 ИЗМЕНЕНО

4.2. Функция reset(self, seed=None, options=None)
Найдите эти строки:
        self.current_step = 10  # Начинаем с 10-й свечи для xLSTM
        self.balance = self.initial_balance
        # ...
        # 🔥 НОВЫЕ FEATURE_COLUMNS - ТОЛЬКО ИНДИКАТОРЫ (для RL среды)
        self.feature_columns = [
            # ...
        ]

Замените их на (обновляем до 30, добавляем self.sequence_length):
        self.sequence_length = 30 # 🔥 ИЗМЕНЕНО: Задаем здесь явно 30
        self.current_step = self.sequence_length # 🔥 ИЗМЕНЕНО: Начинаем с SEQUENCE_LENGTH-й свечи
        self.balance = self.initial_balance
        # ...
        # 🔥 НОВЫЕ FEATURE_COLUMNS - ТОЛЬКО ИНДИКАТОРЫ (для RL среды)
        self.feature_columns = [
            # ...
        ]

Примечание: self.sequence_length в TradingEnvRL должен быть установлен равным config.SEQUENCE_LENGTH. Хотя вы его уже задаете в __init__, явное указание в reset (или передача из config) сделает его более надежным.
5. Файл: train_model.py
Здесь нужно обновить sequence_length в argparse и закомментировать весь блок imblearn.
5.1. argparse и sequence_length
Найдите эту строку:
    parser.add_argument('--sequence_length', type=int, default=10, help='Длина последовательности')

Замените ее на (обновляем значение по умолчанию):
    parser.add_argument('--sequence_length', type=int, default=30, help='Длина последовательности') # 🔥 ИЗМЕНЕНО: default=30

5.2. Блок IMBLEARN
Найдите этот блок (начинается с try:):
    # === НОВЫЙ БЛОК: ИСПОЛЬЗОВАНИЕ IMBLEARN ДЛЯ БАЛАНСИРОВКИ КЛАССОВ ===
    try:
        from imblearn.over_sampling import SMOTE
        from imblearn.under_sampling import RandomUnderSampler
        from imblearn.pipeline import Pipeline
        from collections import Counter

        print("\n🔄 Применяю Oversampling/Undersampling для балансировки классов...")
        
        # ... (весь код imblearn) ...

    except ImportError:
        print("⚠️ imbalanced-learn не установлен. Пропустил oversampling/undersampling. Установите: pip install imbalanced-learn")
    except Exception as e:
        print(f"❌ Ошибка при oversampling/undersampling: {e}")
    # === КОНЕЦ НОВОГО БЛОКА IMBLEARN ===

Закомментируйте его полностью:
    # === НОВЫЙ БЛОК: ИСПОЛЬЗОВАНИЕ IMBLEARN ДЛЯ БАЛАНСИРОВКИ КЛАССОВ ===
    # 🔥 ЗАКОММЕНТИРОВАНО: Отключаем imblearn
    # try:
    #     from imblearn.over_sampling import SMOTE
    #     from imblearn.under_sampling import RandomUnderSampler
    #     from imblearn.pipeline import Pipeline
    #     from collections import Counter

    #     print("\n🔄 Применяю Oversampling/Undersampling для балансировки классов...")
        
    #     # ... (весь код imblearn) ...

    # except ImportError:
    #     print("⚠️ imbalanced-learn не установлен. Пропустил oversampling/undersampling. Установите: pip install imbalanced-learn")
    # except Exception as e:
    #     print(f"❌ Ошибка при oversampling/undersampling: {e}")
    # === КОНЕЦ НОВОГО БЛОКА IMBLEARN ===

5.3. Обновляем логику генерации целевых меток в prepare_xlstm_rl_data
Здесь мы уже ослабили условия и заменили индикаторы. Дополнительно, давайте еще немного ослабим условие strong_trend и price_change_3_period для переклассификации, чтобы увеличить количество исходных сигналов.
Найдите этот блок:
        # Классические технические фильтры
        strong_trend = df['ADX_14'] > 20 # 🔥 ИЗМЕНЕНО: С 25 до 20 (более мягкий порог)
        
        # Условия для BUY
        rsi_buy_zone = df['RSI_14'] < 40 # 🔥 ИЗМЕНЕНО: С 30 до 40
        macd_buy_signal = (df['MACD_12_26_9'] > df['MACD_signal']) & \
                          (df['MACD_hist'] > 0.0005) # 🔥 ИЗМЕНЕНО: С 0.001 до 0.0005
        willr_buy_signal = df['WILLR_14'] < -80 # 🔥 НОВОЕ: WILLR_14 для BUY
        ao_buy_signal = df['AO_5_34'] > 0 # 🔥 НОВОЕ: AO выше нуля
        
        # Условия для SELL
        rsi_sell_zone = df['RSI_14'] > 60 # 🔥 ИЗМЕНЕНО: С 70 до 60
        macd_sell_signal = (df['MACD_12_26_9'] < df['MACD_signal']) & \
                           (df['MACD_hist'] < -0.0005) # 🔥 ИЗМЕНЕНО: С -0.001 до -0.0005
        willr_sell_signal = df['WILLR_14'] > -20 # 🔥 НОВОЕ: WILLR_14 для SELL
        ao_sell_signal = df['AO_5_34'] < 0 # 🔥 НОВОЕ: AO ниже нуля

        # Условия для BUY/SELL только на основе future_return и классических индикаторов
        buy_condition = (
            (df['future_return'] > df['dynamic_threshold'] * 1.0) &
            (strong_trend & (rsi_buy_zone | macd_buy_signal | willr_buy_signal | ao_buy_signal)) # 🔥 ИЗМЕНЕНО: Смешанные условия с OR
        )

        sell_condition = (
            (df['future_return'] < -df['dynamic_threshold'] * 1.0) &
            (strong_trend & (rsi_sell_zone | macd_sell_signal | willr_sell_signal | ao_sell_signal)) # 🔥 ИЗМЕНЕНО: Смешанные условия с OR
        )

Замените его на (еще более мягкие условия, чтобы получить больше сигналов):
        # Создаем целевые метки на основе будущих цен + индикаторов
        df['future_return'] = (df['close'].shift(-5) - df['close']) / df['close']
        
        # Увеличиваем пороги для генерации торговых сигналов
        df['base_threshold'] = 0.002 # 🔥 ИЗМЕНЕНО: С 0.003 до 0.002 (еще более мягкий порог)
        df['dynamic_threshold'] = np.maximum(
            df['base_threshold'],
            (abs(df['AO_5_34']) / df['close'] * 0.8).fillna(0.002) # 🔥 ИЗМЕНЕНО: Коэффициент 0.8
        )

        # Классические технические фильтры
        strong_trend = df['ADX_14'] > 18 # 🔥 ИЗМЕНЕНО: С 20 до 18 (еще более мягкий порог)
        
        # Условия для BUY
        rsi_buy_zone = df['RSI_14'] < 45 # 🔥 ИЗМЕНЕНО: С 40 до 45
        macd_buy_signal = (df['MACD_12_26_9'] > df['MACD_signal']) # 🔥 ИЗМЕНЕНО: Убрали MACD_hist > 0.0005 для упрощения
        willr_buy_signal = df['WILLR_14'] < -70 # 🔥 ИЗМЕНЕНО: С -80 до -70
        ao_buy_signal = df['AO_5_34'] > 0 # AO выше нуля
        
        # Условия для SELL
        rsi_sell_zone = df['RSI_14'] > 55 # 🔥 ИЗМЕНЕНО: С 60 до 55
        macd_sell_signal = (df['MACD_12_26_9'] < df['MACD_signal']) # 🔥 ИЗМЕНЕНО: Убрали MACD_hist < -0.0005 для упрощения
        willr_sell_signal = df['WILLR_14'] > -30 # 🔥 ИЗМЕНЕНО: С -20 до -30
        ao_sell_signal = df['AO_5_34'] < 0 # AO ниже нуля

        # Условия для BUY/SELL только на основе future_return и классических индикаторов
        buy_condition = (
            (df['future_return'] > df['dynamic_threshold'] * 1.0) &
            (strong_trend & (rsi_buy_zone | macd_buy_signal | willr_buy_signal | ao_buy_signal))
        )

        sell_condition = (
            (df['future_return'] < -df['dynamic_threshold'] * 1.0) &
            (strong_trend & (rsi_sell_zone | macd_sell_signal | willr_sell_signal | ao_sell_signal))
        )

5.4. Блок переклассификации HOLD
Найдите этот блок:
                for idx in reclassify_indices:
                    if idx < 5: continue
                    
                    rsi = df.loc[idx, 'RSI_14']
                    adx = df.loc[idx, 'ADX_14']
                    macd_hist = df.loc[idx, 'MACD_hist']
                    willr = df.loc[idx, 'WILLR_14'] # 🔥 НОВОЕ
                    ao = df.loc[idx, 'AO_5_34']     # 🔥 НОВОЕ
                    price_change_3_period = df['close'].pct_change(3).loc[idx]

                    # Условия для переклассификации (с индикаторами) - теперь с AO и WILLR
                    # 1. RSI + ADX + MACD_hist + WILLR + AO + движение цены
                    if (rsi < 40 and adx > 20 and macd_hist > 0.0005 and willr < -80 and ao > 0 and price_change_3_period > 0.003): # 🔥 ИЗМЕНЕНО
                        df.loc[idx, 'target'] = 0  # BUY
                    elif (rsi > 60 and adx > 20 and macd_hist < -0.0005 and willr > -20 and ao < 0 and price_change_3_period < -0.003): # 🔥 ИЗМЕНЕНО
                        df.loc[idx, 'target'] = 1  # SELL
                    
                    # 2. Сильный тренд по ADX + движение цены (без других индикаторов для более широкого охвата)
                    elif (adx > 30 and abs(price_change_3_period) > 0.005): # 🔥 ИЗМЕНЕНО: Порог ADX и price_change
                        df.loc[idx, 'target'] = 0 if price_change_3_period > 0 else 1

Замените его на (еще более мягкие условия, чтобы получить больше сигналов):
                for idx in reclassify_indices:
                    if idx < 5: continue
                    
                    rsi = df.loc[idx, 'RSI_14']
                    adx = df.loc[idx, 'ADX_14']
                    macd_hist = df.loc[idx, 'MACD_hist']
                    willr = df.loc[idx, 'WILLR_14']
                    ao = df.loc[idx, 'AO_5_34']
                    price_change_3_period = df['close'].pct_change(3).loc[idx]

                    # Условия для переклассификации (с индикаторами) - теперь с AO и WILLR
                    # 🔥 Условия значительно ослаблены для увеличения количества сигналов
                    if (rsi < 45 and adx > 18 and macd_hist > 0 and willr < -70 and ao > 0 and price_change_3_period > 0.002): # 🔥 ИЗМЕНЕНО
                        df.loc[idx, 'target'] = 0  # BUY
                    elif (rsi > 55 and adx > 18 and macd_hist < 0 and willr > -30 and ao < 0 and price_change_3_period < -0.002): # 🔥 ИЗМЕНЕНО
                        df.loc[idx, 'target'] = 1  # SELL
                    
                    # 2. Сильный тренд по ADX + движение цены (без других индикаторов для более широкого охвата)
                    elif (adx > 25 and abs(price_change_3_period) > 0.003): # 🔥 ИЗМЕНЕНО: Порог ADX и price_change
                        df.loc[idx, 'target'] = 0 if price_change_3_period > 0 else 1

6. Файл: market_regime_detector.py
Здесь уже все изменения (AO, WILLR, нет ATR/BBANDS, паттерны закомментированы) реализованы верно. Дополнительных изменений в этом файле не требуется.
