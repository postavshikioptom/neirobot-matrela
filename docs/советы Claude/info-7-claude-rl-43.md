
План действий (с вашими предложениями и моими дополнениями):
Шаг 1: Устранение Боллинджера, замена ATR на AO, добавление WILLR, новые логи (реализовано)

Это уже сделано.

Шаг 2: Упрощение архитектуры xLSTM и усиление регуляризации
Это наш следующий главный приоритет, наряду с отладкой целевой функции.
2.1. Файл: models/xlstm_rl_model.py
Найдите функцию build_model(self):
        # Первый xLSTM слой с внешней памятью (уменьшаем units и memory_size)
        xlstm1 = XLSTMLayer(
            units=self.memory_units // 2, # ИЗМЕНЕНО: Уменьшаем units
            memory_size=self.memory_size // 2, # ИЗМЕНЕНО: Уменьшаем memory_size
            return_sequences=True,
            name='xlstm_memory_layer_1'
        )(inputs)
        xlstm1 = LayerNormalization()(xlstm1)
        xlstm1 = Dropout(0.2)(xlstm1) # ИЗМЕНЕНО: Добавлен Dropout
        
        # Второй xLSTM слой (уменьшаем units и memory_size)
        xlstm2 = XLSTMLayer(
            units=self.memory_units // 4, # ИЗМЕНЕНО: Уменьшаем units
            memory_size=self.memory_size // 4, # ИЗМЕНЕНО: Уменьшаем memory_size
            return_sequences=True,
            name='xlstm_memory_layer_2'
        )(xlstm1)
        xlstm2 = LayerNormalization()(xlstm2)
        xlstm2 = Dropout(0.2)(xlstm2) # ИЗМЕНЕНО: Добавлен Dropout
        
        # Механизм внимания
        attention = Attention(name='attention_mechanism')([xlstm2, xlstm2])
        
        # Финальный xLSTM слой (уменьшаем units и memory_size)
        xlstm_final = XLSTMLayer(
            units=self.attention_units // 2, # ИЗМЕНЕНО: Уменьшаем units
            memory_size=self.attention_units // 2, # ИЗМЕНЕНО: Уменьшаем memory_size
            return_sequences=False,
            name='xlstm_memory_final'
        )(attention)
        xlstm_final = LayerNormalization()(xlstm_final)
        xlstm_final = Dropout(0.2)(xlstm_final) # ИЗМЕНЕНО: Добавлен Dropout
        
        # УСИЛЕННАЯ РЕГУЛЯРИЗАЦИЯ
        dense1 = Dense(64, activation='relu', kernel_regularizer=l2(0.003), name='dense_1')(xlstm_final)  # ИЗМЕНЕНО: L2 до 0.003
        dropout1 = Dropout(0.6)(dense1)  # ИЗМЕНЕНО: Dropout до 0.6
        
        dense2 = Dense(32, activation='relu', kernel_regularizer=l2(0.003), name='dense_2')(dropout1)  # ИЗМЕНЕНО: L2 до 0.003
        dropout2 = Dropout(0.5)(dense2)  # ИЗМЕНЕНО: Dropout до 0.5
        
        dense3 = Dense(16, activation='relu', kernel_regularizer=l2(0.002), name='dense_3')(dropout2) # ИЗМЕНЕНО: L2 до 0.002
        dropout3 = Dropout(0.4)(dense3) # ИЗМЕНЕНО: Dropout до 0.4

Измените его на (более простая архитектура, усиленная регуляризация):
        # ✅ Фиксируем входную форму для стабильности
        inputs = Input(shape=self.input_shape, name='input_layer')
        
        # Упрощаем: Первый xLSTM слой с меньшим количеством юнитов и более сильным Dropout
        xlstm1 = XLSTMLayer(
            units=self.memory_units // 4, # 🔥 ИЗМЕНЕНО: Еще сильнее уменьшаем units
            memory_size=self.memory_size // 4, # 🔥 ИЗМЕНЕНО: Еще сильнее уменьшаем memory_size
            return_sequences=True,
            name='xlstm_memory_layer_1'
        )(inputs)
        xlstm1 = LayerNormalization()(xlstm1)
        xlstm1 = Dropout(0.3)(xlstm1) # 🔥 ИЗМЕНЕНО: Увеличиваем Dropout

        # ❌ УДАЛЕНО: Второй xLSTM слой (упрощаем архитектуру)
        # xlstm2 = XLSTMLayer(
        #     units=self.memory_units // 4,
        #     memory_size=self.memory_size // 4,
        #     return_sequences=True,
        #     name='xlstm_memory_layer_2'
        # )(xlstm1)
        # xlstm2 = LayerNormalization()(xlstm2)
        # xlstm2 = Dropout(0.2)(xlstm2)
        
        # Механизм внимания теперь после первого xLSTM слоя
        attention = Attention(name='attention_mechanism')([xlstm1, xlstm1]) # 🔥 ИЗМЕНЕНО: Используем xlstm1
        
        # Финальный xLSTM слой с меньшим количеством юнитов
        xlstm_final = XLSTMLayer(
            units=self.attention_units // 4, # 🔥 ИЗМЕНЕНО: Еще сильнее уменьшаем units
            memory_size=self.attention_units // 4, # 🔥 ИЗМЕНЕНО: Еще сильнее уменьшаем memory_size
            return_sequences=False,
            name='xlstm_memory_final'
        )(attention)
        xlstm_final = LayerNormalization()(xlstm_final)
        xlstm_final = Dropout(0.3)(xlstm_final) # 🔥 ИЗМЕНЕНО: Увеличиваем Dropout
        
        # УСИЛЕННАЯ РЕГУЛЯРИЗАЦИЯ И УПРОЩЕНИЕ ПЛОТНЫХ СЛОЕВ
        dense1 = Dense(32, activation='relu', kernel_regularizer=l2(0.005), name='dense_1')(xlstm_final)  # 🔥 ИЗМЕНЕНО: Меньше юнитов, сильнее L2
        dropout1 = Dropout(0.7)(dense1)  # 🔥 ИЗМЕНЕНО: Сильнее Dropout
        
        # ❌ УДАЛЕНО: Второй и третий плотные слои (дальнейшее упрощение)
        # dense2 = Dense(32, activation='relu', kernel_regularizer=l2(0.003), name='dense_2')(dropout1)
        # dropout2 = Dropout(0.5)(dense2)
        
        # dense3 = Dense(16, activation='relu', kernel_regularizer=l2(0.002), name='dense_3')(dropout2)
        # dropout3 = Dropout(0.4)(dense3)
        
        # НОВЫЙ КОД
        outputs = Dense(3, activation='softmax', name='output_layer')(dropout1) # 🔥 ИЗМЕНЕНО: Используем dropout1

2.2. Изменение оптимизатора и функции потерь
Найдите в функции train(self, ...):
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=0.0002,  # ИЗМЕНЕНО: Уменьшаем LR с 0.0005 до 0.0002
            clipnorm=0.5,
            weight_decay=0.0001
        )
        xlstm_model.model.compile(
            optimizer=optimizer,
            loss=CustomFocalLoss(gamma=1.0, alpha=0.3, class_weights=[1.2, 1.2, 0.8]), # ИЗМЕНЕНО: Используем класс
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

Измените его на (более низкий LR, categorical_crossentropy):
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=0.00005,  # 🔥 ИЗМЕНЕНО: Еще сильнее уменьшаем LR
            clipnorm=1.0, # 🔥 ИЗМЕНЕНО: Увеличиваем clipnorm для большей стабильности
            weight_decay=0.0001
        )
        xlstm_model.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy', # 🔥 ИЗМЕНЕНО: Временно переключаемся на стандартную функцию потерь
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

Шаг 3: Модифицируем train_model.py
3.1. Убираем class_weight из xlstm_model.train
Поскольку мы отключили imblearn и переключились на categorical_crossentropy, передавать class_weight_dict в model.fit не нужно.
Найдите вызов xlstm_model.train:
    history = xlstm_model.train(
        X_train_to_model, y_train,
        X_val_to_model, y_val,
        epochs=80,
        batch_size=32,
        class_weight=class_weight_dict, # <-- Эту строку нужно удалить или закомментировать
        custom_callbacks=callbacks
    )

Измените его на:
    history = xlstm_model.train(
        X_train_to_model, y_train,
        X_val_to_model, y_val,
        epochs=80,
        batch_size=32,
        # class_weight=class_weight_dict, # 🔥 ЗАКОММЕНТИРОВАНО: class_weight больше не нужен
        custom_callbacks=callbacks
    )

===========

План действий: Увеличение sequence_length до 60, отключение imblearn, упрощение модели, настройка весов и дополнительные логи
1. Файл: config.py
Увеличиваем длину последовательности для моделей и количество загружаемых свечей.
Найдите эти строки:
SEQUENCE_LENGTH = 30 # Длина последовательности для моделей
REQUIRED_CANDLES = 35 

Замените их на:
SEQUENCE_LENGTH = 60 # 🔥 ИЗМЕНЕНО: Длина последовательности для моделей (с 30 до 60)
REQUIRED_CANDLES = 65 # 🔥 ИЗМЕНЕНО: Количество свечей для загрузки (должно быть больше SEQUENCE_LENGTH)


2. Файл: feature_engineering.py
В этом файле уже все изменения (AO, WILLR, нет ATR/BBANDS, паттерны закомментированы) реализованы верно. Дополнительных изменений в этом файле не требуется.

3. Файл: run_live_trading.py
Здесь нужно обновить, сколько свечей передается в decision_maker.get_decision.
Найдите эти строки (в функциях manage_active_positions и process_new_signal):
            # Принимаем решение через новую гибридную систему
            decision = decision_maker.get_decision(features_df.tail(15), confidence_threshold=CONFIDENCE_THRESHOLD)

Замените их на (используем config.SEQUENCE_LENGTH):
            # Принимаем решение через новую гибридную систему
            decision = decision_maker.get_decision(features_df.tail(config.SEQUENCE_LENGTH), confidence_threshold=CONFIDENCE_THRESHOLD) # 🔥 ИЗМЕНЕНО: Используем config.SEQUENCE_LENGTH


4. Файл: trading_env.py
Здесь необходимо обновить sequence_length в нескольких местах.
4.1. Функция __init__(self, ...)
Убедитесь, что self.sequence_length инициализируется из config.SEQUENCE_LENGTH.
Найдите эту строку:
    def __init__(self, df: pd.DataFrame, xlstm_model, initial_balance=10000, commission=0.0008):
        super(TradingEnvRL, self).__init__()
        
        self.df = df.copy()
        self.xlstm_model = xlstm_model
        self.initial_balance = initial_balance
        self.commission = commission
        
        # Пространство действий: 0=SELL, 1=BUY, 2=HOLD
        self.action_space = gym.spaces.Discrete(3)
        
        # Пространство наблюдений: xLSTM выход + портфель
        # xLSTM выход (3) + портфель (4) = 7 признаков
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32
        )
        
        self.reset()

Замените ее на (добавляем self.sequence_length из конфига):
    def __init__(self, df: pd.DataFrame, xlstm_model, initial_balance=10000, commission=0.0008):
        super(TradingEnvRL, self).__init__()
        
        self.df = df.copy()
        self.xlstm_model = xlstm_model
        self.initial_balance = initial_balance
        self.commission = commission
        self.sequence_length = config.SEQUENCE_LENGTH # 🔥 НОВОЕ: Инициализируем sequence_length из конфига
        
        # Пространство действий: 0=SELL, 1=BUY, 2=HOLD
        self.action_space = gym.spaces.Discrete(3)
        
        # Пространство наблюдений: xLSTM выход + портфель (размер может измениться, если меняется sequence_length)
        # xLSTM выход (3) + портфель (4) = 7 признаков (пока оставляем 7, если не меняется состав портфеля)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32
        )
        
        self.reset()

4.2. Функция _get_xlstm_prediction(self)
Найдите эти строки:
        if self.current_step < self.sequence_length:
            return np.array([0.33, 0.33, 0.34])
        
        # Берем последние SEQUENCE_LENGTH свечей для xLSTM
        sequence_data = self.df.iloc[self.current_step-self.sequence_length:self.current_step]
        
        # Подготавливаем данные для модели (нужно адаптировать под ваши признаки)
        features = sequence_data[self.feature_columns].values
        features_reshaped = features.reshape(1, self.sequence_length, len(self.feature_columns))
        
        return self.xlstm_model.predict(features_reshaped)[0]

Замените их на (убеждаемся, что sequence_length используется корректно):
        if self.current_step < self.sequence_length:
            return np.array([0.33, 0.33, 0.34])
        
        # Берем последние SEQUENCE_LENGTH свечей для xLSTM
        sequence_data = self.df.iloc[self.current_step-self.sequence_length:self.current_step]
        
        # Подготавливаем данные для модели (нужно адаптировать под ваши признаки)
        features = sequence_data[self.feature_columns].values
        features_reshaped = features.reshape(1, self.sequence_length, len(self.feature_columns)) # 🔥 Убеждаемся, что здесь используется self.sequence_length
        
        return self.xlstm_model.predict(features_reshaped)[0]

Примечание: это изменение уже было в предыдущей итерации, просто убедитесь, что оно сохранилось.
4.3. Функция reset(self, seed=None, options=None)
Найдите эти строки:
        self.sequence_length = 30 # 🔥 ИЗМЕНЕНО: Задаем здесь явно 30
        self.current_step = self.sequence_length # 🔥 ИЗМЕНЕНО: Начинаем с SEQUENCE_LENGTH-й свечи
        self.balance = self.initial_balance

Замените их на (используем self.sequence_length, которая уже инициализирована в __init__):
        # self.sequence_length = 30 # 🔥 УДАЛЕНО: Уже инициализируется в __init__
        self.current_step = self.sequence_length # 🔥 Используем self.sequence_length
        self.balance = self.initial_balance

Примечание: self.sequence_length должен быть инициализирован из config.py один раз в __init__ и затем использоваться. Если вы явно зададите self.sequence_length = 30 в reset, это переопределит значение из конфига.

5. Файл: train_model.py
Здесь нужно обновить sequence_length в argparse, закомментировать весь блок imblearn и внести изменения в архитектуру модели и функцию потерь.
5.1. argparse и sequence_length
Найдите эту строку:
    parser.add_argument('--sequence_length', type=int, default=30, help='Длина последовательности')

Замените ее на (обновляем значение по умолчанию до 60):
    parser.add_argument('--sequence_length', type=int, default=60, help='Длина последовательности') # 🔥 ИЗМЕНЕНО: default=60

5.2. Блок IMBLEARN
Убедитесь, что этот блок полностью закомментирован:
    # === НОВЫЙ БЛОК: ИСПОЛЬЗОВАНИЕ IMBLEARN ДЛЯ БАЛАНСИРОВКИ КЛАССОВ ===
    # 🔥 ЗАКОММЕНТИРОВАНО: Отключаем imblearn
    # try:
    #     from imblearn.over_sampling import SMOTE
    #     from imblearn.under_sampling import RandomUnderSampler
    #     from imblearn.pipeline import Pipeline
    #     from collections import Counter
    #
    #     print("\n🔄 Применяю Oversampling/Undersampling для балансировки классов...")
    #
    #     # ... (весь код imblearn) ...
    #
    # except ImportError:
    #     print("⚠️ imbalanced-learn не установлен. Пропустил oversampling/undersampling. Установите: pip install imbalanced-learn")
    # except Exception as e:
    #     print(f"❌ Ошибка при oversampling/undersampling: {e}")
    # === КОНЕЦ НОВОГО БЛОКА IMBLEARN ===

Примечание: Если вы уже закомментировали его в предыдущей итерации, просто убедитесь, что он остался закомментированным.
5.3. Обновляем логику генерации целевых меток в prepare_xlstm_rl_data
Здесь мы уже ослабили условия и заменили индикаторы. Дополнительно, давайте еще немного ослабим условие strong_trend и price_change_3_period для переклассификации, чтобы увеличить количество исходных сигналов.
Найдите этот блок:
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

Замените его на (еще более мягкие условия, чтобы получить больше сигналов):
        # Создаем целевые метки на основе будущих цен + индикаторов
        df['future_return'] = (df['close'].shift(-5) - df['close']) / df['close']
        
        # Увеличиваем пороги для генерации торговых сигналов
        df['base_threshold'] = 0.0015 # 🔥 ИЗМЕНЕНО: С 0.002 до 0.0015 (еще более мягкий порог)
        df['dynamic_threshold'] = np.maximum(
            df['base_threshold'],
            (abs(df['AO_5_34']) / df['close'] * 0.7).fillna(0.0015) # 🔥 ИЗМЕНЕНО: Коэффициент 0.7
        )

        # Классические технические фильтры
        strong_trend = df['ADX_14'] > 15 # 🔥 ИЗМЕНЕНО: С 18 до 15 (еще более мягкий порог)
        
        # Условия для BUY
        rsi_buy_zone = df['RSI_14'] < 50 # 🔥 ИЗМЕНЕНО: С 45 до 50
        macd_buy_signal = (df['MACD_12_26_9'] > df['MACD_signal']) # MACD_hist уже убрали
        willr_buy_signal = df['WILLR_14'] < -60 # 🔥 ИЗМЕНЕНО: С -70 до -60
        ao_buy_signal = df['AO_5_34'] > 0 # AO выше нуля
        
        # Условия для SELL
        rsi_sell_zone = df['RSI_14'] > 50 # 🔥 ИЗМЕНЕНО: С 55 до 50
        macd_sell_signal = (df['MACD_12_26_9'] < df['MACD_signal']) # MACD_hist уже убрали
        willr_sell_signal = df['WILLR_14'] > -40 # 🔥 ИЗМЕНЕНО: С -30 до -40
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
                    if (rsi < 50 and adx > 15 and macd_hist > 0 and willr < -60 and ao > 0 and price_change_3_period > 0.0015): # 🔥 ИЗМЕНЕНО
                        df.loc[idx, 'target'] = 0  # BUY
                    elif (rsi > 50 and adx > 15 and macd_hist < 0 and willr > -40 and ao < 0 and price_change_3_period < -0.0015): # 🔥 ИЗМЕНЕНО
                        df.loc[idx, 'target'] = 1  # SELL
                    
                    # 2. Сильный тренд по ADX + движение цены (без других индикаторов для более широкого охвата)
                    elif (adx > 20 and abs(price_change_3_period) > 0.002): # 🔥 ИЗМЕНЕНО: Порог ADX и price_change
                        df.loc[idx, 'target'] = 0 if price_change_3_period > 0 else 1


6. Файл: models/xlstm_rl_model.py
Здесь мы упростим архитектуру модели и усилим регуляризацию.
6.1. Функция build_model(self)
Найдите этот блок:
        # Первый xLSTM слой с внешней памятью (уменьшаем units и memory_size)
        xlstm1 = XLSTMLayer(
            units=self.memory_units // 4, # 🔥 ИЗМЕНЕНО: Еще сильнее уменьшаем units
            memory_size=self.memory_size // 4, # 🔥 ИЗМЕНЕНО: Еще сильнее уменьшаем memory_size
            return_sequences=True,
            name='xlstm_memory_layer_1'
        )(inputs)
        xlstm1 = LayerNormalization()(xlstm1)
        xlstm1 = Dropout(0.3)(xlstm1) # 🔥 ИЗМЕНЕНО: Увеличиваем Dropout

        # ❌ УДАЛЕНО: Второй xLSTM слой (упрощаем архитектуру)
        # xlstm2 = XLSTMLayer(
        #     units=self.memory_units // 4,
        #     memory_size=self.memory_size // 4,
        #     return_sequences=True,
        #     name='xlstm_memory_layer_2'
        # )(xlstm1)
        # xlstm2 = LayerNormalization()(xlstm2)
        # xlstm2 = Dropout(0.2)(xlstm2)
        
        # Механизм внимания теперь после первого xLSTM слоя
        attention = Attention(name='attention_mechanism')([xlstm1, xlstm1]) # 🔥 ИЗМЕНЕНО: Используем xlstm1
        
        # Финальный xLSTM слой с меньшим количеством юнитов
        xlstm_final = XLSTMLayer(
            units=self.attention_units // 4, # 🔥 ИЗМЕНЕНО: Еще сильнее уменьшаем units
            memory_size=self.attention_units // 4, # 🔥 ИЗМЕНЕНО: Еще сильнее уменьшаем memory_size
            return_sequences=False,
            name='xlstm_memory_final'
        )(attention)
        xlstm_final = LayerNormalization()(xlstm_final)
        xlstm_final = Dropout(0.3)(xlstm_final) # 🔥 ИЗМЕНЕНО: Увеличиваем Dropout
        
        # УСИЛЕННАЯ РЕГУЛЯРИЗАЦИЯ И УПРОЩЕНИЕ ПЛОТНЫХ СЛОЕВ
        dense1 = Dense(32, activation='relu', kernel_regularizer=l2(0.005), name='dense_1')(xlstm_final)  # 🔥 ИЗМЕНЕНО: Меньше юнитов, сильнее L2
        dropout1 = Dropout(0.7)(dense1)  # 🔥 ИЗМЕНЕНО: Сильнее Dropout
        
        # ❌ УДАЛЕНО: Второй и третий плотные слои (дальнейшее упрощение)
        # dense2 = Dense(32, activation='relu', kernel_regularizer=l2(0.003), name='dense_2')(dropout1)
        # dropout2 = Dropout(0.5)(dense2)
        
        # dense3 = Dense(16, activation='relu', kernel_regularizer=l2(0.002), name='dense_3')(dropout2)
        # dropout3 = Dropout(0.4)(dense3)
        
        # НОВЫЙ КОД
        outputs = Dense(3, activation='softmax', name='output_layer')(dropout1) # 🔥 ИЗМЕНЕНО: Используем dropout1

Измените его на (более простая архитектура, еще усиленная регуляризация):
        # ✅ Фиксируем входную форму для стабильности
        inputs = Input(shape=self.input_shape, name='input_layer')
        
        # 🔥 УПРОЩЕНО: Один xLSTM слой с меньшим количеством юнитов и сильным Dropout
        xlstm_layer = XLSTMLayer(
            units=self.memory_units // 8, # 🔥 ИЗМЕНЕНО: Еще сильнее уменьшаем units (например, 16 юнитов)
            memory_size=self.memory_size // 8, # 🔥 ИЗМЕНЕНО: Еще сильнее уменьшаем memory_size (например, 8 юнитов)
            return_sequences=True, # Оставляем True для внимания
            name='xlstm_layer_1'
        )(inputs)
        xlstm_layer = LayerNormalization()(xlstm_layer)
        xlstm_layer = Dropout(0.4)(xlstm_layer) # 🔥 ИЗМЕНЕНО: Увеличиваем Dropout
        
        # Механизм внимания после единственного xLSTM слоя
        attention = Attention(name='attention_mechanism')([xlstm_layer, xlstm_layer])
        
        # Финальный xLSTM слой для получения одного вектора
        xlstm_final = XLSTMLayer(
            units=self.attention_units // 8, # 🔥 ИЗМЕНЕНО: Еще сильнее уменьшаем units (например, 8 юнитов)
            memory_size=self.attention_units // 8, # 🔥 ИЗМЕНЕНО: Еще сильнее уменьшаем memory_size (например, 8 юнитов)
            return_sequences=False, # Возвращаем только последний выход
            name='xlstm_final_output'
        )(attention)
        xlstm_final = LayerNormalization()(xlstm_final)
        xlstm_final = Dropout(0.4)(xlstm_final) # 🔥 ИЗМЕНЕНО: Увеличиваем Dropout
        
        # 🔥 УПРОЩЕНО: Один плотный слой с сильной регуляризацией
        dense_output = Dense(16, activation='relu', kernel_regularizer=l2(0.01), name='dense_output_layer')(xlstm_final) # 🔥 ИЗМЕНЕНО: Меньше юнитов, сильнее L2
        dropout_output = Dropout(0.5)(dense_output) # 🔥 ИЗМЕНЕНО: Сильнее Dropout
        
        outputs = Dense(3, activation='softmax', name='final_output')(dropout_output) # 🔥 Используем dropout_output

        self.model = Model(inputs=inputs, outputs=outputs, name='Simplified_xLSTM_RL_Model')

6.2. Изменение оптимизатора и функции потерь
Найдите в функции train(self, ...):
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=0.00005,  # 🔥 ИЗМЕНЕНО: Еще сильнее уменьшаем LR
            clipnorm=1.0, # 🔥 ИЗМЕНЕНО: Увеличиваем clipnorm для большей стабильности
            weight_decay=0.0001
        )
        xlstm_model.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy', # 🔥 ИЗМЕНЕНО: Временно переключаемся на стандартную функцию потерь
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

Замените его на (еще более низкий LR, categorical_crossentropy, но с ручными весами классов):
        # 🔥 ИЗМЕНЕНО: Убираем compute_class_weight('balanced') и задаем веса классов вручную
        # Это должно быть сделано перед вызовом compile, если class_weight будет использоваться в loss.
        # Но поскольку мы используем categorical_crossentropy, class_weight передается в model.fit
        
        # 🔥 НОВЫЕ ВЕСА КЛАССОВ (ручная настройка для борьбы с majority-class bias)
        # Эти веса будут переданы в model.fit
        class_weight_dict = {0: 10.0, 1: 10.0, 2: 1.0} # 🔥 ИЗМЕНЕНО: Очень высокие веса для BUY/SELL
        print(f"📊 Ручные веса классов для обучения: {class_weight_dict}")


        optimizer = tf.keras.optimizers.Adam(
            learning_rate=0.00001,  # 🔥 ИЗМЕНЕНО: Еще сильнее уменьшаем LR
            clipnorm=1.0, 
            weight_decay=0.0001
        )
        xlstm_model.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy', # 🔥 Оставляем стандартную функцию потерь
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

Примечание: Теперь class_weight_dict будет использоваться в model.fit, а не в loss (потому что loss='categorical_crossentropy' не принимает class_weights напрямую).
6.3. Обновляем patience для EarlyStopping
Найдите в функции train(self, ...):
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                mode='min',
                patience=35,  # <--- ИЗМЕНЕНО с 25 на 35 (или даже 40-50)
                restore_best_weights=True,
                verbose=1
            ),

Замените его на (увеличиваем patience):
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                mode='min',
                patience=50,  # 🔥 ИЗМЕНЕНО: Увеличиваем patience, чтобы дать модели больше времени на обучение
                restore_best_weights=True,
                verbose=1
            ),


7. Файл: market_regime_detector.py
Здесь уже все изменения (AO, WILLR, нет ATR/BBANDS, паттерны закомментированы) реализованы верно. Дополнительных изменений в этом файле не требуется.
