Хорошо, давайте проведем детальный анализ всех ваших скриптов, чтобы найти причину дисбабаланса классов, долгого обучения и избыточного логирования. Я укажу, где и что нужно изменить.

🚨 1. Главная проблема: Дисбаланс классов (99.8% HOLD)
Это самая критичная проблема. Модель предсказывает HOLD, потому что видит его в 99.8% случаев и считает, что это самый "безопасный" ответ. Чтобы это исправить, нужно:

Расширить условия для BUY/SELL меток: Ваши текущие пороги (>0.01 и < -0.01 для future_return) слишком строгие. Рынок редко дает такие резкие движения, чтобы их можно было однозначно классифицировать как BUY/SELL.
Ввести взвешивание классов (Class Weighting): Это заставит модель уделять больше внимания редким классам (BUY/SELL) во время обучения.

🔧 Инструкции по изменению (для train_model.py и xlstm_rl_model.py):
Файл: train_model.py


Измените пороги для создания меток BUY/SELL:

Найдите функцию prepare_xlstm_rl_data.
Внутри неё найдите блоки, где определяются buy_condition и sell_condition.
Измените пороги future_return и vsa_strength на более мягкие.
Добавьте логику для балансировки классов, если дисбаланс все еще сильный, путем переклассификации части HOLD-сигналов.

# В prepare_xlstm_rl_data(data_path, sequence_length=10):
# ...
    # Создаем целевые метки на основе будущих цен + VSA подтверждения
    df['future_return'] = (df['close'].shift(-5) - df['close']) / df['close']
    # df['target'] = 2  # По умолчанию HOLD - эту строку мы теперь устанавливаем ниже

    # BUY: положительная доходность + VSA подтверждение покупки (СНИЖЕНЫ ПОРОГИ)
    buy_condition = (
        (df['future_return'] > 0.003) &  # СНИЖЕНО с 0.01 до 0.003 (0.3% роста)
        ((df['vsa_no_supply'] == 1) | (df['vsa_stopping_volume'] == 1) | (df['vsa_strength'] > 0.5)) # СНИЖЕНО с 1 до 0.5
    )
    
    # SELL: отрицательная доходность + VSA подтверждение продажи (СНИЖЕНЫ ПОРОГИ)
    sell_condition = (
        (df['future_return'] < -0.003) &  # СНИЖЕНО с -0.01 до -0.003 (-0.3% падения)
        ((df['vsa_no_demand'] == 1) | (df['vsa_climactic_volume'] == 1) | (df['vsa_strength'] < -0.5)) # СНИЖЕНО с -1 до -0.5
    )
    
    # Сначала устанавливаем все в HOLD, затем переписываем
    df['target'] = 2  # По умолчанию HOLD
    df.loc[buy_condition, 'target'] = 0 # BUY
    df.loc[sell_condition, 'target'] = 1 # SELL

    # ДОБАВЬТЕ: Принудительная балансировка классов (если необходимо)
    # Этот блок можно включать, если после ослабления порогов баланс все еще очень плохой.
    # Он попытается переклассифицировать часть "HOLD" в BUY/SELL на основе других индикаторов.
    # Это может быть "грязным" решением, но иногда необходимо для обучения.
    current_buy_count = (df['target'] == 0).sum()
    current_sell_count = (df['target'] == 1).sum()
    current_hold_count = (df['target'] == 2).sum()

    if current_hold_count > (current_buy_count + current_sell_count) * 2: # Если HOLD в 2+ раза больше
        print(f"⚠️ Сильный дисбаланс классов. Попытка переклассификации части HOLD-сигналов.")
        hold_indices = df[df['target'] == 2].index
        
        import random
        random.seed(42) # Для воспроизводимости
        
        # Переклассифицируем 15% HOLD в BUY/SELL на основе RSI, ADX
        reclassify_count = int(current_hold_count * 0.15)
        if reclassify_count > 0:
            reclassify_indices = random.sample(list(hold_indices), min(reclassify_count, len(hold_indices)))
            
            for idx in reclassify_indices:
                # Простая логика: если RSI < 35 и ADX растет -> BUY
                if df.loc[idx, 'RSI_14'] < 35 and df.loc[idx, 'ADX_14'] > df.loc[idx-1, 'ADX_14']:
                    df.loc[idx, 'target'] = 0  # BUY
                # Если RSI > 65 и ADX растет -> SELL
                elif df.loc[idx, 'RSI_14'] > 65 and df.loc[idx, 'ADX_14'] > df.loc[idx-1, 'ADX_14']:
                    df.loc[idx, 'target'] = 1  # SELL
        
        print(f"Баланс классов после переклассификации:")
        unique, counts = np.unique(df['target'], return_counts=True)
        class_names = ['BUY', 'SELL', 'HOLD']
        for class_idx, count in zip(unique, counts):
            print(f"  {class_names[class_idx]}: {count} ({count/len(df)*100:.1f}%)")
# ...



Файл: xlstm_rl_model.py


Добавьте взвешивание классов в метод train:

Импортируйте compute_class_weight из sklearn.utils.class_weight.
Вычислите веса классов и передайте их в model.fit.

# Вверху файла, среди других импортов:
from sklearn.utils.class_weight import compute_class_weight
# ...

class XLSTMRLModel:
    # ...
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32, custom_callbacks=None):
        # ... (существующий код) ...

        # ДОБАВЬТЕ: Вычисление весов классов для борьбы с дисбалансом
        y_integers = np.argmax(y_train, axis=1) # Преобразуем one-hot в целые числа
        class_weights_array = compute_class_weight(
            'balanced',
            classes=np.unique(y_integers),
            y=y_integers
        )
        class_weight_dict = {i: class_weights_array[i] for i in range(len(class_weights_array))}

        print(f"📊 Веса классов для обучения: {class_weight_dict}")

        # Обучение с нормализованными данными
        history = self.model.fit(
            X_train_scaled, y_train,
            validation_data=(X_val_scaled, y_val),
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weight_dict,  # <-- ДОБАВЛЕНО: передаем веса классов
            callbacks=callbacks,
            verbose=0, # Изменяем на 0 для уменьшения логирования
            shuffle=True
        )
        # ...




⏳ 2. Долгие эпохи и избыточное логирование
Ваши эпохи длятся 10 минут, а логирование показывает каждую итерацию (5/25318, 15/25318 и т.д.). Это происходит из-за параметра verbose=1 в model.fit и stable_baselines3 агенте.
🔧 Инструкции по изменению (для xlstm_rl_model.py, train_model.py, rl_agent.py):
Файл: xlstm_rl_model.py


Отключите детальное логирование model.fit:

В методе train измените verbose в self.model.fit на 0.

# В XLSTMRLModel.train(...):
# ...
    history = self.model.fit(
        X_train_scaled, y_train,
        validation_data=(X_val_scaled, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight_dict, # Уже добавлено выше
        callbacks=callbacks,
        verbose=0,  # <-- ИЗМЕНЕНО: 0 для логирования только по эпохам
        shuffle=True
    )
# ...



Измените частоту сохранения checkpoint'ов для xlstm_checkpoint_epoch:

В методе train измените save_freq на 'epoch' и verbose на 0.

# В XLSTMRLModel.train(...):
# ...
        tf.keras.callbacks.ModelCheckpoint(
            'models/xlstm_checkpoint_epoch_{epoch:02d}.keras',
            monitor='val_loss',
            save_best_only=False, # Можно оставить False, чтобы иметь все эпохи
            save_freq='epoch',  # <-- ИЗМЕНЕНО: сохраняем каждую эпоху
            verbose=0 # <-- ИЗМЕНЕНО: отключаем подробное логирование сохранения
        ),
# ...



Файл: train_model.py


Убедитесь, что DetailedProgressCallback корректно выводит одну строку на эпоху:

В вашем текущем коде DetailedProgressCallback уже настроен правильно для вывода одной строки на эпоху.
MemoryCleanupCallback также хорошо настроен.



Отключите verbose для ReduceLROnPlateau:

Это не влияет на каждую итерацию, но уберет сообщения об изменении Learning Rate, если вы хотите совсем минималистичное логирование.

# В train_xlstm_rl_system(...):
# ...
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=15,
            min_lr=1e-7,
            verbose=0 # <-- ИЗМЕНЕНО: 0 для отключения логирования изменений LR
        )
# ...



Файл: rl_agent.py


Отключите детальное логирование для PPO/SAC:

В методе build_agent измените verbose на 0 и progress_bar на False.

# В IntelligentRLAgent.build_agent(self, vec_env):
# ...
        self.model = PPO(
            'MlpPolicy',
            vec_env,
            # ... (другие параметры) ...
            verbose=0, # <-- ИЗМЕНЕНО: 0 для отключения детального логирования
            tensorboard_log="./tensorboard_logs/",
            progress_bar=False # <-- ДОБАВЛЕНО: отключаем прогресс-бар
        )
# ...
        self.model = SAC(
            'MlpPolicy',
            vec_env,
            # ... (другие параметры) ...
            verbose=0, # <-- ИЗМЕНЕНО: 0 для отключения детального логирования
            tensorboard_log="./tensorboard_logs/",
            progress_bar=False # <-- ДОБАВЛЕНО: отключаем прогресс-бар
        )
# ...



Отключите progress_bar в train_with_callbacks:
# В IntelligentRLAgent.train_with_callbacks(...):
# ...
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=eval_callback,
            progress_bar=False # <-- ИЗМЕНЕНО
        )
# ...




⚡ 3. Ускорение обучения (без потери качества)
Помимо уменьшения логирования, которое само по себе снижает накладные расходы, есть несколько других моментов:

Размер батча (batch_size): Вы уже уменьшили его до 16, что хорошо для стабильности, но может немного замедлить обучение. Если после всех изменений стабильность хорошая, можете попробовать увеличить до 32 или 64.
parallel_iterations в XLSTMLayer: Вы установили parallel_iterations=1. Для GPU/TPU увеличение этого параметра может ускорить обучение, но может вызвать ошибки, если операции не независимы. В вашем случае, XLSTMLayer имеет последовательную логику, поэтому parallel_iterations=1 является правильным выбором.
Оптимизация feature_engineering: Кэширование ATR с lru_cache - отличное решение. Убедитесь, что все операции в feature_engineering максимально векторизованы (используют pandas/numpy напрямую, а не циклы Python). Ваш код уже хорошо векторизован.
Отключение XLA (если вызывает проблемы): В train_model.py у вас есть закомментированная строка os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'. Если после всех изменений возникают проблемы с производительностью или ошибками, попробуйте раскомментировать ее. Но обычно XLA ускоряет работу.


🔍 Общий анализ и возможные улучшения:
Файл: xlstm_memory_cell.py

Состояние: Отлично, все правильно. maximum_iterations для tf.while_loop - это критически важно для XLA.

Файл: xlstm_rl_model.py

Состояние: Отлично. Дубль компиляции убран.

Файл: feature_engineering.py

Состояние: Хорошо. Использование try-except для каждого индикатора делает его очень устойчивым.
Улучшение (необязательно, для скорости): Если calculate_vsa_features и detect_candlestick_patterns вызывают проблемы с производительностью, можно также рассмотреть их частичное кэширование или оптимизацию. Но обычно talib и векторизованные операции pandas достаточно быстры.

Файл: hybrid_decision_maker.py

Состояние: Хорошо. Логика принятия решений комплексная и учитывает все модели.
Улучшение (логика _make_final_decision): Если после балансировки классов xLSTM будет давать более разнообразные предсказания, вы можете более тонко настроить веса и пороги в _make_final_decision. Например, если RL всегда очень консервативен, а xLSTM более агрессивен.

Ваш текущий _make_final_decision сначала проверяет xlstm_conf < threshold, что может привести к тому, что RL решение будет использоваться чаще, если xLSTM все еще недостаточно уверен. Это может быть как преимуществом, так и недостатком, в зависимости от желаемой агрессивности.



Файл: market_regime_detector.py

Состояние: Хорошо. Использование xLSTM предсказаний как признаков для режима - интересная идея.

Файл: parameter_optimizer.py

Состояние: Хорошо. Автоматическая оптимизация параметров - мощная функция.
Улучшение: _objective_function сейчас использует средний Sharpe для похожих параметров. В реальной симуляции можно было бы запускать мини-бэктест с новыми параметрами, но это очень ресурсоемко. Текущий подход — это компромисс.

Файл: run_live_trading.py

Состояние: Хорошо. Общий цикл торговли выглядит надежным.
Улучшение (опционально, для чистоты логов):

В manage_active_positions: Вы уже ограничили вывод до 5 позиций. Если хотите еще меньше, можно изменить displayed_positions = positions_items[:5] на [:1] или [:0] для полного отключения.
В run_trading_loop: print в loop_counter % 10 можно настроить, чтобы выводил только ключевые метрики.



Файл: trade_logger.py


Состояние: Хорошо. FIELDNAMES и log_trade обеспечивают надежное логирование.


Улучшение: Вы определили log_enhanced_trade_with_quality_metrics и calculate_signal_quality, но они не используются в run_live_trading.py. Вам следует заменить вызов trade_logger.log_trade на trade_logger.log_enhanced_trade_with_quality_metrics в run_live_trading.py для использования этих новых метрик.
Изменение в run_live_trading.py:

Найдите log_enhanced_trade(...)
Замените его на trade_logger.log_enhanced_trade_with_quality_metrics(...)



Файл: advanced_simulation_engine.py

Состояние: Хорошо. Комплексная симуляция с разными стратегиями.
Улучшение: Может быть очень медленной из-за множества симуляций. Можно распараллелить символы или стратегии, если доступна многопроцессорность.


📝 Резюме изменений:


train_model.py:

Ослаблены пороги для создания меток BUY/SELL в prepare_xlstm_rl_data.
Добавлен опциональный блок для принудительной балансировки классов в prepare_xlstm_rl_data.



xlstm_rl_model.py:

Добавлено взвешивание классов (class_weight) в train() для борьбы с дисбалансом.
verbose в model.fit() изменен на 0.
save_freq для xlstm_checkpoint_epoch изменен на 'epoch', verbose на 0.



rl_agent.py:

verbose в PPO/SAC и progress_bar в learn() изменены на 0/False.



run_live_trading.py:

(Предложено) Заменить log_enhanced_trade на trade_logger.log_enhanced_trade_with_quality_metrics для более полной аналитики.



Эти изменения должны значительно улучшить баланс классов, уменьшить время обучения эпохи и сократить объем логирования, делая его более информативным и менее навязчивым.