
Инструкции по реализации:
Файл 1: train_model.py
1. Корректировка весов классов (усилить BUY и HOLD, ослабить SELL):
Местоположение: В функции train_xlstm_rl_system, в блоке вычисления class_weight_dict (строка ~350).
ЗАМЕНИТЬ:
# СТАРЫЙ КОД (возвращаем BUY/SELL на равные значения)
    if 0 in class_weight_dict:
        class_weight_dict[0] *= 1.5  # ИЗМЕНЕНО: Возвращаем BUY к 1.5
    if 1 in class_weight_dict:
        class_weight_dict[1] *= 1.5  # ИЗМЕНЕНО: Возвращаем SELL к 1.5
    
    if 2 in class_weight_dict:
        class_weight_dict[2] *= 0.7  # ИЗМЕНЕНО: Возвращаем HOLD к 0.7

НА НОВЫЙ КОД (усилить BUY и HOLD, уменьшить SELL):
# НОВЫЙ КОД - Корректируем веса классов (усилить BUY и HOLD, ослабить SELL)
    if 0 in class_weight_dict:
        class_weight_dict[0] *= 2.5  # ИЗМЕНЕНО: Значительно увеличиваем вес BUY (с 1.5 до 2.5)
    if 1 in class_weight_dict:
        class_weight_dict[1] *= 0.8  # ИЗМЕНЕНО: Значительно уменьшаем вес SELL (с 1.5 до 0.8)
    
    if 2 in class_weight_dict:
        class_weight_dict[2] *= 1.2  # ИЗМЕНЕНО: Увеличиваем вес HOLD (с 0.7 до 1.2)

Объяснение: Это более агрессивная попытка сбалансировать предсказания. Мы даем очень сильный стимул для BUY, ослабляем SELL и значительно усиливаем HOLD, чтобы модель не игнорировала его.
2. Интеграция механизма анализа важности признаков в DetailedProgressCallback:
Мы добавим логику, которая будет анализировать предсказания модели на валидационном наборе и выводить, какие признаки (индикаторы/паттерны) были активны в этих предсказаниях. Это потребует передачи X_val_to_model и feature_cols в DetailedProgressCallback.
Местоположение: Внутри класса DetailedProgressCallback, метод on_epoch_end (строка ~400).
ДОБАВИТЬ В __init__ DetailedProgressCallback:
# НОВЫЙ КОД - Добавляем X_val и feature_cols в __init__
    class DetailedProgressCallback(tf.keras.callbacks.Callback):
        def __init__(self, X_val, feature_cols): # ИЗМЕНЕНО: Добавлены X_val, feature_cols
            super().__init__()
            self.X_val = X_val
            self.feature_cols = feature_cols
            # ... остальной код

Местоположение: Внутри метода on_epoch_end DetailedProgressCallback, после вывода метрик по классам (строка ~430).
ДОБАВИТЬ НОВЫЙ КОД:
# НОВЫЙ КОД - Анализ важности признаков
                if epoch % 5 == 0 and self.X_val is not None and self.feature_cols is not None: # Анализируем каждые 5 эпох
                    print("\n📈 ТОП-10 ВЛИЯТЕЛЬНЫХ ПРИЗНАКОВ (на валидации):")
                    # Получаем предсказания модели на валидационном наборе
                    val_preds = self.model.predict(self.X_val, verbose=0)
                    predicted_classes = np.argmax(val_preds, axis=1)

                    # Для каждого класса, найдем признаки, которые чаще всего были активны
                    class_influence = {0: [], 1: [], 2: []} # BUY, SELL, HOLD

                    for class_id in range(3):
                        # Выбираем только те данные валидации, где модель предсказала этот класс
                        class_indices = np.where(predicted_classes == class_id)[0]
                        if len(class_indices) == 0:
                            continue

                        # Берем соответствующие признаки из X_val
                        # Усредняем активации признаков для этого класса
                        active_features = self.X_val[class_indices, -1, :] # Берем признаки последней свечи в последовательности
                        
                        # Определяем "активность" признака (например, если его значение > 0.5 или просто его значение)
                        # Для простоты, мы будем считать среднее значение признака
                        avg_active_features = np.mean(active_features, axis=0)

                        # Создаем пары (значение, имя_признака)
                        feature_scores = [(avg_active_features[i], self.feature_cols[i]) for i in range(len(self.feature_cols))]
                        
                        # Сортируем по абсолютному значению
                        feature_scores.sort(key=lambda x: abs(x[0]), reverse=True)
                        
                        # Выводим топ-10
                        print(f"  Для класса {self.class_names[class_id]}:")
                        for score, name in feature_scores[:10]:
                            print(f"    - {name}: {score:.4f}")

3. Передача X_val_to_model и feature_cols в DetailedProgressCallback:
Местоположение: В списке callbacks в функции train_xlstm_rl_system (строка ~400).
ЗАМЕНИТЬ:
# СТАРЫЙ КОД
        DetailedProgressCallback(),

НА НОВЫЙ КОД:
# НОВЫЙ КОД
        DetailedProgressCallback(X_val_to_model, feature_cols), # ИЗМЕНЕНО: Передаем X_val_to_model и feature_cols

Файл 3: trading_env.py
1. Корректировка _calculate_advanced_reward для борьбы с дисбалансом:
Мы ослабили SELL и усилили HOLD в весах классов, но функция наград для RL-агента также должна отражать это.
Местоположение: Внутри метода _calculate_advanced_reward (строка ~130).
ЗАМЕНИТЬ:
# СТАРЫЙ КОД
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

НА НОВЫЙ КОД (усиление BUY, ослабление SELL):
# НОВЫЙ КОД - Корректируем функцию наград для RL (усилить BUY, ослабить SELL)
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

Объяснение: Мы корректируем штрафы за overtrading в RL-среде. Теперь слабый BUY-сигнал наказывается сильнее, а слабый SELL-сигнал наказывается меньше. Это должно помочь RL-агенту быть более склонным к BUY. Также, я сделал пороги для MACD_hist более строгими.
