

❌ ЧТО НЕ РЕАЛИЗОВАНО (КРИТИЧНО):
1. Главная проблема: НЕ ИЗМЕНЕНЫ УСЛОВИЯ ГЕНЕРАЦИИ МЕТОК
В train_model.py НЕ РЕАЛИЗОВАНЫ ключевые изменения:
❌ НЕ ИЗМЕНЕНО:
# СТАРЫЙ КОД (все еще в файле):
df['vsa_buy_score'] = (
    0.3 * (df['vsa_no_supply'] == 1) +
    0.3 * (df['vsa_stopping_volume'] == 1) +
    0.4 * (df['vsa_strength'] > 0.1)  # Слишком мягкие условия
)

buy_condition = (
    (df['future_return'] > df['dynamic_future_threshold']) &
    (df['vsa_buy_score'] > 0.2)  # Слишком низкий порог
)

2. ❌ НЕ ИЗМЕНЕНЫ ВЕСА КЛАССОВ
❌ НЕ РЕАЛИЗОВАНО:
# СТАРЫЙ КОД (все еще в файле):
additional_weight_multiplier = 1.0  # Не изменен
# НЕТ увеличения веса HOLD класса

3. ❌ НЕ ИЗМЕНЕНА ПЕРЕКЛАССИФИКАЦИЯ
Блок переклассификации HOLD не изменен согласно рекомендациям.
🚨 СРОЧНЫЕ ИСПРАВЛЕНИЯ:
Файл: train_model.py - строка ~180
ЗАМЕНИТЬ:
# НОВЫЙ КОД - БОЛЕЕ СТРОГИЕ УСЛОВИЯ ДЛЯ BUY/SELL
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

Файл: train_model.py - строка ~350
ЗАМЕНИТЬ:
# НОВЫЙ КОД - БАЛАНСИРУЕМ ВЕСА ПРАВИЛЬНО
# Проблема: HOLD имеет слишком низкий recall, нужно УВЕЛИЧИТЬ его вес

# Уменьшаем веса BUY/SELL (они слишком агрессивные)
weight_reduction_factor = 0.6  # Уменьшаем веса BUY/SELL
if 0 in class_weight_dict:
    class_weight_dict[0] *= weight_reduction_factor
if 1 in class_weight_dict:
    class_weight_dict[1] *= weight_reduction_factor

# УВЕЛИЧИВАЕМ вес HOLD (чтобы модель чаще его предсказывала)
hold_boost_factor = 1.8  # Увеличиваем вес HOLD
if 2 in class_weight_dict:
    class_weight_dict[2] *= hold_boost_factor

print(f"📊 ИСПРАВЛЕННЫЕ веса классов: {class_weight_dict}")

Файл: train_model.py - строка ~220
ЗАМЕНИТЬ блок переклассификации:
# НОВЫЙ КОД - УМЕНЬШАЕМ ПЕРЕКЛАССИФИКАЦИЮ
# Теперь НЕ переклассифицируем, если HOLD составляет меньше 70%
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

Файл: train_model.py - изменить Focal Loss (строка ~50)
ЗАМЕНИТЬ:
# НОВЫЙ КОД - МЕНЕЕ АГРЕССИВНЫЙ FOCAL LOSS
def categorical_focal_loss(gamma=1.5, alpha=0.25):  # Уменьшаем gamma, увеличиваем alpha
    """
    Менее агрессивная версия Focal Loss
    gamma=1.5 - умеренная фокусировка на сложных примерах
    alpha=0.25 - стандартный вес
    """
    def focal_loss_fixed(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        y_true = tf.cast(y_true, tf.float32)
        
        # ИСПРАВЛЕННЫЕ класс-специфичные веса
        # Уменьшаем веса BUY/SELL, увеличиваем вес HOLD
        class_weights = tf.constant([0.8, 0.8, 1.4])  # BUY, SELL, HOLD
        weights = tf.reduce_sum(class_weights * y_true, axis=-1, keepdims=True)
        
        cross_entropy = -y_true * K.log(y_pred)
        loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy * weights
        
        return K.sum(loss, axis=-1)
    
    return focal_loss_fixed

Приоритет реализации:

🔥 КРИТИЧНО: Изменить условия генерации BUY/SELL меток (более строгие пороги)
🔥 КРИТИЧНО: Исправить веса классов (увеличить вес HOLD)
🔥 КРИТИЧНО: Изменить переклассификацию (добавлять HOLD вместо убавления)
🔴 ВАЖНО: Изменить параметры Focal Loss

После этих изменений модель должна:

Чаще предсказывать HOLD (увеличится recall для HOLD)
Реже генерировать ложные торговые сигналы
Улучшить общую точность валидации

Все остальные изменения уже корректно реализованы в коде!