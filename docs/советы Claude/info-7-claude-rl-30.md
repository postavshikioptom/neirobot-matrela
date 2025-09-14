
Файл 1: feature_engineering.py
Этот файл отвечает за расчет всех технических индикаторов и определение паттернов, а также извлечение их признаков.


Добавление вспомогательных функций для анализа свечей:

Местоположение: После функции is_volume_spike (строка ~60).
ДОБАВИТЬ НОВЫЙ КОД:
# НОВЫЙ КОД - Вспомогательные функции для анализа свечей
def get_body_size(open_p, close_p):
    return abs(close_p - open_p)

def get_total_range(high_p, low_p):
    return high_p - low_p

def get_upper_shadow(open_p, high_p, close_p):
    return high_p - np.maximum(open_p, close_p)

def get_lower_shadow(open_p, low_p, close_p):
    return np.minimum(open_p, close_p) - low_p

def is_small_body(open_p, close_p, high_p, low_p, threshold_factor=0.2):
    body = get_body_size(open_p, close_p)
    total_range = get_total_range(high_p, low_p)
    return (body < total_range * threshold_factor).astype(int)

def is_long_shadow(shadow_size, body_size, threshold_factor=2.0):
    return (shadow_size > body_size * threshold_factor).astype(int)





Добавление новых функций для извлечения признаков паттернов:

Местоположение: После функции marubozu_features (строка ~200).
ДОБАВИТЬ НОВЫЙ КОД:
# НОВЫЙ КОД - Функции для извлечения признаков бычьих паттернов
def inverted_hammer_features(df):
    atr = calculate_atr(df['high'], df['low'], df['close'])
    support, _ = find_support_resistance(df['low'], df['high'])
    volume_ratio = calculate_volume_ratio(df['volume'])
    
    features = pd.DataFrame(index=df.index)
    features['ih_f_small_body'] = is_small_body(df['open'], df['close'], df['high'], df['low'])
    features['ih_f_long_upper_shadow'] = is_long_shadow(get_upper_shadow(df['open'], df['high'], df['close']), get_body_size(df['open'], df['close']))
    features['ih_f_on_support'] = is_on_level(df['low'], support, atr, threshold=0.5) # Немного шире для поддержки
    features['ih_f_vol_confirm'] = is_volume_spike(volume_ratio, 1.2)
    return features

def dragonfly_doji_features(df):
    atr = calculate_atr(df['high'], df['low'], df['close'])
    support, _ = find_support_resistance(df['low'], df['high'])
    volume_ratio = calculate_volume_ratio(df['volume'])
    
    features = pd.DataFrame(index=df.index)
    features['dd_f_long_lower_shadow'] = is_long_shadow(get_lower_shadow(df['open'], df['low'], df['close']), get_body_size(df['open'], df['close']), threshold_factor=3.0) # Очень длинная тень
    features['dd_f_on_support'] = is_on_level(df['close'], support, atr, threshold=0.5)
    features['dd_f_vol_confirm'] = is_volume_spike(volume_ratio, 1.5)
    return features

def bullish_pin_bar_features(df):
    atr = calculate_atr(df['high'], df['low'], df['close'])
    support, _ = find_support_resistance(df['low'], df['high'])
    volume_ratio = calculate_volume_ratio(df['volume'])
    
    features = pd.DataFrame(index=df.index)
    features['bpb_f_small_body'] = is_small_body(df['open'], df['close'], df['high'], df['low'])
    features['bpb_f_long_lower_wick'] = is_long_shadow(get_lower_shadow(df['open'], df['low'], df['close']), get_body_size(df['open'], df['close']), threshold_factor=2.5)
    features['bpb_f_on_support'] = is_on_level(df['close'], support, atr, threshold=0.5)
    features['bpb_f_vol_confirm'] = is_volume_spike(volume_ratio, 1.2)
    return features

def bullish_kicker_features(df):
    volume_ratio = calculate_volume_ratio(df['volume'])
    
    features = pd.DataFrame(index=df.index)
    # Бычий кикер - это 2 свечи, поэтому признаки будут относиться к текущей (второй) свече
    # и предыдущей (первой) свече, но мы их агрегируем для текущего индекса.
    # Для упрощения, фокусируемся на характеристиках второй свечи в паттерне
    features['bk_f_strong_bullish_body'] = (get_body_size(df['open'], df['close']) > calculate_atr(df['high'], df['low'], df['close']) * 0.7).astype(int) # Большое тело
    features['bk_f_vol_confirm'] = is_volume_spike(volume_ratio, 1.8) # Высокий объем
    # Признак "гэп вверх" сложно определить без доступа к предыдущей свече в этой функции,
    # но talib.CDLBULLISHKICKING уже это учитывает.
    return features

def bullish_belt_hold_features(df):
    atr = calculate_atr(df['high'], df['low'], df['close'])
    support, _ = find_support_resistance(df['low'], df['high'])
    volume_ratio = calculate_volume_ratio(df['volume'])
    
    features = pd.DataFrame(index=df.index)
    features['bbh_f_long_body'] = (get_body_size(df['open'], df['close']) > atr * 0.8).astype(int) # Длинное тело
    features['bbh_f_open_at_low'] = (abs(df['open'] - df['low']) / get_total_range(df['high'], df['low']) < 0.1).astype(int) # Открытие близко к минимуму
    features['bbh_f_vol_confirm'] = is_volume_spike(volume_ratio, 1.3)
    features['bbh_f_on_support'] = is_on_level(df['close'], support, atr, threshold=0.5)
    return features





Обновление функции add_pattern_features для включения новых паттернов:


Местоположение: Функция add_pattern_features, список feature_columns (строка ~205).


ЗАМЕНИТЬ: (добавить новые строки)
# СТАРЫЙ КОД
    feature_columns = [
        'hammer_f_on_support', 'hammer_f_vol_spike',
        'hangingman_f_on_res', 'hangingman_f_vol_spike',
        'engulfing_f_strong', 'engulfing_f_vol_confirm',
        'doji_f_high_vol', 'doji_f_high_atr',
        'shootingstar_f_on_res',
        'marubozu_f_strong_body', 'marubozu_f_vol_confirm', 'marubozu_f_bullish'
    ]



НА НОВЫЙ КОД:
# НОВЫЙ КОД - Добавляем колонки для новых паттернов
    feature_columns = [
        'hammer_f_on_support', 'hammer_f_vol_spike',
        'hangingman_f_on_res', 'hangingman_f_vol_spike',
        'engulfing_f_strong', 'engulfing_f_vol_confirm',
        'doji_f_high_vol', 'doji_f_high_atr',
        'shootingstar_f_on_res',
        'marubozu_f_strong_body', 'marubozu_f_vol_confirm', 'marubozu_f_bullish',
        # НОВЫЕ БЫЧЬИ ПАТТЕРНЫ
        'ih_f_small_body', 'ih_f_long_upper_shadow', 'ih_f_on_support', 'ih_f_vol_confirm',
        'dd_f_long_lower_shadow', 'dd_f_on_support', 'dd_f_vol_confirm',
        'bpb_f_small_body', 'bpb_f_long_lower_wick', 'bpb_f_on_support', 'bpb_f_vol_confirm',
        'bk_f_strong_bullish_body', 'bk_f_vol_confirm',
        'bbh_f_long_body', 'bbh_f_open_at_low', 'bbh_f_vol_confirm', 'bbh_f_on_support'
    ]



Местоположение: В функции add_pattern_features, после блока marubozu_mask (строка ~280).


ДОБАВИТЬ НОВЫЙ КОД:
# НОВЫЙ КОД - Расчет признаков для новых паттернов
inverted_hammer_mask = df['CDLINVERTEDHAMMER'] != 0
if not df[inverted_hammer_mask].empty:
    df.loc[inverted_hammer_mask, ['ih_f_small_body', 'ih_f_long_upper_shadow', 'ih_f_on_support', 'ih_f_vol_confirm']] = inverted_hammer_features(df[inverted_hammer_mask]).values

dragonfly_doji_mask = df['CDLDRAGONFLYDOJI'] != 0
if not df[dragonfly_doji_mask].empty:
    df.loc[dragonfly_doji_mask, ['dd_f_long_lower_shadow', 'dd_f_on_support', 'dd_f_vol_confirm']] = dragonfly_doji_features(df[dragonfly_doji_mask]).values

# Для бычьего пин-бара используем CDLDRAGONFLYDOJI или CDLHAMMER, если нет специфического TA-Lib
# Если CDLDRAGONFLYDOJI или CDLHAMMER, и это бычья свеча, то это может быть пин-бар
bullish_pin_bar_mask = (df['CDLDRAGONFLYDOJI'] != 0) | ((df['CDLHAMMER'] != 0) & (df['close'] > df['open']))
if not df[bullish_pin_bar_mask].empty:
    df.loc[bullish_pin_bar_mask, ['bpb_f_small_body', 'bpb_f_long_lower_wick', 'bpb_f_on_support', 'bpb_f_vol_confirm']] = bullish_pin_bar_features(df[bullish_pin_bar_mask]).values

bullish_kicker_mask = df['CDLBULLISHKICKING'] != 0
if not df[bullish_kicker_mask].empty:
    df.loc[bullish_kicker_mask, ['bk_f_strong_bullish_body', 'bk_f_vol_confirm']] = bullish_kicker_features(df[bullish_kicker_mask]).values
    
bullish_belt_hold_mask = df['CDLBELTHOLD'] != 0
if not df[bullish_belt_hold_mask].empty:
    df.loc[bullish_belt_hold_mask, ['bbh_f_long_body', 'bbh_f_open_at_low', 'bbh_f_vol_confirm', 'bbh_f_on_support']] = bullish_belt_hold_features(df[bullish_belt_hold_mask]).values



Местоположение: В функции add_pattern_features, блок "Combine features into final scores" (строка ~300).


ЗАМЕНИТЬ: (добавить новые строки)
# СТАРЫЙ КОД
    df['shootingstar_f'] = df['shootingstar_f_on_res'].astype(int)
    df['marubozu_f'] = (df['marubozu_f_strong_body'] + df['marubozu_f_vol_confirm'] + df['marubozu_f_bullish']).astype(int)



НА НОВЫЙ КОД:
# НОВЫЙ КОД - Комбинированные признаки для новых паттернов
    df['shootingstar_f'] = df['shootingstar_f_on_res'].astype(int)
    df['marubozu_f'] = (df['marubozu_f_strong_body'] + df['marubozu_f_vol_confirm'] + df['marubozu_f_bullish']).astype(int)
    
    df['inverted_hammer_f'] = (df['ih_f_small_body'] + df['ih_f_long_upper_shadow'] + df['ih_f_on_support'] + df['ih_f_vol_confirm']).astype(int)
    df['dragonfly_doji_f'] = (df['dd_f_long_lower_shadow'] + df['dd_f_on_support'] + df['dd_f_vol_confirm']).astype(int)
    df['bullish_pin_bar_f'] = (df['bpb_f_small_body'] + df['bpb_f_long_lower_wick'] + df['bpb_f_on_support'] + df['bpb_f_vol_confirm']).astype(int)
    df['bullish_kicker_f'] = (df['bk_f_strong_bullish_body'] + df['bk_f_vol_confirm']).astype(int)
    df['bullish_belt_hold_f'] = (df['bbh_f_long_body'] + df['bbh_f_open_at_low'] + df['bbh_f_vol_confirm'] + df['bbh_f_on_support']).astype(int)





Обновление функции detect_candlestick_patterns для новых паттернов TA-Lib:


Местоположение: Функция detect_candlestick_patterns, блок "Calculate base patterns" (строка ~360).


ЗАМЕНИТЬ: (добавить новые строки)
# СТАРЫЙ КОД
    df['CDLMARUBOZU'] = talib.CDLMARUBOZU(open_prices, high_prices, low_prices, close_prices)



НА НОВЫЙ КОД:
# НОВЫЙ КОД - Добавляем новые TA-Lib паттерны
    df['CDLMARUBOZU'] = talib.CDLMARUBOZU(open_prices, high_prices, low_prices, close_prices)
    # НОВЫЕ БЫЧЬИ ПАТТЕРНЫ
    df['CDLINVERTEDHAMMER'] = talib.CDLINVERTEDHAMMER(open_prices, high_prices, low_prices, close_prices)
    df['CDLDRAGONFLYDOJI'] = talib.CDLDRAGONFLYDOJI(open_prices, high_prices, low_prices, close_prices)
    # Для Bullish Pin Bar нет прямого TA-Lib, используем комбинацию или CDLHAMMER
    df['CDLBULLISHKICKING'] = talib.CDLBULLISHKICKING(open_prices, high_prices, low_prices, close_prices)
    df['CDLBELTHOLD'] = talib.CDLBELTHOLD(open_prices, high_prices, low_prices, close_prices)



Местоположение: Функция detect_candlestick_patterns, список pattern_cols (строка ~375).


ЗАМЕНИТЬ: (добавить новые строки)
# СТАРЫЙ КОД
    pattern_cols = [
        'CDLHAMMER', 'CDLENGULFING', 'CDLDOJI', 'CDLSHOOTINGSTAR',
        'CDLHANGINGMAN', 'CDLMARUBOZU'
    ]



НА НОВЫЙ КОД:
# НОВЫЙ КОД - Добавляем новые паттерны в список
    pattern_cols = [
        'CDLHAMMER', 'CDLENGULFING', 'CDLDOJI', 'CDLSHOOTINGSTAR',
        'CDLHANGINGMAN', 'CDLMARUBOZU',
        # НОВЫЕ БЫЧЬИ ПАТТЕРНЫ
        'CDLINVERTEDHAMMER', 'CDLDRAGONFLYDOJI', 'CDLBULLISHKICKING', 'CDLBELTHOLD'
    ]



Местоположение: Функция detect_candlestick_patterns, список feature_cols (строка ~385).


ЗАМЕНИТЬ: (добавить новые строки)
# СТАРЫЙ КОД
    feature_cols = [
        'hammer_f', 'hangingman_f', 'engulfing_f', 'doji_f',
        'shootingstar_f', 'marubozu_f'
    ]



НА НОВЫЙ КОД:
# НОВЫЙ КОД - Добавляем новые признаки паттернов в список
    feature_cols = [
        'hammer_f', 'hangingman_f', 'engulfing_f', 'doji_f',
        'shootingstar_f', 'marubozu_f',
        # НОВЫЕ БЫЧЬИ ПАТТЕРНЫ
        'inverted_hammer_f', 'dragonfly_doji_f', 'bullish_pin_bar_f', 'bullish_kicker_f', 'bullish_belt_hold_f'
    ]






Файл 2: train_model.py
Этот файл готовит данные и обучает модель.


Обновление списка feature_cols:

Местоположение: Функция prepare_xlstm_rl_data, список feature_cols (строка ~70).
ЗАМЕНИТЬ: (добавить новые строки)
# СТАРЫЙ КОД
    feature_cols = [
        # Технические индикаторы
        'RSI_14', 'MACD_12_26_9', 'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0',
        'ADX_14', 'STOCHk_14_3_3', 'STOCHd_14_3_3',
        'ATR_14', # <--- ДОБАВЛЕНО
        # Паттерны
        'CDLHAMMER', 'CDLENGULFING', 'CDLDOJI', 'CDLSHOOTINGSTAR',
        'CDLHANGINGMAN', 'CDLMARUBOZU',
        'is_event' # <--- ДОБАВЛЕНО: Новый признак
    ]


НА НОВЫЙ КОД:
# НОВЫЙ КОД - Добавляем новые паттерны и их признаки в feature_cols
    feature_cols = [
        # Технические индикаторы
        'RSI_14', 'MACD_12_26_9', 'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0',
        'ADX_14', 'STOCHk_14_3_3', 'STOCHd_14_3_3', 'ATR_14',
        # Базовые паттерны TA-Lib
        'CDLHAMMER', 'CDLENGULFING', 'CDLDOJI', 'CDLSHOOTINGSTAR',
        'CDLHANGINGMAN', 'CDLMARUBOZU',
        # НОВЫЕ БЫЧЬИ ПАТТЕРНЫ TA-Lib
        'CDLINVERTEDHAMMER', 'CDLDRAGONFLYDOJI', 'CDLBULLISHKICKING', 'CDLBELTHOLD',
        # Комбинированные признаки паттернов
        'hammer_f', 'hangingman_f', 'engulfing_f', 'doji_f',
        'shootingstar_f', 'marubozu_f',
        # НОВЫЕ КОМБИНИРОВАННЫЕ ПРИЗНАКИ БЫЧЬИХ ПАТТЕРНОВ
        'inverted_hammer_f', 'dragonfly_doji_f', 'bullish_pin_bar_f', 'bullish_kicker_f', 'bullish_belt_hold_f',
        'is_event'
    ]





Корректировка весов классов (возвращаем BUY/SELL на равные):

Местоположение: В функции train_xlstm_rl_system, в блоке вычисления class_weight_dict (строка ~350).
ЗАМЕНИТЬ: (текущий код, который увеличивал BUY до 2.0)
# СТАРЫЙ КОД
    if 0 in class_weight_dict:
        class_weight_dict[0] *= 2.0  # ИЗМЕНЕНО: Еще сильнее увеличиваем вес BUY (с 1.5 до 2.0)
    if 1 in class_weight_dict:
        class_weight_dict[1] *= 1.0  # ИЗМЕНЕНО: Уменьшаем вес SELL (с 1.5 до 1.0), чтобы не доминировал
    
    if 2 in class_weight_dict:
        class_weight_dict[2] *= 0.8  # ИЗМЕНЕНО: Немного увеличиваем вес HOLD (с 0.7 до 0.8), чтобы не был слишком подавлен


НА НОВЫЙ КОД (возвращаем BUY/SELL на равные значения):
# НОВЫЙ КОД - Возвращаем равные веса для BUY/SELL
    if 0 in class_weight_dict:
        class_weight_dict[0] *= 1.5  # ИЗМЕНЕНО: Возвращаем BUY к 1.5
    if 1 in class_weight_dict:
        class_weight_dict[1] *= 1.5  # ИЗМЕНЕНО: Возвращаем SELL к 1.5
    
    if 2 in class_weight_dict:
        class_weight_dict[2] *= 0.7  # ИЗМЕНЕНО: Возвращаем HOLD к 0.7

Объяснение: Теперь, когда мы добавили больше бычьих паттернов, мы предполагаем, что модель будет естественным образом предсказывать больше BUY. Поэтому возвращаем веса BUY/SELL к равным, чтобы не создавать искусственный дисбаланс, но сохраняем их немного выше веса HOLD, чтобы модель не "ленилась".




Файл 3: hybrid_decision_maker.py
Этот файл управляет гибридными решениями.


Обновление списка feature_columns:

Местоположение: В конструкторе __init__, где определяется self.feature_columns.
ЗАМЕНИТЬ: (добавить новые строки)
# СТАРЫЙ КОД
    self.feature_columns = feature_columns # Этот feature_columns приходит из train_model.py


НА НОВЫЙ КОД: (просто убедитесь, что feature_columns здесь соответствует обновленному списку из train_model.py. Поскольку он передается как аргумент, явных изменений здесь не нужно, но важно понимать, что этот список будет содержать новые признаки.)



Обновление get_decision_explanation:

Местоположение: Метод get_decision_explanation, в блоке VSA сигналы:.
ЗАМЕНИТЬ:
# СТАРЫЙ КОД
        # VSA сигналы отключены
        - VSA сигналы: Отключены


НА НОВЫЙ КОД:
# НОВЫЙ КОД - Обновленное объяснение без VSA
        - Анализ паттернов и индикаторов: Активирован






Файл 4: market_regime_detector.py
Этот файл определяет рыночные режимы.

Обновление списка xlstm_feature_columns:

Местоположение: Метод set_xlstm_context.
НА НОВЫЙ КОД: (просто убедитесь, что xlstm_feature_columns здесь соответствует обновленному списку из train_model.py.)




Файл 5: trading_env.py
Этот файл определяет среду для RL-агента.

Обновление списка self.feature_columns:

Местоположение: В методе reset, где определяется self.feature_columns.
ЗАМЕНИТЬ: (добавить новые строки)
# СТАРЫЙ КОД
    self.feature_columns = [
        'RSI_14', 'MACD_12_26_9', 'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0',
        'ADX_14', 'STOCHk_14_3_3', 'STOCHd_14_3_3',
        'ATR_14', # <--- ДОБАВЛЕНО
        'CDLHAMMER', 'CDLENGULFING', 'CDLDOJI', 'CDLSHOOTINGSTAR',
        'CDLHANGINGMAN', 'CDLMARUBOZU',
    ]


НА НОВЫЙ КОД:
# НОВЫЙ КОД - Добавляем новые паттерны и их признаки в self.feature_columns
    self.feature_columns = [
        # Технические индикаторы
        'RSI_14', 'MACD_12_26_9', 'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0',
        'ADX_14', 'STOCHk_14_3_3', 'STOCHd_14_3_3', 'ATR_14',
        # Базовые паттерны TA-Lib
        'CDLHAMMER', 'CDLENGULFING', 'CDLDOJI', 'CDLSHOOTINGSTAR',
        'CDLHANGINGMAN', 'CDLMARUBOZU',
        # НОВЫЕ БЫЧЬИ ПАТТЕРНЫ TA-Lib
        'CDLINVERTEDHAMMER', 'CDLDRAGONFLYDOJI', 'CDLBULLISHKICKING', 'CDLBELTHOLD',
        # Комбинированные признаки паттернов
        'hammer_f', 'hangingman_f', 'engulfing_f', 'doji_f',
        'shootingstar_f', 'marubozu_f',
        # НОВЫЕ КОМБИНИРОВАННЫЕ ПРИЗНАКИ БЫЧЬИХ ПАТТЕРНОВ
        'inverted_hammer_f', 'dragonfly_doji_f', 'bullish_pin_bar_f', 'bullish_kicker_f', 'bullish_belt_hold_f',
        # Признак is_event не используется напрямую в RL-среде, но может быть добавлен
        # 'is_event' # Если is_event используется в RL-среде, добавьте его здесь
    ]



