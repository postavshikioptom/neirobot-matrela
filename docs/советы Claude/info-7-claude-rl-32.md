Теперь, давай обновим код, чтобы интегрировать этот паттерн и его признаки, а также вернуть веса BUY и SELL к равным значениям.

Файл 1: feature_engineering.py
Этот файл отвечает за расчет всех технических индикаторов и определение паттернов, а также извлечение их признаков.


Модификация функции marubozu_features для включения "kicker-эффекта" и переименование:

Мы будем использовать CDLMARUBOZU как основной детектор, а затем добавим твои фильтры.
Местоположение: Найди функцию marubozu_features (строка ~200).
ЗАМЕНИТЬ ВЕСЬ ЭТОТ МЕТОД:
# СТАРЫЙ КОД (весь метод marubozu_features)
def marubozu_features(df):
    """Признаки для паттерна Marubozu"""
    atr = calculate_atr(df['high'], df['low'], df['close'])
    volume_ratio = calculate_volume_ratio(df['volume'])
    
    features = pd.DataFrame(index=df.index)
    # Сила тренда - насколько сильно тело свечи по отношению к ATR
    body_size = (df['close'] - df['open']).abs()
    features['marubozu_f_strong_body'] = (body_size > atr * 0.5).astype(int)
    # Подтверждение объемом
    features['marubozu_f_vol_confirm'] = is_volume_spike(volume_ratio, 1.3)
    # Направление тренда (бычий/медвежий)
    features['marubozu_f_bullish'] = (df['close'] > df['open']).astype(int)
    return features


НА НОВЫЙ КОД (переименованная функция bullish_marubozu_features с новыми фильтрами):
# НОВЫЙ КОД - Функции для извлечения признаков бычьего Marubozu (с kicker-эффектом)
def bullish_marubozu_features(df):
    atr = calculate_atr(df['high'], df['low'], df['close'])
    # support, _ = find_support_resistance(df['low'], df['high']) # Можно добавить, если нужно
    volume_ratio = calculate_volume_ratio(df['volume'])
    
    features = pd.DataFrame(index=df.index)
    
    # 1. Сильное бычье тело
    body_size = (df['close'] - df['open']).abs()
    features['bm_f_strong_body'] = (body_size > atr * 0.7).astype(int) # Большое тело (увеличено с 0.5 до 0.7)
    
    # 2. Подтверждение объемом
    features['bm_f_vol_confirm'] = is_volume_spike(volume_ratio, 1.5) # Высокий объем (увеличено с 1.3 до 1.5)
    
    # 3. Открытие > закрытия предыдущей свечи (Kicker-эффект)
    # Проверяем, что предыдущая свеча существует
    prev_close = df['close'].shift(1)
    features['bm_f_gap_up'] = ((df['open'] > prev_close) & (prev_close.notna())).astype(int)
    
    # 4. Бычье направление (закрытие > открытие)
    features['bm_f_bullish_dir'] = (df['close'] > df['open']).astype(int)
    
    return features





Удаление функции bullish_kicker_features (если она осталась):

Местоположение: После bullish_pin_bar_features (строка ~180).
УДАЛИТЬ ВЕСЬ ЭТОТ МЕТОД. (Мы его уже удаляли в прошлый раз, но убедись, что его нет).



Обновление функции add_pattern_features для новых признаков Marubozu и удаления bullish_kicker:


Местоположение: Функция add_pattern_features, список feature_columns (строка ~205).


ЗАМЕНИТЬ:
# СТАРЫЙ КОД
        'marubozu_f_strong_body', 'marubozu_f_vol_confirm', 'marubozu_f_bullish',
        # НОВЫЕ БЫЧЬИ ПАТТЕРНЫ
        'ih_f_small_body', 'ih_f_long_upper_shadow', 'ih_f_on_support', 'ih_f_vol_confirm',
        'dd_f_long_lower_shadow', 'dd_f_on_support', 'dd_f_vol_confirm',
        'bpb_f_small_body', 'bpb_f_long_lower_wick', 'bpb_f_on_support', 'bpb_f_vol_confirm',
        'bk_f_strong_bullish_body', 'bk_f_vol_confirm', # <--- ЭТО УДАЛИТЬ
        'bbh_f_long_body', 'bbh_f_open_at_low', 'bbh_f_vol_confirm', 'bbh_f_on_support'



НА НОВЫЙ КОД:
# НОВЫЙ КОД - Обновляем список feature_columns
    feature_columns = [
        'hammer_f_on_support', 'hammer_f_vol_spike',
        'hangingman_f_on_res', 'hangingman_f_vol_spike',
        'engulfing_f_strong', 'engulfing_f_vol_confirm',
        'doji_f_high_vol', 'doji_f_high_atr',
        'shootingstar_f_on_res',
        # НОВЫЕ ПРИЗНАКИ БЫЧЬЕГО MARUBOZU
        'bm_f_strong_body', 'bm_f_vol_confirm', 'bm_f_gap_up', 'bm_f_bullish_dir',
        # Оставшиеся бычьи паттерны
        'ih_f_small_body', 'ih_f_long_upper_shadow', 'ih_f_on_support', 'ih_f_vol_confirm',
        'dd_f_long_lower_shadow', 'dd_f_on_support', 'dd_f_vol_confirm',
        'bpb_f_small_body', 'bpb_f_long_lower_wick', 'bpb_f_on_support', 'bpb_f_vol_confirm',
        'bbh_f_long_body', 'bbh_f_open_at_low', 'bbh_f_vol_confirm', 'bbh_f_on_support'
    ]



Местоположение: В функции add_pattern_features, после блока shootingstar_mask (строка ~280).


ЗАМЕНИТЬ: (блок marubozu_mask и inverted_hammer_mask до bullish_belt_hold_mask)
# СТАРЫЙ КОД (блок marubozu_mask и далее до bullish_belt_hold_mask)
    marubozu_mask = df['CDLMARUBOZU'] != 0
    if not df[marubozu_mask].empty:
        df.loc[marubozu_mask, ['marubozu_f_strong_body', 'marubozu_f_vol_confirm', 'marubozu_f_bullish']] = marubozu_features(df[marubozu_mask]).values

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



НА НОВЫЙ КОД (с новым bullish_marubozu_features и без bullish_kicker):
# НОВЫЙ КОД - Расчет признаков для бычьего Marubozu и других паттернов
    # Расчет признаков для бычьего Marubozu (только для бычьих CDLMARUBOZU)
    bullish_marubozu_mask = (df['CDLMARUBOZU'] == 100) # Фильтруем только бычьи Marubozu
    if not df[bullish_marubozu_mask].empty:
        df.loc[bullish_marubozu_mask, ['bm_f_strong_body', 'bm_f_vol_confirm', 'bm_f_gap_up', 'bm_f_bullish_dir']] = bullish_marubozu_features(df[bullish_marubozu_mask]).values

    inverted_hammer_mask = df['CDLINVERTEDHAMMER'] != 0
    if not df[inverted_hammer_mask].empty:
        df.loc[inverted_hammer_mask, ['ih_f_small_body', 'ih_f_long_upper_shadow', 'ih_f_on_support', 'ih_f_vol_confirm']] = inverted_hammer_features(df[inverted_hammer_mask]).values

    dragonfly_doji_mask = df['CDLDRAGONFLYDOJI'] != 0
    if not df[dragonfly_doji_mask].empty:
        df.loc[dragonfly_doji_mask, ['dd_f_long_lower_shadow', 'dd_f_on_support', 'dd_f_vol_confirm']] = dragonfly_doji_features(df[dragonfly_doji_mask]).values

    bullish_pin_bar_mask = (df['CDLDRAGONFLYDOJI'] != 0) | ((df['CDLHAMMER'] != 0) & (df['close'] > df['open']))
    if not df[bullish_pin_bar_mask].empty:
        df.loc[bullish_pin_bar_mask, ['bpb_f_small_body', 'bpb_f_long_lower_wick', 'bpb_f_on_support', 'bpb_f_vol_confirm']] = bullish_pin_bar_features(df[bullish_pin_bar_mask]).values
    
    # bullish_kicker_mask = df['CDLBULLISHKICKING'] != 0 # <--- УДАЛЕНО
    # if not df[bullish_kicker_mask].empty:
    #     df.loc[bullish_kicker_mask, ['bk_f_strong_bullish_body', 'bk_f_vol_confirm']] = bullish_kicker_features(df[bullish_kicker_mask]).values
        
    bullish_belt_hold_mask = df['CDLBELTHOLD'] != 0
    if not df[bullish_belt_hold_mask].empty:
        df.loc[bullish_belt_hold_mask, ['bbh_f_long_body', 'bbh_f_open_at_low', 'bbh_f_vol_confirm', 'bbh_f_on_support']] = bullish_belt_hold_features(df[bullish_belt_hold_mask]).values



Местоположение: В функции add_pattern_features, блок "Combine features into final scores" (строка ~300).


ЗАМЕНИТЬ: (блок inverted_hammer_f до bullish_belt_hold_f)
# СТАРЫЙ КОД
    df['inverted_hammer_f'] = (df['ih_f_small_body'] + df['ih_f_long_upper_shadow'] + df['ih_f_on_support'] + df['ih_f_vol_confirm']).astype(int)
    df['dragonfly_doji_f'] = (df['dd_f_long_lower_shadow'] + df['dd_f_on_support'] + df['dd_f_vol_confirm']).astype(int)
    df['bullish_pin_bar_f'] = (df['bpb_f_small_body'] + df['bpb_f_long_lower_wick'] + df['bpb_f_on_support'] + df['bpb_f_vol_confirm']).astype(int)
    df['bullish_kicker_f'] = (df['bk_f_strong_bullish_body'] + df['bk_f_vol_confirm']).astype(int) # <--- ЭТО УДАЛИТЬ
    df['bullish_belt_hold_f'] = (df['bbh_f_long_body'] + df['bbh_f_open_at_low'] + df['bbh_f_vol_confirm'] + df['bbh_f_on_support']).astype(int)



НА НОВЫЙ КОД:
# НОВЫЙ КОД - Комбинированные признаки для бычьего Marubozu и других паттернов
    df['bullish_marubozu_f'] = (df['bm_f_strong_body'] + df['bm_f_vol_confirm'] + df['bm_f_gap_up'] + df['bm_f_bullish_dir']).astype(int)
    df['inverted_hammer_f'] = (df['ih_f_small_body'] + df['ih_f_long_upper_shadow'] + df['ih_f_on_support'] + df['ih_f_vol_confirm']).astype(int)
    df['dragonfly_doji_f'] = (df['dd_f_long_lower_shadow'] + df['dd_f_on_support'] + df['dd_f_vol_confirm']).astype(int)
    df['bullish_pin_bar_f'] = (df['bpb_f_small_body'] + df['bpb_f_long_lower_wick'] + df['bpb_f_on_support'] + df['bpb_f_vol_confirm']).astype(int)
    # df['bullish_kicker_f'] = (df['bk_f_strong_bullish_body'] + df['bk_f_vol_confirm']).astype(int) # <--- УДАЛЕНО
    df['bullish_belt_hold_f'] = (df['bbh_f_long_body'] + df['bbh_f_open_at_low'] + df['bbh_f_vol_confirm'] + df['bbh_f_on_support']).astype(int)





Обновление функции detect_candlestick_patterns для новых паттернов TA-Lib и удаления CDLBULLISHKICKING:


Местоположение: Функция detect_candlestick_patterns, блок "Calculate base patterns" (строка ~360).


УДАЛИТЬ:
# СТАРЫЙ КОД
    df['CDLBULLISHKICKING'] = talib.CDLBULLISHKICKING(open_prices, high_prices, low_prices, close_prices)



НА НОВЫЙ КОД: (просто удали эту строку)


Местоположение: Функция detect_candlestick_patterns, список pattern_cols (строка ~375).


ЗАМЕНИТЬ:
# СТАРЫЙ КОД
        'CDLINVERTEDHAMMER', 'CDLDRAGONFLYDOJI', 'CDLBULLISHKICKING', 'CDLBELTHOLD'



НА НОВЫЙ КОД:
# НОВЫЙ КОД - Обновляем список pattern_cols
        'CDLINVERTEDHAMMER', 'CDLDRAGONFLYDOJI', 'CDLBELTHOLD' # ИЗМЕНЕНО: Удален CDLBULLISHKICKING



Местоположение: Функция detect_candlestick_patterns, список feature_cols (комбинированные признаки) (строка ~385).


ЗАМЕНИТЬ:
# СТАРЫЙ КОД
        'inverted_hammer_f', 'dragonfly_doji_f', 'bullish_pin_bar_f', 'bullish_kicker_f', 'bullish_belt_hold_f'



НА НОВЫЙ КОД:
# НОВЫЙ КОД - Обновляем список feature_cols (комбинированные признаки)
        'inverted_hammer_f', 'dragonfly_doji_f', 'bullish_pin_bar_f', 'bullish_belt_hold_f' # ИЗМЕНЕНО: Удален bullish_kicker_f






Файл 2: train_model.py
Этот файл готовит данные и обучает модель.


Обновление списка feature_cols:

Местоположение: Функция prepare_xlstm_rl_data, список feature_cols (строка ~70).
ЗАМЕНИТЬ:
# СТАРЫЙ КОД
        # НОВЫЕ БЫЧЬИ ПАТТЕРНЫ TA-Lib
        'CDLINVERTEDHAMMER', 'CDLDRAGONFLYDOJI', 'CDLBULLISHKICKING', 'CDLBELTHOLD',
        # Комбинированные признаки паттернов
        'hammer_f', 'hangingman_f', 'engulfing_f', 'doji_f',
        'shootingstar_f', 'marubozu_f',
        # НОВЫЕ КОМБИНИРОВАННЫЕ ПРИЗНАКИ БЫЧЬИХ ПАТТЕРНОВ
        'inverted_hammer_f', 'dragonfly_doji_f', 'bullish_pin_bar_f', 'bullish_kicker_f', 'bullish_belt_hold_f',


НА НОВЫЙ КОД:
# НОВЫЙ КОД - Обновленный feature_cols
    feature_cols = [
        # Технические индикаторы
        'RSI_14', 'MACD_12_26_9', 'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0',
        'ADX_14', 'STOCHk_14_3_3', 'STOCHd_14_3_3', 'ATR_14',
        # Базовые паттерны TA-Lib
        'CDLHAMMER', 'CDLENGULFING', 'CDLDOJI', 'CDLSHOOTINGSTAR',
        'CDLHANGINGMAN', 'CDLMARUBOZU', # CDLMARUBOZU теперь используется для бычьего
        # НОВЫЕ БЫЧЬИ ПАТТЕРНЫ TA-Lib (без CDLBULLISHKICKING)
        'CDLINVERTEDHAMMER', 'CDLDRAGONFLYDOJI', 'CDLBELTHOLD',
        # Комбинированные признаки паттернов
        'hammer_f', 'hangingman_f', 'engulfing_f', 'doji_f',
        'shootingstar_f',
        'bullish_marubozu_f', # ИЗМЕНЕНО: Используем новый комбинированный признак
        # НОВЫЕ КОМБИНИРОВАННЫЕ ПРИЗНАКИ БЫЧЬИХ ПАТТЕРНОВ (без bullish_kicker_f)
        'inverted_hammer_f', 'dragonfly_doji_f', 'bullish_pin_bar_f', 'bullish_belt_hold_f',
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






Файл 3: hybrid_decision_maker.py

Обновление списка feature_columns:

Местоположение: В конструкторе __init__, где определяется self.feature_columns.
НА НОВЫЙ КОД: (просто убедитесь, что feature_columns здесь соответствует обновленному списку из train_model.py.)




Файл 4: market_regime_detector.py

Обновление списка xlstm_feature_columns:

Местоположение: Метод set_xlstm_context.
НА НОВЫЙ КОД: (просто убедитесь, что xlstm_feature_columns здесь соответствует обновленному списку из train_model.py.)




Файл 5: trading_env.py

Обновление списка self.feature_columns:

Местоположение: В методе reset, где определяется self.feature_columns.
ЗАМЕНИТЬ: (старый список self.feature_columns)
# СТАРЫЙ КОД
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


НА НОВЫЙ КОД:
# НОВЫЙ КОД - Обновленный self.feature_columns
    self.feature_columns = [
        # Технические индикаторы
        'RSI_14', 'MACD_12_26_9', 'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0',
        'ADX_14', 'STOCHk_14_3_3', 'STOCHd_14_3_3', 'ATR_14',
        # Базовые паттерны TA-Lib
        'CDLHAMMER', 'CDLENGULFING', 'CDLDOJI', 'CDLSHOOTINGSTAR',
        'CDLHANGINGMAN', 'CDLMARUBOZU', # CDLMARUBOZU теперь используется для бычьего
        # НОВЫЕ БЫЧЬИ ПАТТЕРНЫ TA-Lib (без CDLBULLISHKICKING)
        'CDLINVERTEDHAMMER', 'CDLDRAGONFLYDOJI', 'CDLBELTHOLD',
        # Комбинированные признаки паттернов
        'hammer_f', 'hangingman_f', 'engulfing_f', 'doji_f',
        'shootingstar_f',
        'bullish_marubozu_f', # ИЗМЕНЕНО: Используем новый комбинированный признак
        # НОВЫЕ КОМБИНИРОВАННЫЕ ПРИЗНАКИ БЫЧЬИХ ПАТТЕРНОВ (без bullish_kicker_f)
        'inverted_hammer_f', 'dragonfly_doji_f', 'bullish_pin_bar_f', 'bullish_belt_hold_f',
        # 'is_event' # Если is_event используется в RL-среде, добавьте его здесь
    ]



