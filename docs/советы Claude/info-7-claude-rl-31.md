
План по исправлению (удаляем CDLBULLISHKICKING и связанные с ним признаки):
Файл 1: feature_engineering.py


Удаление функции bullish_kicker_features:

Местоположение: После функции dragonfly_doji_features (строка ~180).
УДАЛИТЬ ВЕСЬ БЛОК КОДА:
# СТАРЫЙ КОД (удалить весь блок)
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





Удаление признаков bullish_kicker из add_pattern_features:


Местоположение: Функция add_pattern_features, список feature_columns (строка ~205).


ЗАМЕНИТЬ:
# СТАРЫЙ КОД
        'bk_f_strong_bullish_body', 'bk_f_vol_confirm',



НА НОВЫЙ КОД: (просто удали эти строки)


Местоположение: Функция add_pattern_features, блок расчета признаков для bullish_kicker_mask (строка ~280).


УДАЛИТЬ ВЕСЬ БЛОК КОДА:
# СТАРЫЙ КОД (удалить весь блок)
    bullish_kicker_mask = df['CDLBULLISHKICKING'] != 0
    if not df[bullish_kicker_mask].empty:
        df.loc[bullish_kicker_mask, ['bk_f_strong_bullish_body', 'bk_f_vol_confirm']] = bullish_kicker_features(df[bullish_kicker_mask]).values



Местоположение: Функция add_pattern_features, блок "Combine features into final scores" (строка ~300).


ЗАМЕНИТЬ:
# СТАРЫЙ КОД
    df['bullish_kicker_f'] = (df['bk_f_strong_bullish_body'] + df['bk_f_vol_confirm']).astype(int)



НА НОВЫЙ КОД: (просто удали эту строку)




Удаление CDLBULLISHKICKING из detect_candlestick_patterns:


Местоположение: Функция detect_candlestick_patterns, блок "Calculate base patterns" (строка ~360).


ЗАМЕНИТЬ:
# СТАРЫЙ КОД
    df['CDLBULLISHKICKING'] = talib.CDLBULLISHKICKING(open_prices, high_prices, low_prices, close_prices)



НА НОВЫЙ КОД: (просто удали эту строку)


Местоположение: Функция detect_candlestick_patterns, список pattern_cols (строка ~375).


ЗАМЕНИТЬ:
# СТАРЫЙ КОД
        'CDLINVERTEDHAMMER', 'CDLDRAGONFLYDOJI', 'CDLBULLISHKICKING', 'CDLBELTHOLD'



НА НОВЫЙ КОД:
# НОВЫЙ КОД
        'CDLINVERTEDHAMMER', 'CDLDRAGONFLYDOJI', 'CDLBELTHOLD' # ИЗМЕНЕНО: Удален CDLBULLISHKICKING



Местоположение: Функция detect_candlestick_patterns, список feature_cols (комбинированные признаки) (строка ~385).


ЗАМЕНИТЬ:
# СТАРЫЙ КОД
        'inverted_hammer_f', 'dragonfly_doji_f', 'bullish_pin_bar_f', 'bullish_kicker_f', 'bullish_belt_hold_f'



НА НОВЫЙ КОД:
# НОВЫЙ КОД
        'inverted_hammer_f', 'dragonfly_doji_f', 'bullish_pin_bar_f', 'bullish_belt_hold_f' # ИЗМЕНЕНО: Удален bullish_kicker_f






Файл 2: train_model.py
Этот файл готовит данные и обучает модель.

Удаление признаков bullish_kicker из feature_cols:

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
# НОВЫЙ КОД - Обновленный feature_cols без CDLBULLISHKICKING
    feature_cols = [
        # Технические индикаторы
        'RSI_14', 'MACD_12_26_9', 'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0',
        'ADX_14', 'STOCHk_14_3_3', 'STOCHd_14_3_3', 'ATR_14',
        # Базовые паттерны TA-Lib
        'CDLHAMMER', 'CDLENGULFING', 'CDLDOJI', 'CDLSHOOTINGSTAR',
        'CDLHANGINGMAN', 'CDLMARUBOZU',
        # НОВЫЕ БЫЧЬИ ПАТТЕРНЫ TA-Lib (без CDLBULLISHKICKING)
        'CDLINVERTEDHAMMER', 'CDLDRAGONFLYDOJI', 'CDLBELTHOLD',
        # Комбинированные признаки паттернов
        'hammer_f', 'hangingman_f', 'engulfing_f', 'doji_f',
        'shootingstar_f', 'marubozu_f',
        # НОВЫЕ КОМБИНИРОВАННЫЕ ПРИЗНАКИ БЫЧЬИХ ПАТТЕРНОВ (без bullish_kicker_f)
        'inverted_hammer_f', 'dragonfly_doji_f', 'bullish_pin_bar_f', 'bullish_belt_hold_f',
        'is_event'
    ]






Файл 3: hybrid_decision_maker.py

Обновление списка feature_columns:

Местоположение: В конструкторе __init__, где определяется self.feature_columns.
НА НОВЫЙ КОД: (просто убедитесь, что feature_columns здесь соответствует обновленному списку из train_model.py.)




Файл 4: trading_env.py

Обновление списка self.feature_columns:

Местоположение: В методе reset, где определяется self.feature_columns.
ЗАМЕНИТЬ:
# СТАРЫЙ КОД
        # НОВЫЕ БЫЧЬИ ПАТТЕРНЫ TA-Lib
        'CDLINVERTEDHAMMER', 'CDLDRAGONFLYDOJI', 'CDLBULLISHKICKING', 'CDLBELTHOLD',
        # Комбинированные признаки паттернов
        'hammer_f', 'hangingman_f', 'engulfing_f', 'doji_f',
        'shootingstar_f', 'marubozu_f',
        # НОВЫЕ КОМБИНИРОВАННЫЕ ПРИЗНАКИ БЫЧЬИХ ПАТТЕРНОВ
        'inverted_hammer_f', 'dragonfly_doji_f', 'bullish_pin_bar_f', 'bullish_kicker_f', 'bullish_belt_hold_f',
        # Признак is_event не используется напрямую в RL-среде, но может быть добавлен
        # 'is_event' # Если is_event используется в RL-среде, добавьте его здесь


НА НОВЫЙ КОД:
# НОВЫЙ КОД - Обновленный self.feature_columns без CDLBULLISHKICKING
    self.feature_columns = [
        # Технические индикаторы
        'RSI_14', 'MACD_12_26_9', 'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0',
        'ADX_14', 'STOCHk_14_3_3', 'STOCHd_14_3_3', 'ATR_14',
        # Базовые паттерны TA-Lib
        'CDLHAMMER', 'CDLENGULFING', 'CDLDOJI', 'CDLSHOOTINGSTAR',
        'CDLHANGINGMAN', 'CDLMARUBOZU',
        # НОВЫЕ БЫЧЬИ ПАТТЕРНЫ TA-Lib (без CDLBULLISHKICKING)
        'CDLINVERTEDHAMMER', 'CDLDRAGONFLYDOJI', 'CDLBELTHOLD',
        # Комбинированные признаки паттернов
        'hammer_f', 'hangingman_f', 'engulfing_f', 'doji_f',
        'shootingstar_f', 'marubozu_f',
        # НОВЫЕ КОМБИНИРОВАННЫЕ ПРИЗНАКИ БЫЧЬИХ ПАТТЕРНОВ (без bullish_kicker_f)
        'inverted_hammer_f', 'dragonfly_doji_f', 'bullish_pin_bar_f', 'bullish_belt_hold_f',
        # Признак is_event не используется напрямую в RL-среде, но может быть добавлен
        # 'is_event' # Если is_event используется в RL-среде, добавьте его здесь
    ]




==========
