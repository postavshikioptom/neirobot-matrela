Давайте временно отключим все индикаторы и оставим только паттерны для диагностики.
📋 Подробная инструкция по отключению индикаторов
1. Модифицируем feature_engineering.py
def prepare_xlstm_rl_data(data_path, sequence_length=10):
    """
    Подготавливает данные для единой xLSTM+RL системы - ТОЛЬКО ПАТТЕРНЫ
    """
    print(f"Загрузка данных из {data_path}...")
    full_df = pd.read_csv(data_path)
    
    # 🔥 НОВЫЙ КОД: ТОЛЬКО ПАТТЕРНЫ (все индикаторы отключены)
    feature_cols = [
        # ❌ ОТКЛЮЧЕНЫ ВСЕ ТЕХНИЧЕСКИЕ ИНДИКАТОРЫ
        # 'RSI_14', 'MACD_12_26_9', 'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0',
        # 'ADX_14', 'STOCHk_14_3_3', 'STOCHd_14_3_3', 'ATR_14',
        
        # ✅ ТОЛЬКО БАЗОВЫЕ ПАТТЕРНЫ TA-Lib
        'CDLHAMMER', 'CDLENGULFING', 'CDLDOJI', 'CDLSHOOTINGSTAR',
        'CDLHANGINGMAN', 'CDLMARUBOZU',
        # НОВЫЕ БЫЧЬИ ПАТТЕРНЫ TA-Lib
        'CDLINVERTEDHAMMER', 'CDLDRAGONFLYDOJI', 'CDLBELTHOLD',
        
        # ✅ ТОЛЬКО КОМБИНИРОВАННЫЕ ПРИЗНАКИ ПАТТЕРНОВ
        'hammer_f', 'hangingman_f', 'engulfing_f', 'doji_f',
        'shootingstar_f', 'bullish_marubozu_f',
        # НОВЫЕ КОМБИНИРОВАННЫЕ ПРИЗНАКИ БЫЧЬИХ ПАТТЕРНОВ
        'inverted_hammer_f', 'dragonfly_doji_f', 'bullish_pin_bar_f', 'bullish_belt_hold_f',
        
        # ✅ ОСТАВЛЯЕМ EVENT SAMPLING
        'is_event'
    ]
    
    # ... остальной код без изменений

2. Модифицируем run_live_trading.py
# ❌ СТАРЫЕ FEATURE_COLUMNS С ИНДИКАТОРАМИ
# FEATURE_COLUMNS = [
#     # Технические индикаторы
#     'RSI_14', 'MACD_12_26_9', 'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0',
#     'ADX_14', 'STOCHk_14_3_3', 'STOCHd_14_3_3',
#     'ATR_14',
#     # Паттерны
#     'CDLHAMMER', 'CDLENGULFING', 'CDLDOJI', 'CDLSHOOTINGSTAR',
#     'CDLHANGINGMAN', 'CDLMARUBOZU',
# ]

# 🔥 НОВЫЕ FEATURE_COLUMNS - ТОЛЬКО ПАТТЕРНЫ
FEATURE_COLUMNS = [
    # ✅ ТОЛЬКО ПАТТЕРНЫ (все индикаторы отключены)
    'CDLHAMMER', 'CDLENGULFING', 'CDLDOJI', 'CDLSHOOTINGSTAR',
    'CDLHANGINGMAN', 'CDLMARUBOZU',
    'CDLINVERTEDHAMMER', 'CDLDRAGONFLYDOJI', 'CDLBELTHOLD',
    'hammer_f', 'hangingman_f', 'engulfing_f', 'doji_f',
    'shootingstar_f', 'bullish_marubozu_f',
    'inverted_hammer_f', 'dragonfly_doji_f', 'bullish_pin_bar_f', 'bullish_belt_hold_f',
    'is_event'
]

3. Модифицируем условия генерации сигналов в train_model.py
def prepare_xlstm_rl_data(data_path, sequence_length=10):
    # ... код до создания целевых меток
    
    # 🔥 НОВЫЕ УСЛОВИЯ БЕЗ ИНДИКАТОРОВ - ТОЛЬКО ПАТТЕРНЫ
    # Создаем целевые метки на основе будущих цен + ПАТТЕРНОВ
    df['future_return'] = (df['close'].shift(-5) - df['close']) / df['close']
    
    # Базовые пороги без адаптации по ATR (так как ATR отключен)
    df['base_threshold'] = 0.012  # Увеличиваем базовый порог
    
    # 🔥 УСЛОВИЯ ТОЛЬКО НА ОСНОВЕ ПАТТЕРНОВ И ЦЕНЫ
    # BUY условия - сильные бычьи паттерны
    strong_bullish_patterns = (
        (df['CDLHAMMER'] > 0) | 
        (df['CDLENGULFING'] > 0) |
        (df['CDLINVERTEDHAMMER'] > 0) |
        (df['CDLDRAGONFLYDOJI'] > 0) |
        (df['CDLBELTHOLD'] > 0) |
        (df['hammer_f'] >= 2) |
        (df['inverted_hammer_f'] >= 2) |
        (df['bullish_marubozu_f'] >= 2)
    )
    
    # SELL условия - сильные медвежьи паттерны  
    strong_bearish_patterns = (
        (df['CDLHANGINGMAN'] > 0) |
        (df['CDLSHOOTINGSTAR'] > 0) |
        (df['CDLENGULFING'] < 0) |  # Медвежье поглощение
        (df['hangingman_f'] >= 2) |
        (df['shootingstar_f'] >= 1) |
        (df['doji_f'] >= 2)  # Doji в зоне сопротивления
    )
    
    # Более строгие условия для BUY/SELL
    buy_condition = (
        (df['future_return'] > df['base_threshold']) & 
        strong_bullish_patterns
    )
    
    sell_condition = (
        (df['future_return'] < -df['base_threshold']) & 
        strong_bearish_patterns
    )
    
    # Устанавливаем метки
    df['target'] = 2  # По умолчанию HOLD
    df.loc[buy_condition, 'target'] = 0  # BUY
    df.loc[sell_condition, 'target'] = 1  # SELL
    
    # ... остальной код без изменений

4. Обновляем trading_env.py для RL
class TradingEnvRL(gym.Env):
    def reset(self, seed=None, options=None):
        # ... код инициализации
        
        # 🔥 НОВЫЕ FEATURE_COLUMNS - ТОЛЬКО ПАТТЕРНЫ
        self.feature_columns = [
            # ❌ ВСЕ ИНДИКАТОРЫ ОТКЛЮЧЕНЫ
            # 'RSI_14', 'MACD_12_26_9', 'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0',
            # 'ADX_14', 'STOCHk_14_3_3', 'STOCHd_14_3_3', 'ATR_14',
            
            # ✅ ТОЛЬКО ПАТТЕРНЫ
            'CDLHAMMER', 'CDLENGULFING', 'CDLDOJI', 'CDLSHOOTINGSTAR',
            'CDLHANGINGMAN', 'CDLMARUBOZU',
            'CDLINVERTEDHAMMER', 'CDLDRAGONFLYDOJI', 'CDLBELTHOLD',
            'hammer_f', 'hangingman_f', 'engulfing_f', 'doji_f',
            'shootingstar_f', 'bullish_marubozu_f',
            'inverted_hammer_f', 'dragonfly_doji_f', 'bullish_pin_bar_f', 'bullish_belt_hold_f',
        ]
        
        return self._get_observation(), {}
    
    def _calculate_advanced_reward(self, action, pnl_pct, xlstm_prediction):
        """
        Система наград БЕЗ индикаторов - только на основе паттернов
        """
        base_reward = pnl_pct if pnl_pct != 0 else 0
        
        # ... остальные награды без изменений
        
        # 🔥 УБИРАЕМ АНАЛИЗ ИНДИКАТОРОВ ДЛЯ OVERTRADING
        # Вместо индикаторов используем только паттерны
        if action != 2:  # Если не HOLD
            current_row = self.df.iloc[self.current_step]
            
            # Считаем силу паттернов вместо индикаторов
            bullish_pattern_strength = (
                abs(current_row.get('CDLHAMMER', 0)) +
                abs(current_row.get('CDLENGULFING', 0)) +
                current_row.get('hammer_f', 0) +
                current_row.get('bullish_marubozu_f', 0)
            )
            
            bearish_pattern_strength = (
                abs(current_row.get('CDLHANGINGMAN', 0)) +
                abs(current_row.get('CDLSHOOTINGSTAR', 0)) +
                current_row.get('hangingman_f', 0) +
                current_row.get('shootingstar_f', 0)
            )
            
            # Штраф за торговлю без сильных паттернов
            if action == 1 and bullish_pattern_strength < 2:  # BUY без бычьих паттернов
                overtrading_penalty = -1.0
            elif action == 0 and bearish_pattern_strength < 2:  # SELL без медвежьих паттернов
                overtrading_penalty = -1.0
        
        # ... остальной код без изменений

5. Обновляем функции динамических стопов
def calculate_dynamic_stops(features_row, position_side, entry_price):
    """
    Динамические стоп-лоссы БЕЗ ATR - используем фиксированные значения
    """
    base_sl = STOP_LOSS_PCT
    base_tp = TAKE_PROFIT_PCT
    
    # 🔥 БЕЗ ATR - используем фиксированные или паттерн-адаптивные стопы
    close_price = features_row.get('close', entry_price)
    
    # Адаптация на основе силы паттернов вместо ATR
    pattern_strength = (
        abs(features_row.get('CDLHAMMER', 0)) +
        abs(features_row.get('CDLENGULFING', 0)) +
        abs(features_row.get('CDLHANGINGMAN', 0)) +
        features_row.get('hammer_f', 0) +
        features_row.get('engulfing_f', 0)
    )
    
    # Если паттерны сильные, делаем стопы чуть шире
    if pattern_strength > 2:
        dynamic_sl = base_sl * 1.2  # Увеличиваем SL на 20%
        dynamic_tp = base_tp * 1.1  # Увеличиваем TP на 10%
    else:
        dynamic_sl, dynamic_tp = base_sl, base_tp
        
    # Ограничиваем максимальные значения
    dynamic_sl = max(dynamic_sl, -2.5)
    dynamic_tp = min(dynamic_tp, 2.5)

    return dynamic_sl, dynamic_tp

