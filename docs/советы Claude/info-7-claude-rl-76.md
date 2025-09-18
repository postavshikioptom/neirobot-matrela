
Мы можем сделать множитель для ATR (0.7 в normalized_atr * 0.7) также динамическим, или более агрессивно снижать порог, если ATR очень низкий.
Изменения в feature_engineering.py (метод calculate_adaptive_threshold)
def calculate_adaptive_threshold(self, df, base_threshold=0.005):
    """
    Рассчитывает адаптивный порог на основе волатильности рынка
    ... (существующий код) ...
    
    # Нормализуем ATR относительно текущей цены
    last_price = df['close'].iloc[-1]
    if last_price > 0:
        normalized_atr = atr / last_price
    else:
        normalized_atr = 0.001
    
    # 🔥 ИСПРАВЛЕНО: Более агрессивная настройка множителя для низковолатильных активов
    # Если normalized_atr очень низкий, мы можем использовать больший множитель
    # чтобы сделать порог более чувствительным
    atr_multiplier = config.ADAPTIVE_THRESHOLD_MULTIPLIER
    if normalized_atr < 0.0005:  # Если ATR меньше 0.05%
        atr_multiplier = 1.5   # Увеличиваем множитель, чтобы сделать порог более чувствительным
    elif normalized_atr < 0.001: # Если ATR меньше 0.1%
        atr_multiplier = 1.0
        
    # Рассчитываем адаптивный порог на основе волатильности
    # Минимальный порог 0.01% (0.0001), максимальный 2% (0.02)
    adaptive_threshold = max(0.0001, min(0.02, normalized_atr * atr_multiplier))
    
    # Дополнительная корректировка: если порог все еще слишком высок
    # Мы можем принудительно снизить его до рекомендованного, если он значительно ниже
    recommended_threshold_from_log = self._get_recommended_threshold_from_data(df, future_window=config.FUTURE_WINDOW)
    if recommended_threshold_from_log is not None and recommended_threshold_from_log < adaptive_threshold * 0.5:
        print(f"Принудительно снижаем адаптивный порог до рекомендованного: {recommended_threshold_from_log:.6f}")
        adaptive_threshold = recommended_threshold_from_log
        
    print(f"[ADAPTIVE] Base threshold: {base_threshold:.6f}, ATR: {normalized_atr:.6f}, "
          f"Adaptive threshold: {adaptive_threshold:.6f}")
    
    return adaptive_threshold

def _get_recommended_threshold_from_data(self, df, future_window):
    """
    Вспомогательный метод для получения рекомендованного порога из данных,
    чтобы получить примерно 30% сигналов.
    """
    try:
        prices = df['close'].values
        if len(prices) <= future_window:
            return None
            
        sample_changes = []
        for j in range(len(prices) - future_window):
            cp = float(prices[j])
            fp = float(prices[j+future_window])
            if cp == 0:
                pct = 0.0
            else:
                pct = (fp - cp) / cp
            sample_changes.append(pct)
        
        changes_abs = np.abs(sample_changes)
        if not changes_abs.any(): # Если все изменения нулевые
            return 0.0001 # Возвращаем минимальный порог
            
        changes_sorted = np.sort(changes_abs)
        
        # Находим порог, который бы дал примерно 30% сигналов (не HOLD)
        target_idx = int(len(changes_sorted) * 0.7)  # 70-й процентиль
        if target_idx < len(changes_sorted):
            return changes_sorted[target_idx]
        else:
            return None
    except Exception as e:
        print(f"Ошибка при расчете рекомендованного порога: {e}")
        return None

Изменения в config.py:
# Было:
# ADAPTIVE_THRESHOLD_MIN = 0.003   # Минимальный порог (0.3%)
# ADAPTIVE_THRESHOLD_MAX = 0.015   # Максимальный порог (1.5%)

# Измените на более низкий минимальный порог
ADAPTIVE_THRESHOLD_MIN = 0.0001  # Минимальный порог (0.01%)
ADAPTIVE_THRESHOLD_MAX = 0.02    # Максимальный порог (2%)

2. Увеличение future_window или использование нескольких future_window
future_window = 5 минут на 1-минутном графике - это очень короткий период. Цена может не успеть значительно измениться.
Решение: Попробуйте увеличить future_window до 15-30 минут.
Это позволит захватывать более значимые движения.
Изменения в config.py:
# Было:
FUTURE_WINDOW = 5

# Измените на:
FUTURE_WINDOW = 30 # Изменить на 30, протестируйте разные значения
