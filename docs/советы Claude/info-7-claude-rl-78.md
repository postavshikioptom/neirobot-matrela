
Вам нужно изменить два метода в feature_engineering.py.
1. Метод create_trading_labels
Здесь нужно убрать значения по умолчанию для threshold и future_window из сигнатуры функции и использовать значения из config.py.
# В файле feature_engineering.py

# Было:
# def create_trading_labels(self, df, threshold=0.005, future_window=5):

# Измените на:
def create_trading_labels(self, df): # Убираем значения по умолчанию
    """
    Создает метки для торговли на основе будущих изменений цены
    с использованием адаптивного порога на основе волатильности.
    Использует параметры из config.py.
    
    Args:
        df (pd.DataFrame): DataFrame с данными цен
        
    Returns:
        np.array: Массив меток (0: SELL, 1: HOLD, 2: BUY)
    """
    # Проверяем и сортируем по timestamp, если он есть
    if 'timestamp' in df.columns:
        if not pd.api.types.is_numeric_dtype(df['timestamp']):
            print(f"⚠️ timestamp не в числовом формате: {df['timestamp'].dtype}, преобразуем")
            df['timestamp'] = pd.to_numeric(df['timestamp'])
        df = df.sort_values('timestamp')
    
    # 🔥 ИСПРАВЛЕНО: Используем threshold из config.py
    adaptive_threshold = self.calculate_adaptive_threshold(df, config.PRICE_CHANGE_THRESHOLD)
    
    prices = df['close'].values
    labels = []

    # DEBUG: лог входных параметров и короткого среза цен
    try:
        print(f"[LABELS DEBUG] adaptive_threshold={adaptive_threshold}, future_window={config.FUTURE_WINDOW}, len(prices)={len(prices)}")
        print("[LABELS DEBUG] first 8 closes:", prices[:8].tolist())
        print("[LABELS DEBUG] last 8 closes:", prices[-8:].tolist())
    except Exception:
        pass

    # 🔥 ИСПРАВЛЕНО: Используем future_window из config.py
    for i in range(len(prices) - config.FUTURE_WINDOW):
        current_price = float(prices[i])
        future_price = float(prices[i + config.FUTURE_WINDOW])

        # Защита от деления на ноль
        if current_price == 0 or np.isnan(current_price) or np.isinf(current_price):
            price_change = 0.0
        else:
            price_change = (future_price - current_price) / float(current_price)

        # DEBUG для первых 20 вычислений
        if i < 20:
            print(f"[LABELS DEBUG] i={i}, cur={current_price:.6f}, fut={future_price:.6f}, change={price_change:.6f}")

        # Используем адаптивный порог для определения сигналов
        if price_change > adaptive_threshold:
            labels.append(2)  # BUY
        elif price_change < -adaptive_threshold:
            labels.append(0)  # SELL
        else:
            labels.append(1)  # HOLD

    # Логирование распределения меток
    vals, counts = np.unique(labels, return_counts=True)
    dist = {int(v): int(c) for v, c in zip(vals, counts)}
    print(f"[LABELS DEBUG] label distribution (SELL=0,HOLD=1,BUY=2): {dist}")
    
    # Анализ дисбаланса
    total = len(labels)
    hold_count = dist.get(1, 0)
    hold_percentage = hold_count / total if total > 0 else 0
    
    if hold_percentage > 0.8:
        print(f"[HOLD WARNING] Высокий процент HOLD меток: {hold_percentage:.2%}")
        
        # Дополнительный анализ изменений цены для диагностики
        if total > 0:
            sample_changes = []
            # 🔥 ИСПРАВЛЕНО: Используем config.FUTURE_WINDOW
            for j in range(min(200, len(prices) - config.FUTURE_WINDOW)):
                cp = float(prices[j])
                fp = float(prices[j+config.FUTURE_WINDOW])
                if cp == 0:
                    pct = 0.0
                else:
                    pct = (fp - cp) / cp
                sample_changes.append(pct)
            
            print(f"[HOLD DEBUG] Symbol likely all-HOLD. sample changes (first 50): {np.array(sample_changes)[:50].tolist()}")
            print(f"[HOLD DEBUG] Change stats: min={np.min(sample_changes):.6f}, max={np.max(sample_changes):.6f}, "
                  f"mean={np.mean(sample_changes):.6f}, std={np.std(sample_changes):.6f}")
            print(f"[HOLD DEBUG] Current adaptive threshold: {adaptive_threshold:.6f}")
            
            # Анализируем, какой порог нужен для более сбалансированного распределения
            changes_abs = np.abs(sample_changes)
            changes_sorted = np.sort(changes_abs)
            
            # Находим порог, который бы дал примерно 30% сигналов (не HOLD)
            target_idx = int(len(changes_sorted) * 0.7)  # 70-й процентиль
            if target_idx < len(changes_sorted):
                suggested_threshold = changes_sorted[target_idx]
                print(f"[HOLD DEBUG] Для получения ~30% сигналов рекомендуемый порог: {suggested_threshold:.6f}")
        
    return np.array(labels)

2. Метод prepare_supervised_data
Здесь также нужно убрать значения по умолчанию для threshold и future_window из сигнатуры функции, так как они будут браться из config.py.
# В файле feature_engineering.py

# Было:
# def prepare_supervised_data(self, df, threshold=0.005, future_window=5):

# Измените на:
def prepare_supervised_data(self, df): # Убираем значения по умолчанию
    """
    Подготавливает данные для supervised learning (этап 1)
    с использованием адаптивных порогов.
    Использует параметры из config.py.
    
    Args:
        df (pd.DataFrame): DataFrame с данными цен
        
    Returns:
        tuple: (X, labels) - подготовленные данные и метки
    """
    # Подготавливаем данные (добавляем индикаторы, нормализуем)
    X, _, processed_df = self.prepare_data(df)
    
    # 🔥 ИСПРАВЛЕНО: Создаем метки без передачи threshold и future_window
    labels = self.create_trading_labels(processed_df)
    
    # Убеждаемся, что длины X и labels совпадают
    min_len = min(len(X), len(labels))
    print(f"[PREPARE DEBUG] before trim: len(X)={len(X)}, len(labels)={len(labels)}, using min_len={min_len}")
    X = X[:min_len]
    labels = labels[:min_len]
    
    # Выводим пример первых 30 меток
    print(f"[PREPARE DEBUG] labels sample (first 30): {labels[:30].tolist()}")
    
    return X, labels

3. Метод calculate_adaptive_threshold
Здесь параметр base_threshold также должен браться из config.py.
# В файле feature_engineering.py

# Было:
# def calculate_adaptive_threshold(self, df, base_threshold=0.005):

# Измените на:
def calculate_adaptive_threshold(self, df, base_threshold=None): # Убираем значение по умолчанию, или делаем None
    """
    Рассчитывает адаптивный порог на основе волатильности рынка
    
    Args:
        df (pd.DataFrame): DataFrame с данными цен
        base_threshold (float, optional): Базовый порог для минимального изменения. Если None, берется из config.
        
    Returns:
        float: Адаптивный порог для определения сигналов
    """
    # Если base_threshold не передан, берем его из config
    if base_threshold is None:
        base_threshold = config.PRICE_CHANGE_THRESHOLD
        
    # ... (остальной код метода без изменений, так как он уже использует config.ADAPTIVE_THRESHOLD_MIN/MAX/MULTIPLIER) ...

4. Вызовы prepare_supervised_data в train_model.py
В файле train_model.py в методе load_and_prepare_data вызовы self.feature_eng.prepare_supervised_data также нужно изменить, чтобы не передавать threshold и future_window:
# В файле train_model.py, в методе load_and_prepare_data

# Было:
# X_scaled_sequences, labels = self.feature_eng.prepare_supervised_data(
#     symbol_data, 
#     threshold=config.ADAPTIVE_THRESHOLD_BASE,  # Используем базовый порог
#     future_window=config.FUTURE_WINDOW
# )

# Измените на:
X_scaled_sequences, labels = self.feature_eng.prepare_supervised_data(symbol_data)
