Реализация адаптивных динамических порогов на основе волатильности
1. Изменения в файле feature_engineering.py
Шаг 1: Добавляем новый метод для расчета адаптивного порога
def calculate_adaptive_threshold(self, df, base_threshold=0.005):
    """
    Рассчитывает адаптивный порог на основе волатильности рынка
    
    Args:
        df (pd.DataFrame): DataFrame с данными цен
        base_threshold (float): Базовый порог для минимального изменения
    
    Returns:
        float: Адаптивный порог для определения сигналов
    """
    try:
        # Проверяем наличие необходимых колонок
        required_cols = ['high', 'low', 'close']
        for col in required_cols:
            if col not in df.columns:
                print(f"Ошибка: колонка {col} отсутствует в данных")
                return base_threshold
        
        # Расчет True Range для последних N свечей
        n_periods = min(14, len(df) - 1)  # Используем стандартный период ATR=14 или меньше, если данных недостаточно
        
        tr_values = []
        for i in range(1, n_periods + 1):
            if i >= len(df):
                break
                
            high = df['high'].iloc[-i]
            low = df['low'].iloc[-i]
            prev_close = df['close'].iloc[-(i+1)]
            
            tr1 = high - low
            tr2 = abs(high - prev_close)
            tr3 = abs(low - prev_close)
            
            tr = max(tr1, tr2, tr3)
            tr_values.append(tr)
        
        # Рассчитываем ATR
        if tr_values:
            atr = sum(tr_values) / len(tr_values)
        else:
            print("Недостаточно данных для расчета ATR, используем базовый порог")
            return base_threshold
        
        # Нормализуем ATR относительно текущей цены
        last_price = df['close'].iloc[-1]
        if last_price > 0:
            normalized_atr = atr / last_price
        else:
            normalized_atr = 0.001
        
        # Рассчитываем адаптивный порог на основе волатильности
        # Минимальный порог 0.3% (0.003), максимальный 1.5% (0.015)
        adaptive_threshold = max(0.003, min(0.015, normalized_atr * 0.7))
        
        print(f"[ADAPTIVE] Base threshold: {base_threshold:.6f}, ATR: {normalized_atr:.6f}, "
              f"Adaptive threshold: {adaptive_threshold:.6f}")
        
        return adaptive_threshold
        
    except Exception as e:
        print(f"Ошибка при расчете адаптивного порога: {e}")
        return base_threshold

Шаг 2: Модифицируем метод create_trading_labels для использования адаптивного порога
def create_trading_labels(self, df, threshold=0.005, future_window=5):
    """
    Создает метки для торговли на основе будущих изменений цены
    с использованием адаптивного порога на основе волатильности
    
    Args:
        df (pd.DataFrame): DataFrame с данными цен
        threshold (float): Базовый порог для минимального изменения
        future_window (int): Окно для прогнозирования будущей цены
        
    Returns:
        np.array: Массив меток (0: SELL, 1: HOLD, 2: BUY)
    """
    # Проверяем и сортируем по timestamp, если он есть
    if 'timestamp' in df.columns:
        if not pd.api.types.is_numeric_dtype(df['timestamp']):
            print(f"⚠️ timestamp не в числовом формате: {df['timestamp'].dtype}, преобразуем")
            df['timestamp'] = pd.to_numeric(df['timestamp'])
        df = df.sort_values('timestamp')
    
    # Рассчитываем адаптивный порог
    adaptive_threshold = self.calculate_adaptive_threshold(df, threshold)
    
    prices = df['close'].values
    labels = []

    # DEBUG: лог входных параметров и короткого среза цен
    try:
        print(f"[LABELS DEBUG] adaptive_threshold={adaptive_threshold}, future_window={future_window}, len(prices)={len(prices)}")
        print("[LABELS DEBUG] first 8 closes:", prices[:8].tolist())
        print("[LABELS DEBUG] last 8 closes:", prices[-8:].tolist())
    except Exception:
        pass

    for i in range(len(prices) - future_window):
        current_price = float(prices[i])
        future_price = float(prices[i + future_window])

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
            for j in range(min(200, len(prices) - future_window)):
                cp = float(prices[j])
                fp = float(prices[j+future_window])
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

Шаг 3: Обновляем метод prepare_supervised_data для использования нового функционала
def prepare_supervised_data(self, df, threshold=0.005, future_window=5):
    """
    Подготавливает данные для supervised learning (этап 1)
    с использованием адаптивных порогов
    """
    # Подготавливаем данные (добавляем индикаторы, нормализуем)
    X, _, processed_df = self.prepare_data(df)
    
    # Создаем метки с адаптивным порогом
    labels = self.create_trading_labels(processed_df, threshold, future_window)
    
    # Убеждаемся, что длины X и labels совпадают
    min_len = min(len(X), len(labels))
    print(f"[PREPARE DEBUG] before trim: len(X)={len(X)}, len(labels)={len(labels)}, using min_len={min_len}")
    X = X[:min_len]
    labels = labels[:min_len]
    
    # Выводим пример первых 30 меток
    print(f"[PREPARE DEBUG] labels sample (first 30): {labels[:30].tolist()}")
    
    return X, labels

2. Изменения в файле config.py
Добавим параметры для настройки адаптивного порога:
# Добавить в файл config.py
# Параметры адаптивного порога
ADAPTIVE_THRESHOLD_BASE = 0.005  # Базовый порог для адаптивного алгоритма
ADAPTIVE_THRESHOLD_MIN = 0.003   # Минимальный порог (0.3%)
ADAPTIVE_THRESHOLD_MAX = 0.015   # Максимальный порог (1.5%)
ADAPTIVE_THRESHOLD_MULTIPLIER = 0.7  # Множитель для ATR при расчете порога

3. Изменения в файле train_model.py
Обновим метод load_and_prepare_data для использования адаптивного порога:
def load_and_prepare_data(self):
    """Загружает и подготавливает данные для всех этапов"""
    print("=== ПОДГОТОВКА ДАННЫХ ===")
    
    # Загружаем данные, timestamp должен быть числом
    df = pd.read_csv(self.data_path, dtype={
        'timestamp': np.int64,  # Явно указываем тип для timestamp
        'open': float, 
        'high': float, 
        'low': float, 
        'close': float, 
        'volume': float, 
        'turnover': float,
        'symbol': str
    })
    print(f"Загружено {len(df)} строк данных")
    
    # Проверяем тип timestamp
    print(f"Тип timestamp: {df['timestamp'].dtype}")
    
    symbol_counts = df['symbol'].value_counts()
    valid_symbols = symbol_counts[symbol_counts >= config.MIN_ROWS_PER_SYMBOL].index.tolist()
    
    if len(valid_symbols) == 0:
        valid_symbols = symbol_counts.head(20).index.tolist()
    
    print(f"Используем {len(valid_symbols)} символов: {valid_symbols[:5]}...")
    
    df_filtered = df[df['symbol'].isin(valid_symbols)].copy()
    
    all_X_supervised = []
    all_y_supervised = []
    
    # Для RL-этапа будем хранить X для каждого символа отдельно
    X_data_for_rl = {}
    
    for i, symbol in enumerate(valid_symbols):
        symbol_data = df_filtered[df_filtered['symbol'] == symbol].copy()
        
        if len(symbol_data) < config.SEQUENCE_LENGTH + config.FUTURE_WINDOW + 10:
            print(f"Пропускаем символ {symbol}: недостаточно данных ({len(symbol_data)} строк)")
            continue
        
        try:
            # Используем адаптивный порог для создания меток
            X_scaled_sequences, labels = self.feature_eng.prepare_supervised_data(
                symbol_data, 
                threshold=config.ADAPTIVE_THRESHOLD_BASE,  # Используем базовый порог
                future_window=config.FUTURE_WINDOW
            )
            
            if len(X_scaled_sequences) > 0:
                all_X_supervised.append(X_scaled_sequences)
                all_y_supervised.append(labels)
                
                X_data_for_rl[symbol] = X_scaled_sequences
                
                print(f"Символ {symbol}: {len(X_scaled_sequences)} последовательностей")
                
                # Вывод распределения меток
                try:
                    if labels is not None and len(labels) > 0:
                        u, c = np.unique(labels, return_counts=True)
                        dist = {int(k): int(v) for k, v in zip(u, c)}
                        print(f"[SYMBOL DEBUG] {symbol} labels distribution: {dist} (threshold=adaptive)")
                    else:
                        print(f"[SYMBOL DEBUG] {symbol} produced no labels")
                except Exception as e:
                    print(f"[SYMBOL DEBUG] error computing dist for {symbol}: {e}")
                
        except Exception as e:
            print(f"❌ Ошибка при обработке символа {symbol}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Проверка на наличие данных перед объединением
    if all_X_supervised and all_y_supervised:
        X_supervised = np.vstack(all_X_supervised)
        y_supervised = np.concatenate(all_y_supervised)
        
        # Анализ глобального распределения меток
        u, c = np.unique(y_supervised, return_counts=True)
        global_dist = {int(k): int(v) for k, v in zip(u, c)}
        total = y_supervised.shape[0]
        print(f"[GLOBAL LABELS] distribution: {global_dist}, total={total}")
        
        # Предупреждение, если HOLD все еще доминирует
        hold_count = global_dist.get(1, 0)
        if hold_count / total > 0.8:
            print(f"⚠️ [GLOBAL WARNING] HOLD fraction is very high: {hold_count}/{total} = {hold_count/total:.2%}")
            print(f"⚠️ Рекомендуется проверить данные или уменьшить базовый порог в config.py")
        
        print(f"Итого подготовлено для Supervised: X={X_supervised.shape}, y={y_supervised.shape}")
        print(f"Распределение классов: SELL={np.sum(y_supervised==0)}, HOLD={np.sum(y_supervised==1)}, BUY={np.sum(y_supervised==2)}")
        
        # Остальной код без изменений...
    else:
        print("❌ Нет данных для обучения. Проверьте обработку символов выше.")
        return False
    
    # Остальной код без изменений...
