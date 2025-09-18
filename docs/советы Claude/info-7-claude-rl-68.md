
FeatureEngineering.create_trading_labels — добавить детальный лог и проверку threshold/типов
Файл: feature_engineering.py
Замените тело метода на это (или вставьте дополнения в начале цикла):

def create_trading_labels(self, df, threshold=0.01, future_window=5):
    df = df.sort_values('timestamp')
    prices = df['close'].values
    labels = []

    # DEBUG: лог входных параметров и короткого среза цен
    try:
        print(f"[LABELS DEBUG] threshold={threshold}, future_window={future_window}, len(prices)={len(prices)}")
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

        # DEBUG первых 20 вычислений
        if i < 20:
            print(f"[LABELS DEBUG] i={i}, cur={current_price:.6f}, fut={future_price:.6f}, change={price_change:.6f}")

        if price_change > threshold:
            labels.append(2)  # BUY
        elif price_change < -threshold:
            labels.append(0)  # SELL
        else:
            labels.append(1)  # HOLD

    # Очень важный лог — распределение меток для текущего символа
    try:
        vals, counts = np.unique(labels, return_counts=True)
        dist = {int(v): int(c) for v, c in zip(vals, counts)}
        print(f"[LABELS DEBUG] label distribution (SELL=0,HOLD=1,BUY=2): {dist}")
    except Exception:
        pass

    return np.array(labels)


Prepare_supervised_data — логировать lengths и first label sample
Файл: feature_engineering.py
В prepare_supervised_data сразу после создания X и labels добавьте:

min_len = min(len(X), len(labels))
print(f"[PREPARE DEBUG] before trim: len(X)={len(X)}, len(labels)={len(labels)}, using min_len={min_len}")
X = X[:min_len]
labels = labels[:min_len]
# Покажем первые 30 меток
print(f"[PREPARE DEBUG] labels sample (first 30): {labels[:30].tolist()}")


В ThreeStageTrainer.load_and_prepare_data — лог распределения меток по каждому символу
Файл: train_model.py (в loop где обрабатываете символ)
Добавьте прямо после получения labels:

# DEBUG per-symbol label distribution
try:
    if labels is not None and len(labels) > 0:
        u, c = np.unique(labels, return_counts=True)
        dist = {int(k): int(v) for k, v in zip(u, c)}
        print(f"[SYMBOL DEBUG] {symbol} labels distribution: {dist} (threshold={config.PRICE_CHANGE_THRESHOLD})")
    else:
        print(f"[SYMBOL DEBUG] {symbol} produced no labels")
except Exception as e:
    print(f"[SYMBOL DEBUG] error computing dist for {symbol}: {e}")


Validation: early stop if overall labels are suspicious (optional warning only)
Файл: train_model.py, after building global y_supervised and before augmentation add:

# Global labels distribution check
u, c = np.unique(y_supervised, return_counts=True)
global_dist = {int(k): int(v) for k, v in zip(u, c)}
total = y_supervised.shape[0]
print(f"[GLOBAL LABELS] distribution: {global_dist}, total={total}")
# Warn if HOLD dominant > 85%
hold_count = global_dist.get(1, 0)
if hold_count / total > 0.85:
    print(f"⚠️ [GLOBAL WARNING] HOLD fraction is very high: {hold_count}/{total} = {hold_count/total:.2%}")

(вы просили без вмешательства в баланс — это только предупреждение)

Исправление проверки памяти и логики остановки аугментации
Файл: train_model.py — функция augment_sequences_batched
Замените вычисление current_memory_gb и условие на:

# Перед циклом: get total and threshold once
total_memory_gb = psutil.virtual_memory().total / (1024**3)
# inside loop, replace current_memory_gb calculation with:
used_memory_gb = (psutil.virtual_memory().total - psutil.virtual_memory().available) / (1024**3)
available_memory_gb = psutil.virtual_memory().available / (1024**3)
# Then the check:
if available_memory_gb < (max_memory_gb * 0.2):  # stop if available < 20% of configured limit
    print(f"⚠️ Достигнут лимит памяти: available={available_memory_gb:.2f}GB, required buffer={max_memory_gb*0.2:.2f}GB")
    break

Также уменьшите начальный batch_size выбор разумнее (не min(1000, len(X)//10) — это могло дать 1000). Пример:
batch_size = min(500, max(64, len(X)//50))
print(f"Начинаем батчевую аугментацию с размером батча {batch_size}")


Исправить опечатки инкрементов (prediction_count, step_count, fallback_retry_count)
Файлы: models/xlstm_rl_model.py, feature_engineering.py, trading_env.py
Замените все occurrences:


self.prediction_count += 1  → self.prediction_count += 1
self.step_count += 1 → self.step_count += 1
self.fallback_retry_count += 1 → self.fallback_retry_count += 1
self.total_trades += 1 → self.total_trades += 1
self.total_profit += profit → self.total_profit += profit
self.balance += ... → self.balance += ...

(Иначе эти выражения ничего не делают — причина неожиданных значений)

Улучшенное логирование при заполнении NaN (safe_fill_nan_inf)
Файл: feature_engineering.py — уже есть log_nan_inf_stats; добавьте при возврате списка cols с >0 NaN чтобы сигнализировать какие индикаторы существенно пострадали:

nan_stats, inf_stats = log_nan_inf_stats(df, "До очистки")
# after safe_fill:
post_nan, post_inf = log_nan_inf_stats(df_clean, "После очистки")
# if for any important indicator post_nan still > 0:
for col, stats in post_nan.items():
    if stats['percentage'] > 0.5:
        print(f"⚠️ После очистки {col} имеет {stats['percentage']:.2f}% NaN — проверить источник данных")


Исправление сообщений ModelCheckpoint monitor='val_f1_score' → заменить на валидный метрик имя
Файл: train_model.py — stage1_supervised_pretraining
ModelCheckpoint currently monitors 'val_f1_score' — такой метрики нет в model.metrics (мы compute F1 in callback). Из‑за этого Keras может бросать warning/error. Замените на мониторинг 'val_loss' или 'val_accuracy'. Например:

tf.keras.callbacks.ModelCheckpoint(
    'models/best_supervised_model.keras',
    save_best_only=True,
    monitor='val_loss'
),


Логирование sample price_changes при подозрительном символе
Файл: feature_engineering.create_trading_labels — при вычислении label distribution per symbol, если HOLD fraction > 0.85 добавьте:

if dist.get(1,0) / sum(dist.values()) > 0.85:
    # покажем 50 случайных price_change значений для диагностики
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

