# --- Bybit API Keys & URLs ---
# ВАЖНО: Всегда используйте ключи для ДЕМО-счета (Testnet) во время разработки и тестирования.
# Никогда не публикуйте свои реальные ключи в открытом доступе (например, на GitHub).

BYBIT_API_KEY = "OOofB1HzYVpySyMPom"
BYBIT_API_SECRET = "e4AkAz9x1ycOMCtXKa1milmShfk61KZxJyhG"

# Эндпоинты для ДЕМО-ТОРГОВЛИ
API_URL = "https://api-demo.bybit.com"
# WebSocket v5 для публичных данных (свечи) по споту
WEBSOCKET_URL = "wss://stream.bybit.com/v5/public/linear"

# --- Trading Parameters ---
ORDER_USDT_AMOUNT = 11.0 # Сумма одного ордера в USDT
LEVERAGE = "2"  # Устанавливаем плечо. ВАЖНО: это значение должно быть в виде строки.
REQUIRED_CANDLES = 100 # Сколько свечей нужно для анализа
# Пока оставим один символ для простоты. В будущем можно будет сделать список: ["BTCUSDT", "ETHUSDT"]
SYMBOLS = ["ADAUSDT"]
TIMEFRAME = "1"  # Таймфрейм в минутах (1, 5, 15, 60, 240, "D", "W", "M")


# 🔥 ДОБАВЛЕНО: Параметры для предотвращения memory leak
MEMORY_CLEANUP_FREQUENCY = 100  # Очистка каждые 100 предсказаний
MAX_PREDICTION_BATCH_SIZE = 32  # Максимальный размер батча

# Параметры модели
SEQUENCE_LENGTH = 60
XLSTM_MEMORY_SIZE = 48  # Уменьшено с 64 до 48 (-25% для ускорения)
XLSTM_MEMORY_UNITS = 96  # Уменьшено с 128 до 96 (-25% для ускорения)

# Параметры обучения
SUPERVISED_EPOCHS = 50
SUPERVISED_BATCH_SIZE = 64  # Увеличено с 32 до 64 (компенсация за уменьшение модели)
SUPERVISED_VALIDATION_SPLIT = 0.15

REWARD_MODEL_EPOCHS = 30
REWARD_MODEL_BATCH_SIZE = 64

RL_EPISODES = 100
RL_BATCH_SIZE = 32

# План v3 — Этап A/B/C флаги и параметры
CLASS_BALANCED_BATCHING = True
TARGET_CLASS_RATIOS = [0.33, 0.33, 0.34]  # 🎯 ПОЧТИ РАВНЫЕ: Пытаемся достичь баланса через равномерность
DYNAMIC_CLASS_WEIGHTS = False  # 🚨 ВРЕМЕННО ОТКЛЮЧЕНО: Система слишком нестабильна с динамическими весами
DYNAMIC_WEIGHT_STEP = 0.015  # 🔄 УМЕНЬШЕНО: Очень осторожная корректировка после экстремального дисбаланса
USE_HARD_NEGATIVE_MINING = True
HNM_TOP_K_FRACTION = 0.05  # 🔥 ВОЗВРАТ К СТАНДАРТУ: 5% для стабильности

# Ассиметричная Focal Loss и class-conditional label smoothing (БЕЗ КОНФЛИКТА С SAMPLE_WEIGHTS)
AFL_ALPHA = [1.02, 1.08, 0.94]    # 🎯 ТОЧНАЯ НАСТРОЙКА: BUY 1.00→0.94 против доминирования 72%, ищем золотую середину между 0.90(2%) и 1.00(72%)
AFL_GAMMA = [2.0, 2.0, 2.0]   # 🔥 БАЗОВЫЕ ЗНАЧЕНИЯ: Равный фокус на всех классах  
CLASS_SMOOTHING = [0.05, 0.05, 0.05]  # 🔄 ВОЗВРАТ К РАВНЫМ: Убираем экспериментальные изменения
ENTROPY_PENALTY_LAMBDA = 0.03  # 🔥 УМЕНЬШЕНО: Меньший штраф для стабильности

# Валидация с TTA и Temperature Scaling
USE_TTA_VALIDATION = True
TTA_TRANSFORMS = ['identity', 'zscore_window', 'gaussian_smooth']
USE_TEMPERATURE_SCALING = True

# SMOTE — chunked режим
USE_CHUNKED_SMOTE = True
CHUNKED_SMOTE_MINORITY_CLASSES = [0,1,2]  # # Делаем ВСЕ классы равноправными (никого не пережимаем) (раньше пережимали)
CHUNKED_SMOTE_MAX_SYNTH_PER_CLASS = 10000  # 🔄 ВОЗВРАТ К СТАНДАРТУ: Избегаем переобучения после дисбаланса

# Параметры индикаторов
RSI_PERIOD = 14
MACD_FASTPERIOD = 12
MACD_SLOWPERIOD = 26
MACD_SIGNALPERIOD = 9

# Новые параметры индикаторов
EMA_PERIODS = [7, 14, 21]  # Периоды для EMA
KAMA_PERIOD = 14           # Период для KAMA

# SMOTE параметры
USE_SMOTE = True  # Включить SMOTE для балансировки классов
SUPERTREND_PERIOD = 10     # Период для SuperTrend
SUPERTREND_MULTIPLIER = 3.0 # Множитель для SuperTrend
CMO_PERIOD = 14            # Период для CMO
ROC_PERIOD = 5             # Период для ROC
ATR_PERIOD = 14            # Период для ATR
NATR_PERIOD = 14           # Период для NATR
STDDEV_PERIOD = 20         # Период для стандартного отклонения

# Параметры данных на этапе формирования (настроены нормально 25% 50% 25%
MIN_ROWS_PER_SYMBOL = 100
FUTURE_WINDOW = 25 # Я сам увеличил до 25. Уменьшено с 30 до 20 (более быстрые сигналы, меньше задержка)
PRICE_CHANGE_THRESHOLD = 0.004  # До скрипта оптимизации - 0.010 / Увеличено до 1% изменения цены за FUTURE_WINDOW свечей — больше нейтральной зоны (HOLD)

# Параметры адаптивного порога
ADAPTIVE_THRESHOLD_MIN = 0.0035   # До скрипта оптимизации - 0.003 Чуть повышаем минимум, чтобы не сжимать HOLD при низкой волатильности
ADAPTIVE_THRESHOLD_MAX = 0.012    # До скрипта оптимизации - 0.02 Максимальный порог (2%). Чем меньше  тем меньше HOLD
ADAPTIVE_THRESHOLD_MULTIPLIER = 1.2 # До скрипта оптимизации - 1.2 Чуть расширяем нейтральную зону для HOLD
