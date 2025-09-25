# Проект: neirobot-matrela — метадок репозитория

## Обзор
- **Назначение**: Пайплайн для обучения xLSTM-классификатора с HNM, EMA и плавным переходом CE→AFL, последующей RL-фазой, и гибридным использованием в торговом боте (xLSTM + RL + правила). Есть скрипты для сбора данных Bybit, офлайн-оптимизации порогов разметки и запуска live-трейдинга.
- **Ключевые этапы**:
  - Supervised: балансировка классов, HNM, аугментации, EMA-валидация, LR Warmup+Cosine, CE→AFL.
  - RL: дообучение актора/критика поверх предобученной модели.
  - Live: гибридные решения и управление позициями.

## Точки входа
- `train_model.py` — подготовка данных, обучение supervised, сохранение моделей/логов.
- `run_live_trading.py` — рабочий цикл live-трейдинга с HybridDecisionMaker.
- `data_loader.py` — массовая выгрузка исторических данных с Bybit в CSV.
- `optimize_label_thresholds.py` — офлайн-поиск порогов разметки (PRICE_CHANGE_THRESHOLD и ADAPTIVE_*).

## Конфигурация
- Файл: `config.py`
- Важное:
  - **Обучение**: SEQUENCE_LENGTH, SUPERVISED_EPOCHS, SUPERVISED_BATCH_SIZE, LR_BASE/LR_MIN/LR_WARMUP_EPOCHS
  - **Потери**: AFL_WARMUP_EPOCHS, CE_WEIGHT_START/END, AFL_ALPHA/GAMMA, CLASS_SMOOTHING, ENTROPY_PENALTY_LAMBDA
  - **Балансировка/HNM**: TARGET_CLASS_RATIOS, CLASS_BALANCED_BATCHING, USE_HARD_NEGATIVE_MINING, HNM_* параметры
  - **Регуляризация**: WEIGHT_DECAY_L2, DROPOUT_RNN1/2
  - **Валидация**: USE_EMA_VALIDATION, USE_TTA_VALIDATION, TTA_TRANSFORMS, USE_TEMPERATURE_SCALING
  - **Аугментации**: USE_AUGMENTATIONS, AUG_* (noise, shift, mask)
  - **Разметка**: PRICE_CHANGE_THRESHOLD, ADAPTIVE_THRESHOLD_MIN/MAX/MULTIPLIER
  - **Данные**: MIN_ROWS_PER_SYMBOL, FUTURE_WINDOW
  - **Live/API**: BYBIT_API_KEY/SECRET (использовать демо-ключи), API_URL, WEBSOCKET_URL

## Основные модули
- `models/xlstm_rl_model.py`
  - Классы: FocalLoss (AFL с per-class alpha/gamma и smoothing), WarmUpCosineDecayScheduler, EMAWeightsCallback, XLSTMRLModel (актор/критик).
  - Особенности: clipnorm, bias инициализация по TARGET_CLASS_RATIOS, конфигурируемые DROPOUT_RNN1/2 и WEIGHT_DECAY_L2, базовый LR из config.
- `feature_engineering.py`
  - Признаки: base OHLCV + обширный набор индикаторов (EMA, MACD, KAMA, SuperTrend, RSI, CMO, ROC, OBV, MFI, ATR, NATR, STDDEV, HT_* и дополнительные — Bollinger, StochRSI, WILLR).
  - Безопасность: валидация данных, кэш, заполнение NaN/Inf, RobustScaler.
  - Метки: `create_trading_labels` с адаптивными порогами и окном FUTURE_WINDOW.
- `train_model.py`
  - Подготовка данных, формирование последовательностей, глобальная статистика меток.
  - Батчинг: класс-стратификация по TARGET_CLASS_RATIOS, планируемый HNM, опциональная символ-стратификация (SYMBOL_STRATIFIED_BATCHING).
  - Аугментации: легкие, с контролем памяти через psutil.
  - Коллбеки: EMA, валидационные метрики, LR-контроль.
- `rl_agent.py`
  - Простая A2C-подобная тренировка актора/критика, буфер опыта, epsilon-стратегия, сохранение/загрузка.
- `simulation_engine.py`
  - Эмуляция эпизодов, сбор метрик, опциональный SHAP-анализ важности признаков, визуализации.
- `run_live_trading.py`
  - Гибридные сигналы, управление позициями, динамические стопы/TP, логирование и уведомления.
- `validation_metrics.py`
  - Подробные метрики по эпохам (confusion matrix, F1, распределение предсказаний).
- `notification_system.py`, `trade_manager.py`, `trade_logger.py` (если присутствуют) — инструменты для продакшн-потока.

## Данные
- Исторические CSV: `historical_data*.csv` в корне; используются для обучения/аналитики.
- Промежуточные/результаты: `results/`, `plots/`, логи (`training.log`, `three_stage_training.log` и т.п.).

## Скрипт оптимизации порогов
- `optimize_label_thresholds.py`
  - Вход: `--data` путь к CSV (timestamp, open, high, low, close, volume, turnover, symbol)
  - Параметры: `--target` цели классов (например, 0.3,0.4,0.3), `--max-symbols` ограничение по символам
  - Выход: `threshold_sweep_results.json` с лучшими порогами и распределениями
  - Патч: печатает строки для обновления `config.py` параметрами PRICE_CHANGE_THRESHOLD и ADAPTIVE_*

## Требования окружения
- Python, TensorFlow 2.19, TA-Lib, scikit-learn, numpy/pandas, pybit, shap/lime и др. (см. `requirements.txt`).
- ГПУ не обязателен, но поддерживается; параметры подбираются автоматически.

## Рекомендации по запуску (кратко)
- Обучение: использовать `train_model.py`, предварительно подготовив `historical_data.csv`.
- Оптимизация меток: запустить `optimize_label_thresholds.py` для приведения долей к TARGET_CLASS_RATIOS.
- Live: `run_live_trading.py` с корректными DEMO ключами Bybit и подготовленными моделями в `models/`.

## Замечания безопасности
- Хранить реальные ключи вне репозитория. Для демонстрации использовать только тестовые ключи.

## Статус плана (v4)
- Все 8 пунктов реализованы и отмечены в `3-XLSTM_IMPROVEMENT_PLAN-4.md`. Настройки вынесены в `config.py`.