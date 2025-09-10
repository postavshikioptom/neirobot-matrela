# План Переименования Файлов

## Текущая структура файлов

```
active_positions.json
chat.txt
config.py
dqn_trading_model.zip
feature_engineering.py
get_all_symbols.py
historical_data.csv
hotlist.txt
live_data.json
output.txt
requirements.txt
run_live_trading.py
screener.py
symbols.txt
trade_logger.py
trade_manager.py
trade_statistics.py
trader_status.txt
trading_env.py
train_model.py
train_xgboost_historical.py
xgboost_historical_model.json
xgboost_model.json
xgboost_model.py
инфо.txt
Сравнение-матмоделей-в-связке-Tensroflow.md
что нужно для создание нейробота.txt
```

## Новая структура файлов с префиксами

### Файлы, связанные с XGBoost
- xgboost_model.py (уже правильно)
- xgboost_model.json (уже правильно)
- xgboost_historical_model.json (уже правильно)
- train_xgboost_historical.py (уже правильно)

### Файлы, связанные с Kalman Filter (новые)
- kalman_filter.py (будет создан)
- kalman_model.json (будет создан)
- train_kalman.py (будет создан)

### Файлы, связанные с LSTM (новые)
- lstm_model.py (будет создан)
- lstm_model.h5 (будет создан)
- train_lstm.py (будет создан)

### Файлы, связанные с GPR (новые)
- gpr_model.py (будет создан)
- gpr_model.json (будет создан)
- train_gpr.py (будет создан)

### Основные файлы системы (без изменений)
- run_live_trading.py
- feature_engineering.py
- trading_env.py
- train_model.py
- trade_logger.py
- trade_manager.py
- trade_statistics.py
- screener.py
- config.py

### Вспомогательные файлы (без изменений)
- active_positions.json
- hotlist.txt
- live_data.json
- symbols.txt
- trader_status.txt
- historical_data.csv
- requirements.txt

### Документация (без изменений)
- docs/ARCHITECTURE_ENHANCED_TRADING_SYSTEM.md
- docs/ARCHITECTURE_XGBOOST.md
- docs/plan_xgboost_training.md
- docs/QDN_MODEL_DOCUMENTATION.md
- Сравнение-матмоделей-в-связке-Tensroflow.md
- что нужно для создание нейробота.txt
- инфо.txt

## План переименования существующих файлов

После анализа текущих файлов, все файлы, связанные с XGBoost, уже имеют правильные имена с префиксом "xgboost".
Остальные файлы не требуют переименования, так как они являются основными компонентами системы или вспомогательными файлами.

## Создание новых файлов

Будут созданы следующие новые файлы для реализации Kalman Filter, LSTM и GPR:

1. kalman_filter.py - реализация фильтра Калмана
2. kalman_model.json - сохраненная модель Kalman Filter
3. train_kalman.py - скрипт для обучения Kalman Filter
4. lstm_model.py - реализация LSTM модели
5. lstm_model.h5 - сохраненная LSTM модель
6. train_lstm.py - скрипт для обучения LSTM
7. gpr_model.py - реализация Gaussian Process Regression
8. gpr_model.json - сохраненная GPR модель
9. train_gpr.py - скрипт для обучения GPR