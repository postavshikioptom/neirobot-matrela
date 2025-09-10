# План исправления модели LSTM

## Проблема
Модель LSTM возвращает `NaN` при предсказании в `run_live_trading.py`, несмотря на то, что она была обучена в `retrain_lstm_model.py`. Это происходит из-за того, что скейлер, используемый при предсказании, не соответствует скейлеру, использованному при обучении.

## Причина
В `retrain_lstm_model.py` неправильно вызывается метод `save_model`:
```python
# Неправильный код
with open(MODEL_FILE, 'wb') as f:
    lstm_model.save_model(MODEL_FILE)
```

Метод `save_model` уже сам сохраняет модель в файл, и нет необходимости оборачивать это в `with open`. Это может привести к проблемам с созданием файла скейлера.

## Решение
Исправить `retrain_lstm_model.py`, убрав `with open` и просто вызвав `lstm_model.save_model(MODEL_FILE)`.

## Изменения в `retrain_lstm_model.py`

### Текущий код (строки 119-122):
```python
# Сохраняем модель в файл
with open(MODEL_FILE, 'wb') as f:
    lstm_model.save_model(MODEL_FILE)
print(f"Модель успешно сохранена в {MODEL_FILE}")
```

### Исправленный код:
```python
# Сохраняем модель и скейлер
lstm_model.save_model(MODEL_FILE)
print(f"Модель и скейлер успешно сохранены в {MODEL_FILE} и {MODEL_FILE}_scaler.pkl")
```

## Дополнительные проверки
Добавить проверку существования файлов после сохранения:
```python
import os

# Проверяем, что файлы были созданы
if os.path.exists(MODEL_FILE):
    print(f"Файл модели {MODEL_FILE} успешно создан.")
else:
    print(f"Ошибка: файл модели {MODEL_FILE} не был создан.")

scaler_file = MODEL_FILE + "_scaler.pkl"
if os.path.exists(scaler_file):
    print(f"Файл скейлера {scaler_file} успешно создан.")
else:
    print(f"Ошибка: файл скейлера {scaler_file} не был создан.")
```

## Тестирование
После внесения изменений:
1. Запустить `retrain_lstm_model.py` для переобучения модели.
2. Проверить, что созданы файлы `lstm_model.pkl` и `lstm_model.pkl_scaler.pkl`.
3. Запустить `run_live_trading.py` и убедиться, что модель больше не возвращает `NaN`.