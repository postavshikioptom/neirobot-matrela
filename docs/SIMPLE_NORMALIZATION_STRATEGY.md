# Простая стратегия нормализации данных для уменьшения влияния OBV и quantity

## Проблема

OBV и quantity доминируют по важности среди других признаков. Необходимо нормализовать их так, чтобы они были сопоставимы по масштабу с другими признаками.

## Решение

Для решения проблемы предлагается использовать простое деление на фиксированные значения:

1. OBV делится на 10000
2. Quantity делится на 1000

Это должно привести к тому, что:
- OBV будет находиться в диапазоне 5-8
- Quantity будет находиться в диапазоне 1
- Все признаки будут сопоставимы по масштабу

## Необходимые изменения

### 1. Модификация train_model.py

#### Модификация функции prepare_state
Создать новую функцию для подготовки состояния с простой нормализацией:
```python
def prepare_simple_normalized_state(data_row, scaler, numeric_feature_cols):
    """
    Подготавливает нормализованное состояние с простой нормализацией OBV и quantity
    """
    # Получаем базовые нормализованные данные
    state_raw = data_row[numeric_feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0).values.reshape(1, -1)
    state_scaled = scaler.transform(state_raw).flatten()
    
    # Применяем простую нормализацию для OBV
    obv_index = numeric_feature_cols.index('OBV')
    obv_value = float(data_row['OBV'])
    # Нормализация OBV делением на 1000
    obv_normalized = obv_value / 10000.0
    state_scaled[obv_index] = obv_normalized
    
    # Применяем простую нормализацию для quantity
    quantity_index = numeric_feature_cols.index('quantity')
    quantity_value = float(data_row['quantity'])
    # Нормализация quantity делением на 1000
    quantity_normalized = quantity_value / 1000.0
    state_scaled[quantity_index] = quantity_normalized
    
    return state_scaled
```

#### Модификация процесса обучения
В функции `train_on_logs()` заменить стандартную нормализацию на простую:
```python
# State - это состояние на момент открытия сделки
state = prepare_simple_normalized_state(opening_trade, scaler, numeric_feature_cols)

# Next State - это состояние на момент закрытия сделки
next_state = prepare_simple_normalized_state(closing_trade, scaler, numeric_feature_cols)
```

### 2. Модификация run_live_trading.py

#### Модификация подготовки данных
В функции, где подготавливаются данные для модели, применить простую нормализацию:
```python
# Получаем числовые данные для нормализации
numeric_data_raw = np.array([log_data.get(key, 0.0) for key in numeric_feature_cols], dtype=np.float32).reshape(1, -1)
# Нормализуем числовые данные стандартным скейлером
numeric_data_scaled = scaler.transform(numeric_data_raw).flatten()

# Применяем простую нормализацию для OBV
obv_index = numeric_feature_cols.index('OBV')
obv_value = float(log_data.get('OBV', 0.0))
# Нормализация OBV делением на 10000
obv_normalized = obv_value / 10000.0
numeric_data_scaled[obv_index] = obv_normalized

# Применяем простую нормализацию для quantity
quantity_index = numeric_feature_cols.index('quantity')
quantity_value = float(log_data.get('quantity', 0.0))
# Нормализация quantity делением на 1000
quantity_normalized = quantity_value / 1000.0
numeric_data_scaled[quantity_index] = quantity_normalized

# Заполняем полный вектор признаков
numeric_idx = 0
for i, col in enumerate(feature_cols):
    if col in numeric_feature_cols:
        obs_full[i] = numeric_data_scaled[numeric_idx]
        numeric_idx += 1
obs = obs_full
```

### 3. Модификация feature_engineering.py (если необходимо)

Если данные для обучения и live trading формируются в feature_engineering.py, то там также нужно применить простую нормализацию:
```python
# В функции, где формируются признаки
features['OBV'] = features['OBV'] / 10000.0
features['quantity'] = features['quantity'] / 1000.0
```

## Преимущества подхода

1. **Простота реализации**: Требует минимальных изменений в коде
2. **Предсказуемость**: Результаты нормализации легко интерпретируемы
3. **Сопоставимость масштабов**: OBV и quantity будут сопоставимы по масштабу с другими признаками
4. **Сохранение информации**: Относительные отношения между значениями сохраняются

## Параметры для настройки

1. **Коэффициент нормализации OBV**: 10000.0 (может быть изменен при необходимости)
2. **Коэффициент нормализации quantity**: 1000.0 (может быть изменен при необходимости)

## Тестирование подхода

1. Создать новую модель с простой нормализацией
2. Провести обучение
3. Выполнить SHAP анализ для проверки распределения важности признаков
4. Сравнить результаты с предыдущими версиями

Ожидается, что после применения простой нормализации:
- OBV будет находиться в диапазоне 5-8
- Quantity будет находиться в диапазоне 1
- Все признаки будут сопоставимы по масштабу
- Модель будет учитывать все признаки более равномерно