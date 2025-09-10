# Улучшенная стратегия нормализации данных для уменьшения влияния OBV и quantity

## Проблема

SHAP анализ показал, что признаки OBV и quantity продолжают доминировать по важности среди других признаков, занимая более 10% от всех решений. Необходимо нормализовать эти признаки так, чтобы они не доминировали.

## Решение

Для решения проблемы предлагается использовать комбинацию методов нормализации:

1. Для OBV - логарифмическая нормализация
2. Для quantity - деление на фиксированное значение

## Подходы к нормализации

### 1. Логарифмическая нормализация для OBV

OBV (On-Balance Volume) может принимать очень большие значения. Логарифмическая нормализация поможет уменьшить разницу между большими и средними значениями:

```python
# Для положительных значений OBV
obv_normalized = np.log(obv + 1)

# Для значений OBV, которые могут быть отрицательными
obv_normalized = np.sign(obv) * np.log(np.abs(obv) + 1)
```

### 2. Нормализация quantity делением на фиксированное значение

Quantity обычно имеет значения в диапазоне от нескольких десятков до нескольких сотен. Разделив на фиксированное значение, мы можем привести его к диапазону, близкому к другим признакам:

```python
# Предполагая, что типичное значение quantity составляет 1000
quantity_normalized = quantity / 1000.0
```

## Необходимые изменения

### 1. Модификация train_model.py

#### Импорты
Добавить необходимые импорты:
```python
import numpy as np
```

#### Модификация функции prepare_state
Создать новую функцию для подготовки состояния с улучшенной нормализацией:
```python
def prepare_normalized_state(data_row, scaler, numeric_feature_cols):
    """
    Подготавливает нормализованное состояние с особым учетом OBV и quantity
    """
    # Получаем базовые нормализованные данные
    state_raw = data_row[numeric_feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0).values.reshape(1, -1)
    state_scaled = scaler.transform(state_raw).flatten()
    
    # Применяем специальную нормализацию для OBV
    obv_index = numeric_feature_cols.index('OBV')
    obv_value = float(data_row['OBV'])
    # Логарифмическая нормализация OBV
    obv_normalized = np.sign(obv_value) * np.log(np.abs(obv_value) + 1)
    state_scaled[obv_index] = obv_normalized
    
    # Применяем специальную нормализацию для quantity
    quantity_index = numeric_feature_cols.index('quantity')
    quantity_value = float(data_row['quantity'])
    # Нормализация quantity делением на фиксированное значение
    quantity_normalized = quantity_value / 100.0
    state_scaled[quantity_index] = quantity_normalized
    
    return state_scaled
```

#### Модификация процесса обучения
В функции `train_on_logs()` заменить стандартную нормализацию на улучшенную:
```python
# State - это состояние на момент открытия сделки
state = prepare_normalized_state(opening_trade, scaler, numeric_feature_cols)

# Next State - это состояние на момент закрытия сделки
next_state = prepare_normalized_state(closing_trade, scaler, numeric_feature_cols)
```

### 2. Модификация run_live_trading.py

#### Модификация подготовки данных
В функции, где подготавливаются данные для модели, применить улучшенную нормализацию:
```python
# Получаем числовые данные для нормализации
numeric_data_raw = np.array([log_data.get(key, 0.0) for key in numeric_feature_cols], dtype=np.float32).reshape(1, -1)
# Нормализуем числовые данные стандартным скейлером
numeric_data_scaled = scaler.transform(numeric_data_raw).flatten()

# Применяем специальную нормализацию для OBV
obv_index = numeric_feature_cols.index('OBV')
obv_value = float(log_data.get('OBV', 0.0))
# Логарифмическая нормализация OBV
obv_normalized = np.sign(obv_value) * np.log(np.abs(obv_value) + 1)
numeric_data_scaled[obv_index] = obv_normalized

# Применяем специальную нормализацию для quantity
quantity_index = numeric_feature_cols.index('quantity')
quantity_value = float(log_data.get('quantity', 0.0))
# Нормализация quantity делением на фиксированное значение
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

## Преимущества подхода

1. **Эффективное уменьшение масштаба OBV**: Логарифмическая нормализация значительно уменьшает значения OBV, приводя их к диапазону, близкому к другим признакам.

2. **Простота нормализации quantity**: Деление на фиксированное значение легко реализуемо и эффективно приводит quantity к разумным значениям.

3. **Сохранение информации**: Оба метода сохраняют относительные отношения между значениями, что важно для обучения модели.

4. **Устойчивость к выбросам**: Логарифмическая нормализация делает OBV более устойчивым к очень большим значениям.

## Параметры для настройки

1. **Коэффициент нормализации quantity**: Значение 1000.0 может быть изменено в зависимости от типичных значений quantity в данных.

2. **Обработка нулевых значений OBV**: При логарифмической нормализации добавляется 1 к абсолютному значению, чтобы избежать логарифма от нуля.

## Тестирование подхода

1. Создать новую модель с улучшенной нормализацией
2. Провести обучение
3. Выполнить SHAP анализ для проверки распределения важности признаков
4. Сравнить результаты с предыдущими версиями