# Стратегия нормализации данных для улучшения модели DQN

## Проблема

SHAP анализ показал, что признак OBV доминирует по важности среди других признаков, несмотря на использование StandardScaler. Также quantity имеет значения, которые значительно превышают значения других признаков.

## Решение

Для улучшения нормализации данных предлагается использовать RobustScaler для признаков OBV и quantity, так как он более устойчив к выбросам и большим значениям.

## Необходимые изменения

### 1. Модификация train_model.py

#### Импорты
Добавить импорт RobustScaler:
```python
from sklearn.preprocessing import StandardScaler, RobustScaler
```

#### Создание скейлеров
В функции `create_blank_model()` добавить создание отдельных скейлеров для OBV и quantity:
```python
# Создаем и сохраняем начальный скейлер.
# Он будет обучен на реальных данных при первом запуске train_on_logs().
scaler = StandardScaler()
# Создаем отдельные скейлеры для OBV и quantity
obv_scaler = RobustScaler()
quantity_scaler = RobustScaler()
# Создаем фиктивные данные (одна строка нулей) для инициализации скейлеров
# с правильным количеством признаков.
dummy_data = np.zeros((1, len(numeric_feature_cols)))
scaler.fit(dummy_data)
# Для OBV и quantity создаем отдельные фиктивные данные
obv_dummy = np.zeros((1, 1))
quantity_dummy = np.zeros((1, 1))
obv_scaler.fit(obv_dummy)
quantity_scaler.fit(quantity_dummy)

with open(SCALER_PATH, 'wb') as f:
    pickle.dump({
        'standard_scaler': scaler,
        'obv_scaler': obv_scaler,
        'quantity_scaler': quantity_scaler
    }, f)
```

#### Обучение скейлеров
В функции `train_on_logs()` модифицировать обучение скейлеров:
```python
# Загружаем скейлеры
if os.path.exists(SCALER_PATH):
    with open(SCALER_PATH, 'rb') as f:
        scalers = pickle.load(f)
        scaler = scalers['standard_scaler']
        obv_scaler = scalers['obv_scaler']
        quantity_scaler = scalers['quantity_scaler']
    print("Скейлеры загружены.")
    # Переобучаем скейлеры на всех доступных данных
    all_numeric_data = log_df[numeric_feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0).values
    scaler.fit(all_numeric_data)
    
    # Обучаем отдельные скейлеры для OBV и quantity
    obv_data = log_df[['OBV']].apply(pd.to_numeric, errors='coerce').fillna(0).values
    quantity_data = log_df[['quantity']].apply(pd.to_numeric, errors='coerce').fillna(0).values
    obv_scaler.fit(obv_data)
    quantity_scaler.fit(quantity_data)
    
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump({
            'standard_scaler': scaler,
            'obv_scaler': obv_scaler,
            'quantity_scaler': quantity_scaler
        }, f)
    print("Скейлеры переобучены и сохранены.")

else:
    scaler = StandardScaler()
    obv_scaler = RobustScaler()
    quantity_scaler = RobustScaler()
    
    all_numeric_data = log_df[numeric_feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0).values
    scaler.fit(all_numeric_data)
    
    # Обучаем отдельные скейлеры для OBV и quantity
    obv_data = log_df[['OBV']].apply(pd.to_numeric, errors='coerce').fillna(0).values
    quantity_data = log_df[['quantity']].apply(pd.to_numeric, errors='coerce').fillna(0).values
    obv_scaler.fit(obv_data)
    quantity_scaler.fit(quantity_data)
    
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump({
            'standard_scaler': scaler,
            'obv_scaler': obv_scaler,
            'quantity_scaler': quantity_scaler
        }, f)
    print("Скейлеры обучены и сохранены.")
```

#### Применение скейлеров
В функции `train_on_logs()` модифицировать применение скейлеров:
```python
# State - это состояние на момент открытия сделки
state_raw = opening_trade[numeric_feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0).values.reshape(1, -1)
# Применяем стандартный скейлер
state_scaled = scaler.transform(state_raw).flatten()
# Применяем отдельные скейлеры для OBV и quantity
obv_value = opening_trade['OBV']
quantity_value = opening_trade['quantity']
obv_scaled = obv_scaler.transform([[obv_value]])[0][0]
quantity_scaled = quantity_scaler.transform([[quantity_value]])[0][0]

# Заменяем значения в state_scaled
obv_index = numeric_feature_cols.index('OBV')
quantity_index = numeric_feature_cols.index('quantity')
state_scaled[obv_index] = obv_scaled
state_scaled[quantity_index] = quantity_scaled
state = state_scaled

# Next State - это состояние на момент закрытия сделки
next_state_raw = closing_trade[numeric_feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0).values.reshape(1, -1)
# Применяем стандартный скейлер
next_state_scaled = scaler.transform(next_state_raw).flatten()
# Применяем отдельные скейлеры для OBV и quantity
obv_value = closing_trade['OBV']
quantity_value = closing_trade['quantity']
obv_scaled = obv_scaler.transform([[obv_value]])[0][0]
quantity_scaled = quantity_scaler.transform([[quantity_value]])[0][0]

# Заменяем значения в next_state_scaled
obv_index = numeric_feature_cols.index('OBV')
quantity_index = numeric_feature_cols.index('quantity')
next_state_scaled[obv_index] = obv_scaled
next_state_scaled[quantity_index] = quantity_scaled
next_state = next_state_scaled
```

### 2. Модификация run_live_trading.py

#### Загрузка скейлеров
В функции, где загружается скейлер, модифицировать загрузку:
```python
# Загружаем скейлеры
if os.path.exists("dqn_scaler.pkl"):
    with open("dqn_scaler.pkl", 'rb') as f:
        scalers = pickle.load(f)
        scaler = scalers['standard_scaler']
        obv_scaler = scalers['obv_scaler']
        quantity_scaler = scalers['quantity_scaler']
    
    # Формируем полный вектор признаков (длиной 34)
    obs_full = np.zeros(len(feature_cols), dtype=np.float32)
    
    # Получаем числовые данные для нормализации
    numeric_data_raw = np.array([log_data.get(key, 0.0) for key in numeric_feature_cols], dtype=np.float32).reshape(1, -1)
    # Нормализуем числовые данные стандартным скейлером
    numeric_data_scaled = scaler.transform(numeric_data_raw).flatten()
    
    # Применяем отдельные скейлеры для OBV и quantity
    obv_value = log_data.get('OBV', 0.0)
    quantity_value = log_data.get('quantity', 0.0)
    obv_scaled = obv_scaler.transform([[obv_value]])[0][0]
    quantity_scaled = quantity_scaler.transform([[quantity_value]])[0][0]
    
    # Заменяем значения в numeric_data_scaled
    obv_index = numeric_feature_cols.index('OBV')
    quantity_index = numeric_feature_cols.index('quantity')
    numeric_data_scaled[obv_index] = obv_scaled
    numeric_data_scaled[quantity_index] = quantity_scaled
    
    # Заполняем полный вектор признаков
    numeric_idx = 0
    for i, col in enumerate(feature_cols):
        if col in numeric_feature_cols:
            obs_full[i] = numeric_data_scaled[numeric_idx]
            numeric_idx += 1
    obs = obs_full
```

## Преимущества подхода

1. **Устойчивость к выбросам**: RobustScaler использует медиану и интерквартильный размах, что делает его более устойчивым к большим значениям.
2. **Сохранение структуры данных**: Остальные признаки продолжают нормализоваться с помощью StandardScaler.
3. **Гибкость**: Можно легко настроить параметры нормализации для каждого признака отдельно.