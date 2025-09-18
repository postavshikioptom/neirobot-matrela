# План обновления индикаторов для xLSTM модели

## Обзор изменений

Текущие индикаторы заменяются на новый набор для улучшения качества торговых сигналов:

### Текущие индикаторы (удаляются):
- RSI (14)
- MACD (12,26,9) 
- STOCH_K, STOCH_D (5,3,3)
- WILLR (14)
- AO (5,34)

### Новые индикаторы (добавляются):

#### Тренд:
- **EMA(7,14,21)** - Экспоненциальные скользящие средние
- **MACD(12,26,9)** - Сохраняется, но обновляется реализация
- **KAMA** - Адаптивная скользящая средняя Кауфмана
- **SuperTrend** - Трендовый индикатор на основе ATR

#### Momentum:
- **RSI** - Сохраняется
- **CMO(14)** - Осциллятор моментума Чанде
- **ROC(5)** - Скорость изменения цены

#### Volume:
- **OBV** - Баланс объема
- **MFI** - Индекс денежного потока

#### Volatility:
- **ATR** - Средний истинный диапазон
- **NATR** - Нормализованный ATR

#### Statistical:
- **STDDEV** - Стандартное отклонение

#### Cycle:
- **HT_DCPERIOD** - Доминирующий период цикла (преобразование Гильберта)
- **HT_SINE** - Синусоидальная волна (преобразование Гильберта)

## Детальный план реализации

### 1. Обновление config.py

Добавить новые параметры индикаторов:

```python
# Новые параметры индикаторов
EMA_PERIODS = [7, 14, 21]  # Периоды для EMA
KAMA_PERIOD = 14           # Период для KAMA
SUPERTREND_PERIOD = 10     # Период для SuperTrend
SUPERTREND_MULTIPLIER = 3.0 # Множитель для SuperTrend
CMO_PERIOD = 14            # Период для CMO
ROC_PERIOD = 5             # Период для ROC
ATR_PERIOD = 14            # Период для ATR
NATR_PERIOD = 14           # Период для NATR
STDDEV_PERIOD = 20         # Период для стандартного отклонения
```

### 2. Обновление feature_engineering.py

#### 2.1 Изменение метода _calculate_all_indicators_batch()

Заменить текущие индикаторы на новые:

```python
def _calculate_all_indicators_batch(self, df):
    """Обновленный расчет всех индикаторов"""
    try:
        # Предварительно заполняем NaN значения
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                df[col] = df[col].ffill().bfill()
        
        close_prices = df['close'].values
        high_prices = df['high'].values
        low_prices = df['low'].values
        volume_prices = df['volume'].values
        
        indicators = {}
        
        # === ТРЕНДОВЫЕ ИНДИКАТОРЫ ===
        
        # EMA (7, 14, 21)
        for period in config.EMA_PERIODS:
            indicators[f'EMA_{period}'] = talib.EMA(close_prices, timeperiod=period)
        
        # MACD (сохраняется)
        macd, macdsignal, macdhist = talib.MACD(
            close_prices,
            fastperiod=config.MACD_FASTPERIOD,
            slowperiod=config.MACD_SLOWPERIOD,
            signalperiod=config.MACD_SIGNALPERIOD
        )
        indicators['MACD'] = macd
        indicators['MACDSIGNAL'] = macdsignal
        indicators['MACDHIST'] = macdhist
        
        # KAMA
        indicators['KAMA'] = talib.KAMA(close_prices, timeperiod=config.KAMA_PERIOD)
        
        # SuperTrend (требует кастомной реализации)
        indicators['SUPERTREND'] = self._calculate_supertrend(
            high_prices, low_prices, close_prices, 
            config.SUPERTREND_PERIOD, config.SUPERTREND_MULTIPLIER
        )
        
        # === MOMENTUM ИНДИКАТОРЫ ===
        
        # RSI (сохраняется)
        indicators['RSI'] = talib.RSI(close_prices, timeperiod=config.RSI_PERIOD)
        
        # CMO
        indicators['CMO'] = talib.CMO(close_prices, timeperiod=config.CMO_PERIOD)
        
        # ROC
        indicators['ROC'] = talib.ROC(close_prices, timeperiod=config.ROC_PERIOD)
        
        # === VOLUME ИНДИКАТОРЫ ===
        
        # OBV
        indicators['OBV'] = talib.OBV(close_prices, volume_prices)
        
        # MFI
        indicators['MFI'] = talib.MFI(
            high_prices, low_prices, close_prices, volume_prices, 
            timeperiod=config.RSI_PERIOD
        )
        
        # === VOLATILITY ИНДИКАТОРЫ ===
        
        # ATR
        indicators['ATR'] = talib.ATR(
            high_prices, low_prices, close_prices, 
            timeperiod=config.ATR_PERIOD
        )
        
        # NATR
        indicators['NATR'] = talib.NATR(
            high_prices, low_prices, close_prices, 
            timeperiod=config.NATR_PERIOD
        )
        
        # === STATISTICAL ИНДИКАТОРЫ ===
        
        # STDDEV
        indicators['STDDEV'] = talib.STDDEV(close_prices, timeperiod=config.STDDEV_PERIOD)
        
        # === CYCLE ИНДИКАТОРЫ ===
        
        # HT_DCPERIOD
        indicators['HT_DCPERIOD'] = talib.HT_DCPERIOD(close_prices)
        
        # HT_SINE
        sine, leadsine = talib.HT_SINE(close_prices)
        indicators['HT_SINE'] = sine
        indicators['HT_LEADSINE'] = leadsine
        
        # Добавляем все индикаторы в DataFrame
        for name, values in indicators.items():
            df[name] = values
            if np.isnan(values).any():
                median_value = np.nanmedian(values)
                df[name] = df[name].fillna(median_value)
        
        return True
        
    except Exception as e:
        print(f"Ошибка при расчете индикаторов: {e}")
        return False
```

#### 2.2 Добавление метода для SuperTrend

```python
def _calculate_supertrend(self, high, low, close, period=10, multiplier=3.0):
    """Расчет SuperTrend индикатора"""
    try:
        # Расчет ATR
        atr = talib.ATR(high, low, close, timeperiod=period)
        
        # Расчет базовых линий
        hl2 = (high + low) / 2
        upper_band = hl2 + (multiplier * atr)
        lower_band = hl2 - (multiplier * atr)
        
        # Инициализация массивов
        supertrend = np.zeros_like(close)
        direction = np.ones_like(close)
        
        # Расчет SuperTrend
        for i in range(1, len(close)):
            # Обновление верхней полосы
            if upper_band[i] < upper_band[i-1] or close[i-1] > upper_band[i-1]:
                upper_band[i] = upper_band[i]
            else:
                upper_band[i] = upper_band[i-1]
            
            # Обновление нижней полосы
            if lower_band[i] > lower_band[i-1] or close[i-1] < lower_band[i-1]:
                lower_band[i] = lower_band[i]
            else:
                lower_band[i] = lower_band[i-1]
            
            # Определение направления тренда
            if close[i] <= lower_band[i-1]:
                direction[i] = -1
            elif close[i] >= upper_band[i-1]:
                direction[i] = 1
            else:
                direction[i] = direction[i-1]
            
            # Расчет SuperTrend
            if direction[i] == 1:
                supertrend[i] = lower_band[i]
            else:
                supertrend[i] = upper_band[i]
        
        return supertrend
        
    except Exception as e:
        print(f"Ошибка при расчете SuperTrend: {e}")
        return np.zeros_like(close)
```

#### 2.3 Обновление списка признаков

```python
# В методе _add_technical_indicators обновить:
self.feature_columns = self.base_features + [
    # Трендовые индикаторы
    'EMA_7', 'EMA_14', 'EMA_21',
    'MACD', 'MACDSIGNAL', 'MACDHIST',
    'KAMA', 'SUPERTREND',
    
    # Momentum индикаторы
    'RSI', 'CMO', 'ROC',
    
    # Volume индикаторы
    'OBV', 'MFI',
    
    # Volatility индикаторы
    'ATR', 'NATR',
    
    # Statistical индикаторы
    'STDDEV',
    
    # Cycle индикаторы
    'HT_DCPERIOD', 'HT_SINE', 'HT_LEADSINE'
]
```

### 3. Обновление trading_env.py

#### 3.1 Обновление системы наград

Изменить метод `_calculate_advanced_reward()` для использования новых индикаторов:

```python
def _calculate_advanced_reward(self, action, pnl_pct, xlstm_prediction):
    """Обновленная система наград с новыми индикаторами"""
    current_row = self.df.iloc[self.current_step]
    
    # Используем новые индикаторы для определения силы сигнала
    buy_signal_strength = (
        (current_row.get('RSI', 50) < 30) +
        (current_row.get('CMO', 0) < -50) +
        (current_row.get('ROC', 0) > 0) +
        (current_row.get('MACD', 0) > current_row.get('MACDSIGNAL', 0)) +
        (current_row.get('SUPERTREND', 0) < current_row.get('close', 0))
    )
    
    sell_signal_strength = (
        (current_row.get('RSI', 50) > 70) +
        (current_row.get('CMO', 0) > 50) +
        (current_row.get('ROC', 0) < 0) +
        (current_row.get('MACD', 0) < current_row.get('MACDSIGNAL', 0)) +
        (current_row.get('SUPERTREND', 0) > current_row.get('close', 0))
    )
    
    # Остальная логика наград...
```

### 4. Обновление других файлов

#### 4.1 train_model.py
- Обновить размеры входных данных для модели
- Проверить совместимость с новым количеством признаков

#### 4.2 simulation_engine.py
- Обновить обработку новых индикаторов в симуляции

#### 4.3 run_live_trading.py
- Обновить логику принятия решений с новыми индикаторами

### 5. Тестирование

#### 5.1 Проверка корректности расчетов
- Сравнить результаты с TradingView или другими источниками
- Проверить обработку NaN значений
- Тестировать на различных временных рядах

#### 5.2 Тестирование производительности
- Проверить скорость расчета новых индикаторов
- Оптимизировать при необходимости

#### 5.3 Интеграционное тестирование
- Тестировать на этапе обучения
- Тестировать симуляцию торговли
- Тестировать онлайн торговлю

## Порядок выполнения

1. **Обновить config.py** - добавить новые параметры
2. **Обновить feature_engineering.py** - реализовать новые индикаторы
3. **Обновить trading_env.py** - адаптировать систему наград
4. **Обновить train_model.py** - проверить совместимость
5. **Обновить simulation_engine.py** - адаптировать симуляцию
6. **Обновить run_live_trading.py** - адаптировать онлайн торговлю
7. **Тестирование** - проверить все этапы работы

## Риски и меры предосторожности

1. **Совместимость моделей** - существующие обученные модели могут не работать с новыми признаками
2. **Производительность** - новые индикаторы могут замедлить расчеты
3. **Стабильность** - некоторые индикаторы могут давать нестабильные результаты
4. **Память** - увеличение количества признаков требует больше памяти

## Рекомендации

1. Создать резервную копию текущего кода
2. Тестировать изменения поэтапно
3. Сравнивать результаты с текущей версией
4. Документировать все изменения
5. Подготовить план отката в случае проблем
