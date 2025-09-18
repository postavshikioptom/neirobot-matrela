
Улучшим заполнение NaN значений в методе _calculate_all_indicators_batch:
def _calculate_all_indicators_batch(self, df):
    """🔥 ДОБАВЛЕНО: Массовый расчет всех индикаторов"""
    try:
        # Предварительно заполняем NaN значения в исходных данных
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                # Используем forward fill, затем backward fill для заполнения NaN
                df[col] = df[col].ffill().bfill()
        
        close_prices = df['close'].values
        high_prices = df['high'].values
        low_prices = df['low'].values
        
        # Массовый расчет всех индикаторов
        indicators = {}
        
        # RSI
        indicators['RSI'] = talib.RSI(close_prices, timeperiod=config.RSI_PERIOD)
        
        # MACD
        macd, macdsignal, macdhist = talib.MACD(
            close_prices,
            fastperiod=config.MACD_FASTPERIOD,
            slowperiod=config.MACD_SLOWPERIOD,
            signalperiod=config.MACD_SIGNALPERIOD
        )
        indicators['MACD'] = macd
        indicators['MACDSIGNAL'] = macdsignal
        indicators['MACDHIST'] = macdhist
        
        # Stochastic
        stoch_k, stoch_d = talib.STOCH(
            high_prices, low_prices, close_prices,
            fastk_period=config.STOCH_K_PERIOD,
            slowk_period=config.STOCH_K_PERIOD,
            slowd_period=config.STOCH_D_PERIOD
        )
        indicators['STOCH_K'] = stoch_k
        indicators['STOCH_D'] = stoch_d
        
        # Williams %R
        indicators['WILLR'] = talib.WILLR(high_prices, low_prices, close_prices, timeperiod=config.WILLR_PERIOD)
        
        # Awesome Oscillator
        median_price = (high_prices + low_prices) / 2
        sma_5 = talib.SMA(median_price, timeperiod=config.AO_FASTPERIOD)
        sma_34 = talib.SMA(median_price, timeperiod=config.AO_SLOWPERIOD)
        indicators['AO'] = sma_5 - sma_34
        
        # Добавляем все индикаторы в DataFrame и заполняем NaN
        for name, values in indicators.items():
            df[name] = values
            # Заполняем NaN значения медианой для каждого индикатора
            if np.isnan(values).any():
                median_value = np.nanmedian(values)
                df[name] = df[name].fillna(median_value)
        
        return True
        
    except Exception as e:
        print(f"Ошибка при массовом расчете индикаторов: {e}")
        return False

3. Улучшение логирования для отладки
Добавьте более подробное логирование для лучшего понимания процесса:
# В методе load_and_prepare_data в train_model.py
# После обработки всех символов
print(f"\n=== ИТОГОВАЯ СТАТИСТИКА ПО СИМВОЛАМ ===")
print(f"Всего обработано символов: {len(valid_symbols)}")
print(f"Успешно создано последовательностей: {sum(len(data) for data in all_X_supervised)}")
print(f"Распределение меток по всем символам:")

# Объединяем все метки для анализа
all_labels = np.concatenate(all_y_supervised) if all_y_supervised else np.array([])
if len(all_labels) > 0:
    unique, counts = np.unique(all_labels, return_counts=True)
    distribution = {int(u): int(c) for u, c in zip(unique, counts)}
    total = len(all_labels)
    print(f"SELL (0): {distribution.get(0, 0)} ({distribution.get(0, 0)/total:.2%})")
    print(f"HOLD (1): {distribution.get(1, 0)} ({distribution.get(1, 0)/total:.2%})")
    print(f"BUY (2): {distribution.get(2, 0)} ({distribution.get(2, 0)/total:.2%})")
else:
    print("Нет данных для анализа")
