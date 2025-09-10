# Отладочный скрипт для проверки паттернов (версия 3)

## Описание проблемы
Паттерны (CDLHAMMER, CDLENGULFING, CDLMORNINGSTAR, CDLEVENINGSTAR, CDL3WHITESOLDIERS) не отображаются в логах, хотя в отладочном скрипте некоторые паттерны имеют ненулевые значения.

## Новый отладочный скрипт
Следующий скрипт будет имитировать работу бота и проверять передачу данных в логи:

```python
import pandas as pd
import json
import feature_engineering
import trade_logger

LIVE_DATA_FILE = 'live_data.json'

def test_patterns_logging(symbol='NMRUSDT'):
    print(f"--- Тестирование передачи паттернов в логи для {symbol} ---")
    
    # Загрузим данные из live_data.json
    try:
        with open(LIVE_DATA_FILE, 'r') as f:
            live_data = json.load(f)
    except FileNotFoundError:
        print(f"Файл {LIVE_DATA_FILE} не найден.")
        return
    
    symbol_data = live_data.get(symbol)
    if not symbol_data:
        print(f"Нет данных для символа {symbol}")
        return
    
    kline_list = symbol_data.get('klines')
    if not kline_list:
        print(f"Нет свечей для символа {symbol}")
        return
    
    print(f"Загружено {len(kline_list)} свечей для {symbol}")
    
    # Преобразуем данные в DataFrame, как это делается в run_live_trading.py
    kline_df = pd.DataFrame(kline_list, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
    # Преобразуем timestamp в читаемый формат
    kline_df['timestamp'] = pd.to_datetime(kline_df['timestamp'], unit='ms')
    print(f"DataFrame создан. Первые 5 строк:")
    print(kline_df.head())
    
    # Рассчитаем индикаторы
    features_df = feature_engineering.calculate_features(kline_df.copy())
    print(f"\nПосле расчета индикаторов. Размер: {len(features_df)} строк")
    
    # Рассчитаем паттерны
    features_df = feature_engineering.detect_candlestick_patterns(features_df)
    print(f"После расчета паттернов. Размер: {len(features_df)} строк")
    
    # Проверим последние значения
    latest_features = features_df.iloc[-1]
    print(f"\nПоследние значения признаков:")
    pattern_columns = ['CDLHAMMER', 'CDLENGULFING', 'CDLMORNINGSTAR', 'CDLEVENINGSTAR', 'CDL3WHITESOLDIERS']
    indicator_columns = ['RSI_14', 'MACD_12_26_9', 'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0', 'ADX_14', 'STOCHk_14_3_3', 'STOCHd_14_3_3']
    
    print("Индикаторы:")
    for col in indicator_columns:
        if col in latest_features:
            print(f"  {col}: {latest_features[col]}")
        else:
            print(f"  {col}: отсутствует")
    
    print("Паттерны:")
    for col in pattern_columns:
        if col in latest_features:
            print(f"  {col}: {latest_features[col]}")
        else:
            print(f"  {col}: отсутствует")
    
    # Имитируем передачу данных в лог
    print(f"\n--- Имитация передачи данных в лог ---")
    log_data = {
        'symbol': symbol,
        'decision': 'TEST',
        'order_type': 'BUY',
        'price': latest_features['close'],
        'quantity': 1,
        'usdt_amount': latest_features['close'],
        'bybit_order_id': 'test_id',
        'status': 'SUCCESS',
        'lstm_decision': 'BUY',
        'lstm_confidence': 0.8,
        'xlstm_decision': 'SELL',
        'xlstm_confidence': 0.7,
        'consensus_decision': 'BUY',
        'consensus_confidence': 0.75
    }
    
    # Добавляем индикаторы и паттерны, как это делается в run_live_trading.py
    log_data.update(latest_features[indicator_columns].to_dict())
    log_data.update(latest_features[pattern_columns].to_dict())
    
    print("Данные для лога:")
    for key, value in log_data.items():
        print(f"  {key}: {value}")
    
    # Запишем в тестовый лог файл
    test_log_file = 'test_trade_log.csv'
    trade_logger.LOG_FILE = test_log_file
    trade_logger.log_trade(log_data)
    print(f"\nДанные записаны в тестовый лог файл: {test_log_file}")

if __name__ == '__main__':
    test_patterns_logging()