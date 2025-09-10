import csv
from datetime import datetime
import os
import time

LOG_FILE = 'trade_log.csv'

# --- Фиксированный список всех возможных колонок ---
# Это гарантирует, что заголовок всегда будет одинаковым.
FIELDNAMES = [
    'timestamp', 'symbol', 'decision', 'order_type', 'price', 'quantity',
    'usdt_amount', 'bybit_order_id', 'status', 'error_message', 'pnl',
    # --- Модели ---
    'xlstm_pattern_decision', 'xlstm_pattern_confidence',
    'xlstm_indicator_decision', 'xlstm_indicator_confidence',
    'consensus_decision', 'consensus_confidence',
    # --- Индикаторы ---
    'RSI_14', 'MACD_12_26_9', 'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0', 'ADX_14', 'STOCHk_14_3_3', 'STOCHd_14_3_3',
    # --- Паттерны ---
    'CDLHAMMER', 'CDLENGULFING', 'CDLDOJI', 'CDLSHOOTINGSTAR',
    'CDLHANGINGMAN', 'CDLMARUBOZU',  # Заменено CDL3BLACKCROWS
    'hammer_f', 'hangingman_f', 'engulfing_f', 'doji_f',
    'shootingstar_f', 'marubozu_f',  # Заменено 3blackcrows_f
    # --- Детальные признаки паттернов ---
    'hammer_f_on_support', 'hammer_f_vol_spike',
    'hangingman_f_on_res', 'hangingman_f_vol_spike',
    'engulfing_f_strong', 'engulfing_f_vol_confirm',
    'doji_f_high_vol', 'doji_f_high_atr',
    'shootingstar_f_on_res',
    'marubozu_f_strong_body', 'marubozu_f_vol_confirm', 'marubozu_f_bullish'  # Новые признаки Marubozu
]

def log_trade(log_data):
    """
    Записывает информацию о сделке в CSV файл.
    """
    log_row = {field: log_data.get(field, 'N/A') for field in FIELDNAMES}
    log_row['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # 4. Записываем в CSV
    try:
        file_exists = os.path.isfile(LOG_FILE)
        
        with open(LOG_FILE, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            
            if not file_exists or os.stat(LOG_FILE).st_size == 0:
                writer.writeheader()
            
            writer.writerow(log_row)

    except Exception as e:
        print(f"!!! ОШИБКА ЗАПИСИ ЛОГА: {e} !!!")