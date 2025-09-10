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
def log_enhanced_trade_with_quality_metrics(log_data):
    """
    Расширенное логирование с метриками качества сигналов
    """
    # Вычисляем метрики качества сигнала
    signal_quality = calculate_signal_quality(log_data)
    log_data.update(signal_quality)
    
    # Стандартное логирование
    log_trade(log_data)
    
    # Дополнительная аналитика
    update_signal_quality_stats(signal_quality)

def calculate_signal_quality(log_data):
    """
    Вычисляет метрики качества торгового сигнала
    """
    quality_metrics = {}
    
    # VSA качество (0-100)
    vsa_signals_count = sum([
        log_data.get('vsa_no_demand', 0),
        log_data.get('vsa_no_supply', 0),
        log_data.get('vsa_stopping_volume', 0),
        log_data.get('vsa_climactic_volume', 0)
    ])
    vsa_strength = abs(log_data.get('vsa_strength', 0))
    quality_metrics['vsa_quality'] = min(100, (vsa_signals_count * 25) + (vsa_strength * 10))
    
    # xLSTM уверенность качество
    xlstm_confidence = log_data.get('xlstm_confidence', 0)
    quality_metrics['xlstm_quality'] = xlstm_confidence * 100
    
    # Согласованность моделей
    xlstm_decision = log_data.get('final_decision', 'HOLD')
    rl_decision = log_data.get('rl_decision', 'HOLD')
    quality_metrics['model_consensus'] = 100 if xlstm_decision == rl_decision else 50
    
    # Общее качество сигнала
    quality_metrics['overall_signal_quality'] = (
        quality_metrics['vsa_quality'] * 0.4 +
        quality_metrics['xlstm_quality'] * 0.4 +
        quality_metrics['model_consensus'] * 0.2
    )
    
    return quality_metrics

def update_signal_quality_stats(quality_metrics):
    """
    Обновляет статистику качества сигналов
    """
    stats_file = 'signal_quality_stats.json'
    
    try:
        with open(stats_file, 'r') as f:
            stats = json.load(f)
    except:
        stats = {'total_signals': 0, 'quality_sum': 0, 'quality_history': []}
    
    stats['total_signals'] += 1
    stats['quality_sum'] += quality_metrics['overall_signal_quality']
    stats['quality_history'].append(quality_metrics['overall_signal_quality'])
    
    # Сохраняем только последние 1000 сигналов
    if len(stats['quality_history']) > 1000:
        stats['quality_history'] = stats['quality_history'][-1000:]
        
    stats['average_quality'] = stats['quality_sum'] / stats['total_signals']
    
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)