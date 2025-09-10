import pandas as pd
import os

# --- Константы ---
LOG_FILE = 'trade_log.csv'
REPORT_FILE = 'trade_statistics.xlsx'
# Стандартная комиссия тейкера на Bybit (0.055%)
COMMISSION_RATE = 0.00055

def analyze_trades():
    """
    Анализирует лог торгов, обогащает его данными о PnL и сохраняет в Excel.
    Отображает все исходные столбцы из лога.
    """
    print(f"--- Анализ торгового лога: {LOG_FILE} ---")
    if not os.path.exists(LOG_FILE) or os.path.getsize(LOG_FILE) == 0:
        print(f"Файл лога {LOG_FILE} не найден или пуст.")
        return

    try:
        log_df = pd.read_csv(LOG_FILE)
        # Преобразуем timestamp в читаемый формат, сохраняя исходный для совместимости
        log_df['timestamp_dt'] = pd.to_datetime(log_df['timestamp'], errors='coerce')
    except Exception as e:
        print(f"Ошибка при чтении {LOG_FILE}: {e}")
        return

    # Инициализация новых столбцов для анализа PnL
    pnl_columns = ['open_price', 'close_price', 'side', 'net_pnl_usdt', 'net_pnl_pct', 'commission_usdt']
    for col in pnl_columns:
        log_df[col] = None

    closed_trades = log_df[log_df['decision'].str.contains('CLOSE', na=False)].copy()
    print(f"Найдено {len(closed_trades)} закрытых сделок для анализа.")

    trade_pairs = []

    for idx, closing_trade in closed_trades.iterrows():
        if pd.isna(closing_trade['price']) or closing_trade['price'] == 'N/A':
            continue

        close_price = float(closing_trade['price'])
        close_time = closing_trade['timestamp_dt']
        symbol = closing_trade['symbol']

        # Ищем последнюю открывающую сделку для этого символа перед закрытием
        opening_trades = log_df[
            (log_df['symbol'] == symbol) &
            (log_df['decision'].str.contains('OPEN', na=False)) &
            (log_df['timestamp_dt'] < close_time)
        ].copy()

        if opening_trades.empty:
            continue

        opening_trade = opening_trades.iloc[-1]
        
        # Проверяем, не была ли эта открывающая сделка уже сопоставлена
        if opening_trade.name in [p['open_idx'] for p in trade_pairs]:
            continue

        open_price = float(opening_trade['price'])
        quantity = float(opening_trade['quantity'])
        side = 'Buy' if opening_trade['order_type'] == 'BUY' else 'Sell'
        usdt_amount = open_price * quantity

        if side == 'Buy':
            gross_pnl_pct = (close_price - open_price) / open_price
        else:
            gross_pnl_pct = (open_price - close_price) / open_price

        commission_open = usdt_amount * COMMISSION_RATE
        commission_close = (close_price * quantity) * COMMISSION_RATE
        total_commission_usdt = commission_open + commission_close
        gross_pnl_usdt = usdt_amount * gross_pnl_pct
        net_pnl_usdt = gross_pnl_usdt - total_commission_usdt
        net_pnl_pct = (net_pnl_usdt / usdt_amount) * 100 if usdt_amount != 0 else 0

        # Сохраняем индексы, чтобы не использовать их снова
        trade_pairs.append({'open_idx': opening_trade.name, 'close_idx': idx})

        # Обновляем строки в основном DataFrame
        log_df.loc[idx, 'open_price'] = open_price
        log_df.loc[idx, 'close_price'] = close_price
        log_df.loc[idx, 'side'] = side
        log_df.loc[idx, 'net_pnl_usdt'] = net_pnl_usdt
        log_df.loc[idx, 'net_pnl_pct'] = net_pnl_pct
        log_df.loc[idx, 'commission_usdt'] = total_commission_usdt

    # --- Подготовка данных для Excel ---
    # Создаем копию для отчета, чтобы не изменять оригинальный DataFrame
    report_df = log_df.copy()
    
    # Добавляем все новые столбцы из логов в отчет
    new_columns = [
        'xlstm_pattern_decision', 'xlstm_pattern_confidence',
        'xlstm_indicator_decision', 'xlstm_indicator_confidence',
        'consensus_decision', 'consensus_confidence',
        'RSI_14', 'MACD_12_26_9', 'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0',
        'ADX_14', 'STOCHk_14_3_3', 'STOCHd_14_3_3',
        'CDLHAMMER', 'CDLENGULFING', 'CDLDOJI', 'CDLSHOOTINGSTAR',
        'CDLHANGINGMAN', 'CDLMARUBOZU',  # Заменено CDL3BLACKCROWS
        'hammer_f', 'hangingman_f', 'engulfing_f', 'doji_f',
        'shootingstar_f', 'marubozu_f',  # Заменено 3blackcrows_f
        'hammer_f_on_support', 'hammer_f_vol_spike',
        'hangingman_f_on_res', 'hangingman_f_vol_spike',
        'engulfing_f_strong', 'engulfing_f_vol_confirm',
        'doji_f_high_vol', 'doji_f_high_atr',
        'shootingstar_f_on_res',
        'marubozu_f_strong_body', 'marubozu_f_vol_confirm', 'marubozu_f_bullish'  # Новые признаки
    ]
    
    # Проверяем какие столбцы есть в данных и добавляем только их
    available_new_columns = [col for col in new_columns if col in report_df.columns]
    
    # Форматируем дату
    report_df['timestamp_dt'] = report_df['timestamp_dt'].dt.strftime('%Y-%m-%d %H:%M:%S')

    # --- Создание отдельной таблицы для статистики ---
    completed_trades_df = report_df.dropna(subset=['net_pnl_usdt']).copy()
    
    if not completed_trades_df.empty:
        total_net_pnl_usdt = completed_trades_df['net_pnl_usdt'].sum()
        avg_pnl_pct = completed_trades_df['net_pnl_pct'].mean()
        wins = completed_trades_df[completed_trades_df['net_pnl_usdt'] > 0]
        losses = completed_trades_df[completed_trades_df['net_pnl_usdt'] <= 0]
        win_rate = (len(wins) / len(completed_trades_df)) * 100 if not completed_trades_df.empty else 0

        print("\n--- Итоговая статистика ---")
        print(f"Всего завершенных сделок: {len(completed_trades_df)}")
        print(f"Прибыльных сделок: {len(wins)}")
        print(f"Убыточных сделок: {len(losses)}")
        print(f"Винрейт: {win_rate:.2f}%")
        print(f"ИТОГОВЫЙ PNL: {total_net_pnl_usdt:.4f} USDT")
        print("---------------------------")

        summary_data = {
            'Metric': [
                'Всего сделок', 'Прибыльных сделок', 'Убыточных сделок',
                'Винрейт (%)', 'Итоговый PnL (USDT)', 'Средний PnL (%)'
            ],
            'Value': [
                len(completed_trades_df), len(wins), len(losses),
                f"{win_rate:.2f}", f"{total_net_pnl_usdt:.4f}", f"{avg_pnl_pct:.4f}"
            ]
        }
        stats_summary_df = pd.DataFrame(summary_data)

        # Добавляем строку "Итого" в основной отчет
        total_row_data = {'symbol': ['ИТОГО']}
        
        # Добавляем обязательные столбцы PNL
        total_row_data.update({
            'net_pnl_usdt': [total_net_pnl_usdt],
            'net_pnl_pct': [avg_pnl_pct]
        })
        
        # Добавляем пустые значения для всех остальных столбцов
        for col in report_df.columns:
            if col not in total_row_data:
                total_row_data[col] = [None]
                
        total_row = pd.DataFrame(total_row_data)
        report_df = pd.concat([report_df, total_row], ignore_index=True)

    else:
        print("Нет завершенных сделок для создания статистики.")
        stats_summary_df = pd.DataFrame({'Message': ['Нет данных для статистики']})


    # --- Сохранение в Excel ---
    try:
        with pd.ExcelWriter(REPORT_FILE, engine='xlsxwriter') as writer:
            report_df.to_excel(writer, index=False, sheet_name='Full_Trade_Log')
            if not completed_trades_df.empty:
                stats_summary_df.to_excel(writer, index=False, sheet_name='Statistics_Summary')
        
        print(f"\nОтчет успешно сохранен в файл: {REPORT_FILE}")
    except Exception as e:
        print(f"\nОшибка при сохранении Excel-файла: {e}")

if __name__ == '__main__':
    analyze_trades()