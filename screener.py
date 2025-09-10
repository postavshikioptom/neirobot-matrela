import time
import pandas as pd
from pybit.unified_trading import HTTP
import config
import json
import os

# --- Константы ---
TRADER_STATUS_FILE = 'trader_status.txt'
LIVE_DATA_FILE = 'live_data.json'
LIVE_DATA_FILE_TMP = LIVE_DATA_FILE + '.tmp'
HOTLIST_FILE = 'hotlist.txt'

def get_trader_status():
    try:
        with open(TRADER_STATUS_FILE, 'r') as f:
            return f.read().strip()
    except FileNotFoundError:
        return 'DONE'

def set_trader_status(status):
    with open(TRADER_STATUS_FILE, 'w') as f:
        f.write(status)

def fetch_and_write_data(symbol, session):
    try:
        print(f"\n!!! НАЙДЕНА ГОРЯЧАЯ МОНЕТА: {symbol}. Загружаю данные... !!!")
        kline_data = session.get_kline(
            category="linear",
            symbol=symbol,
            interval=config.TIMEFRAME,
            limit=config.REQUIRED_CANDLES
        )['result']['list']

        if len(kline_data) < config.REQUIRED_CANDLES:
            print(f"Недостаточно истории для {symbol} ({len(kline_data)}/{config.REQUIRED_CANDLES}). Пропускаю.")
            return False

        kline_data.reverse()
        formatted_klines = []
        for k in kline_data:
            formatted_klines.append([
                int(k[0]), float(k[1]), float(k[2]),
                float(k[3]), float(k[4]), float(k[5]),
                float(k[6])
            ])
        
        live_data_for_one_symbol = {symbol: {'klines': formatted_klines}}

        with open(LIVE_DATA_FILE_TMP, 'w') as f:
            json.dump(live_data_for_one_symbol, f, indent=4)
        os.replace(LIVE_DATA_FILE_TMP, LIVE_DATA_FILE)

        with open(HOTLIST_FILE, 'w') as f:
            f.write(f"{symbol}\n")
        
        print(f"Данные для {symbol} успешно записаны. Передаю задачу трейдеру.")
        return True
    except Exception as e:
        print(f"\n!!! Ошибка при загрузке данных для {symbol}: {e} !!!")
        return False

def check_turnover_spike(df):
    if len(df) < 12: return False
    last_closed_candle = df.iloc[-2]
    previous_10_candles = df.iloc[-12:-2]
    average_turnover = previous_10_candles['turnover'].mean()
    return average_turnover > 0 and last_closed_candle['turnover'] >= (average_turnover * 1.7)

def run_screener():
    try:
        with open("symbols.txt", 'r') as f:
            symbols = [line.strip() for line in f.readlines()]
        print(f"Скринер запущен. Загружено {len(symbols)} контрактов.")
    except FileNotFoundError:
        print("Ошибка: Файл symbols.txt не найден.")
        return

    session = HTTP(testnet=False)
    blacklist = {"BTCUSDT", "ETHUSDT", "SOLUSDT", "PAXGUSDT", "XAUTUSDT", "TUTUSDT", "USDEUSDT", "USDCUSDT", "USD1USDT", "BCHUSDT", "MOTHER2USDT"}
    print(f"Исключены из поиска: {', '.join(blacklist)}")

    set_trader_status('DONE')
    symbol_index = 0

    while True:
        trader_status = get_trader_status()
        if trader_status in ['BUSY', 'MANAGING_ONLY']:
            print(f"Трейдер занят (статус: {trader_status}). Ожидаю...", end='\r')
            time.sleep(5)
            continue

        if symbol_index >= len(symbols):
            print("\nПроверили все символы. Начинаем с начала через 60 секунд.")
            symbol_index = 0
            time.sleep(60)
        
        symbol = symbols[symbol_index]
        symbol_index += 1

        if symbol in blacklist:
            continue

        print(f"({symbol_index}/{len(symbols)}) Проверка {symbol}... ", end='\r')
        try:
            kline_data_for_check = session.get_kline(category="linear", symbol=symbol, interval=config.TIMEFRAME, limit=12)['result']['list']
            if len(kline_data_for_check) < 12: continue

            kline_data_for_check.reverse()
            df = pd.DataFrame(kline_data_for_check, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
            df[list(df.columns)] = df[list(df.columns)].apply(pd.to_numeric)

            if check_turnover_spike(df):
                if fetch_and_write_data(symbol, session):
                    set_trader_status('BUSY')
            
            time.sleep(0.2)

        except Exception as e:
            print(f"\nКритическая ошибка в цикле для {symbol}: {e}")
            time.sleep(10)
            pass

if __name__ == "__main__":
    run_screener()
