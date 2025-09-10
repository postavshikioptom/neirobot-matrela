import pandas as pd
from pybit.unified_trading import HTTP
import time
import math

# --- Конфигурация ---
TARGET_TOTAL_ROWS = 100000
SYMBOLS_TO_LOAD = 500  # Количество символов для загрузки
KLINE_LIMIT_PER_REQUEST = 100  # Максимум от Bybit
OUTPUT_FILENAME = 'historical_data.csv'

# --- Инициализация API ---
session = HTTP(testnet=False) # Используем основной, а не демо-счет для исторических данных

def get_all_usdt_symbols():
    """Получает полный список бессрочных контрактов USDT с Bybit."""
    try:
        markets = session.get_instruments_info(category='linear')
        if markets['retCode'] == 0:
            return [s['symbol'] for s in markets['result']['list'] if s['quoteCoin'] == 'USDT']
        else:
            print(f"Ошибка API при получении символов: {markets['retMsg']}")
            return []
    except Exception as e:
        print(f"Критическая ошибка при получении символов: {e}")
        return []

def get_historical_data_for_symbol(symbol, total_rows_needed):
    """Загружает все доступные исторические данные для одного символа до лимита."""
    all_symbol_data = []
    requests_needed = math.ceil(total_rows_needed / KLINE_LIMIT_PER_REQUEST)
    end_time = None

    print(f"  Требуется {requests_needed} запросов для {symbol}...")

    for i in range(requests_needed):
        try:
            response = session.get_kline(
                category='linear',
                symbol=symbol,
                interval='1', # 1-минутные свечи
                limit=KLINE_LIMIT_PER_REQUEST,
                end=end_time
            )

            if response['retCode'] == 0 and response['result']['list']:
                data = response['result']['list']
                all_symbol_data.extend(data)
                
                # Устанавливаем end_time для следующего запроса
                end_time = int(data[-1][0])

                # Прерываемся, если данных меньше, чем лимит (значит, дошли до конца истории)
                if len(data) < KLINE_LIMIT_PER_REQUEST:
                    print(f"  Достигнут конец истории для {symbol}.")
                    break
            else:
                if response['retCode'] != 0:
                    print(f"  Ошибка API для {symbol}: {response['retMsg']}")
                break # Прерываем цикл для этого символа при ошибке или отсутствии данных

            time.sleep(0.2) # Пауза между запросами

        except Exception as e:
            print(f"  Критическая ошибка при загрузке {symbol}: {e}")
            break
            
    df = pd.DataFrame(all_symbol_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
    df['timestamp'] = pd.to_datetime(pd.to_numeric(df['timestamp']), unit='ms')
    df['symbol'] = symbol
    return df

if __name__ == "__main__":
    print("Получение списка всех USDT символов...")
    usdt_symbols = get_all_usdt_symbols()
    
    if not usdt_symbols:
        print("Не удалось получить список символов. Выход.")
    else:
        symbols_to_process = usdt_symbols[:SYMBOLS_TO_LOAD]
        rows_per_symbol = TARGET_TOTAL_ROWS // len(symbols_to_process)
        print(f"Планируется загрузить ~{rows_per_symbol} строк для каждого из {len(symbols_to_process)} символов.")

        all_data_frames = []
        for i, symbol in enumerate(symbols_to_process):
            print(f"({i+1}/{len(symbols_to_process)}) Загрузка данных для {symbol}...")
            symbol_df = get_historical_data_for_symbol(symbol, rows_per_symbol)
            
            if not symbol_df.empty:
                all_data_frames.append(symbol_df)
                print(f"  Загружено {len(symbol_df)} строк для {symbol}.")

        if all_data_frames:
            master_df = pd.concat(all_data_frames)
            master_df.drop_duplicates(inplace=True)
            master_df.sort_values('timestamp', inplace=True)
            master_df.to_csv(OUTPUT_FILENAME, index=False)
            print(f"\nВсе данные ({len(master_df)} строк) успешно сохранены в файл: {OUTPUT_FILENAME}")
        else:
            print("\nНе удалось загрузить какие-либо данные.")