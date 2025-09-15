import pandas as pd
from pybit.unified_trading import HTTP
import time
import math
import logging
from typing import List, Optional

# --- Настройка логирования ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Конфигурация ---
TARGET_TOTAL_ROWS = 350000
SYMBOLS_TO_LOAD = 500  # Количество символов для загрузки
KLINE_LIMIT_PER_REQUEST = 100  # Увеличено до максимума Bybit
OUTPUT_FILENAME = 'historical_data.csv'
REQUEST_DELAY = 0.2  # Уменьшена задержка

# --- Инициализация API ---
session = HTTP(testnet=False)

def get_all_usdt_symbols() -> List[str]:
    """Получает список всех торговых инструментов USDT с Bybit."""
    try:
        markets = session.get_instruments_info(category='linear')
        if markets['retCode'] == 0:
            symbols = [s['symbol'] for s in markets['result']['list'] 
                      if s['quoteCoin'] == 'USDT' and s['status'] == 'Trading']
            logger.info(f"Найдено {len(symbols)} активных USDT символов")
            return symbols
        else:
            logger.error(f"Ошибка API при получении символов: {markets['retMsg']}")
            return []
    except Exception as e:
        logger.error(f"Неожиданная ошибка при получении символов: {e}")
        return []

def get_historical_data_for_symbol(symbol: str, total_rows_needed: int) -> pd.DataFrame:
    """Загружает все доступные исторические данные для одного символа за период."""
    all_symbol_data = []
    requests_needed = math.ceil(total_rows_needed / KLINE_LIMIT_PER_REQUEST)
    end_time = None
    
    logger.info(f"Выполняем {requests_needed} запросов для {symbol}...")

    for i in range(requests_needed):
        try:
            response = session.get_kline(
                category='linear',
                symbol=symbol,
                interval='1',  # 1-минутные свечи
                limit=KLINE_LIMIT_PER_REQUEST,
                end=end_time
            )

            if response['retCode'] == 0 and response['result']['list']:
                data = response['result']['list']
                all_symbol_data.extend(data)
                
                # Устанавливаем end_time для следующего запроса (самая старая свеча)
                end_time = int(data[-1][0]) - 1

                # Прерываемся, если данных меньше, чем лимит
                if len(data) < KLINE_LIMIT_PER_REQUEST:
                    logger.info(f"Достигнут конец истории для {symbol}")
                    break
                    
                # Прогресс
                if (i + 1) % 10 == 0:
                    logger.info(f"  Загружено {len(all_symbol_data)} записей для {symbol}")
                    
            else:
                if response['retCode'] != 0:
                    logger.warning(f"Ошибка API для {symbol}: {response['retMsg']}")
                break

            time.sleep(REQUEST_DELAY)

        except Exception as e:
            logger.error(f"Неожиданная ошибка для символа {symbol}: {e}")
            break
    
    if not all_symbol_data:
        logger.warning(f"Не удалось загрузить данные для {symbol}")
        return pd.DataFrame()
        
    # Создаем DataFrame
    df = pd.DataFrame(all_symbol_data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
    ])
    
    # Преобразуем типы данных
    numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'turnover']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df['timestamp'] = pd.to_datetime(pd.to_numeric(df['timestamp']), unit='ms')
    df['symbol'] = symbol
    
    # Удаляем дубликаты и сортируем
    df.drop_duplicates(subset=['timestamp'], inplace=True)
    df.sort_values('timestamp', inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    return df

def save_data_incrementally(data_frames: List[pd.DataFrame], filename: str) -> None:
    """Сохраняет данные инкрементально для экономии памяти."""
    if not data_frames:
        logger.warning("Нет данных для сохранения")
        return
        
    logger.info(f"Сохраняем данные в {filename}...")
    
    # Объединяем все данные
    master_df = pd.concat(data_frames, ignore_index=True)
    
    # Удаляем дубликаты по timestamp и symbol
    master_df.drop_duplicates(subset=['timestamp', 'symbol'], inplace=True)
    master_df.sort_values(['symbol', 'timestamp'], inplace=True)
    master_df.reset_index(drop=True, inplace=True)
    
    # Сохраняем
    master_df.to_csv(filename, index=False)
    logger.info(f"Сохранено {len(master_df)} записей в файл {filename}")

def main():
    """Основная функция программы."""
    logger.info("Получаем список всех USDT символов...")
    usdt_symbols = get_all_usdt_symbols()
    
    if not usdt_symbols:
        logger.error("Не удалось получить список символов. Выход.")
        return
    
    # Ограничиваем количество символов
    symbols_to_process = usdt_symbols[:SYMBOLS_TO_LOAD]
    rows_per_symbol = TARGET_TOTAL_ROWS // len(symbols_to_process)
    
    logger.info(f"Планируется загрузить ~{rows_per_symbol} строк для каждого из {len(symbols_to_process)} символов")

    all_data_frames = []
    successful_symbols = 0
    
    for i, symbol in enumerate(symbols_to_process):
        logger.info(f"({i+1}/{len(symbols_to_process)}) Загружаем данные для {symbol}...")
        
        symbol_df = get_historical_data_for_symbol(symbol, rows_per_symbol)
        
        if not symbol_df.empty:
            all_data_frames.append(symbol_df)
            successful_symbols += 1
            logger.info(f"Загружено {len(symbol_df)} строк для {symbol}")
            
            # Сохраняем промежуточные результаты каждые 50 символов
            if len(all_data_frames) % 50 == 0:
                temp_filename = f"temp_{OUTPUT_FILENAME}"
                save_data_incrementally(all_data_frames.copy(), temp_filename)
        else:
            logger.warning(f"Пропускаем {symbol} - нет данных")

    # Финальное сохранение
    if all_data_frames:
        save_data_incrementally(all_data_frames, OUTPUT_FILENAME)
        logger.info(f"\nУспешно обработано {successful_symbols} символов из {len(symbols_to_process)}")
    else:
        logger.error("\nНе удалось загрузить какие-либо данные")

if __name__ == "__main__":
    main()
