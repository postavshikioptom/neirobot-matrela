import config
from pybit.unified_trading import HTTP

def get_all_linear_symbols():
    """
    Получает все торгуемые линейные фьючерсные контракты, оканчивающиеся на USDT, с Bybit
    и сохраняет их в файл 'symbols.txt'.
    """
    print("Подключение к Bybit для получения списка всех фьючерсных контрактов...")
    # Используем testnet=True, так как работаем с демо-счетом
    session = HTTP(testnet=True)
    
    all_symbols = []
    cursor = ""
    
    while True:
        try:
            response = session.get_instruments_info(
                category="linear", # <<<--- ИЗМЕНЕНО НА 'linear'
                limit=1000,      # Максимальный лимит за один запрос
                cursor=cursor
            )
            
            if 'result' in response and 'list' in response['result']:
                symbol_list = response['result']['list']
                if not symbol_list:
                    break # Список пуст, выходим

                all_symbols.extend(symbol_list)
                cursor = response['result'].get('nextPageCursor', "")
                
                print(f"Получено {len(symbol_list)} контрактов, общее количество: {len(all_symbols)}. Наличие следующей страницы: {bool(cursor)}")

                if not cursor:
                    break  # Выходим из цикла, если курсора больше нет
            else:
                print("Ошибка: Не удалось получить список символов. Ответ API:")
                print(response)
                break

        except Exception as e:
            print(f"Произошла ошибка при запросе к API: {e}")
            break
            
    if all_symbols:
        # Фильтруем только те, что торгуются и базовый актив - USDT
        usdt_symbols = [
            s['symbol'] for s in all_symbols 
            if s['quoteCoin'] == 'USDT' and s['status'] == 'Trading'
        ]
        
        print(f"\nНайдено {len(usdt_symbols)} активных контрактов с USDT.")
        
        # Сохраняем в файл
        file_path = "symbols.txt"
        with open(file_path, 'w') as f:
            for symbol in usdt_symbols:
                f.write(f"{symbol}\n")
        print(f"Список контрактов сохранен в файл '{file_path}'.")
        
    else:
        print("Не удалось получить ни одного контракта.")

if __name__ == "__main__":
    get_all_linear_symbols()
