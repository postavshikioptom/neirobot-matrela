"""
test_patterns.py - Скрипт для тестирования паттернов на исторических данных

Этот скрипт создает симуляцию, которая:
1. Загружает исторические данные (первые 100 строк для анализа структуры)
2. Генерирует 100,000 строк тестовых данных на основе реальных паттернов
3. Прогоняет все данные через систему расчета паттернов
4. Выводит результаты в test_patterns_result.csv для анализа
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import warnings
import traceback
warnings.filterwarnings('ignore')

# Импортируем наши модули для расчета паттернов
import feature_engineering

def load_and_analyze_historical_data(file_path='historical_data.csv', sample_size=100):
    """
    Загружает и анализирует первые строки исторических данных
    для понимания структуры и диапазонов значений
    """
    print(f"Загружаем первые {sample_size} строк из {file_path}...")
    
    try:
        # Проверяем существование файла
        if not os.path.exists(file_path):
            print(f"Файл {file_path} не найден. Используем значения по умолчанию.")
            return None, []
            
        # Читаем только первые строки для анализа
        df = pd.read_csv(file_path, nrows=sample_size)
        print(f"Загружено {len(df)} строк для анализа")
        
        # Анализируем структуру данных
        print("\nСтруктура данных:")
        print(df.info())
        
        print("\nСтатистика по числовым колонкам:")
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            if col in df.columns:
                print(f"{col}: min={df[col].min():.6f}, max={df[col].max():.6f}, mean={df[col].mean():.6f}")
        
        # Извлекаем уникальные символы
        symbols = df['symbol'].unique() if 'symbol' in df.columns else ['BTCUSDT']
        print(f"\nНайдено символов: {len(symbols)}")
        print(f"Примеры символов: {symbols[:10]}")
        
        return df, symbols
        
    except Exception as e:
        print(f"Ошибка при загрузке данных: {e}")
        return None, []

def generate_realistic_ohlcv_data(base_price=100, num_rows=100000, volatility=0.02):
    """
    Генерирует реалистичные OHLCV данные для тестирования паттернов
    
    Параметры:
    - base_price: базовая цена для генерации
    - num_rows: количество строк для генерации
    - volatility: волатильность (стандартное отклонение изменений цены)
    """
    print(f"Генерируем {num_rows} строк тестовых данных...")
    
    np.random.seed(42)  # Для воспроизводимости результатов
    
    data = []
    current_price = base_price
    base_volume = 10000
    
    # Генерируем временные метки
    start_time = datetime(2024, 1, 1)
    
    for i in range(num_rows):
        # Генерируем изменение цены (случайное блуждание с трендом)
        price_change = np.random.normal(0, volatility * current_price)
        
        # Добавляем небольшой восходящий тренд
        trend = 0.0001 * current_price
        new_close = current_price + price_change + trend
        
        # Генерируем OHLC на основе цены закрытия
        # Open = предыдущая цена закрытия с небольшим гэпом
        open_price = current_price + np.random.normal(0, 0.001 * current_price)
        
        # High и Low создаем с учетом внутридневной волатильности
        intraday_range = abs(np.random.normal(0, 0.01 * current_price))
        high_price = max(open_price, new_close) + intraday_range * np.random.random()
        low_price = min(open_price, new_close) - intraday_range * np.random.random()
        
        # Убеждаемся, что High >= max(Open, Close) и Low <= min(Open, Close)
        high_price = max(high_price, open_price, new_close)
        low_price = min(low_price, open_price, new_close)
        
        # ИСПРАВЛЕНИЕ: Убеждаемся, что цены положительные
        if new_close <= 0:
            new_close = current_price * 0.99
        if open_price <= 0:
            open_price = current_price * 0.99
        if high_price <= 0:
            high_price = max(open_price, new_close) * 1.01
        if low_price <= 0:
            low_price = min(open_price, new_close) * 0.99
        
        # Генерируем объем (коррелирует с волатильностью)
        volume_multiplier = 1 + abs(price_change) / (volatility * current_price + 1e-8)  # Избегаем деления на 0
        volume = max(1, int(base_volume * volume_multiplier * (0.5 + np.random.random())))  # Минимум 1
        
        # Turnover = volume * average_price
        avg_price = (high_price + low_price + open_price + new_close) / 4
        turnover = volume * avg_price
        
        # Создаем временную метку
        timestamp = start_time + timedelta(minutes=i)
        
        data.append({
            'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'open': round(max(0.000001, open_price), 6),  # Минимальная цена
            'high': round(max(0.000001, high_price), 6),
            'low': round(max(0.000001, low_price), 6),
            'close': round(max(0.000001, new_close), 6),
            'volume': volume,
            'turnover': round(max(0, turnover), 6),  # Минимум 0
            'symbol': f'TEST{(i % 10) + 1:02d}USDT'  # Создаем 10 разных тестовых символов
        })
        
        current_price = new_close
        
        # Выводим прогресс каждые 10000 строк
        if (i + 1) % 10000 == 0:
            print(f"Сгенерировано {i + 1} строк...")
    
    df = pd.DataFrame(data)
    print(f"Генерация завершена. Создано {len(df)} строк данных")
    
    return df

def add_pattern_scenarios(df):
    """
    Добавляет специальные сценарии для тестирования конкретных паттернов
    Это гарантирует, что некоторые паттерны точно будут обнаружены
    """
    print("Добавляем специальные паттерны для тестирования...")
    
    # Создаем копию для модификации
    test_df = df.copy()
    
    # ИСПРАВЛЕНИЕ: Добавляем более выраженные паттерны
    
    # Добавляем Hammer паттерны (каждые 500 строк)
    hammer_count = 0
    for i in range(100, len(test_df), 500):
        if i < len(test_df):
            # Hammer: маленькое тело, длинная нижняя тень
            close_prev = test_df.iloc[i-1]['close'] if i > 0 else test_df.iloc[i]['close']
            open_price = close_prev
            close_price = open_price * 1.01  # Небольшое бычье тело
            high_price = close_price * 1.005
            low_price = open_price * 0.97  # ДЛИННАЯ нижняя тень (3% вниз)
            
            test_df.iloc[i, test_df.columns.get_loc('open')] = open_price
            test_df.iloc[i, test_df.columns.get_loc('high')] = high_price
            test_df.iloc[i, test_df.columns.get_loc('low')] = low_price
            test_df.iloc[i, test_df.columns.get_loc('close')] = close_price
            hammer_count += 1
    
    # Добавляем Shooting Star паттерны (каждые 700 строк)
    shooting_star_count = 0
    for i in range(200, len(test_df), 700):
        if i < len(test_df):
            # Shooting Star: маленькое тело, длинная верхняя тень
            close_prev = test_df.iloc[i-1]['close'] if i > 0 else test_df.iloc[i]['close']
            open_price = close_prev
            close_price = open_price * 0.99  # Небольшое медвежье тело
            low_price = close_price * 0.995
            high_price = open_price * 1.03  # ДЛИННАЯ верхняя тень (3% вверх)
            
            test_df.iloc[i, test_df.columns.get_loc('open')] = open_price
            test_df.iloc[i, test_df.columns.get_loc('high')] = high_price
            test_df.iloc[i, test_df.columns.get_loc('low')] = low_price
            test_df.iloc[i, test_df.columns.get_loc('close')] = close_price
            shooting_star_count += 1
    
    # Добавляем Doji паттерны (каждые 600 строк)
    doji_count = 0
    for i in range(300, len(test_df), 600):
        if i < len(test_df):
            # Doji: Open ≈ Close
            close_prev = test_df.iloc[i-1]['close'] if i > 0 else test_df.iloc[i]['close']
            open_price = close_prev
            close_price = open_price * (1 + np.random.normal(0, 0.0005))  # Почти равны
            high_price = open_price * 1.015  # Длинные тени
            low_price = open_price * 0.985
            
            test_df.iloc[i, test_df.columns.get_loc('open')] = open_price
            test_df.iloc[i, test_df.columns.get_loc('high')] = high_price
            test_df.iloc[i, test_df.columns.get_loc('low')] = low_price
            test_df.iloc[i, test_df.columns.get_loc('close')] = close_price
            doji_count += 1
    
    # ДОБАВЛЯЕМ: Engulfing паттерны (каждые 800 строк)
    engulfing_count = 0
    for i in range(1, len(test_df), 800):
        if i < len(test_df) - 1:
            # Bullish Engulfing: большая белая свеча поглощает предыдущую черную
            # Первая свеча (медвежья)
            prev_open = test_df.iloc[i-1]['close']
            prev_close = prev_open * 0.98
            test_df.iloc[i-1, test_df.columns.get_loc('open')] = prev_open
            test_df.iloc[i-1, test_df.columns.get_loc('close')] = prev_close
            test_df.iloc[i-1, test_df.columns.get_loc('high')] = prev_open * 1.002
            test_df.iloc[i-1, test_df.columns.get_loc('low')] = prev_close * 0.998
            
            # Вторая свеча (бычья, поглощающая)
            curr_open = prev_close * 0.995  # Открытие ниже закрытия предыдущей
            curr_close = prev_open * 1.02   # Закрытие выше открытия предыдущей
            test_df.iloc[i, test_df.columns.get_loc('open')] = curr_open
            test_df.iloc[i, test_df.columns.get_loc('close')] = curr_close
            test_df.iloc[i, test_df.columns.get_loc('high')] = curr_close * 1.002
            test_df.iloc[i, test_df.columns.get_loc('low')] = curr_open * 0.998
            engulfing_count += 1
    
    # ДОБАВЛЯЕМ: Marubozu паттерны (каждые 900 строк)
    marubozu_count = 0
    for i in range(400, len(test_df), 900):
        if i < len(test_df):
            # Marubozu: Open ≈ Low, Close ≈ High (или наоборот для медвежьего)
            close_prev = test_df.iloc[i-1]['close'] if i > 0 else test_df.iloc[i]['close']
            
            # Случайно выбираем бычий или медвежий Marubozu
            is_bullish = np.random.choice([True, False])
            
            if is_bullish:  # Бычий Marubozu
                open_price = close_prev * 0.99
                close_price = open_price * 1.04  # Сильный рост 4%
                low_price = open_price * 1.001   # Почти равен Open
                high_price = close_price * 0.999 # Почти равен Close
            else:  # Медвежий Marubozu
                open_price = close_prev * 1.01
                close_price = open_price * 0.96  # Сильное падение 4%
                high_price = open_price * 0.999  # Почти равен Open
                low_price = close_price * 1.001  # Почти равен Close
            
            test_df.iloc[i, test_df.columns.get_loc('open')] = open_price
            test_df.iloc[i, test_df.columns.get_loc('high')] = high_price
            test_df.iloc[i, test_df.columns.get_loc('low')] = low_price
            test_df.iloc[i, test_df.columns.get_loc('close')] = close_price
            marubozu_count += 1
    
    print(f"Добавлено специальных паттернов:")
    print(f"  - Hammer: {hammer_count}")
    print(f"  - Shooting Star: {shooting_star_count}")
    print(f"  - Doji: {doji_count}")
    print(f"  - Engulfing: {engulfing_count}")
    print(f"  - Marubozu: {marubozu_count}")  # Заменено 3 Black Crows
    
    return test_df

def analyze_pattern_results(final_df):
    """
    НОВАЯ ФУНКЦИЯ: Детальный анализ результатов паттернов
    """
    print(f"\n=== ДЕТАЛЬНЫЙ АНАЛИЗ РЕЗУЛЬТАТОВ ===")
    
    # Находим все колонки с паттернами
    pattern_columns = [col for col in final_df.columns if col.startswith(('CDL', 'hammer_', 'hangingman_', 'engulfing_', 'doji_', 'shootingstar_', 'marubozu_'))]
    
    print(f"Найдено колонок с паттернами: {len(pattern_columns)}")
    
    # Разделяем на основные паттерны и признаки
    base_patterns = [col for col in pattern_columns if col.startswith('CDL')]
    feature_patterns = [col for col in pattern_columns if not col.startswith('CDL')]
    
    print(f"\nОсновные паттерны TA-Lib:")
    always_zero = []
    working_patterns = []
    
    for col in sorted(base_patterns):
        count = final_df[col].ne(0).sum()
        unique_values = sorted(final_df[col].unique())  # Сортируем значения
        
        if count == 0:
            always_zero.append(col)
        else:
            working_patterns.append(f"  - {col}: {count} раз (значения: {unique_values})")
    
    if working_patterns:
        print("\n  Рабочие паттерны (обнаружены > 0 раз):")
        for p in working_patterns:
            print(p)
    
    if always_zero:
        print("\n  Паттерны, которые всегда возвращали 0 (возможно, не работают или не были сгенерированы):")
        for col in always_zero:
            print(f"  - {col}")
            
    # ДОБАВЛЯЕМ: Специальная проверка для CDLSHOOTINGSTAR и CDL3BLACKCROWS
    print(f"\n=== СПЕЦИАЛЬНАЯ ПРОВЕРКА ПРОБЛЕМНЫХ ПАТТЕРНОВ ===")
    problem_patterns = ['CDLSHOOTINGSTAR', 'CDLMARUBOZU']
    for pattern in problem_patterns:
        if pattern in final_df.columns:
            count = final_df[pattern].ne(0).sum()
            unique_vals = sorted(final_df[pattern].unique())
            max_val = final_df[pattern].max()
            min_val = final_df[pattern].min()
            print(f"{pattern}:")
            print(f"  - Найдено: {count} раз")
            print(f"  - Уникальные значения: {unique_vals}")
            print(f"  - Диапазон: {min_val} до {max_val}")
            if count == 0:
                print(f"  - ⚠️  ПРОБЛЕМА: Паттерн никогда не обнаруживается!")
        else:
            print(f"{pattern}: колонка не найдена в данных")
            
    print(f"\nПризнаки на основе паттернов:")
    feature_working = 0
    feature_total = len(feature_patterns)
    
    for col in sorted(feature_patterns):
        count = final_df[col].ne(0).sum()
        if count > 0:
            print(f"  - {col}: {count} раз")
            feature_working += 1
        else:
            print(f"  - {col}: 0 раз (не работает)")
    
    print(f"\n=== ИТОГОВАЯ СТАТИСТИКА ===")
    print(f"Основных паттернов TA-Lib: {len(base_patterns)}")
    print(f"  - Работающих: {len(working_patterns)}")
    print(f"  - Всегда ноль: {len(always_zero)}")
    print(f"Признаков паттернов: {feature_total}")
    print(f"  - Работающих: {feature_working}")
    print(f"  - Не работающих: {feature_total - feature_working}")

def create_detailed_report(final_df, output_file):
    """
    Создает подробный отчет в текстовом файле
    """
    report_file = output_file.replace('.csv', '_report.txt')
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=== ОТЧЕТ О ТЕСТИРОВАНИИ ПАТТЕРНОВ ===\n")
        f.write(f"Дата создания: {datetime.now()}\n")
        f.write(f"Общее количество строк: {len(final_df)}\n\n")
        
        # Анализ паттернов
        pattern_columns = [col for col in final_df.columns if col.startswith(('CDL', 'hammer_', 'hangingman_', 'engulfing_', 'doji_', 'shootingstar_', 'marubozu_'))]
        base_patterns = [col for col in pattern_columns if col.startswith('CDL')]
        
        f.write("=== АНАЛИЗ ОСНОВНЫХ ПАТТЕРНОВ ===\n")
        for col in sorted(base_patterns):
            count = final_df[col].ne(0).sum()
            unique_values = sorted(final_df[col].unique())
            f.write(f"{col}: {count} обнаружений, значения: {unique_values}\n")
        
        f.write(f"\nОтчет сохранен в: {os.path.abspath(report_file)}")
    
    print(f"Подробный отчет сохранен в: {report_file}")

def run_pattern_simulation():
    """
    Основная функция симуляции паттернов
    """
    print("=== ЗАПУСК СИМУЛЯЦИИ ТЕСТИРОВАНИЯ ПАТТЕРНОВ ===")
    print(f"Время начала: {datetime.now()}")
    
    # Шаг 1: Анализ исторических данных
    historical_df, symbols = load_and_analyze_historical_data()
    
    if historical_df is None:
        print("Не удалось загрузить исторические данные, используем значения по умолчанию")
        base_price = 50.0
    else:
        # Используем среднюю цену из исторических данных как базу
        base_price = historical_df['close'].mean() if 'close' in historical_df.columns else 50.0
    
    print(f"Базовая цена для генерации: {base_price}")
    
    # Шаг 2: Генерация тестовых данных
    test_data = generate_realistic_ohlcv_data(
        base_price=base_price, 
        num_rows=100000, 
        volatility=0.02
    )
    
    # Шаг 3: Добавление специальных паттернов
    test_data = add_pattern_scenarios(test_data)
    
    # Шаг 4: Обработка данных по символам и расчет паттернов
    print("\n=== НАЧИНАЕМ РАСЧЕТ ПАТТЕРНОВ ===")
    
    all_results = []
    symbols_in_data = test_data['symbol'].unique()
    
    for i, symbol in enumerate(symbols_in_data):
        print(f"\nОбрабатываем символ {symbol} ({i+1}/{len(symbols_in_data)})...")
        
        # Фильтруем данные по символу
        symbol_data = test_data[test_data['symbol'] == symbol].copy().reset_index(drop=True)
        print(f"Строк для символа {symbol}: {len(symbol_data)}")
        
        try:
            # Шаг 4.1: Расчет технических индикаторов
            print("  - Расчет технических индикаторов...")
            features_df = feature_engineering.calculate_features(symbol_data)
            
            if features_df.empty:
                print(f"  - Пропускаем {symbol}: нет данных после расчета индикаторов")
                continue
            
            print(f"  - Строк после расчета индикаторов: {len(features_df)}")
            
            # Шаг 4.2: Обнаружение паттернов свечей
            print("  - Обнаружение паттернов свечей...")
            patterns_df = feature_engineering.detect_candlestick_patterns(features_df)
            
            if patterns_df.empty:
                print(f"  - Пропускаем {symbol}: нет данных после обнаружения паттернов")
                continue
            
            print(f"  - Строк после обнаружения паттернов: {len(patterns_df)}")
            
            # Добавляем символ обратно для идентификации
            patterns_df['symbol'] = symbol
            
            # Собираем результаты
            all_results.append(patterns_df)
            
        except Exception as e:
            print(f"  - ОШИБКА при обработке {symbol}: {e}")
            traceback.print_exc()  # ДОБАВЛЯЕМ: Подробную информацию об ошибке
            continue
    
    # Шаг 5: Объединение всех результатов
    if not all_results:
        print("\nОШИБКА: Нет результатов для сохранения!")
        return
    
    print(f"\n=== ОБЪЕДИНЯЕМ РЕЗУЛЬТАТЫ ===")
    final_df = pd.concat(all_results, ignore_index=True)
    print(f"Общее количество обработанных строк: {len(final_df)}")
    
    # Шаг 6: Анализ результатов
    analyze_pattern_results(final_df)
            
    # Шаг 7: Сохранение результатов
    output_file = 'test_patterns_result.csv'
    print(f"\nСохраняем результаты в файл: {output_file}")
    final_df.to_csv(output_file, index=False)
    print(f"Результаты успешно сохранены в {os.path.abspath(output_file)}")
    
    # ДОБАВЛЯЕМ: Создание подробного отчета
    create_detailed_report(final_df, output_file)
    
    print(f"\n=== СИМУЛЯЦИЯ ЗАВЕРШЕНА ===")
    print(f"Время окончания: {datetime.now()}")
    print(f"\nФайлы созданы:")
    print(f"  - Данные: {os.path.abspath(output_file)}")
    print(f"  - Отчет: {os.path.abspath(output_file.replace('.csv', '_report.txt'))}")

if __name__ == '__main__':
    run_pattern_simulation()