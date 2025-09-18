import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
import pickle
import os
import talib # 🔥 ДОБАВЛЕНО: Импорт TA-Lib
import tensorflow as tf # 🔥 ДОБАВЛЕНО: Импорт Tensorflow
import config # 🔥 ДОБАВЛЕНО: Импорт config для параметров индикаторов
import gc
from collections import deque
import logging

def log_nan_inf_stats(df, stage_name="Unknown"):
    """🔥 ДОБАВЛЕНО: Логирование статистики NaN и inf значений"""
    nan_stats = {}
    inf_stats = {}
    
    for col in df.columns:
        if df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
            nan_count = df[col].isna().sum()
            inf_count = np.isinf(df[col]).sum()
            
            if nan_count > 0:
                nan_stats[col] = {
                    'count': nan_count,
                    'percentage': nan_count / len(df) * 100
                }
            
            if inf_count > 0:
                inf_stats[col] = {
                    'count': inf_count,
                    'percentage': inf_count / len(df) * 100
                }
    
    if nan_stats:
        print(f"⚠️ NaN обнаружены в {stage_name}:")
        for col, stats in nan_stats.items():
            print(f"  {col}: {stats['count']} ({stats['percentage']:.2f}%)")
    
    if inf_stats:
        print(f"⚠️ Inf обнаружены в {stage_name}:")
        for col, stats in inf_stats.items():
            print(f"  {col}: {stats['count']} ({stats['percentage']:.2f}%)")
    
    return nan_stats, inf_stats

def safe_fill_nan_inf(df, method='median'):
    """🔥 ДОБАВЛЕНО: Безопасное заполнение NaN и inf значений"""
    df_clean = df.copy()
    
    # Логируем статистику до очистки
    nan_stats, inf_stats = log_nan_inf_stats(df_clean, "До очистки")
    
    for col in df_clean.columns:
        if df_clean[col].dtype in ['float64', 'float32', 'int64', 'int32']:
            # Заменяем inf на NaN
            df_clean[col] = df_clean[col].replace([np.inf, -np.inf], np.nan)
            
            # Заполняем NaN в зависимости от метода
            if method == 'median':
                fill_value = df_clean[col].median()
                if pd.isna(fill_value):
                    # Если медиана тоже NaN, используем среднее
                    fill_value = df_clean[col].mean()
                    if pd.isna(fill_value):
                        # Если и среднее NaN, используем 0
                        fill_value = 0.0
            elif method == 'mean':
                fill_value = df_clean[col].mean()
                if pd.isna(fill_value):
                    fill_value = 0.0
            else:  # method == 'zero'
                fill_value = 0.0
            
            df_clean[col] = df_clean[col].fillna(fill_value)
    
    # Логируем статистику после очистки
    post_nan, post_inf = log_nan_inf_stats(df_clean, "После очистки")
    # if for any important indicator post_nan still > 0:
    for col, stats in post_nan.items():
        if stats['percentage'] > 0.5:
            print(f"⚠️ После очистки {col} имеет {stats['percentage']:.2f}% NaN — проверить источник данных")
    
    return df_clean

class FeatureEngineering:
    """
    Класс для обработки и подготовки признаков для модели, включая технические индикаторы
    """
    def __init__(self, sequence_length=60):
        self.sequence_length = sequence_length
        self.scaler = RobustScaler()  # Более устойчив к выбросам чем StandardScaler
        # 🔥 ИЗМЕНЕНО: Исходные колонки для расчета индикаторов
        self.base_features = ['open', 'high', 'low', 'close', 'volume', 'turnover']
        self.feature_columns = list(self.base_features) # Будут обновлены после добавления индикаторов
        # 🔥 ДОБАВЛЕНО: Кэширование результатов
        self.indicator_cache = {}
        self.cache_max_size = 1000
        self.fallback_retry_count = 0
        self.max_fallback_retries = 3
    
    def _validate_data_for_indicators(self, df):
        """🔥 ДОБАВЛЕНО: Предварительная валидация данных"""
        if df is None or df.empty:
            return False, "Пустой DataFrame"
        
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            return False, f"Отсутствуют колонки: {missing_cols}"
        
        if len(df) < config.RSI_PERIOD + 5:
            return False, f"Недостаточно данных: {len(df)} < {config.RSI_PERIOD + 5}"
        
        # Проверка на слишком много NaN
        for col in required_cols:
            nan_percent = df[col].isna().sum() / len(df)
            if nan_percent > 0.5:
                return False, f"Слишком много NaN в колонке {col}: {nan_percent:.2%}"
        
        return True, "OK"
    
    def _get_cache_key(self, df):
        """🔥 ДОБАВЛЕНО: Создание ключа кэша для DataFrame"""
        try:
            # Используем hash от первых/последних значений и размера
            first_vals = tuple(df.iloc[0][['open', 'high', 'low', 'close']].values)
            last_vals = tuple(df.iloc[-1][['open', 'high', 'low', 'close']].values)
            return hash((first_vals, last_vals, len(df)))
        except:
            return None
        
    def _add_technical_indicators(self, df):
        """
        Добавляет технические индикаторы с предварительной валидацией и кэшированием
        """
        try:
            # 🔥 ДОБАВЛЕНО: Предварительная валидация
            is_valid, error_msg = self._validate_data_for_indicators(df)
            if not is_valid:
                print(f"Валидация данных не пройдена: {error_msg}")
                return self._create_fallback_indicators_df()
            
            # 🔥 ДОБАВЛЕНО: Проверка кэша
            cache_key = self._get_cache_key(df)
            if cache_key and cache_key in self.indicator_cache:
                print("Используем кэшированные индикаторы")
                cached_result = self.indicator_cache[cache_key].copy()
                return cached_result
            
            # Убедимся, что все необходимые колонки в числовом формате
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 🔥 ИСПРАВЛЕНО: Массовый расчет всех индикаторов
            indicators_success = self._calculate_all_indicators_batch(df)
            
            if not indicators_success:
                print("Ошибка при массовом расчете индикаторов, используем fallback")
                return self._create_fallback_indicators_df()
            
            # Обновляем список признаков
            self.feature_columns = self.base_features + [
                # Трендовые индикаторы
                'EMA_7', 'EMA_14', 'EMA_21',
                'MACD', 'MACDSIGNAL', 'MACDHIST',
                'KAMA', 'SUPERTREND',
                
                # Momentum индикаторы
                'RSI', 'CMO', 'ROC',
                
                # Volume индикаторы
                'OBV', 'MFI',
                
                # Volatility индикаторы
                'ATR', 'NATR',
                
                # Statistical индикаторы
                'STDDEV',
                
                # Cycle индикаторы
                'HT_DCPERIOD', 'HT_SINE', 'HT_LEADSINE'
            ]
            
            # 🔥 ДОБАВЛЕНО: Кэшируем результат
            if cache_key and len(self.indicator_cache) < self.cache_max_size:
                self.indicator_cache[cache_key] = df.copy()
            elif len(self.indicator_cache) >= self.cache_max_size:
                # Очищаем кэш при переполнении
                self.indicator_cache.clear()
                gc.collect()
            
        except Exception as e:
            print(f"❌ Критическая ошибка в _add_technical_indicators: {e}")
            return self._create_fallback_indicators_df()
        
        # Надёжная обработка NaN
        try:
            df = safe_fill_nan_inf(df, method='median')
        except Exception as e:
            print(f"Ошибка при безопасной обработке NaN/inf: {e}")
            df = df.fillna(0)
        
        return df
    
    def _calculate_all_indicators_batch(self, df):
        """🔥 ОБНОВЛЕНО: Массовый расчет всех новых индикаторов"""
        try:
            # Предварительно заполняем NaN значения в исходных данных
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in df.columns:
                    # Используем forward fill, затем backward fill для заполнения NaN
                    df[col] = df[col].ffill().bfill()
            
            # Преобразуем в float64 для talib
            close_prices = df['close'].astype(np.float64).values
            high_prices = df['high'].astype(np.float64).values
            low_prices = df['low'].astype(np.float64).values
            volume_prices = df['volume'].astype(np.float64).values
            
            # Заменяем нулевые цены на минимальные положительные значения для предотвращения деления на ноль
            close_prices = np.where(close_prices <= 0, 1e-8, close_prices)
            high_prices = np.where(high_prices <= 0, 1e-8, high_prices)
            low_prices = np.where(low_prices <= 0, 1e-8, low_prices)
            volume_prices = np.where(volume_prices <= 0, 1e-8, volume_prices)
            
            # Массовый расчет всех индикаторов
            indicators = {}
            
            # === ТРЕНДОВЫЕ ИНДИКАТОРЫ ===
            
            # EMA (7, 14, 21)
            for period in config.EMA_PERIODS:
                indicators[f'EMA_{period}'] = talib.EMA(close_prices, timeperiod=period)
            
            # MACD (сохраняется)
            macd, macdsignal, macdhist = talib.MACD(
                close_prices,
                fastperiod=config.MACD_FASTPERIOD,
                slowperiod=config.MACD_SLOWPERIOD,
                signalperiod=config.MACD_SIGNALPERIOD
            )
            indicators['MACD'] = macd
            indicators['MACDSIGNAL'] = macdsignal
            indicators['MACDHIST'] = macdhist
            
            # KAMA
            indicators['KAMA'] = talib.KAMA(close_prices, timeperiod=config.KAMA_PERIOD)
            
            # SuperTrend (кастомная реализация)
            indicators['SUPERTREND'] = self._calculate_supertrend(
                high_prices, low_prices, close_prices, 
                config.SUPERTREND_PERIOD, config.SUPERTREND_MULTIPLIER
            )
            
            # === MOMENTUM ИНДИКАТОРЫ ===
            
            # RSI (сохраняется)
            indicators['RSI'] = talib.RSI(close_prices, timeperiod=config.RSI_PERIOD)
            
            # CMO
            indicators['CMO'] = talib.CMO(close_prices, timeperiod=config.CMO_PERIOD)
            
            # ROC
            indicators['ROC'] = talib.ROC(close_prices, timeperiod=config.ROC_PERIOD)
            
            # === VOLUME ИНДИКАТОРЫ ===
            
            # OBV
            indicators['OBV'] = talib.OBV(close_prices, volume_prices)
            
            # MFI
            indicators['MFI'] = talib.MFI(
                high_prices, low_prices, close_prices, volume_prices, 
                timeperiod=config.RSI_PERIOD
            )
            
            # === VOLATILITY ИНДИКАТОРЫ ===
            
            # ATR
            indicators['ATR'] = talib.ATR(
                high_prices, low_prices, close_prices, 
                timeperiod=config.ATR_PERIOD
            )
            
            # NATR
            indicators['NATR'] = talib.NATR(
                high_prices, low_prices, close_prices, 
                timeperiod=config.NATR_PERIOD
            )
            
            # === STATISTICAL ИНДИКАТОРЫ ===
            
            # STDDEV
            indicators['STDDEV'] = talib.STDDEV(close_prices, timeperiod=config.STDDEV_PERIOD)
            
            # === CYCLE ИНДИКАТОРЫ ===
            
            # HT_DCPERIOD
            indicators['HT_DCPERIOD'] = talib.HT_DCPERIOD(close_prices)
            
            # HT_SINE
            sine, leadsine = talib.HT_SINE(close_prices)
            indicators['HT_SINE'] = sine
            indicators['HT_LEADSINE'] = leadsine
            
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
    
    def _calculate_supertrend(self, high, low, close, period=10, multiplier=3.0):
        """Расчет SuperTrend индикатора"""
        try:
            # Расчет ATR
            atr = talib.ATR(high, low, close, timeperiod=period)
            
            # Расчет базовых линий
            hl2 = (high + low) / 2
            upper_band = hl2 + (multiplier * atr)
            lower_band = hl2 - (multiplier * atr)
            
            # Инициализация массивов
            supertrend = np.zeros_like(close)
            direction = np.ones_like(close)
            
            # Расчет SuperTrend
            for i in range(1, len(close)):
                # Обновление верхней полосы
                if upper_band[i] < upper_band[i-1] or close[i-1] > upper_band[i-1]:
                    upper_band[i] = upper_band[i]
                else:
                    upper_band[i] = upper_band[i-1]
                
                # Обновление нижней полосы
                if lower_band[i] > lower_band[i-1] or close[i-1] < lower_band[i-1]:
                    lower_band[i] = lower_band[i]
                else:
                    lower_band[i] = lower_band[i-1]
                
                # Определение направления тренда
                if close[i] <= lower_band[i-1]:
                    direction[i] = -1
                elif close[i] >= upper_band[i-1]:
                    direction[i] = 1
                else:
                    direction[i] = direction[i-1]
                
                # Расчет SuperTrend
                if direction[i] == 1:
                    supertrend[i] = lower_band[i]
                else:
                    supertrend[i] = upper_band[i]
            
            return supertrend
            
        except Exception as e:
            print(f"Ошибка при расчете SuperTrend: {e}")
            return np.zeros_like(close)

    def _create_fallback_indicators_df(self, df=None):
        """🔥 ИСПРАВЛЕНО: Создает DataFrame с fallback значениями без рекурсии"""
        self.fallback_retry_count += 1
        
        if self.fallback_retry_count > self.max_fallback_retries:
            print(f"❌ Превышено максимальное количество попыток fallback: {self.max_fallback_retries}")
            # Сбрасываем счетчик для следующих вызовов
            self.fallback_retry_count = 0
            return None
        
        if df is None:
            # Создаем минимальный DataFrame
            df = pd.DataFrame({
                'open': [100.0],
                'high': [101.0],
                'low': [99.0],
                'close': [100.5],
                'volume': [1000.0],
                'turnover': [100000.0]
            })
        
        # Добавляем fallback индикаторы с медианными значениями
        # Трендовые индикаторы
        for period in config.EMA_PERIODS:
            df[f'EMA_{period}'] = 100.0
        df['MACD'] = 0.0
        df['MACDSIGNAL'] = 0.0
        df['MACDHIST'] = 0.0
        df['KAMA'] = 100.0
        df['SUPERTREND'] = 100.0
        
        # Momentum индикаторы
        df['RSI'] = 50.0
        df['CMO'] = 0.0
        df['ROC'] = 0.0
        
        # Volume индикаторы
        df['OBV'] = 0.0
        df['MFI'] = 50.0
        
        # Volatility индикаторы
        df['ATR'] = 1.0
        df['NATR'] = 1.0
        
        # Statistical индикаторы
        df['STDDEV'] = 1.0
        
        # Cycle индикаторы
        df['HT_DCPERIOD'] = 20.0
        df['HT_SINE'] = 0.0
        df['HT_LEADSINE'] = 0.0
        
        self.feature_columns = self.base_features + [
            # Трендовые индикаторы
            'EMA_7', 'EMA_14', 'EMA_21',
            'MACD', 'MACDSIGNAL', 'MACDHIST',
            'KAMA', 'SUPERTREND',
            
            # Momentum индикаторы
            'RSI', 'CMO', 'ROC',
            
            # Volume индикаторы
            'OBV', 'MFI',
            
            # Volatility индикаторы
            'ATR', 'NATR',
            
            # Statistical индикаторы
            'STDDEV',
            
            # Cycle индикаторы
            'HT_DCPERIOD', 'HT_SINE', 'HT_LEADSINE'
        ]
        
        # Сбрасываем счетчик после успешного создания
        self.fallback_retry_count = 0
        
        return df

    def prepare_data(self, df):
        """
        Подготовка данных: добавление индикаторов, нормализация и создание последовательностей
        """
        # Проверяем, что timestamp в числовом формате
        if 'timestamp' in df.columns:
            if not pd.api.types.is_numeric_dtype(df['timestamp']):
                print(f"⚠️ timestamp не в числовом формате: {df['timestamp'].dtype}, преобразуем")
                df['timestamp'] = pd.to_numeric(df['timestamp'])
            print(f"Тип timestamp: {df['timestamp'].dtype}")
        
        # Сортируем по timestamp (числовому)
        df = df.sort_values('timestamp')
        
        # 🔥 ИЗМЕНЕНО: Добавляем индикаторы
        df_with_indicators = self._add_technical_indicators(df.copy())
        
        # Преобразуем признаки в числовой формат
        for col in self.feature_columns:
            df_with_indicators[col] = pd.to_numeric(df_with_indicators[col], errors='coerce')
        
        # Получаем только нужные колонки
        data = df_with_indicators[self.feature_columns].values
        
        # Обучаем скейлер на всех данных (теперь включая индикаторы)
        scaled_data = self.scaler.fit_transform(data)
        
        # Создаем последовательности
        X, y_close = self._create_sequences(scaled_data)
        
        return X, y_close, df_with_indicators
    
    def prepare_test_data(self, df):
        """
        Подготовка тестовых данных с использованием уже обученного скейлера и индикаторов
        """
        df = df.sort_values('timestamp')
        
        # 🔥 ИЗМЕНЕНО: Добавляем индикаторы
        df_with_indicators = self._add_technical_indicators(df.copy())
        
        # Преобразуем признаки в числовой формат
        for col in self.feature_columns:
            df_with_indicators[col] = pd.to_numeric(df_with_indicators[col], errors='coerce')
        
        # Получаем только нужные колонки
        data = df_with_indicators[self.feature_columns].values
        
        # Применяем уже обученный скейлер
        scaled_data = self.scaler.transform(data)
        
        # Создаем последовательности
        X, y_close = self._create_sequences(scaled_data)
        
        return X, y_close, df_with_indicators
    
    def _create_sequences(self, data):
        """Создает последовательности для обучения с обработкой ошибок"""
        try:
            if data is None or len(data) == 0:
                print("❌ Пустые данные переданы в _create_sequences")
                return np.array([]), np.array([])
            
            # print(f"Создание последовательностей из данных формы {data.shape}") # 🔥 Убрано лишнее логирование
            
            if len(data) <= self.sequence_length:
                print(f"❌ Недостаточно данных для создания последовательностей: {len(data)} <= {self.sequence_length}")
                if len(data) > 10:
                    reduced_sequence_length = len(data) - 5
                    print(f"Пробуем создать последовательности с уменьшенной длиной {reduced_sequence_length}")
                    X = []
                    y_close = []
                    
                    close_index = 3
                    try:
                        if hasattr(self, 'base_features') and 'close' in self.base_features:
                            close_index = self.base_features.index('close')
                    except (ValueError, AttributeError):
                        pass
                    
                    X.append(data[:reduced_sequence_length])
                    y_close.append(data[reduced_sequence_length, close_index])
                    
                    return np.array(X), np.array(y_close)
                else:
                    return np.array([]), np.array([])
            
            X = []
            y_close = []
            
            close_index = 3
            try:
                if hasattr(self, 'base_features') and 'close' in self.base_features:
                    close_index = self.base_features.index('close')
            except (ValueError, AttributeError):
                close_index = 3
            
            for i in range(len(data) - self.sequence_length):
                try:
                    # Создаем последовательность
                    sequence = data[i:i+self.sequence_length]
                    
                    # Проверяем на NaN/inf в последовательности
                    if np.isnan(sequence).any() or np.isinf(sequence).any():
                        print(f"Обнаружен NaN/inf в последовательности {i}, пропускаем")
                        continue
                    
                    X.append(sequence)
                    y_close.append(data[i+self.sequence_length, close_index])
                except (IndexError, ValueError) as e:
                    print(f"Ошибка при создании последовательности {i}: {e}")
                    continue
            
            if len(X) == 0:
                print("❌ Не удалось создать ни одной последовательности")
                return np.array([]), np.array([])
            
            return np.array(X), np.array(y_close)
            
        except Exception as e:
            print(f"❌ Критическая ошибка в _create_sequences: {e}")
            return np.array([]), np.array([])
    
    def calculate_adaptive_threshold(self, df, base_threshold=None):
        """
        Рассчитывает адаптивный порог на основе волатильности рынка
        
        Args:
            df (pd.DataFrame): DataFrame с данными цен
            base_threshold (float, optional): Базовый порог для минимального изменения. Если None, берется из config.
            
        Returns:
            float: Адаптивный порог для определения сигналов
        """
        # Если base_threshold не передан, берем его из config
        if base_threshold is None:
            base_threshold = config.PRICE_CHANGE_THRESHOLD
        try:
            # Проверяем наличие необходимых колонок
            required_cols = ['high', 'low', 'close']
            for col in required_cols:
                if col not in df.columns:
                    print(f"Ошибка: колонка {col} отсутствует в данных")
                    return base_threshold
            
            # Расчет True Range для последних N свечей
            n_periods = min(14, len(df) - 1)  # Используем стандартный период ATR=14 или меньше, если данных недостаточно
            
            tr_values = []
            for i in range(1, n_periods + 1):
                if i >= len(df):
                    break
                    
                high = df['high'].iloc[-i]
                low = df['low'].iloc[-i]
                prev_close = df['close'].iloc[-(i+1)]
                
                tr1 = high - low
                tr2 = abs(high - prev_close)
                tr3 = abs(low - prev_close)
                
                tr = max(tr1, tr2, tr3)
                tr_values.append(tr)
            
            # Рассчитываем ATR
            if tr_values:
                atr = sum(tr_values) / len(tr_values)
            else:
                print("Недостаточно данных для расчета ATR, используем базовый порог")
                return base_threshold
            
            # Нормализуем ATR относительно текущей цены
            last_price = df['close'].iloc[-1]
            if last_price > 0:
                normalized_atr = atr / last_price
            else:
                normalized_atr = 0.001
            
            # 🔥 ИСПРАВЛЕНО: Более консервативная настройка множителя для низковолатильных активов
            # Уменьшаем множитель, чтобы не делать порог слишком чувствительным
            atr_multiplier = config.ADAPTIVE_THRESHOLD_MULTIPLIER
            # Если normalized_atr очень низкий, мы можем немного увеличить множитель,
            # но не так агрессивно, как раньше, чтобы не получить слишком много BUY/SELL
            if normalized_atr < 0.0005:  # Если ATR меньше 0.05%
                atr_multiplier = 0.9   # Немного увеличиваем множитель
            elif normalized_atr < 0.001: # Если ATR меньше 0.1%
                atr_multiplier = 0.8
                
            # Рассчитываем адаптивный порог на основе волатильности
            adaptive_threshold = max(
                config.ADAPTIVE_THRESHOLD_MIN, 
                min(config.ADAPTIVE_THRESHOLD_MAX, normalized_atr * atr_multiplier)
            )
            
            # 🔥 ИСПРАВЛЕНО: Убираем принудительное снижение порога,
            # так как теперь мы хотим его немного увеличить для HOLD
            # recommended_threshold_from_log = self._get_recommended_threshold_from_data(df, future_window=config.FUTURE_WINDOW)
            # if recommended_threshold_from_log is not None and recommended_threshold_from_log < adaptive_threshold * 0.5:
            #     print(f"Принудительно снижаем адаптивный порог до рекомендованного: {recommended_threshold_from_log:.6f}")
            #     adaptive_threshold = recommended_threshold_from_log
                
            print(f"[ADAPTIVE] Base threshold: {base_threshold:.6f}, ATR: {normalized_atr:.6f}, "
                  f"Adaptive threshold: {adaptive_threshold:.6f}")
            
            return adaptive_threshold
            
        except Exception as e:
            print(f"Ошибка при расчете адаптивного порога: {e}")
            return base_threshold

    def _get_recommended_threshold_from_data(self, df, future_window):
        """
        Вспомогательный метод для получения рекомендованного порога из данных,
        чтобы получить примерно 30% сигналов.
        """
        try:
            prices = df['close'].values
            if len(prices) <= future_window:
                return None
                
            sample_changes = []
            for j in range(len(prices) - future_window):
                cp = float(prices[j])
                fp = float(prices[j+future_window])
                if cp == 0:
                    pct = 0.0
                else:
                    pct = (fp - cp) / cp
                sample_changes.append(pct)
            
            changes_abs = np.abs(sample_changes)
            if not changes_abs.any(): # Если все изменения нулевые
                return 0.0001 # Возвращаем минимальный порог
                
            changes_sorted = np.sort(changes_abs)
            
            # Находим порог, который бы дал примерно 30% сигналов (не HOLD)
            target_idx = int(len(changes_sorted) * 0.7)  # 70-й процентиль
            if target_idx < len(changes_sorted):
                return changes_sorted[target_idx]
            else:
                return None
        except Exception as e:
            print(f"Ошибка при расчете рекомендованного порога: {e}")
            return None

    def create_trading_labels(self, df):
        """
        Создает метки для торговли на основе будущих изменений цены
        с использованием адаптивного порога на основе волатильности.
        Использует параметры из config.py.
        
        Args:
            df (pd.DataFrame): DataFrame с данными цен
            
        Returns:
            np.array: Массив меток (0: SELL, 1: HOLD, 2: BUY)
        """
        # Проверяем и сортируем по timestamp, если он есть
        if 'timestamp' in df.columns:
            if not pd.api.types.is_numeric_dtype(df['timestamp']):
                print(f"⚠️ timestamp не в числовом формате: {df['timestamp'].dtype}, преобразуем")
                df['timestamp'] = pd.to_numeric(df['timestamp'])
            df = df.sort_values('timestamp')
        
        # 🔥 ИСПРАВЛЕНО: Используем threshold из config.py
        adaptive_threshold = self.calculate_adaptive_threshold(df, config.PRICE_CHANGE_THRESHOLD)
        
        prices = df['close'].values
        labels = []

        # DEBUG: лог входных параметров и короткого среза цен
        try:
            print(f"[LABELS DEBUG] adaptive_threshold={adaptive_threshold}, future_window={config.FUTURE_WINDOW}, len(prices)={len(prices)}")
            # print("[LABELS DEBUG] first 8 closes:", prices[:8].tolist())
            # print("[LABELS DEBUG] last 8 closes:", prices[-8:].tolist())
        except Exception:
            pass

        # 🔥 ИСПРАВЛЕНО: Используем future_window из config.py
        for i in range(len(prices) - config.FUTURE_WINDOW):
            current_price = float(prices[i])
            future_price = float(prices[i + config.FUTURE_WINDOW])

            # Защита от деления на ноль
            if current_price == 0 or np.isnan(current_price) or np.isinf(current_price):
                price_change = 0.0
            else:
                price_change = (future_price - current_price) / float(current_price)

        # DEBUG для первых 20 вычислений
        # if i < 20:
        #     print(f"[LABELS DEBUG] i={i}, cur={current_price:.6f}, fut={future_price:.6f}, change={price_change:.6f}")

            # Используем адаптивный порог для определения сигналов
            if price_change > adaptive_threshold:
                labels.append(2)  # BUY
            elif price_change < -adaptive_threshold:
                labels.append(0)  # SELL
            else:
                labels.append(1)  # HOLD

        # Логирование распределения меток
        vals, counts = np.unique(labels, return_counts=True)
        dist = {int(v): int(c) for v, c in zip(vals, counts)}
        print(f"[LABELS DEBUG] label distribution (SELL=0,HOLD=1,BUY=2): {dist}")
        
        # Анализ дисбаланса
        total = len(labels)
        hold_count = dist.get(1, 0)
        hold_percentage = hold_count / total if total > 0 else 0
        
        if hold_percentage > 0.8:
            print(f"[HOLD WARNING] Высокий процент HOLD меток: {hold_percentage:.2%}")
            
            # Дополнительный анализ изменений цены для диагностики
            if total > 0:
                sample_changes = []
                # 🔥 ИСПРАВЛЕНО: Используем config.FUTURE_WINDOW
                for j in range(min(200, len(prices) - config.FUTURE_WINDOW)):
                    cp = float(prices[j])
                    fp = float(prices[j+config.FUTURE_WINDOW])
                    if cp == 0:
                        pct = 0.0
                    else:
                        pct = (fp - cp) / cp
                    sample_changes.append(pct)
                
                # print(f"[HOLD DEBUG] Symbol likely all-HOLD. sample changes (first 50): {np.array(sample_changes)[:50].tolist()}")
                # print(f"[HOLD DEBUG] Change stats: min={np.min(sample_changes):.6f}, max={np.max(sample_changes):.6f}, "
                #       f"mean={np.mean(sample_changes):.6f}, std={np.std(sample_changes):.6f}")
                print(f"[HOLD DEBUG] Current adaptive threshold: {adaptive_threshold:.6f}")
                
                # Анализируем, какой порог нужен для более сбалансированного распределения
                changes_abs = np.abs(sample_changes)
                changes_sorted = np.sort(changes_abs)
                
                # Находим порог, который бы дал примерно 30% сигналов (не HOLD)
                target_idx = int(len(changes_sorted) * 0.7)  # 70-й процентиль
                if target_idx < len(changes_sorted):
                    suggested_threshold = changes_sorted[target_idx]
                    print(f"[HOLD DEBUG] Для получения ~30% сигналов рекомендуемый порог: {suggested_threshold:.6f}")
        
        return np.array(labels)
    
    def prepare_supervised_data(self, df):
        """
        Подготавливает данные для supervised learning (этап 1)
        с использованием адаптивных порогов.
        Использует параметры из config.py.
        
        Args:
            df (pd.DataFrame): DataFrame с данными цен
            
        Returns:
            tuple: (X, labels) - подготовленные данные и метки
        """
        # Подготавливаем данные (добавляем индикаторы, нормализуем)
        X, _, processed_df = self.prepare_data(df)
        
        # 🔥 ИСПРАВЛЕНО: Создаем метки без передачи threshold и future_window
        labels = self.create_trading_labels(processed_df)
        
        # Убеждаемся, что длины X и labels совпадают
        min_len = min(len(X), len(labels))
        print(f"[PREPARE DEBUG] before trim: len(X)={len(X)}, len(labels)={len(labels)}, using min_len={min_len}")
        X = X[:min_len]
        labels = labels[:min_len]
        
        # Выводим пример первых 30 меток
        print(f"[PREPARE DEBUG] labels sample (first 30): {labels[:30].tolist()}")
        
        return X, labels
    
    def save_scaler(self, path='models'):
        """
        Сохраняет обученный скейлер
        """
        if not os.path.exists(path):
            os.makedirs(path)
        
        with open(os.path.join(path, 'scaler.pkl'), 'wb') as f:
            pickle.dump(self.scaler, f)
        
        print(f"Скейлер сохранен в {path}/scaler.pkl")

    def load_scaler(self, path='models'):
        """
        Загружает обученный скейлер
        """
        scaler_path = os.path.join(path, 'scaler.pkl')
        
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            print("Скейлер успешно загружен")
            return True
        else:
            print("Не удалось найти сохраненный скейлер")
            return False