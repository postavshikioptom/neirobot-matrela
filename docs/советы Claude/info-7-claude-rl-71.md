Вот изменения, которые нужно внести в каждый файл:
1. Файл feature_engineering.py
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
    
    # Сортируем по числовому timestamp
    if 'timestamp' in df.columns:
        df = df.sort_values('timestamp')
    
    # Добавляем индикаторы
    df_with_indicators = self._add_technical_indicators(df.copy())
    
    # Преобразуем признаки в числовой формат
    for col in self.feature_columns:
        if col in df_with_indicators.columns:
            df_with_indicators[col] = pd.to_numeric(df_with_indicators[col], errors='coerce')
    
    # Получаем только нужные колонки
    data = df_with_indicators[self.feature_columns].values
    
    # Обучаем скейлер на всех данных
    scaled_data = self.scaler.fit_transform(data)
    
    # Создаем последовательности
    X, y_close = self._create_sequences(scaled_data)
    
    return X, y_close, df_with_indicators

2. Файл create_trading_labels в feature_engineering.py
def create_trading_labels(self, df, threshold=0.01, future_window=5):
    """Создает метки для торговли на основе будущих изменений цены"""
    # Проверяем, что timestamp в числовом формате
    if 'timestamp' in df.columns:
        if not pd.api.types.is_numeric_dtype(df['timestamp']):
            print(f"⚠️ timestamp не в числовом формате: {df['timestamp'].dtype}, преобразуем")
            df['timestamp'] = pd.to_numeric(df['timestamp'])
    
    # Сортируем данные по числовому timestamp
    if 'timestamp' in df.columns:
        df = df.sort_values('timestamp')
    
    prices = df['close'].values
    labels = []
    
    # DEBUG: лог входных параметров и короткого среза цен
    try:
        print(f"[LABELS DEBUG] threshold={threshold}, future_window={future_window}, len(prices)={len(prices)}")
        print("[LABELS DEBUG] first 8 closes:", prices[:8].tolist())
        print("[LABELS DEBUG] last 8 closes:", prices[-8:].tolist())
    except Exception:
        pass
    
    # Остальной код без изменений...

3. В файле train_model.py в методе load_and_prepare_data
def load_and_prepare_data(self):
    """Загружает и подготавливает данные для всех этапов"""
    print("=== ПОДГОТОВКА ДАННЫХ ===")
    
    # Загружаем данные, timestamp должен быть числом
    df = pd.read_csv(self.data_path, dtype={
        'timestamp': np.int64,  # Явно указываем тип для timestamp
        'open': float, 
        'high': float, 
        'low': float, 
        'close': float, 
        'volume': float, 
        'turnover': float,
        'symbol': str
    })
    print(f"Загружено {len(df)} строк данных")
    
    # Проверяем тип timestamp
    print(f"Тип timestamp: {df['timestamp'].dtype}")
    
    # Остальной код без изменений...

4. В файле run_live_trading.py в функции fetch_latest_data
def fetch_latest_data(session, symbol, timeframe, limit=100):
    """Получает последние свечи с биржи"""
    try:
        response = session.get_kline(
            category="linear",
            symbol=symbol,
            interval=timeframe,
            limit=limit
        )
        
        if response['retCode'] == 0:
            data = response['result']['list']
            
            # Преобразуем данные в DataFrame
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
            ])
            
            # Преобразуем типы данных
            numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'turnover']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # НЕ преобразуем timestamp в datetime, оставляем как число
            df['timestamp'] = pd.to_numeric(df['timestamp'])
            df['symbol'] = symbol
            
            # Сортируем по времени
            df.sort_values('timestamp', inplace=True)
            
            return df
        else:
            print(f"Ошибка при получении данных: {response['retMsg']}")
            return None
    
    except Exception as e:
        print(f"Ошибка при получении данных: {e}")
        return None
