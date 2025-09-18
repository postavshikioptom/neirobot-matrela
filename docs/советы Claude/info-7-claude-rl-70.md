
В файле train_model.py в методе load_and_prepare_data нужно правильно загрузить данные с числовым timestamp:

def load_and_prepare_data(self):
    """Загружает и подготавливает данные для всех этапов"""
    print("=== ПОДГОТОВКА ДАННЫХ ===")
    
    # Загружаем данные без преобразования timestamp
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


В файле feature_engineering.py в методе prepare_data нужно правильно обрабатывать числовой timestamp:

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
    
    # Остальной код без изменений...


В файле feature_engineering.py в методе create_trading_labels также нужно учесть числовой формат timestamp:

def create_trading_labels(self, df, threshold=0.01, future_window=5):
    """Создает метки для торговли на основе будущих изменений цены"""
    # Проверяем тип timestamp
    if 'timestamp' in df.columns:
        if not pd.api.types.is_numeric_dtype(df['timestamp']):
            print(f"⚠️ timestamp не в числовом формате: {df['timestamp'].dtype}, преобразуем")
            df['timestamp'] = pd.to_numeric(df['timestamp'])
    
    # Сортируем данные по числовому timestamp
    if 'timestamp' in df.columns:
        df = df.sort_values('timestamp')
    
    prices = df['close'].values
    labels = []
    
    # Остальной код без изменений...
