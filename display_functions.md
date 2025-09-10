def display_position_info(symbol, pos, latest_price, entry_price, pnl_pct, display_counter, price_series, latest_features):
    """
    Выводит информацию о позиции с ограничением на количество отображаемых позиций.
    
    Args:
        symbol (str): Символ торговой пары
        pos (dict): Информация о позиции
        latest_price (float): Текущая цена
        entry_price (float): Цена входа
        pnl_pct (float): Процент прибыли/убытка
        display_counter (int): Счетчик отображенных позиций
        price_series (pd.Series): Временной ряд цен
        latest_features (pd.Series): Последние технические индикаторы
        
    Returns:
        int: Обновленный счетчик отображенных позиций
    """
    # Ограничиваем вывод в консоль
    if display_counter < 10:
        print(f"  - {symbol}: PnL {pnl_pct:.2f}% | Вход: {entry_price} | Сейчас: {latest_price}")
        
        # Вывод информации от моделей с ограничением
        # 1. Обработка данных через Kalman Filter
        kalman_result = process_with_kalman_filter(price_series)
        
        # 2. Обработка данных через LSTM
        lstm_result = process_with_lstm(price_series)
        
        # 3. Обработка данных через GPR
        # В новой архитектуре GPR использует оригинальные данные, а не сглаженные от Kalman Filter
        gpr_result = process_with_gpr(price_series)
        
        # 4. Передаем результаты Kalman Filter в GPR (дополнительно)
        # Для этого мы можем использовать тренд от Kalman Filter как дополнительный признак
        kalman_trend = kalman_result['trend']
        
        # 5. Передаем только технические индикаторы в XGBoost
        # Модель XGBoost обучена только на технических индикаторах
        feature_names = ["open","high","low","close","volume","turnover","BBL_20_2.0","BBM_20_2.0","BBU_20_2.0","BBB_20_2.0","BBP_20_2.0","MACD_12_26_9","MACDh_12_26_9","MACDs_12_26_9","OBV","ATRr_14","WILLR_14","RSI_14","CCI_20_0.015","ADX_14","DMP_14","DMN_14"]
        
        # 6. Получаем предсказание от XGBoost
        # Выбираем только те признаки, которые ожидает модель
        xgboost_features = latest_features[feature_names].to_frame().T
        # Преобразуем типы данных для XGBoost
        for col in xgboost_features.columns:
            xgboost_features[col] = pd.to_numeric(xgboost_features[col], errors='coerce')
        xgboost_features.fillna(0, inplace=True)
        xgboost_prediction = xgboost_model.predict(xgboost_features)
        print(f"--- {symbol} | Уверенность XGBoost: {xgboost_prediction} ---")
        print(f"--- {symbol} | Результаты Kalman Filter: Цена={kalman_result['kalman_price']}, Тренд={kalman_result['trend']} ---")
        print(f"--- {symbol} | Результаты LSTM: Предсказание={lstm_result['prediction']}, Уверенность={lstm_result['confidence']} ---")
        print(f"--- {symbol} | Результаты GPR: Предсказание={gpr_result['prediction']}, Уверенность={gpr_result['confidence']} ---")
        
        # Возвращаем обновленный счетчик
        return display_counter + 1
    else:
        # Если лимит исчерпан, просто возвращаем счетчик без изменений
        return display_counter