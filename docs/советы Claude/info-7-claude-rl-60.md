Нужно заменить вызовы df.fillna(method='ffill', inplace=True) и df.fillna(method='bfill', inplace=True) на df.ffill(inplace=True) и df.bfill(inplace=True) соответственно.
Файл, который нужно изменить: feature_engineering.py
Вот исправленный фрагмент кода для feature_engineering.py (только те части, которые меняются):
# В feature_engineering.py, в методе _add_technical_indicators(self, df):
# ... (расчет индикаторов) ...

        # Обновляем список признаков
        self.feature_columns = self.base_features + [
            'RSI', 'MACD', 'MACDSIGNAL', 'MACDHIST', 
            'STOCH_K', 'STOCH_D', 'WILLR', 'AO'
        ]
        
        # 🔥 ИЗМЕНЕНО: Более надежная обработка NaN с использованием ffill() и bfill()
        # Сначала заполняем NaN предыдущими валидными значениями
        df.ffill(inplace=True) # 🔥 ИЗМЕНЕНО
        # Затем заполняем оставшиеся NaN (если они в самом начале ряда) последующими валидными значениями
        df.bfill(inplace=True) # 🔥 ИЗМЕНЕНО
        # Если все еще есть NaN (например, если весь столбец полностью состоит из NaN), заполняем 0
        df.fillna(0, inplace=True) 
        
        return df

# ... (остальной код) ...

Объяснение исправлений:

df.ffill(inplace=True): Выполняет ту же операцию, что и df.fillna(method='ffill', inplace=True), но является предпочтительным способом в новых версиях Pandas.
df.bfill(inplace=True): Аналогично, выполняет ту же операцию, что и df.fillna(method='bfill', inplace=True).

Эти изменения устранят FutureWarning и сделают ваш код более современным и устойчивым к будущим обновлениям Pandas.