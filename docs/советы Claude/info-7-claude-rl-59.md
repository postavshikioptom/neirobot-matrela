Как исправить:
Нужно убедиться, что индикаторы добавлены в DataFrame перед тем, как вы пытаетесь получить к ним доступ или масштабировать их.
Файл, который нужно изменить: train_model.py
Вот исправленный фрагмент кода для train_model.py (только те части, которые меняются):
# В train_model.py, в методе load_and_prepare_data():
# ...
        for i, symbol in enumerate(valid_symbols):
            symbol_data = df_filtered[df_filtered['symbol'] == symbol].copy()
            
            if len(symbol_data) < config.SEQUENCE_LENGTH + config.FUTURE_WINDOW + 10:
                print(f"Пропускаем символ {symbol}: недостаточно данных ({len(symbol_data)} строк)")
                continue
            
            try:
                # 🔥 ИЗМЕНЕНО: Исправлена логика использования скейлера
                if i == 0:
                    # Для первого символа обучаем скейлер и подготавливаем данные
                    X_scaled_sequences, labels = self.feature_eng.prepare_supervised_data(
                        symbol_data, 
                        threshold=config.PRICE_CHANGE_THRESHOLD,
                        future_window=config.FUTURE_WINDOW
                    )
                else:
                    # Для остальных символов используем уже обученный скейлер
                    # и только трансформируем данные, затем генерируем метки
                    
                    # 🔥 КЛЮЧЕВОЕ ИЗМЕНЕНИЕ ЗДЕСЬ: Сначала добавляем индикаторы
                    # Это гарантирует, что все колонки (базовые + индикаторы) существуют
                    # перед тем, как мы пытаемся к ним обратиться или масштабировать
                    symbol_data_with_indicators = self.feature_eng._add_technical_indicators(symbol_data.copy()) # 🔥 ДОБАВЛЕНО
                    
                    # Теперь преобразуем все колонки (включая индикаторы) в числовой формат
                    temp_df_for_scaling = symbol_data_with_indicators.copy() # 🔥 ИЗМЕНЕНО: используем df с индикаторами
                    for col in self.feature_eng.feature_columns:
                        temp_df_for_scaling[col] = pd.to_numeric(temp_df_for_scaling[col], errors='coerce')
                    
                    # Применяем обученный скейлер
                    scaled_data = self.feature_eng.scaler.transform(temp_df_for_scaling[self.feature_eng.feature_columns].values)
                    
                    # Создаем последовательности из трансформированных данных
                    X_scaled_sequences, _ = self.feature_eng._create_sequences(scaled_data)
                    
                    # Создаем метки на основе оригинальных цен
                    # 🔥 ИЗМЕНЕНО: передаем original symbol_data, а не temp_df_for_scaling
                    labels = self.feature_eng.create_trading_labels(
                        symbol_data, # 🔥 ИЗМЕНЕНО: Используем оригинальный df для меток
                        threshold=config.PRICE_CHANGE_THRESHOLD,
                        future_window=config.FUTURE_WINDOW
                    )
                    
                    # Обрезаем до минимальной длины
                    min_len = min(len(X_scaled_sequences), len(labels))
                    X_scaled_sequences = X_scaled_sequences[:min_len]
                    labels = labels[:min_len]
                
                if len(X_scaled_sequences) > 0:
                    all_X_supervised.append(X_scaled_sequences)
                    all_y_supervised.append(labels)
                    
                    X_data_for_rl[symbol] = X_scaled_sequences
                    
                    print(f"Символ {symbol}: {len(X_scaled_sequences)} последовательностей")
                    
            except Exception as e:
                print(f"❌ Ошибка при обработке символа {symbol}: {e}")
                import traceback
                traceback.print_exc()
                continue
# ... (остальной код) ...

Объяснение исправлений:

symbol_data_with_indicators = self.feature_eng._add_technical_indicators(symbol_data.copy()): Эта строка теперь вызывается перед циклом for col in self.feature_eng.feature_columns:. Это гарантирует, что все колонки индикаторов ('RSI', 'MACD' и т.д.) будут существовать в symbol_data_with_indicators до того, как мы попытаемся к ним обратиться.
temp_df_for_scaling = symbol_data_with_indicators.copy(): Теперь мы копируем DataFrame, который уже содержит индикаторы.
labels = self.feature_eng.create_trading_labels(symbol_data, ...): Для создания меток мы должны использовать оригинальный symbol_data (который содержит оригинальные цены), а не temp_df_for_scaling (который уже масштабирован и может иметь измененные цены из-за заполнения NaN).

Эти изменения должны устранить KeyError: 'RSI' и обеспечить корректную последовательность добавления индикаторов и масштабирования данных.