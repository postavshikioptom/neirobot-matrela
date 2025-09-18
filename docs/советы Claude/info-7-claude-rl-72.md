

2. Исправление ошибки "need at least one array to concatenate"
В файле train_model.py в методе load_and_prepare_data добавьте проверку перед вызовом np.vstack:
# Проверка перед объединением
if all_X_supervised:  # Проверяем, что список не пустой
    X_supervised = np.vstack(all_X_supervised)
    y_supervised = np.concatenate(all_y_supervised)
    
    print(f"Итого подготовлено для Supervised: X={X_supervised.shape}, y={y_supervised.shape}")
    print(f"Распределение классов: SELL={np.sum(y_supervised==0)}, HOLD={np.sum(y_supervised==1)}, BUY={np.sum(y_supervised==2)}")
    
    # Остальной код обработки...
else:
    print("❌ Нет данных для обучения. Проверьте обработку символов выше.")
    return False

3. Исправление проблемы с недостаточными данными
В файле feature_engineering.py улучшите обработку ошибок в методе _create_sequences:
def _create_sequences(self, data):
    """Создает последовательности для обучения с обработкой ошибок"""
    try:
        if data is None or len(data) == 0:
            print("❌ Пустые данные переданы в _create_sequences")
            return np.array([]), np.array([])
        
        print(f"Создание последовательностей из данных формы {data.shape}")
        
        if len(data) <= self.sequence_length:
            print(f"❌ Недостаточно данных для создания последовательностей: {len(data)} <= {self.sequence_length}")
            # Если данных недостаточно, но они есть, пробуем создать хотя бы одну последовательность
            if len(data) > 10:  # Минимально допустимая длина последовательности
                reduced_sequence_length = len(data) - 5  # Уменьшаем требуемую длину
                print(f"Пробуем создать последовательности с уменьшенной длиной {reduced_sequence_length}")
                X = []
                y_close = []
                
                # Безопасное определение индекса 'close'
                close_index = 3  # По умолчанию 3 (обычно это 'close')
                try:
                    if hasattr(self, 'base_features') and 'close' in self.base_features:
                        close_index = self.base_features.index('close')
                except (ValueError, AttributeError):
                    pass
                
                # Создаем одну последовательность с уменьшенной длиной
                X.append(data[:reduced_sequence_length])
                y_close.append(data[reduced_sequence_length, close_index])
                
                return np.array(X), np.array(y_close)
            else:
                return np.array([]), np.array([])
        
        # Стандартное создание последовательностей
        X = []
        y_close = []
        
        # Безопасное определение индекса 'close'
        try:
            close_index = self.base_features.index('close')
        except (ValueError, AttributeError):
            close_index = 3  # Fallback к индексу 3
        
        for i in range(len(data) - self.sequence_length):
            try:
                X.append(data[i:i+self.sequence_length])
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

4. Исправление проблемы с "all-HOLD" метками
В файле config.py измените параметр PRICE_CHANGE_THRESHOLD:
# Было:
PRICE_CHANGE_THRESHOLD = 0.01

# Измените на:
PRICE_CHANGE_THRESHOLD = 0.005  # Уменьшаем порог для более чувствительной классификации
