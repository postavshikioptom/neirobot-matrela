📊 АНАЛИЗ КРИТИЧЕСКИХ ОШИБОК - СОГЛАСИЕ/НЕСОГЛАСИЕ
✅ СОГЛАСЕН (8/10 пунктов):
1. Memory Leak в predict методах ✅ ЧАСТИЧНО СОГЛАСЕН
Проблема: tf.keras.backend.clear_session() действительно слишком агрессивен
Исправление в models/xlstm_rl_model.py:
import gc

class XLSTMRLModel:
    def __init__(self, input_shape, memory_size=64, memory_units=128):
        # ... существующий код ...
        self.prediction_count = 0
    
    def predict_action(self, state):
        """Предсказывает действие на основе состояния"""
        if len(state.shape) == 2:
            state = np.expand_dims(state, axis=0)
        
        if state.dtype != np.float32:
            state = state.astype(np.float32)
        
        action_probs = self.actor_model.predict(state, verbose=0)[0]
        
        # 🔥 ИСПРАВЛЕНО: Мягкая очистка памяти вместо clear_session()
        self.prediction_count += 1
        if self.prediction_count % 100 == 0:
            gc.collect()  # 🔥 ИЗМЕНЕНО: Используем gc.collect() вместо clear_session()
            print(f"Мягкая очистка памяти после {self.prediction_count} предсказаний")
        
        return action_probs
    
    def predict_value(self, state):
        """Предсказывает значение состояния"""
        if len(state.shape) == 2:
            state = np.expand_dims(state, axis=0)
        
        if state.dtype != np.float32:
            state = state.astype(np.float32)
        
        value = self.critic_model.predict(state, verbose=0)[0]
        
        # 🔥 ДОБАВЛЕНО: Аналогичная очистка для critic
        self.prediction_count += 1
        if self.prediction_count % 100 == 0:
            gc.collect()
        
        return value

2. Переполнение памяти в TradingEnvironment ✅ СОГЛАСЕН
Исправление в trading_env.py:
class TradingEnvironment(gym.Env):
    def __init__(self, data_by_symbol, sequence_length=60, initial_balance=10000, transaction_fee=0.001, max_memory_size=1000):
        # ... существующий код ...
        self.max_memory_size = max_memory_size  # 🔥 ДОБАВЛЕНО: Лимит памяти
        self.memory_buffer = []  # 🔥 ДОБАВЛЕНО: Буфер для ограничения памяти
    
    def step(self, action):
        # ... существующий код до обновления памяти ...
        
        # 🔥 ДОБАВЛЕНО: Ограничение размера памяти
        self.memory_buffer.append({
            'state': observation,
            'action': action,
            'reward': reward,
            'done': done
        })
        
        # Ограничиваем размер буфера памяти
        if len(self.memory_buffer) > self.max_memory_size:
            self.memory_buffer.pop(0)  # Удаляем самые старые данные
        
        # Периодическая очистка памяти
        if len(self.memory_buffer) % 100 == 0:
            gc.collect()
        
        return observation, reward, done, False, info

3. Неправильная обработка NaN в XLSTMMemoryCell ✅ СОГЛАСЕН
Исправление в models/xlstm_memory_cell.py:
class XLSTMMemoryCell(layers.Layer):
    def call(self, inputs, states):
        # 🔥 ДОБАВЛЕНО: Проверка входных данных на NaN
        inputs = tf.debugging.check_numerics(inputs, "NaN detected in XLSTMMemoryCell inputs")
        
        # Предыдущие состояния
        h_prev, memory_prev = states
        
        # 🔥 ДОБАВЛЕНО: Проверка состояний на NaN
        h_prev = tf.debugging.check_numerics(h_prev, "NaN detected in h_prev")
        memory_prev = tf.debugging.check_numerics(memory_prev, "NaN detected in memory_prev")
        
        # Вычисляем гейты с экспоненциальной активацией для стабильности
        i = tf.nn.sigmoid(tf.matmul(inputs, self.Wi) + tf.matmul(h_prev, self.Ui) + self.bi)
        f = tf.nn.sigmoid(tf.matmul(inputs, self.Wf) + tf.matmul(h_prev, self.Uf) + self.bf)
        o = tf.nn.sigmoid(tf.matmul(inputs, self.Wo) + tf.matmul(h_prev, self.Uo) + self.bo)
        
        # 🔥 ДОБАВЛЕНО: Проверка гейтов на NaN
        i = tf.debugging.check_numerics(i, "NaN detected in input gate")
        f = tf.debugging.check_numerics(f, "NaN detected in forget gate")
        o = tf.debugging.check_numerics(o, "NaN detected in output gate")
        
        # Кандидат на новое значение ячейки
        c_tilde = tf.nn.tanh(tf.matmul(inputs, self.Wc) + tf.matmul(h_prev, self.Uc) + self.bc)
        c_tilde = tf.debugging.check_numerics(c_tilde, "NaN detected in c_tilde")
        
        # ... остальной код с аналогичными проверками ...
        
        # 🔥 ДОБАВЛЕНО: Финальная проверка выходов
        h = tf.debugging.check_numerics(h, "NaN detected in output h")
        memory_new = tf.debugging.check_numerics(memory_new, "NaN detected in memory_new")
        
        return h, [h, memory_new]

4. Утечка ресурсов в валидации ✅ СОГЛАСЕН
Исправление в validation_metrics_callback.py:
import gc

class ValidationMetricsCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 5 == 0:
            print(f"\n📊 Детальные метрики на эпохе {epoch+1}:")
            
            y_pred_probs = self.model.predict(self.X_val, verbose=0)
            y_pred_classes = np.argmax(y_pred_probs, axis=1)
            
            if self.y_val.ndim > 1 and self.y_val.shape[1] > 1:
                y_true_classes = np.argmax(self.y_val, axis=1)
            else:
                y_true_classes = self.y_val
            
            # ... код для confusion matrix и classification report ...
            
            # 🔥 ДОБАВЛЕНО: Очистка памяти после валидации
            del y_pred_probs, y_pred_classes, y_true_classes
            gc.collect()
            print("Память очищена после валидации")

5. Критическая ошибка в TradingEnvironment.reset() ✅ СОГЛАСЕН
Исправление в trading_env.py:
class TradingEnvironment(gym.Env):
    def reset(self, seed=None):
        """Сбрасывает среду в начальное состояние"""
        super().reset(seed=seed)
        
        # 🔥 ДОБАВЛЕНО: Проверка на пустые символы
        if not self.symbols or len(self.symbols) == 0:
            print("❌ Нет доступных символов для торговли")
            # Создаем dummy данные для предотвращения краха
            dummy_shape = (self.sequence_length, 10)  # 10 признаков по умолчанию
            observation = np.zeros(dummy_shape, dtype=np.float32)
            return observation, {}
        
        # Выбираем случайный символ при каждом сбросе
        try:
            self.current_symbol = random.choice(self.symbols)
            self.current_data = self.data_by_symbol[self.current_symbol]
        except (KeyError, IndexError) as e:
            print(f"❌ Ошибка при выборе символа: {e}")
            # Fallback к первому доступному символу
            if self.symbols:
                self.current_symbol = self.symbols[0]
                self.current_data = self.data_by_symbol.get(self.current_symbol, None)
        
        # 🔥 ДОБАВЛЕНО: Проверка на корректность данных
        if self.current_data is None or len(self.current_data) == 0:
            print(f"❌ Нет данных для символа {self.current_symbol}")
            dummy_shape = (self.sequence_length, 10)
            observation = np.zeros(dummy_shape, dtype=np.float32)
            return observation, {}
        
        # ... остальной код reset() ...

6. Неконтролируемый рост батчей ✅ СОГЛАСЕН
Исправление в train_model.py:
import psutil

class ThreeStageTrainer:
    def load_and_prepare_data(self):
        # ... существующий код до аугментации ...
        
        # 🔥 ДОБАВЛЕНО: Проверка доступной памяти перед аугментацией
        def check_memory_before_augmentation(data_size_mb):
            available_memory_gb = psutil.virtual_memory().available / (1024**3)
            required_memory_gb = (data_size_mb * 2) / 1024  # Аугментация удваивает размер
            
            if required_memory_gb > available_memory_gb * 0.8:  # Оставляем 20% буфер
                print(f"⚠️ Недостаточно памяти для аугментации: нужно {required_memory_gb:.2f}GB, доступно {available_memory_gb:.2f}GB")
                return False
            return True
        
        def augment_sequences(X, y, factor=2):
            # 🔥 ДОБАВЛЕНО: Проверка памяти
            data_size_mb = X.nbytes / (1024**2)
            if not check_memory_before_augmentation(data_size_mb):
                print("Пропускаем аугментацию из-за нехватки памяти")
                return X, y
            
            augmented_X, augmented_y = [], []
            for i in range(len(X)):
                # Оригинальные данные
                augmented_X.append(X[i])
                augmented_y.append(y[i])
                
                # Добавляем шум только если есть память
                if i % 1000 == 0:  # Проверяем каждые 1000 образцов
                    if not check_memory_before_augmentation(len(augmented_X) * X[0].nbytes / (1024**2)):
                        print("Останавливаем аугментацию из-за нехватки памяти")
                        break
                
                noise = np.random.normal(0, 0.05 * np.std(X[i]), X[i].shape)
                augmented_X.append(X[i] + noise)
                augmented_y.append(y[i])
            
            return np.array(augmented_X), np.array(augmented_y)
        
        # Применяем безопасную аугментацию
        X_supervised, y_supervised = augment_sequences(X_supervised, y_supervised)

7. Потенциальное переполнение в XLSTMMemoryCell ✅ СОГЛАСЕН
Исправление в models/xlstm_memory_cell.py:
class XLSTMMemoryCell(layers.Layer):
    def call(self, inputs, states):
        # ... код до экспоненциальных операций ...
        
        # 🔥 ДОБАВЛЕНО: Ограничение значений перед экспоненциальными операциями
        # Ограничиваем входы для предотвращения overflow
        inputs = tf.clip_by_value(inputs, -10.0, 10.0)
        
        # Вычисляем гейты с ограничением значений
        i_logits = tf.matmul(inputs, self.Wi) + tf.matmul(h_prev, self.Ui) + self.bi
        i_logits = tf.clip_by_value(i_logits, -10.0, 10.0)  # 🔥 ДОБАВЛЕНО
        i = tf.nn.sigmoid(i_logits)
        
        f_logits = tf.matmul(inputs, self.Wf) + tf.matmul(h_prev, self.Uf) + self.bf
        f_logits = tf.clip_by_value(f_logits, -10.0, 10.0)  # 🔥 ДОБАВЛЕНО
        f = tf.nn.sigmoid(f_logits)
        
        o_logits = tf.matmul(inputs, self.Wo) + tf.matmul(h_prev, self.Uo) + self.bo
        o_logits = tf.clip_by_value(o_logits, -10.0, 10.0)  # 🔥 ДОБАВЛЕНО
        o = tf.nn.sigmoid(o_logits)
        
        # Кандидат на новое значение ячейки с ограничением
        c_tilde_logits = tf.matmul(inputs, self.Wc) + tf.matmul(h_prev, self.Uc) + self.bc
        c_tilde_logits = tf.clip_by_value(c_tilde_logits, -10.0, 10.0)  # 🔥 ДОБАВЛЕНО
        c_tilde = tf.nn.tanh(c_tilde_logits)
        
        # Экспоненциальный гейт с ограничением
        e_logits = tf.matmul(inputs, self.We) + tf.matmul(h_prev, self.Ue) + self.be
        e_logits = tf.clip_by_value(e_logits, -10.0, 10.0)  # 🔥 ДОБАВЛЕНО
        e = tf.nn.softmax(e_logits)
        
        # ... остальной код ...


8. Отсутствие обработки исключений ✅ СОГЛАСЕН
Исправление в feature_engineering.py:
class FeatureEngineering:
    def _add_technical_indicators(self, df):
        """
        Добавляет технические индикаторы в DataFrame с использованием TA-Lib.
        """
        try:
            # 🔥 ДОБАВЛЕНО: Проверка входных данных
            if df is None or df.empty:
                print("❌ Пустой DataFrame передан в _add_technical_indicators")
                return self._create_fallback_indicators_df()
            
            # Проверяем наличие достаточных данных для RSI
            if len(df) < config.RSI_PERIOD + 5:
                print(f"Недостаточно данных для RSI: {len(df)} строк, нужно минимум {config.RSI_PERIOD + 5}")
                return self._create_fallback_indicators_df(df)
            
            # Убедимся, что все необходимые колонки в числовом формате
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col not in df.columns:
                    print(f"❌ Отсутствует колонка {col}")
                    return self._create_fallback_indicators_df(df)
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Проверка на NaN перед расчётом индикаторов
            if df['close'].isna().sum() > len(df) * 0.5:
                print(f"Слишком много NaN в данных, используем fallback")
                return self._create_fallback_indicators_df(df)
            
            # 🔥 ДОБАВЛЕНО: Обработка каждого индикатора отдельно
            try:
                df['RSI'] = talib.RSI(df['close'].ffill(), timeperiod=config.RSI_PERIOD)
            except Exception as e:
                print(f"Ошибка при расчёте RSI: {e}")
                df['RSI'] = 50.0  # Нейтральное значение RSI
            
            try:
                macd, macdsignal, macdhist = talib.MACD(
                    df['close'].ffill(), 
                    fastperiod=config.MACD_FASTPERIOD, 
                    slowperiod=config.MACD_SLOWPERIOD, 
                    signalperiod=config.MACD_SIGNALPERIOD
                )
                df['MACD'] = macd
                df['MACDSIGNAL'] = macdsignal
                df['MACDHIST'] = macdhist
            except Exception as e:
                print(f"Ошибка при расчёте MACD: {e}")
                df['MACD'] = 0.0
                df['MACDSIGNAL'] = 0.0
                df['MACDHIST'] = 0.0
            
            try:
                stoch_k, stoch_d = talib.STOCH(
                    df['high'], df['low'], df['close'],
                    fastk_period=config.STOCH_K_PERIOD,
                    slowk_period=config.STOCH_K_PERIOD,
                    slowd_period=config.STOCH_D_PERIOD
                )
                df['STOCH_K'] = stoch_k
                df['STOCH_D'] = stoch_d
            except Exception as e:
                print(f"Ошибка при расчёте Stochastic: {e}")
                df['STOCH_K'] = 50.0
                df['STOCH_D'] = 50.0
            
            try:
                df['WILLR'] = talib.WILLR(df['high'], df['low'], df['close'], timeperiod=config.WILLR_PERIOD)
            except Exception as e:
                print(f"Ошибка при расчёте Williams %R: {e}")
                df['WILLR'] = -50.0  # Нейтральное значение Williams %R
            
            try:
                median_price = (df['high'] + df['low']) / 2
                sma_5 = talib.SMA(median_price, timeperiod=config.AO_FASTPERIOD)
                sma_34 = talib.SMA(median_price, timeperiod=config.AO_SLOWPERIOD)
                df['AO'] = sma_5 - sma_34
            except Exception as e:
                print(f"Ошибка при расчёте AO: {e}")
                df['AO'] = 0.0
            
            # Обновляем список признаков
            self.feature_columns = self.base_features + [
                'RSI', 'MACD', 'MACDSIGNAL', 'MACDHIST', 
                'STOCH_K', 'STOCH_D', 'WILLR', 'AO'
            ]
            
        except Exception as e:
            print(f"❌ Критическая ошибка в _add_technical_indicators: {e}")
            return self._create_fallback_indicators_df(df)
        
        # Надёжная обработка NaN
        try:
            df = df.ffill().bfill().fillna(0)
        except Exception as e:
            print(f"Ошибка при обработке NaN: {e}")
            df = df.fillna(0)
        
        return df
    
    def _create_fallback_indicators_df(self, df=None):
        """🔥 ДОБАВЛЕНО: Создает DataFrame с fallback значениями индикаторов"""
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
        
        # Добавляем fallback индикаторы
        df['RSI'] = 50.0  # Нейтральный RSI
        df['MACD'] = 0.0
        df['MACDSIGNAL'] = 0.0
        df['MACDHIST'] = 0.0
        df['STOCH_K'] = 50.0
        df['STOCH_D'] = 50.0
        df['WILLR'] = -50.0
        df['AO'] = 0.0
        
        self.feature_columns = self.base_features + [
            'RSI', 'MACD', 'MACDSIGNAL', 'MACDHIST', 
            'STOCH_K', 'STOCH_D', 'WILLR', 'AO'
        ]
        
        return df
    
    def _create_sequences(self, data):
        """Создает последовательности для обучения с обработкой ошибок"""
        try:
            if data is None or len(data) == 0:
                print("❌ Пустые данные переданы в _create_sequences")
                return np.array([]), np.array([])
            
            X = []
            y_close = []
            
            # Безопасное определение индекса 'close'
            try:
                close_index = self.base_features.index('close')
            except ValueError:
                print("❌ Колонка 'close' не найдена в base_features")
                close_index = 3  # Fallback к индексу 3
            
            if len(data) <= self.sequence_length:
                print(f"❌ Недостаточно данных для создания последовательностей: {len(data)} <= {self.sequence_length}")
                return np.array([]), np.array([])
            
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

