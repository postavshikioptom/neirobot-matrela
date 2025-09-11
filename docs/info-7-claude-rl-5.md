Вижу проблему! Ошибка в том, что в XLSTMLayer.call() мы пытаемся использовать range(seq_len), где seq_len это символический тензор, а не обычное число. В TensorFlow 2.x нужно использовать специальные функции для работы с динамическими размерностями.
🔧 ИСПРАВЛЕНИЕ ОШИБКИ
Замените файл models/xlstm_memory_cell.py на исправленную версию:
import tensorflow as tf
from tensorflow.keras.layers import Layer
import numpy as np

class XLSTMMemoryCell(Layer):
    """
    Настоящая xLSTM ячейка с расширенной памятью
    """
    
    def __init__(self, units, memory_size=64, **kwargs):
        super(XLSTMMemoryCell, self).__init__(**kwargs)
        self.units = units
        self.memory_size = memory_size
        self.state_size = [units, units, memory_size]  # h, c, memory
        
    def build(self, input_shape):
        # Основные веса LSTM
        self.W_i = self.add_weight(shape=(input_shape[-1], self.units), name='W_i')
        self.W_f = self.add_weight(shape=(input_shape[-1], self.units), name='W_f')  
        self.W_c = self.add_weight(shape=(input_shape[-1], self.units), name='W_c')
        self.W_o = self.add_weight(shape=(input_shape[-1], self.units), name='W_o')
        
        # Рекуррентные веса
        self.U_i = self.add_weight(shape=(self.units, self.units), name='U_i')
        self.U_f = self.add_weight(shape=(self.units, self.units), name='U_f')
        self.U_c = self.add_weight(shape=(self.units, self.units), name='U_c')  
        self.U_o = self.add_weight(shape=(self.units, self.units), name='U_o')
        
        # Веса внешней памяти (ключевое отличие xLSTM)
        self.W_mem = self.add_weight(shape=(self.memory_size, self.units), name='W_mem')
        self.U_mem = self.add_weight(shape=(self.units, self.memory_size), name='U_mem')
        
        # Bias
        self.b_i = self.add_weight(shape=(self.units,), name='b_i')
        self.b_f = self.add_weight(shape=(self.units,), name='b_f')
        self.b_c = self.add_weight(shape=(self.units,), name='b_c')
        self.b_o = self.add_weight(shape=(self.units,), name='b_o')
        
        super(XLSTMMemoryCell, self).build(input_shape)
        
    def call(self, inputs, states):
        h_prev, c_prev, memory_prev = states
        
        # Читаем из внешней памяти
        memory_read = tf.matmul(memory_prev, self.W_mem)
        
        # Основные вычисления LSTM с памятью
        i = tf.nn.sigmoid(tf.matmul(inputs, self.W_i) + tf.matmul(h_prev, self.U_i) +
                         memory_read + self.b_i)
        f = tf.nn.sigmoid(tf.matmul(inputs, self.W_f) + tf.matmul(h_prev, self.U_f) + self.b_f)
        c_tilde = tf.nn.tanh(tf.matmul(inputs, self.W_c) + tf.matmul(h_prev, self.U_c) + self.b_c)
        o = tf.nn.sigmoid(tf.matmul(inputs, self.W_o) + tf.matmul(h_prev, self.U_o) + self.b_o)
        
        # Обновляем состояние ячейки
        c_new = f * c_prev + i * c_tilde
        h_new = o * tf.nn.tanh(c_new)
        
        # Обновляем внешнюю память (ключевое отличие xLSTM!)
        memory_update = tf.matmul(tf.expand_dims(h_new, 1), tf.expand_dims(self.U_mem, 0))
        memory_new = memory_prev + 0.1 * tf.squeeze(memory_update, 1)  # Медленное обновление
        
        return h_new, [h_new, c_new, memory_new]

class XLSTMLayer(Layer):
    """
    Слой xLSTM с использованием кастомной ячейки памяти - ИСПРАВЛЕННАЯ ВЕРСИЯ
    """
    
    def __init__(self, units, memory_size=64, return_sequences=False, **kwargs):
        super(XLSTMLayer, self).__init__(**kwargs)
        self.units = units
        self.memory_size = memory_size
        self.return_sequences = return_sequences
        self.cell = XLSTMMemoryCell(units, memory_size)
        
    def build(self, input_shape):
        # Строим ячейку
        self.cell.build(input_shape)
        super(XLSTMLayer, self).build(input_shape)
        
    def compute_output_shape(self, input_shape):
        if self.return_sequences:
            return (input_shape[0], input_shape[1], self.units)
        else:
            return (input_shape[0], self.units)
    
    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        seq_len = tf.shape(inputs)[1]
        
        # Инициализируем состояния
        h = tf.zeros((batch_size, self.units))
        c = tf.zeros((batch_size, self.units))
        memory = tf.zeros((batch_size, self.memory_size))
        
        states = [h, c, memory]
        
        # ✅ ИСПРАВЛЕНИЕ: Используем tf.TensorArray для накопления выходов
        if self.return_sequences:
            outputs = tf.TensorArray(dtype=tf.float32, size=seq_len)
        
        # ✅ ИСПРАВЛЕНИЕ: Используем tf.while_loop вместо обычного range
        def step_fn(t, states, outputs_ta):
            # Извлекаем текущий временной шаг
            current_input = inputs[:, t, :]
            
            # Применяем ячейку
            output, new_states = self.cell(current_input, states)
            
            # Записываем выход если нужно
            if self.return_sequences:
                outputs_ta = outputs_ta.write(t, output)
            
            return t + 1, new_states, outputs_ta
        
        # Условие остановки цикла
        def condition(t, states, outputs_ta):
            return t < seq_len
        
        if self.return_sequences:
            # Выполняем цикл
            _, final_states, outputs = tf.while_loop(
                condition, step_fn, 
                [0, states, outputs],
                parallel_iterations=1
            )
            
            # Собираем все выходы
            outputs = outputs.stack()  # (seq_len, batch_size, units)
            outputs = tf.transpose(outputs, [1, 0, 2])  # (batch_size, seq_len, units)
            return outputs
        else:
            # Только последний выход
            def step_fn_last(t, states, last_output):
                current_input = inputs[:, t, :]
                output, new_states = self.cell(current_input, states)
                return t + 1, new_states, output
            
            _, _, last_output = tf.while_loop(
                condition, step_fn_last,
                [0, states, tf.zeros((batch_size, self.units))],
                parallel_iterations=1
            )
            
            return last_output

🔧 ДОПОЛНИТЕЛЬНОЕ ИСПРАВЛЕНИЕ
Также нужно обновить models/xlstm_rl_model.py для лучшей совместимости:
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, Attention
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
import pickle
import os
from .xlstm_memory_cell import XLSTMLayer  # Импортируем настоящий xLSTM

class XLSTMRLModel:
    """
    Настоящая xLSTM модель с расширенной памятью - ИСПРАВЛЕННАЯ ВЕРСИЯ
    """
    
    def __init__(self, input_shape, memory_units=128, memory_size=64, attention_units=64):
        self.input_shape = input_shape
        self.memory_units = memory_units
        self.memory_size = memory_size
        self.attention_units = attention_units
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def build_model(self):
        """
        Строит настоящую xLSTM архитектуру с памятью - ИСПРАВЛЕННАЯ ВЕРСИЯ
        """
        # ✅ Фиксируем входную форму для стабильности
        inputs = Input(shape=self.input_shape, name='input_layer')
        
        # Первый xLSTM слой с внешней памятью
        xlstm1 = XLSTMLayer(
            units=self.memory_units,
            memory_size=self.memory_size,
            return_sequences=True,
            name='xlstm_memory_layer_1'
        )(inputs)
        
        # Второй xLSTM слой
        xlstm2 = XLSTMLayer(
            units=self.memory_units // 2,
            memory_size=self.memory_size // 2,
            return_sequences=True,
            name='xlstm_memory_layer_2'
        )(xlstm1)
        
        # Механизм внимания
        attention = Attention(name='attention_mechanism')([xlstm2, xlstm2])
        
        # Финальный xLSTM слой
        xlstm_final = XLSTMLayer(
            units=self.attention_units,
            memory_size=self.attention_units,
            return_sequences=False,
            name='xlstm_memory_final'
        )(attention)
        
        # Классификационные слои
        dense1 = Dense(64, activation='relu', name='dense_1')(xlstm_final)
        dropout1 = Dropout(0.3)(dense1)
        
        dense2 = Dense(32, activation='relu', name='dense_2')(dropout1)
        dropout2 = Dropout(0.2)(dense2)
        
        # Выходной слой
        outputs = Dense(3, activation='softmax', name='output_layer')(dropout2)
        
        self.model = Model(inputs=inputs, outputs=outputs, name='True_xLSTM_RL_Model')
        
        # Компиляция
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        print("✅ Настоящая xLSTM модель с памятью создана!")
        return self.model
    
    # ... остальные методы остаются без изменений
    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=50, batch_size=32):
        """
        Обучает модель
        """
        if self.model is None:
            self.build_model()
            
        # Нормализация данных
        X_train_scaled = self.scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
        
        if X_val is not None:
            X_val_scaled = self.scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
            validation_data = (X_val_scaled, y_val)
        else:
            validation_data = None
        
        # Обучение
        history = self.model.fit(
            X_train_scaled, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,
            shuffle=True
        )
        
        self.is_trained = True
        return history
    
    def predict(self, X):
        """
        Предсказание
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
            
        X_scaled = self.scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
        return self.model.predict(X_scaled, verbose=0)
    
    def save_model(self, model_path='models/xlstm_rl_model.keras', scaler_path='models/xlstm_rl_scaler.pkl'):
        """
        Сохранение модели и скейлера
        """
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        self.model.save(model_path)
        
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
            
        print(f"xLSTM-RL модель сохранена: {model_path}")
        print(f"Скейлер сохранен: {scaler_path}")
    
    def load_model(self, model_path='models/xlstm_rl_model.keras', scaler_path='models/xlstm_rl_scaler.pkl'):
        """
        Загрузка модели и скейлера
        """
        self.model = tf.keras.models.load_model(model_path, custom_objects={'XLSTMLayer': XLSTMLayer})
        
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
            
        self.is_trained = True
        print(f"xLSTM-RL модель загружена: {model_path}")

🚀 ТЕПЕРЬ ЗАПУСКАЙТЕ ОБУЧЕНИЕ
После внесения этих исправлений запустите:
python train_model.py

Теперь система должна работать правильно! Исправления решают проблему с символическими тензорами, используя tf.while_loop вместо обычного Python цикла, что является правильным подходом для TensorFlow 2.x.