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
                         tf.reduce_mean(memory_read, axis=0, keepdims=True) + self.b_i)
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
    Слой xLSTM с использованием кастомной ячейки памяти
    """
    
    def __init__(self, units, memory_size=64, return_sequences=False, **kwargs):
        super(XLSTMLayer, self).__init__(**kwargs)
        self.units = units
        self.memory_size = memory_size
        self.return_sequences = return_sequences
        self.cell = XLSTMMemoryCell(units, memory_size)
        
    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        seq_len = tf.shape(inputs)[1]
        
        # Инициализируем состояния
        h = tf.zeros((batch_size, self.units))
        c = tf.zeros((batch_size, self.units))
        memory = tf.zeros((batch_size, self.memory_size))
        
        states = [h, c, memory]
        outputs = []
        
        # Проходим по временным шагам
        for t in range(seq_len):
            output, states = self.cell(inputs[:, t, :], states)
            outputs.append(output)
            
        if self.return_sequences:
            return tf.stack(outputs, axis=1)
        else:
            return outputs[-1]