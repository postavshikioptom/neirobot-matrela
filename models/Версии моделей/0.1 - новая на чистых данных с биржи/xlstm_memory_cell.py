import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

class XLSTMMemoryCell(layers.Layer):
    """
    Расширенная ячейка памяти xLSTM с улучшенной структурой памяти
    """
    def __init__(self, units, memory_size=64, **kwargs):
        super(XLSTMMemoryCell, self).__init__(**kwargs)
        self.units = units
        self.memory_size = memory_size
        self.state_size = [tf.TensorShape([units]), tf.TensorShape([memory_size, units])]
        self.output_size = units

    def build(self, input_shape):
        input_dim = input_shape[-1]
        
        # Веса для входного гейта
        self.Wi = self.add_weight(shape=(input_dim, self.units),
                                  name='Wi', initializer='glorot_uniform')
        self.Ui = self.add_weight(shape=(self.units, self.units),
                                 name='Ui', initializer='glorot_uniform')
        self.bi = self.add_weight(shape=(self.units,),
                                 name='bi', initializer='zeros')
        
        # Веса для гейта забывания
        self.Wf = self.add_weight(shape=(input_dim, self.units),
                                  name='Wf', initializer='glorot_uniform')
        self.Uf = self.add_weight(shape=(self.units, self.units),
                                 name='Uf', initializer='glorot_uniform')
        self.bf = self.add_weight(shape=(self.units,),
                                 name='bf', initializer='ones')
        
        # Веса для выходного гейта
        self.Wo = self.add_weight(shape=(input_dim, self.units),
                                  name='Wo', initializer='glorot_uniform')
        self.Uo = self.add_weight(shape=(self.units, self.units),
                                 name='Uo', initializer='glorot_uniform')
        self.bo = self.add_weight(shape=(self.units,),
                                 name='bo', initializer='zeros')
        
        # Веса для кандидата на обновление ячейки
        self.Wc = self.add_weight(shape=(input_dim, self.units),
                                  name='Wc', initializer='glorot_uniform')
        self.Uc = self.add_weight(shape=(self.units, self.units),
                                 name='Uc', initializer='glorot_uniform')
        self.bc = self.add_weight(shape=(self.units,),
                                 name='bc', initializer='zeros')
        
        # Веса для расширенной памяти
        self.Wm = self.add_weight(shape=(self.memory_size, self.units),
                                 name='Wm', initializer='glorot_uniform')
        
        # Экспоненциальные гейты для улучшенной памяти
        self.We = self.add_weight(shape=(input_dim, self.memory_size),
                                 name='We', initializer='glorot_uniform')
        self.Ue = self.add_weight(shape=(self.units, self.memory_size),
                                 name='Ue', initializer='glorot_uniform')
        self.be = self.add_weight(shape=(self.memory_size,),
                                 name='be', initializer='zeros')
        
        self.built = True

    def call(self, inputs, states):
        # Предыдущие состояния
        h_prev, memory_prev = states
        
        # Вычисляем гейты с экспоненциальной активацией для стабильности
        i = tf.nn.sigmoid(tf.matmul(inputs, self.Wi) + tf.matmul(h_prev, self.Ui) + self.bi)
        f = tf.nn.sigmoid(tf.matmul(inputs, self.Wf) + tf.matmul(h_prev, self.Uf) + self.bf)
        o = tf.nn.sigmoid(tf.matmul(inputs, self.Wo) + tf.matmul(h_prev, self.Uo) + self.bo)
        
        # Кандидат на новое значение ячейки
        c_tilde = tf.nn.tanh(tf.matmul(inputs, self.Wc) + tf.matmul(h_prev, self.Uc) + self.bc)
        
        # Экспоненциальный гейт для расширенной памяти
        e = tf.nn.softmax(tf.matmul(inputs, self.We) + tf.matmul(h_prev, self.Ue) + self.be)
        e = tf.reshape(e, [-1, self.memory_size, 1])
        
        # Обновление расширенной памяти с использованием механизма внимания
        memory_contribution = tf.reduce_sum(memory_prev * e, axis=1)
        
        # Обновление состояния ячейки с учетом расширенной памяти
        c = f * memory_contribution + i * c_tilde
        
        # Обновление выхода
        h = o * tf.nn.tanh(c)
        
        # Обновление памяти - сдвигаем старые значения и добавляем новое
        new_memory_item = tf.expand_dims(c, axis=1)
        memory_new = tf.concat([new_memory_item, memory_prev[:, :-1, :]], axis=1)
        
        return h, [h, memory_new]

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        """Создает начальное состояние для RNN ячейки"""
        # Определяем batch_size
        if inputs is not None:
            batch_size = tf.shape(inputs)[0]
            dtype = inputs.dtype
        elif batch_size is None:
            batch_size = 1
        
        # Определяем dtype
        if dtype is None:
            dtype = tf.float32
        
        # Создаем начальные состояния
        h_init = tf.zeros([batch_size, self.units], dtype=dtype)
        memory_init = tf.zeros([batch_size, self.memory_size, self.units], dtype=dtype)
        
        return [h_init, memory_init]

    def get_config(self):
        config = super(XLSTMMemoryCell, self).get_config()
        config.update({
            'units': self.units,
            'memory_size': self.memory_size
        })
        return config