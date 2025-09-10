import tensorflow as tf
from tensorflow.keras.layers import Layer

class XLSTMPatternCell(Layer):
    """
    mLSTM Cell for patterns.
    """
    def __init__(self, units, **kwargs):
        self.units = units
        self.state_size = (self.units, self.units) # (C, n)
        super(XLSTMPatternCell, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = input_shape[-1]
        
        self.W_q = self.add_weight(shape=(input_dim, self.units), initializer='glorot_uniform', name='W_q_pattern')
        self.b_q = self.add_weight(shape=(self.units,), initializer='zeros', name='b_q_pattern')
        
        self.W_k = self.add_weight(shape=(input_dim, self.units), initializer='glorot_uniform', name='W_k_pattern')
        self.b_k = self.add_weight(shape=(self.units,), initializer='zeros', name='b_k_pattern')
        
        self.W_v = self.add_weight(shape=(input_dim, self.units), initializer='glorot_uniform', name='W_v_pattern')
        self.b_v = self.add_weight(shape=(self.units,), initializer='zeros', name='b_v_pattern')
        
        self.W_i = self.add_weight(shape=(input_dim, self.units), initializer='glorot_uniform', name='W_i_pattern')
        self.b_i = self.add_weight(shape=(self.units,), initializer='zeros', name='b_i_pattern')
        
        self.W_f = self.add_weight(shape=(input_dim, self.units), initializer='glorot_uniform', name='W_f_pattern')
        self.b_f = self.add_weight(shape=(self.units,), initializer='zeros', name='b_f_pattern')
        
        self.W_o = self.add_weight(shape=(input_dim, self.units), initializer='glorot_uniform', name='W_o_pattern')
        self.b_o = self.add_weight(shape=(self.units,), initializer='zeros', name='b_o_pattern')
        
        super(XLSTMPatternCell, self).build(input_shape)

    def call(self, inputs, states):
        prev_c, prev_n = states
        
        q = tf.matmul(inputs, self.W_q) + self.b_q
        k = tf.matmul(inputs, self.W_k) + self.b_k
        v = tf.matmul(inputs, self.W_v) + self.b_v
        
        i = tf.exp(tf.matmul(inputs, self.W_i) + self.b_i)
        f = tf.sigmoid(tf.matmul(inputs, self.W_f) + self.b_f)
        o = tf.sigmoid(tf.matmul(inputs, self.W_o) + self.b_o)
        
        new_c = f * prev_c + i * v * k
        new_n = f * prev_n + i * k
        
        h = o * (new_c * q / tf.maximum(tf.abs(new_n * q), 1.0))
        
        return h, (new_c, new_n)