import tensorflow as tf
from tensorflow.keras.layers import Layer

class sLSTMCell(Layer):
    """
    sLSTM Cell.
    """
    def __init__(self, units, **kwargs):
        self.units = units
        self.state_size = (self.units, self.units) # (c, n)
        super(sLSTMCell, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = input_shape[-1]
        
        self.W_z = self.add_weight(shape=(input_dim, self.units), initializer='glorot_uniform', name='W_z')
        self.R_z = self.add_weight(shape=(self.units, self.units), initializer='orthogonal', name='R_z')
        self.b_z = self.add_weight(shape=(self.units,), initializer='zeros', name='b_z')
        
        self.W_i = self.add_weight(shape=(input_dim, self.units), initializer='glorot_uniform', name='W_i')
        self.R_i = self.add_weight(shape=(self.units, self.units), initializer='orthogonal', name='R_i')
        self.b_i = self.add_weight(shape=(self.units,), initializer='zeros', name='b_i')
        
        self.W_f = self.add_weight(shape=(input_dim, self.units), initializer='glorot_uniform', name='W_f')
        self.R_f = self.add_weight(shape=(self.units, self.units), initializer='orthogonal', name='R_f')
        self.b_f = self.add_weight(shape=(self.units,), initializer='zeros', name='b_f')
        
        self.W_o = self.add_weight(shape=(input_dim, self.units), initializer='glorot_uniform', name='W_o')
        self.R_o = self.add_weight(shape=(self.units, self.units), initializer='orthogonal', name='R_o')
        self.b_o = self.add_weight(shape=(self.units,), initializer='zeros', name='b_o')
        
        super(sLSTMCell, self).build(input_shape)

    def call(self, inputs, states):
        prev_h, prev_c, prev_n = states
        
        z = tf.tanh(tf.matmul(inputs, self.W_z) + tf.matmul(prev_h, self.R_z) + self.b_z)
        i = tf.exp(tf.matmul(inputs, self.W_i) + tf.matmul(prev_h, self.R_i) + self.b_i)
        f = tf.sigmoid(tf.matmul(inputs, self.W_f) + tf.matmul(prev_h, self.R_f) + self.b_f)
        o = tf.sigmoid(tf.matmul(inputs, self.W_o) + tf.matmul(prev_h, self.R_o) + self.b_o)
        
        new_c = f * prev_c + i * z
        new_n = f * prev_n + i
        
        h = o * (new_c / new_n)
        
        return h, (h, new_c, new_n)

class mLSTMCell(Layer):
    """
    mLSTM Cell.
    """
    def __init__(self, units, **kwargs):
        self.units = units
        self.state_size = (self.units, self.units) # (C, n)
        super(mLSTMCell, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = input_shape[-1]
        
        self.W_q = self.add_weight(shape=(input_dim, self.units), initializer='glorot_uniform', name='W_q')
        self.b_q = self.add_weight(shape=(self.units,), initializer='zeros', name='b_q')
        
        self.W_k = self.add_weight(shape=(input_dim, self.units), initializer='glorot_uniform', name='W_k')
        self.b_k = self.add_weight(shape=(self.units,), initializer='zeros', name='b_k')
        
        self.W_v = self.add_weight(shape=(input_dim, self.units), initializer='glorot_uniform', name='W_v')
        self.b_v = self.add_weight(shape=(self.units,), initializer='zeros', name='b_v')
        
        self.W_i = self.add_weight(shape=(input_dim, self.units), initializer='glorot_uniform', name='W_i')
        self.b_i = self.add_weight(shape=(self.units,), initializer='zeros', name='b_i')
        
        self.W_f = self.add_weight(shape=(input_dim, self.units), initializer='glorot_uniform', name='W_f')
        self.b_f = self.add_weight(shape=(self.units,), initializer='zeros', name='b_f')
        
        self.W_o = self.add_weight(shape=(input_dim, self.units), initializer='glorot_uniform', name='W_o')
        self.b_o = self.add_weight(shape=(self.units,), initializer='zeros', name='b_o')
        
        super(mLSTMCell, self).build(input_shape)

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