import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import os

class XLSTMMemoryCell(layers.Layer):
    """
    Расширенная ячейка памяти xLSTM с улучшенной структурой памяти
    """
    def __init__(self, units, memory_size=64, debug_mode=False, **kwargs):
        super(XLSTMMemoryCell, self).__init__(**kwargs)
        self.units = units
        self.memory_size = memory_size
        # 🔥 ДОБАВЛЕНО: Режим отладки
        self.debug_mode = debug_mode or os.environ.get('XLSTM_DEBUG', 'False').lower() == 'true'
        self.state_size = [tf.TensorShape([units]), tf.TensorShape([memory_size, units])]
        self.output_size = units
        # 🔥 ИСПРАВЛЕНО: Убрано лишнее логирование из init
        # print(f"DEBUG XLSTM CELL __init__: units={self.units}, memory_size={self.memory_size}, state_size={self.state_size}, output_size={self.output_size}")

    def build(self, input_shape):
        input_dim = input_shape[-1]
        
        # 🔥 ИСПРАВЛЕНО: Убрано лишнее логирование из build
        # print(f"DEBUG XLSTM build: input_shape={input_shape}, input_dim={input_dim}, units={self.units}, memory_size={self.memory_size}")
        
        # Веса для входного гейта
        self.Wi = self.add_weight(shape=(input_dim, self.units),
                                  name='Wi', initializer='glorot_uniform')
        self.Ui = self.add_weight(shape=(self.units, self.units),
                                 name='Ui', initializer='glorot_uniform')
        self.bi = self.add_weight(shape=(self.units,),
                                 name='bi', initializer='zeros')
        # 🔥 ИСПРАВЛЕНО: Убрано лишнее логирование форм весов после создания
        # print(f"DEBUG XLSTM build: Wi shape={self.Wi.shape}, Ui shape={self.Ui.shape}, bi shape={self.bi.shape}")
        
        # Веса для гейта забывания
        self.Wf = self.add_weight(shape=(input_dim, self.units),
                                  name='Wf', initializer='glorot_uniform')
        self.Uf = self.add_weight(shape=(self.units, self.units),
                                 name='Uf', initializer='glorot_uniform')
        self.bf = self.add_weight(shape=(self.units,),
                                 name='bf', initializer='ones')
        # print(f"DEBUG XLSTM build: Wf shape={self.Wf.shape}, Uf shape={self.Uf.shape}, bf shape={self.bf.shape}")
        
        # Веса для выходного гейта
        self.Wo = self.add_weight(shape=(input_dim, self.units),
                                  name='Wo', initializer='glorot_uniform')
        self.Uo = self.add_weight(shape=(self.units, self.units),
                                 name='Uo', initializer='glorot_uniform')
        self.bo = self.add_weight(shape=(self.units,),
                                 name='bo', initializer='zeros')
        # print(f"DEBUG XLSTM build: Wo shape={self.Wo.shape}, Uo shape={self.Uo.shape}, bo shape={self.bo.shape}")
        
        # Веса для кандидата на обновление ячейки
        self.Wc = self.add_weight(shape=(input_dim, self.units),
                                  name='Wc', initializer='glorot_uniform')
        self.Uc = self.add_weight(shape=(self.units, self.units),
                                 name='Uc', initializer='glorot_uniform')
        self.bc = self.add_weight(shape=(self.units,),
                                 name='bc', initializer='zeros')
        # print(f"DEBUG XLSTM build: Wc shape={self.Wc.shape}, Uc shape={self.Uc.shape}, bc shape={self.bc.shape}")
        
        # Веса для расширенной памяти
        self.Wm = self.add_weight(shape=(input_dim, self.units),
                                 name='Wm', initializer='glorot_uniform')
        # print(f"DEBUG XLSTM build: Wm shape={self.Wm.shape}")
        
        # Экспоненциальные гейты для улучшенной памяти
        self.We = self.add_weight(shape=(input_dim, self.memory_size),
                                 name='We', initializer='glorot_uniform')
        self.Ue = self.add_weight(shape=(self.units, self.memory_size),
                                 name='Ue', initializer='glorot_uniform')
        self.be = self.add_weight(shape=(self.memory_size,),
                                 name='be', initializer='zeros')
        # print(f"DEBUG XLSTM build: We shape={self.We.shape}, Ue shape={self.Ue.shape}, be shape={self.be.shape}")
        
        self.built = True

    def call(self, inputs, states):
        # 🔥 ИСПРАВЛЕНО: Убрано лишнее логирование из call
        # print(f"\nDEBUG XLSTM call (start): inputs_shape={inputs.shape}, states_shape={[s.shape for s in states]}")
        
        # 🔥 ИСПРАВЛЕНО: Проверки NaN только в debug режиме
        if self.debug_mode:
            inputs = tf.debugging.check_numerics(inputs, "NaN detected in XLSTMMemoryCell inputs")
        
        # Предыдущие состояния
        h_prev, memory_prev = states
        
        if self.debug_mode:
            h_prev = tf.debugging.check_numerics(h_prev, "NaN detected in h_prev")
            memory_prev = tf.debugging.check_numerics(memory_prev, "NaN detected in memory_prev")
        
        # Ограничиваем входы для предотвращения overflow (всегда активно)
        inputs = tf.clip_by_value(inputs, -10.0, 10.0)
        
        # Вычисляем гейты с ограничением значений
        i_logits = tf.matmul(inputs, self.Wi) + tf.matmul(h_prev, self.Ui) + self.bi
        i_logits = tf.clip_by_value(i_logits, -10.0, 10.0)
        i = tf.nn.sigmoid(i_logits)
        # print(f"DEBUG XLSTM call: i_logits_shape={i_logits.shape}, i_shape={i.shape}")
        
        f_logits = tf.matmul(inputs, self.Wf) + tf.matmul(h_prev, self.Uf) + self.bf
        f_logits = tf.clip_by_value(f_logits, -10.0, 10.0)
        f = tf.nn.sigmoid(f_logits)
        # print(f"DEBUG XLSTM call: f_logits_shape={f_logits.shape}, f_shape={f.shape}")
        
        o_logits = tf.matmul(inputs, self.Wo) + tf.matmul(h_prev, self.Uo) + self.bo
        o_logits = tf.clip_by_value(o_logits, -10.0, 10.0)
        o = tf.nn.sigmoid(o_logits)
        # print(f"DEBUG XLSTM call: o_logits_shape={o_logits.shape}, o_shape={o.shape}")
        
        # 🔥 ИСПРАВЛЕНО: Проверки только в debug режиме
        if self.debug_mode:
            i = tf.debugging.check_numerics(i, "NaN detected in input gate")
            f = tf.debugging.check_numerics(f, "NaN detected in forget gate")
            o = tf.debugging.check_numerics(o, "NaN detected in output gate")
        
        # Кандидат на новое значение ячейки
        c_tilde_logits = tf.matmul(inputs, self.Wc) + tf.matmul(h_prev, self.Uc) + self.bc
        c_tilde_logits = tf.clip_by_value(c_tilde_logits, -10.0, 10.0)
        c_tilde = tf.nn.tanh(c_tilde_logits)
        # print(f"DEBUG XLSTM call: c_tilde_logits_shape={c_tilde_logits.shape}, c_tilde_shape={c_tilde.shape}")
        
        # Экспоненциальный гейт для расширенной памяти
        e_logits = tf.matmul(inputs, self.We) + tf.matmul(h_prev, self.Ue) + self.be
        e_logits = tf.clip_by_value(e_logits, -10.0, 10.0)
        e = tf.nn.softmax(e_logits)
        e = tf.reshape(e, [-1, self.memory_size, 1])
        # print(f"DEBUG XLSTM call: e_logits_shape={e_logits.shape}, e_reshaped_shape={e.shape}")
        
        # Обновление расширенной памяти с использованием механизма внимания
        # print(f"DEBUG XLSTM call: memory_prev_shape={memory_prev.shape}, e_shape={e.shape}")
        memory_attention = tf.reduce_sum(memory_prev * e, axis=1)
        # print(f"DEBUG XLSTM call: memory_attention_shape={memory_attention.shape}")
        
        memory_input = tf.matmul(inputs, self.Wm)
        # print(f"DEBUG XLSTM call: memory_input_shape={memory_input.shape}")
        memory_contribution = memory_attention + memory_input
        # print(f"DEBUG XLSTM call: memory_contribution_shape={memory_contribution.shape}")
        
        # Обновление состояния ячейки с учетом расширенной памяти
        c = f * memory_contribution + i * c_tilde
        # print(f"DEBUG XLSTM call: c_shape={c.shape}")
        
        # Обновление выхода
        h = o * tf.nn.tanh(c)
        # print(f"DEBUG XLSTM call: h_shape={h.shape}")
        
        # Обновление памяти - сдвигаем старые значения и добавляем новое
        new_memory_item = tf.expand_dims(c, axis=1)
        memory_new = tf.concat([new_memory_item, memory_prev[:, :-1, :]], axis=1)
        # print(f"DEBUG XLSTM call: new_memory_item_shape={new_memory_item.shape}, memory_new_shape={memory_new.shape}")
        
        # 🔥 ИСПРАВЛЕНО: Финальные проверки только в debug режиме
        if self.debug_mode:
            h = tf.debugging.check_numerics(h, "NaN detected in output h")
            memory_new = tf.debugging.check_numerics(memory_new, "NaN detected in memory_new")
        
        # print(f"DEBUG XLSTM call (end): h_shape={h.shape}, memory_new_shape={memory_new.shape}")
        
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