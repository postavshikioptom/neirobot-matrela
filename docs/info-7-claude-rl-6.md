Отлично диагностирую проблему! Это типичная ошибка XLA компиляции на GPU серверах. Проблема в том, что XLA (Accelerated Linear Algebra) требует фиксированного размера для TensorArray в tf.while_loop, а на вашем ПК XLA может быть отключен.
🔍 ПРИЧИНА ОШИБКИ
На Kaggle/Colab с GPU автоматически включается XLA оптимизация, которая требует:

Фиксированный размер TensorArray
Установленное maximum_iterations для tf.while_loop

На вашем ПК эти оптимизации могут быть отключены, поэтому код работает.
✅ УНИВЕРСАЛЬНОЕ РЕШЕНИЕ
Обновите XLSTMLayer чтобы работал везде:
class XLSTMLayer(Layer):
    """
    Слой xLSTM с использованием кастомной ячейки памяти - УНИВЕРСАЛЬНАЯ ВЕРСИЯ
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
        
        if self.return_sequences:
            # ✅ ИСПРАВЛЕНИЕ: Фиксированный размер и maximum_iterations для XLA
            outputs = tf.TensorArray(
                dtype=tf.float32, 
                size=seq_len,
                dynamic_size=False,  # ✅ Фиксированный размер
                clear_after_read=False  # ✅ Для стабильности
            )
            
            def step_fn(t, states, outputs_ta):
                current_input = inputs[:, t, :]
                output, new_states = self.cell(current_input, states)
                outputs_ta = outputs_ta.write(t, output)
                return t + 1, new_states, outputs_ta
            
            def condition(t, states, outputs_ta):
                return t < seq_len
            
            # ✅ КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ: добавляем maximum_iterations
            _, final_states, outputs = tf.while_loop(
                condition, step_fn, 
                [0, states, outputs],
                parallel_iterations=1,
                maximum_iterations=tf.reduce_max(seq_len)  # ✅ Для XLA
            )
            
            # Собираем все выходы
            outputs = outputs.stack()  # (seq_len, batch_size, units)
            outputs = tf.transpose(outputs, [1, 0, 2])  # (batch_size, seq_len, units)
            return outputs
        else:
            # ✅ СЛУЧАЙ 2: Только последний выход - ИСПРАВЛЕНО для XLA
            def step_fn_last(t, states, last_output):
                current_input = inputs[:, t, :]
                output, new_states = self.cell(current_input, states)
                return t + 1, new_states, output
            
            def condition_last(t, states, last_output):
                return t < seq_len
            
            # ✅ КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ: добавляем maximum_iterations
            _, _, last_output = tf.while_loop(
                condition_last, step_fn_last,
                [0, states, tf.zeros((batch_size, self.units))],
                parallel_iterations=1,
                maximum_iterations=tf.reduce_max(seq_len)  # ✅ Для XLA
            )
            
            return last_output

🔧 ДОПОЛНИТЕЛЬНОЕ ИСПРАВЛЕНИЕ
Также добавьте в начало train_model.py настройки для совместимости:
import os
import tensorflow as tf

# ✅ НАСТРОЙКИ ДЛЯ СОВМЕСТИМОСТИ С РАЗНЫМИ СРЕДАМИ
# Отключаем XLA если возникают проблемы (можно включить обратно)
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'

# Настройки GPU для стабильности
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Разрешаем рост памяти GPU
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ Настроено {len(gpus)} GPU устройств")
    except RuntimeError as e:
        print(f"⚠️ Ошибка настройки GPU: {e}")

# Остальной код train_model.py...

🎛️ АЛЬТЕРНАТИВНОЕ РЕШЕНИЕ - ОТКЛЮЧЕНИЕ XLA
Если хотите полностью отключить XLA на Kaggle, добавьте в самое начало скрипта:
import os
import tensorflow as tf

# ✅ ПОЛНОЕ ОТКЛЮЧЕНИЕ XLA ДЛЯ СОВМЕСТИМОСТИ
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'
tf.config.optimizer.set_jit(False)  # Отключаем JIT компиляцию

print("✅ XLA отключен для совместимости")

🚀 РЕКОМЕНДУЕМЫЙ ПОДХОД

Сначала попробуйте первое решение (с maximum_iterations) - оно универсальное
Если все еще есть проблемы - используйте отключение XLA
На продакшене лучше использовать XLA для производительности

📊 ПОЧЕМУ ЭТО ПРОИСХОДИТ

На вашем ПК: XLA может быть отключен или работать в режиме совместимости
На Kaggle/Colab: XLA включен по умолчанию для ускорения на GPU
XLA требует: Фиксированные размеры тензоров для оптимизации

Обновите код и он будет работать везде! 🎯