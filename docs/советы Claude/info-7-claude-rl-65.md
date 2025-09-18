Проблема: tf.shape(x)[-1] возвращает тензор, а не число, поэтому сравнение x_shape != 64 не работает корректно.
Исправление в models/xlstm_rl_model.py в методе _build_actor_model():
def _build_actor_model(self):
    # ... существующий код до residual connection ...
    
    dense2 = layers.Dense(
        64, 
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(self.weight_decay)
    )(dense1)
    
    # 🔥 ИСПРАВЛЕНО: Используем статическую размерность вместо динамической
    x_static_shape = x.shape[-1]  # Получаем статическую размерность
    if x_static_shape != 64:
        # Приводим x к размеру 64
        x_resized = layers.Dense(64, kernel_regularizer=tf.keras.regularizers.l2(self.weight_decay))(x)
    else:
        x_resized = x
    
    # Residual connection
    x = layers.Add()([x_resized, dense2])
    
    # ... остальной код ...

Альтернативное решение (если статическая размерность недоступна):
# Если x.shape[-1] возвращает None (динамическая размерность)
try:
    x_static_shape = x.shape[-1]
    if x_static_shape is None or x_static_shape != 64:
        # Всегда приводим к размеру 64 для безопасности
        x_resized = layers.Dense(64, kernel_regularizer=tf.keras.regularizers.l2(self.weight_decay))(x)
    else:
        x_resized = x
except:
    # Fallback: всегда приводим к нужному размеру
    x_resized = layers.Dense(64, kernel_regularizer=tf.keras.regularizers.l2(self.weight_decay))(x)

# Residual connection
x = layers.Add()([x_resized, dense2])

Это исправление гарантирует корректное сравнение размерностей и правильную работу residual connection.