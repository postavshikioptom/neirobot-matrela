Давайте добавим логирование форм тензоров непосредственно перед слоем layers.Add() в файле models/xlstm_rl_model.py, в методе _build_actor_model.
# В файле models/xlstm_rl_model.py

class XLSTMRLModel:
    # ... (другой код) ...

    def _build_actor_model(self):
        # ... (существующий код до residual connections) ...
        
        # 🔥 ИСПРАВЛЕНО: Правильные residual connections с проверкой размерностей
        dense1 = layers.Dense(
            128, 
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(self.weight_decay)
        )(x)
        dense1 = layers.Dropout(0.2)(dense1)
        
        dense2 = layers.Dense(
            64, 
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(self.weight_decay)
        )(dense1)
        
        # 🔥 ДОБАВЛЕНО: Логирование форм перед операцией сложения
        print(f"DEBUG Actor Model: x (before resize) shape={x.shape}")
        
        x_static_shape = x.shape[-1]
        if x_static_shape != 64:
            x_resized = layers.Dense(64, kernel_regularizer=tf.keras.regularizers.l2(self.weight_decay))(x)
            print(f"DEBUG Actor Model: x_resized shape={x_resized.shape}")
        else:
            x_resized = x
            print(f"DEBUG Actor Model: x_resized (no change) shape={x_resized.shape}")
        
        print(f"DEBUG Actor Model: dense2 shape={dense2.shape}")
        
        # Residual connection
        # 🔥 ДОБАВЛЕНО: Дополнительная проверка на совместимость форм перед сложением
        if x_resized.shape[-1] != dense2.shape[-1]:
            print(f"⚠️ Ошибка: Формы для residual connection не совпадают: x_resized={x_resized.shape}, dense2={dense2.shape}")
            # Можно добавить слой для приведения форм, если это необходимо, или вызвать ошибку
            # Например: dense2_aligned = layers.Dense(x_resized.shape[-1])(dense2)
        
        x = layers.Add()([x_resized, dense2])
        print(f"DEBUG Actor Model: x (after add) shape={x.shape}")
        
        # ... (остальной код) ...
