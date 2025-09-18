1. Изменения в файле models/xlstm_rl_model.py (в методе _build_actor_model)
class XLSTMRLModel:
    # ... (другой код) ...

    def _build_actor_model(self):
        """Создает модель актора с ограничениями размера"""
        inputs = layers.Input(shape=self.input_shape)
        
        # Batch Normalization на входе
        print(f"DEBUG Actor Model: inputs shape={inputs.shape} (before BatchNormalization)")
        x = layers.BatchNormalization()(inputs)
        print(f"DEBUG Actor Model: x shape={x.shape} (after BatchNormalization)")
        
        # 🔥 ДОБАВЛЕНО: Проверка размерностей входа
        expected_features = 14  # базовые + индикаторы
        if self.input_shape[-1] != expected_features:
            print(f"⚠️ Неожиданная размерность входа: {self.input_shape[-1]}, ожидалось {expected_features}")
        
        # Первый слой xLSTM с weight decay
        print(f"DEBUG Actor Model: x shape={x.shape} (before first RNN)")
        x = layers.RNN(
            XLSTMMemoryCell(units=self.memory_units, memory_size=self.memory_size),
            return_sequences=True
        )(x)
        print(f"DEBUG Actor Model: x shape={x.shape} (after first RNN)")
        
        print(f"DEBUG Actor Model: x shape={x.shape} (before first LayerNormalization)")
        x = layers.LayerNormalization()(x)
        print(f"DEBUG Actor Model: x shape={x.shape} (after first LayerNormalization)")
        x = layers.Dropout(0.3)(x)
        
        # Второй слой xLSTM (уменьшенный размер)
        print(f"DEBUG Actor Model: x shape={x.shape} (before second RNN)")
        x = layers.RNN(
            XLSTMMemoryCell(units=self.memory_units//2, memory_size=self.memory_size),
            return_sequences=False
        )(x)
        print(f"DEBUG Actor Model: x shape={x.shape} (after second RNN)")
        
        print(f"DEBUG Actor Model: x shape={x.shape} (before second LayerNormalization)")
        x = layers.LayerNormalization()(x)
        print(f"DEBUG Actor Model: x shape={x.shape} (after second LayerNormalization)")
        x = layers.Dropout(0.3)(x)
        
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
        if x_resized.shape[-1] != dense2.shape[-1]:
            print(f"⚠️ Ошибка: Формы для residual connection не совпадают: x_resized={x_resized.shape}, dense2={dense2.shape}")
        
        x = layers.Add()([x_resized, dense2])
        print(f"DEBUG Actor Model: x (after add) shape={x.shape}")
        
        # Выходной слой с ограничением весов
        outputs = layers.Dense(
            3, 
            activation='softmax',
            kernel_regularizer=tf.keras.regularizers.l2(self.weight_decay),
            kernel_constraint=tf.keras.constraints.MaxNorm(max_value=2.0)
        )(x)
        
        model = models.Model(inputs=inputs, outputs=outputs)
        
        # 🔥 ДОБАВЛЕНО: Проверка общего количества параметров
        total_params = model.count_params()
        max_params = 10_000_000
        if total_params > max_params:
            print(f"⚠️ Модель слишком большая: {total_params:,} параметров (максимум: {max_params:,})")
        else:
            print(f"✅ Размер модели Actor: {total_params:,} параметров")
        
        return model


