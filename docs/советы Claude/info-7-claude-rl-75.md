

🔧 ЧТО НУЖНО ИСПРАВИТЬ:
Замените ваш _build_critic_model() на:
def _build_critic_model(self):
    """Создает модель критика для оценки действий"""
    inputs = layers.Input(shape=self.input_shape)
    
    # Нормализация входных данных
    x = layers.LayerNormalization()(inputs)
    
    # Первый слой xLSTM
    x = layers.RNN(XLSTMMemoryCell(units=self.memory_units,
                                   memory_size=self.memory_size),
                  return_sequences=True)(x)
    x = layers.LayerNormalization()(x)
    x = layers.Dropout(0.2)(x)
    
    # Второй слой xLSTM
    x = layers.RNN(XLSTMMemoryCell(units=self.memory_units,
                                   memory_size=self.memory_size),
                  return_sequences=False)(x)
    x = layers.LayerNormalization()(x)
    x = layers.Dropout(0.2)(x)
    
    # 🔥 ДОБАВИТЬ: Полносвязные слои С РЕГУЛЯРИЗАЦИЕЙ
    x = layers.Dense(64, activation='relu',
                    kernel_regularizer=tf.keras.regularizers.l2(self.weight_decay))(x)
    x = layers.Dense(32, activation='relu',
                    kernel_regularizer=tf.keras.regularizers.l2(self.weight_decay))(x)
    
    # 🔥 ДОБАВИТЬ: Выходной слой С РЕГУЛЯРИЗАЦИЕЙ
    outputs = layers.Dense(1, 
                          kernel_regularizer=tf.keras.regularizers.l2(self.weight_decay))(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    
    # 🔥 ДОБАВИТЬ: Проверка параметров
    total_params = model.count_params()
    max_params = 10_000_000
    if total_params > max_params:
        print(f"⚠️ Critic модель слишком большая: {total_params:,} параметров")
    else:
        print(f"✅ Размер модели Critic: {total_params:,} параметров")
    
    return model
