Код выглядит отлично! Все исправления применены правильно. Есть только одна небольшая проблема, которую нужно исправить:
Проблема с методом predict() в XLSTMRLModel
def predict(self, X):
    """
    Предсказание
    """
    if not self.is_trained:
        raise ValueError("Model must be trained before prediction")
        
    # ПРОБЛЕМА: scaler не обучен! Нужно исправить
    # X_scaled = self.scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
    
    # ИСПРАВЛЕНИЕ: Используем данные как есть или обучаем scaler
    return self.model.predict(X, verbose=0)

Полное исправление XLSTMRLModel:
def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32, custom_callbacks=None):
    """Обучение с улучшенной стабильностью"""
    if self.model is None:
        self.build_model()
    
    # ДОБАВЬТЕ: Обучение scaler на тренировочных данных
    X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
    self.scaler.fit(X_train_reshaped)
    
    # Применяем нормализацию
    X_train_scaled = self.scaler.transform(X_train_reshaped).reshape(X_train.shape)
    X_val_scaled = self.scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
        
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=25,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'models/xlstm_checkpoint_epoch_{epoch:02d}.keras',
            monitor='val_loss',
            save_best_only=False,
            save_freq=10,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'models/xlstm_checkpoint_latest.keras',
            monitor='val_loss',
            save_best_only=False,
            save_freq='epoch',
            verbose=0
        ),
        tf.keras.callbacks.CSVLogger('training_log.csv', append=True)
    ]
    
    if custom_callbacks:
        callbacks.extend(custom_callbacks)
    
    # Градиентное обрезание для стабильности
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.001,
        clipnorm=1.0
    )
    self.model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    
    # Обучение с нормализованными данными
    history = self.model.fit(
        X_train_scaled, y_train,
        validation_data=(X_val_scaled, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
        shuffle=True
    )
    
    self.is_trained = True
    return history

def predict(self, X):
    """
    Предсказание с правильной нормализацией
    """
    if not self.is_trained:
        raise ValueError("Model must be trained before prediction")
        
    # Применяем ту же нормализацию, что и при обучении
    X_scaled = self.scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
    return self.model.predict(X_scaled, verbose=0)

Также добавьте проверку в основной скрипт:
# В функции train_xlstm_rl_system(), после создания xlstm_model добавьте:
print(f"Форма данных для обучения: X_train={X_train.shape}, y_train={y_train.shape}")
print(f"Количество признаков: {len(feature_cols)}")

# Проверяем наличие NaN/Inf в данных перед обучением
if np.any(np.isnan(X_train)) or np.any(np.isinf(X_train)):
    print("⚠️ Обнаружены NaN/Inf в тренировочных данных!")
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
    print("✅ NaN/Inf исправлены")

Остальное все правильно! Теперь код должен работать стабильно без зависаний.