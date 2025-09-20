import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os
import tensorflow.keras.backend as K
import gc

# Безопасный импорт
try:
    from models.xlstm_memory_cell import XLSTMMemoryCell
except ImportError as e:
    print(f"❌ Ошибка импорта XLSTMMemoryCell: {e}")
    print("Убедитесь, что файл models/xlstm_memory_cell.py существует")
    raise ImportError("XLSTMMemoryCell не найден")


class FocalLoss(tf.keras.losses.Loss):
    """
    Focal Loss для борьбы с дисбалансом классов
    Автоматически адаптируется к любому распределению классов
    """
    def __init__(self, alpha=0.25, gamma=2.0, label_smoothing=0.1, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
    
    def call(self, y_true, y_pred):
        # Label smoothing для регуляризации
        num_classes = tf.cast(tf.shape(y_pred)[-1], tf.float32)
        y_true_smooth = y_true * (1.0 - self.label_smoothing) + self.label_smoothing / num_classes
        
        # Вычисляем cross-entropy loss
        ce_loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=False)
        
        # Вычисляем p_t (вероятность правильного класса)
        p_t = tf.exp(-ce_loss)
        
        # Focal Loss формула: FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
        focal_loss = self.alpha * tf.pow(1 - p_t, self.gamma) * ce_loss
        
        return focal_loss
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'alpha': self.alpha,
            'gamma': self.gamma,
            'label_smoothing': self.label_smoothing
        })
        return config


class WarmUpCosineDecayScheduler(tf.keras.callbacks.Callback):
    """
    Learning Rate Scheduler с WarmUp и Cosine Decay
    Улучшает стабильность обучения и конвергенцию
    """
    def __init__(self, warmup_epochs=5, total_epochs=50, base_lr=0.001, min_lr=1e-6):
        super().__init__()
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr
        
    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.warmup_epochs:
            # WarmUp phase: линейное увеличение LR
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            # Cosine Decay phase
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
        
        self.model.optimizer.learning_rate.assign(lr)
        print(f"Epoch {epoch + 1}: Установлена скорость обучения: {lr:.6f}")





class XLSTMRLModel:
    """
    Модель xLSTM с RL для торговли - ТРЁХЭТАПНАЯ АРХИТЕКТУРА
    """
    def __init__(self, input_shape, memory_size=64, memory_units=128, weight_decay=1e-4, gradient_clip_norm=1.0):
        self.input_shape = input_shape
        self.memory_size = memory_size
        self.memory_units = memory_units
        # 🔥 ДОБАВЛЕНО: Параметры регуляризации
        self.weight_decay = 5e-4
        self.gradient_clip_norm = gradient_clip_norm
        
        # 🔥 ИСПРАВЛЕНО: Добавьте эти строки для инициализации моделей
        self.actor_model = self._build_actor_model()
        self.critic_model = self._build_critic_model()
        
        # Настройка оптимизаторов с учетом доступного устройства
        self._configure_optimizers()
        
        # 🔥 ДОБАВЛЕНО: Буфер для батчевых предсказаний
        self.prediction_count = 0
        self.batch_predictions = []
        self.batch_size = 32

    def _configure_optimizers(self):
        """Настраивает оптимизаторы с учетом доступного устройства"""
        try:
            # Проверяем, доступны ли GPU
            physical_devices = tf.config.list_physical_devices('GPU')
            if len(physical_devices) > 0:
                # Для GPU используем более агрессивные настройки
                self.supervised_optimizer = tf.keras.optimizers.Adam(
                    clipnorm=self.gradient_clip_norm
                )
                self.supervised_optimizer.learning_rate = tf.Variable(0.001)
                self.actor_optimizer = tf.keras.optimizers.Adam(
                    clipnorm=self.gradient_clip_norm
                )
                self.actor_optimizer.learning_rate = tf.Variable(0.0005)
                self.critic_optimizer = tf.keras.optimizers.Adam(
                    clipnorm=self.gradient_clip_norm
                )
                self.critic_optimizer.learning_rate = tf.Variable(0.001)
                print("Настроены оптимизаторы для GPU")
            else:
                # Для CPU используем более консервативные настройки
                self.supervised_optimizer = tf.keras.optimizers.Adam(
                    clipnorm=self.gradient_clip_norm
                )
                self.supervised_optimizer.learning_rate = tf.Variable(0.0005)
                self.actor_optimizer = tf.keras.optimizers.Adam(
                    clipnorm=self.gradient_clip_norm
                )
                self.actor_optimizer.learning_rate = tf.Variable(0.0001)
                self.critic_optimizer = tf.keras.optimizers.Adam(
                    clipnorm=self.gradient_clip_norm
                )
                self.critic_optimizer.learning_rate = tf.Variable(0.0005)
                print("Настроены оптимизаторы для CPU")
        except Exception as e:
            print(f"Ошибка при настройке оптимизаторов: {e}")
            # Fallback на стандартные настройки
            self.supervised_optimizer = tf.keras.optimizers.Adam(
                clipnorm=self.gradient_clip_norm
            )
            self.supervised_optimizer.learning_rate = tf.Variable(0.001)
            self.actor_optimizer = tf.keras.optimizers.Adam(
                clipnorm=self.gradient_clip_norm
            )
            self.actor_optimizer.learning_rate = tf.Variable(0.001)
            self.critic_optimizer = tf.keras.optimizers.Adam(
                clipnorm=self.gradient_clip_norm
            )
            self.critic_optimizer.learning_rate = tf.Variable(0.001)

    def _build_actor_model(self):
        """Создает модель актора с ограничениями размера"""
        inputs = layers.Input(shape=self.input_shape)
        
        # Batch Normalization на входе
        # print(f"DEBUG Actor Model: inputs shape={inputs.shape} (before BatchNormalization)")
        x = layers.BatchNormalization()(inputs)
        # print(f"DEBUG Actor Model: x shape={x.shape} (after BatchNormalization)")
        
        # 🔥 ДОБАВЛЕНО: Проверка размерностей входа (оставляем)
        expected_features = 14  # базовые + индикаторы
        if self.input_shape[-1] != expected_features:
            print(f"⚠️ Неожиданная размерность входа: {self.input_shape[-1]}, ожидалось {expected_features}")
        
        # Первый слой xLSTM с weight decay
        # print(f"DEBUG Actor Model: x shape={x.shape} (before first RNN)")
        x = layers.RNN(
            XLSTMMemoryCell(units=self.memory_units, memory_size=self.memory_size),
            return_sequences=True
        )(x)
        # print(f"DEBUG Actor Model: x shape={x.shape} (after first RNN)")
        
        x = layers.LayerNormalization()(x)
        x = layers.Dropout(0.4)(x)
        
        # Второй слой xLSTM (уменьшенный размер)
        # print(f"DEBUG Actor Model: x shape={x.shape} (before second RNN)")
        x = layers.RNN(
            XLSTMMemoryCell(units=self.memory_units//2, memory_size=self.memory_size),
            return_sequences=False
        )(x)
        # print(f"DEBUG Actor Model: x shape={x.shape} (after second RNN)")
        
        x = layers.LayerNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        # 🔥 ИСПРАВЛЕНО: Правильные residual connections с проверкой размерностей
        dense1 = layers.Dense(
            128, 
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(self.weight_decay)
        )(x)
        dense1 = layers.Dropout(0.3)(dense1)
        
        dense2 = layers.Dense(
            64, 
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(self.weight_decay)
        )(dense1)
        
        # 🔥 ИСПРАВЛЕНО: Убрано лишнее логирование форм перед операцией сложения
        # print(f"DEBUG Actor Model: x (before resize) shape={x.shape}")
        
        x_static_shape = x.shape[-1]
        if x_static_shape != 64:
            x_resized = layers.Dense(64, kernel_regularizer=tf.keras.regularizers.l2(self.weight_decay))(x)
            # print(f"DEBUG Actor Model: x_resized shape={x_resized.shape}")
        else:
            x_resized = x
            # print(f"DEBUG Actor Model: x_resized (no change) shape={x_resized.shape}")
        
        # print(f"DEBUG Actor Model: dense2 shape={dense2.shape}")
        
        # Residual connection
        # print(f"DEBUG Actor Model: x_resized.shape={x_resized.shape}, dense2.shape={dense2.shape}") # Оставляем, если хотим проверять формы перед сложением
        if x_resized.shape[-1] != dense2.shape[-1]:
            print(f"⚠️ Ошибка: Формы для residual connection не совпадают: x_resized={x_resized.shape}, dense2={dense2.shape}")
        
        x = layers.Add()([x_resized, dense2])
        # print(f"DEBUG Actor Model: x (after add) shape={x.shape}")
        
        # Выходной слой с ограничением весов
        outputs = layers.Dense(
            3, 
            activation='softmax',
            kernel_regularizer=tf.keras.regularizers.l2(self.weight_decay),
            kernel_constraint=tf.keras.constraints.MaxNorm(max_value=2.0) # 🔥 РАСКОММЕНТИРОВАТЬ
        )(x)
        
        model = models.Model(inputs=inputs, outputs=outputs)
        
        # 🔥 ДОБАВЛЕНО: Проверка общего количества параметров (оставляем)
        total_params = model.count_params()
        max_params = 10_000_000  # 10M параметров максимум
        if total_params > max_params:
            print(f"⚠️ Модель слишком большая: {total_params:,} параметров (максимум: {max_params:,})")
        else:
            print(f"✅ Размер модели Actor: {total_params:,} параметров")
        
        return model

    def _build_critic_model(self):
        """Создает модель критика для оценки действий"""
        inputs = layers.Input(shape=self.input_shape)
        
        # Нормализация входных данных
        # print(f"DEBUG Critic Model: inputs shape={inputs.shape} (before LayerNormalization)")
        x = layers.LayerNormalization()(inputs)
        # print(f"DEBUG Critic Model: x shape={x.shape} (after LayerNormalization)")
        
        # Первый слой xLSTM
        # print(f"DEBUG Critic Model: x shape={x.shape} (before first RNN)")
        x = layers.RNN(XLSTMMemoryCell(units=self.memory_units,
                                       memory_size=self.memory_size),
                      return_sequences=True)(x)
        # print(f"DEBUG Critic Model: x shape={x.shape} (after first RNN)")
        
        x = layers.LayerNormalization()(x)
        x = layers.Dropout(0.2)(x)
        
        # Второй слой xLSTM
        # print(f"DEBUG Critic Model: x shape={x.shape} (before second RNN)")
        x = layers.RNN(XLSTMMemoryCell(units=self.memory_units,
                                       memory_size=self.memory_size),
                      return_sequences=False)(x)
        # print(f"DEBUG Critic Model: x shape={x.shape} (after second RNN)")
        
        x = layers.LayerNormalization()(x)
        x = layers.Dropout(0.2)(x)
        
        # Полносвязные слои С РЕГУЛЯРИЗАЦИЕЙ
        # print(f"DEBUG Critic Model: x shape={x.shape} (before first Dense)")
        x = layers.Dense(64, activation='relu',
                        kernel_regularizer=tf.keras.regularizers.l2(self.weight_decay))(x)
        # print(f"DEBUG Critic Model: x shape={x.shape} (after first Dense)")
        
        # print(f"DEBUG Critic Model: x shape={x.shape} (before second Dense)")
        x = layers.Dense(32, activation='relu',
                        kernel_regularizer=tf.keras.regularizers.l2(self.weight_decay))(x)
        # print(f"DEBUG Critic Model: x shape={x.shape} (after second Dense)")
        
        # Выходной слой С РЕГУЛЯРИЗАЦИЕЙ
        # print(f"DEBUG Critic Model: x shape={x.shape} (before output Dense)")
        outputs = layers.Dense(1, 
                              kernel_regularizer=tf.keras.regularizers.l2(self.weight_decay))(x)
        # print(f"DEBUG Critic Model: outputs shape={outputs.shape} (after output Dense)")
        
        model = models.Model(inputs=inputs, outputs=outputs)
        
        # 🔥 ДОБАВИТЬ: Проверка параметров (оставляем)
        total_params = model.count_params()
        max_params = 10_000_000
        if total_params > max_params:
            print(f"⚠️ Critic модель слишком большая: {total_params:,} параметров")
        else:
            print(f"✅ Размер модели Critic: {total_params:,} параметров")
        
        return model

    def compile_for_supervised_learning(self):
        """Компилирует модель для этапа 1: Supervised Learning с Focal Loss"""
        # Создаем Focal Loss для борьбы с дисбалансом классов
        focal_loss = FocalLoss(alpha=0.25, gamma=2.0, label_smoothing=0.1)
        
        self.actor_model.compile(
            optimizer=self.supervised_optimizer,
            loss=focal_loss,  # Заменили на Focal Loss
            metrics=['accuracy'],
            run_eagerly=False
        )
        print("✅ Модель скомпилирована для supervised learning с Focal Loss (α=0.25, γ=2.0)")

    def compile_for_reward_modeling(self):
        """Компилирует модель для этапа 2: Reward Model Training"""
        optimizer = tf.keras.optimizers.Adam(
            clipnorm=self.gradient_clip_norm  # 🔥 ДОБАВЛЕНО: Gradient clipping
        )
        optimizer.learning_rate = tf.Variable(0.001)
        self.critic_model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )
        print("✅ Модель скомпилирована для reward modeling")

    def get_training_callbacks(self, total_epochs=50, patience=10):
        """
        Возвращает улучшенные callbacks для обучения
        """
        callbacks = [
            # WarmUp + Cosine Decay Learning Rate
            WarmUpCosineDecayScheduler(
                warmup_epochs=5,
                total_epochs=total_epochs,
                base_lr=0.001,
                min_lr=1e-6
            ),
            
            # Early Stopping для предотвращения переобучения
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Model Checkpoint для сохранения лучшей модели
            tf.keras.callbacks.ModelCheckpoint(
                filepath='models/best_xlstm_model.keras',
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            
            # Reduce LR on Plateau (дополнительная защита)
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        return callbacks

    def save(self, path='models', stage=""):
        """Сохраняет модель с указанием этапа"""
        if not os.path.exists(path):
            os.makedirs(path)
        
        # Сохранение моделей с указанием этапа
        actor_name = f'xlstm_rl_actor{stage}.keras'
        critic_name = f'xlstm_rl_critic{stage}.keras'
        
        self.actor_model.save(os.path.join(path, actor_name))
        self.critic_model.save(os.path.join(path, critic_name))
        
        print(f"Модели сохранены в {path} (этап: {stage})")

    def load(self, path='models', stage=""):
        """Загружает модель с указанием этапа"""
        actor_name = f'xlstm_rl_actor{stage}.keras'
        critic_name = f'xlstm_rl_critic{stage}.keras'
        
        actor_path = os.path.join(path, actor_name)
        critic_path = os.path.join(path, critic_name)
        
        # 🔥 ИСПРАВЛЕНО: Безопасная загрузка с проверкой
        try:
            if os.path.exists(actor_path) and os.path.exists(critic_path):
                self.actor_model = tf.keras.models.load_model(
                    actor_path, 
                    custom_objects={'XLSTMMemoryCell': XLSTMMemoryCell}
                )
                self.critic_model = tf.keras.models.load_model(
                    critic_path,
                    custom_objects={'XLSTMMemoryCell': XLSTMMemoryCell}
                )
                print(f"Модели успешно загружены (этап: {stage})")
            else:
                print(f"Не удалось найти сохраненные модели для этапа: {stage}")
        except Exception as e:
            print(f"❌ Ошибка при загрузке моделей: {e}")
            return False

    def predict_action(self, state):
        """Предсказывает действие на основе состояния с батчевой обработкой"""
        if len(state.shape) == 2:
            state = np.expand_dims(state, axis=0)
        
        if state.dtype != np.float32:
            state = state.astype(np.float32)
        
        # 🔥 ИСПРАВЛЕНО: Используем predict_on_batch для лучшей производительности
        try:
            action_probs = self.actor_model.predict_on_batch(state)[0]
        except Exception as e:
            print(f"Ошибка в predict_on_batch, используем fallback: {e}")
            action_probs = self.actor_model.predict(state, verbose=0)[0]
        
        # 🔥 ДОБАВЛЕНО: Агрессивная очистка памяти
        self.prediction_count += 1
        if self.prediction_count % 50 == 0:  # Увеличена частота очистки
            gc.collect()
            if self.prediction_count % 500 == 0:  # Глубокая очистка каждые 500 вызовов
                tf.keras.backend.clear_session()
                print(f"Глубокая очистка памяти после {self.prediction_count} предсказаний")
        
        return action_probs
    
    def predict_value(self, state):
        """Предсказывает значение состояния с батчевой обработкой"""
        if len(state.shape) == 2:
            state = np.expand_dims(state, axis=0)
        
        if state.dtype != np.float32:
            state = state.astype(np.float32)
        
        # 🔥 ИСПРАВЛЕНО: Используем predict_on_batch
        try:
            value = self.critic_model.predict_on_batch(state)[0]
        except Exception as e:
            print(f"Ошибка в predict_on_batch для critic: {e}")
            value = self.critic_model.predict(state, verbose=0)[0]
        
        # Аналогичная очистка памяти
        self.prediction_count += 1
        if self.prediction_count % 50 == 0:
            gc.collect()
            if self.prediction_count % 500 == 0:
                tf.keras.backend.clear_session()
        
        return value
    
    def predict_batch_actions(self, states):
        """🔥 ДОБАВЛЕНО: Батчевое предсказание для множественных состояний"""
        if len(states) == 0:
            return np.array([])
        
        states = np.array(states, dtype=np.float32)
        if len(states.shape) == 2:
            states = np.expand_dims(states, axis=1)
        
        action_probs = self.actor_model.predict_on_batch(states)
        
        # Очистка после батчевого предсказания
        gc.collect()
        
        return action_probs