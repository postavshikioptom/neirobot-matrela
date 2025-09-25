import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os
import tensorflow.keras.backend as K
import gc
import config

# Безопасный импорт
try:
    from models.xlstm_memory_cell import XLSTMMemoryCell
except ImportError as e:
    print(f"❌ Ошибка импорта XLSTMMemoryCell: {e}")
    print("Убедитесь, что файл models/xlstm_memory_cell.py существует")
    raise ImportError("XLSTMMemoryCell не найден")


class FocalLoss(tf.keras.losses.Loss):
    """
    Ассиметричная Focal Loss с поддержкой per-class alpha/gamma,
    Class-Conditional Label Smoothing и опциональным штрафом за
    сверх-уверенные BUY (entropy penalty).
    """
    def __init__(self,
                 alpha=0.25,
                 gamma=2.0,
                 label_smoothing=0.1,
                 per_class_alpha=None,   # [α_SELL, α_HOLD, α_BUY]
                 per_class_gamma=None,   # [γ_SELL, γ_HOLD, γ_BUY]
                 per_class_smoothing=None,  # [s_SELL, s_HOLD, s_BUY]
                 entropy_penalty_lambda=0.0,  # λ for BUY confidence penalty
                 **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.per_class_alpha = per_class_alpha
        self.per_class_gamma = per_class_gamma
        self.per_class_smoothing = per_class_smoothing
        self.entropy_penalty_lambda = entropy_penalty_lambda
    
    def call(self, y_true, y_pred):
        # Ensure y_true is rank-1 (sparse labels)
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        num_classes = tf.shape(y_pred)[-1]
        num_classes_f = tf.cast(num_classes, tf.float32)
        
        # One-hot and per-class label smoothing
        y_true_onehot = tf.one_hot(y_true, depth=num_classes, dtype=tf.float32)
        if self.per_class_smoothing is not None:
            smoothing_vec = tf.gather(tf.constant(self.per_class_smoothing, dtype=tf.float32), y_true)
        else:
            smoothing_vec = tf.fill(tf.shape(y_true), tf.cast(self.label_smoothing, tf.float32))
        smoothing_vec = tf.expand_dims(smoothing_vec, axis=-1)
        y_true_smooth = y_true_onehot * (1.0 - smoothing_vec) + smoothing_vec / num_classes_f
        
        # Cross-entropy with smoothed labels
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        ce_loss = -tf.reduce_sum(y_true_smooth * tf.math.log(y_pred), axis=-1)
        
        # p_t: probability assigned to the true class
        p_t = tf.reduce_sum(y_true_onehot * y_pred, axis=-1)
        
        # Per-class alpha and gamma
        if self.per_class_alpha is not None:
            alpha_t = tf.gather(tf.constant(self.per_class_alpha, dtype=tf.float32), y_true)
        else:
            alpha_t = tf.fill(tf.shape(y_true), tf.cast(self.alpha, tf.float32))
        
        if self.per_class_gamma is not None:
            gamma_t = tf.gather(tf.constant(self.per_class_gamma, dtype=tf.float32), y_true)
        else:
            gamma_t = tf.fill(tf.shape(y_true), tf.cast(self.gamma, tf.float32))
        
        focal_factor = tf.pow(1.0 - p_t, gamma_t)
        focal_loss = alpha_t * focal_factor * ce_loss
        
        # 🔥 УЛУЧШЕНО: Entropy penalty для всех классов с высокой уверенностью
        if self.entropy_penalty_lambda > 0.0:
            entropy = -tf.reduce_sum(y_pred * tf.math.log(y_pred), axis=-1)
            # Применяем штраф к любому классу с уверенностью > 0.8
            max_pred = tf.reduce_max(y_pred, axis=-1)
            high_confidence = tf.cast(tf.greater(max_pred, 0.8), tf.float32)
            penalty = self.entropy_penalty_lambda * (1.0 - entropy) * high_confidence
            focal_loss = focal_loss + penalty
        
        return focal_loss
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'alpha': self.alpha,
            'gamma': self.gamma,
            'label_smoothing': self.label_smoothing,
            'per_class_alpha': self.per_class_alpha,
            'per_class_gamma': self.per_class_gamma,
            'per_class_smoothing': self.per_class_smoothing,
            'entropy_penalty_lambda': self.entropy_penalty_lambda,
        })
        return config


class WarmUpCosineDecayScheduler(tf.keras.callbacks.Callback):
    """
    Learning Rate Scheduler с WarmUp и Cosine Decay
    Улучшает стабильность обучения и конвергенцию. Пик берём из config.LR_BASE.
    """
    def __init__(self, warmup_epochs=2, total_epochs=50, base_lr=None, min_lr=1e-6):
        super().__init__()
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        # Base LR может приходить извне, иначе из config
        self.base_lr = base_lr if base_lr is not None else getattr(config, 'LR_BASE', 6e-4)
        self.min_lr = min_lr
        
    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.warmup_epochs:
            # WarmUp phase: линейное увеличение LR
            lr = self.base_lr * (epoch + 1) / max(1, self.warmup_epochs)
        else:
            # Cosine Decay phase
            progress = (epoch - self.warmup_epochs) / max(1, (self.total_epochs - self.warmup_epochs))
            progress = np.clip(progress, 0.0, 1.0)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
        
        try:
            self.model.optimizer.learning_rate.assign(lr)
            print(f"Epoch {epoch + 1}: Установлена скорость обучения: {lr:.6f}")
        except Exception as e:
            print(f"[LR SCHED] skip assign: {e}")





class EMAWeightsCallback(tf.keras.callbacks.Callback):
    """Экспоненциальное сглаживание весов. Поддерживает валидацию на EMA-весах.
    - Обновляет EMA после каждого батча
    - На валидации может временно подменять веса на EMA и возвращать обратно
    """
    def __init__(self, decay=None, use_for_validation=None):
        super().__init__()
        self.decay = float(decay if decay is not None else getattr(config, 'EMA_DECAY', 0.999))
        self.use_for_validation = bool(use_for_validation if use_for_validation is not None else getattr(config, 'USE_EMA_VALIDATION', True))
        self.ema_weights = None
        self.train_weights = None
    
    def on_train_begin(self, logs=None):
        # model.get_weights() уже возвращает numpy arrays, не нужно вызывать .numpy()
        weights = self.model.get_weights()
        self.ema_weights = [w.copy() if hasattr(w, 'copy') else np.array(w).copy() for w in weights]
    
    def on_train_batch_end(self, batch, logs=None):
        current = self.model.get_weights()
        for i in range(len(self.ema_weights)):
            self.ema_weights[i] = self.decay * self.ema_weights[i] + (1.0 - self.decay) * current[i]
    
    def on_test_begin(self, logs=None):
        if not self.use_for_validation:
            return
        try:
            self.train_weights = self.model.get_weights()
            self.model.set_weights(self.ema_weights)
        except Exception as e:
            print(f"[EMA] skip swap to EMA on validation begin: {e}")
    
    def on_test_end(self, logs=None):
        if not self.use_for_validation:
            return
        try:
            if self.train_weights is not None:
                self.model.set_weights(self.train_weights)
                self.train_weights = None
        except Exception as e:
            print(f"[EMA] skip restore train weights after validation: {e}")


class XLSTMRLModel:
    """
    Модель xLSTM с RL для торговли - ТРЁХЭТАПНАЯ АРХИТЕКТУРА
    """
    def __init__(self, input_shape, memory_size=64, memory_units=128, weight_decay=1e-4, gradient_clip_norm=1.0):
        self.input_shape = input_shape
        self.memory_size = memory_size
        self.memory_units = memory_units
        # 🔥 ДОБАВЛЕНО: Параметры регуляризации (подключены к config)
        self.weight_decay = float(getattr(config, 'WEIGHT_DECAY_L2', weight_decay))
        self.dropout_rnn1 = float(getattr(config, 'DROPOUT_RNN1', 0.6))
        self.dropout_rnn2 = float(getattr(config, 'DROPOUT_RNN2', 0.5))
        self.gradient_clip_norm = gradient_clip_norm
        
        # 🔥 ИСПРАВЛЕНО: Добавьте эти строки для инициализации моделей
        self.actor_model = self._build_actor_model()
        self.critic_model = self._build_critic_model()
        
        # Инициализация bias последнего слоя под целевые доли (SELL,HOLD,BUY)
        try:
            priors = np.array(getattr(config, 'TARGET_CLASS_RATIOS', [0.33, 0.40, 0.33]), dtype=np.float32)
            logits_bias = np.log(np.clip(priors, 1e-8, 1.0))
            last_dense = self.actor_model.get_layer('classifier')
            # Если softmax, сдвиг bias допустим
            last_dense.bias.assign(logits_bias)
            print(f"[INIT] classifier bias set to log-priors: {priors}")
        except Exception as e:
            print(f"[INIT] skip classifier bias init: {e}")
        
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
                # Use configurable base LR for supervised phase (stabilize early training)
                self.supervised_optimizer.learning_rate = tf.Variable(getattr(config, 'LR_BASE', 6e-4))
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
                # Use configurable base LR for CPU as well
                self.supervised_optimizer.learning_rate = tf.Variable(getattr(config, 'LR_BASE', 6e-4))
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
        
        # 🔥 ОБНОВЛЕНО: Динамическая проверка размерности без жёсткого числа фичей
        expected_features = self.input_shape[-1]
        # print(f"[INFO] Входных признаков: {expected_features}")
        
        # Первый слой xLSTM с weight decay
        # print(f"DEBUG Actor Model: x shape={x.shape} (before first RNN)")
        x = layers.RNN(
            XLSTMMemoryCell(units=self.memory_units, memory_size=self.memory_size),
            return_sequences=True
        )(x)
        # print(f"DEBUG Actor Model: x shape={x.shape} (after first RNN)")
        
        x = layers.LayerNormalization()(x)
        # Dropout after first RNN, configurable via config
        x = layers.Dropout(self.dropout_rnn1)(x)
        
        # Второй слой xLSTM (уменьшенный размер)
        # print(f"DEBUG Actor Model: x shape={x.shape} (before second RNN)")
        x = layers.RNN(
            XLSTMMemoryCell(units=self.memory_units//2, memory_size=self.memory_size),
            return_sequences=False
        )(x)
        # print(f"DEBUG Actor Model: x shape={x.shape} (after second RNN)")
        
        x = layers.LayerNormalization()(x)
        # Dropout after second RNN, configurable via config
        x = layers.Dropout(self.dropout_rnn2)(x)
        
        # 🔥 ИСПРАВЛЕНО: Правильные residual connections с проверкой размерностей
        dense1 = layers.Dense(
            128, 
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(self.weight_decay)
        )(x)
        dense1 = layers.Dropout(0.5)(dense1)
        
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
            kernel_constraint=tf.keras.constraints.MaxNorm(max_value=2.0),
            name='classifier'
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
        x = layers.Dropout(0.4)(x)
        
        # Второй слой xLSTM
        # print(f"DEBUG Critic Model: x shape={x.shape} (before second RNN)")
        x = layers.RNN(XLSTMMemoryCell(units=self.memory_units,
                                       memory_size=self.memory_size),
                      return_sequences=False)(x)
        # print(f"DEBUG Critic Model: x shape={x.shape} (after second RNN)")

        x = layers.LayerNormalization()(x)
        x = layers.Dropout(0.4)(x)
        
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
        """Компилирует модель для этапа 1: Supervised Learning с плавным переходом CE→AFL"""
        # Этап 1–N_warm: смешанная потеря CE и AFL для более стабильного старта
        warm_epochs = int(getattr(config, 'AFL_WARMUP_EPOCHS', 5))
        ce_weight_start = float(getattr(config, 'CE_WEIGHT_START', 0.8))  # начальный вес CE
        ce_weight_end = float(getattr(config, 'CE_WEIGHT_END', 0.0))      # к концу warmup CE уходит
        
        # Базовые компоненты потерь
        focal_loss = FocalLoss(
            per_class_alpha=getattr(config, 'AFL_ALPHA', None),
            per_class_gamma=getattr(config, 'AFL_GAMMA', None),
            per_class_smoothing=getattr(config, 'CLASS_SMOOTHING', None),
            entropy_penalty_lambda=getattr(config, 'ENTROPY_PENALTY_LAMBDA', 0.0),
            alpha=0.25,
            gamma=2.0,
            label_smoothing=0.1,
        )
        ce_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        
        class MixedLoss(tf.keras.losses.Loss):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                # Создаем tf.Variable на том же устройстве, что и модель
                with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
                    self.epoch_var = tf.Variable(0.0, trainable=False, dtype=tf.float32, name='mixed_loss_epoch')
                
            def set_epoch(self, epoch):
                """Устанавливает текущую эпоху"""
                self.epoch_var.assign(float(epoch))
                
            def call(self, y_true, y_pred):
                # Линейная интерполяция веса CE в зависимости от текущей эпохи
                if warm_epochs > 0:
                    t = tf.clip_by_value(self.epoch_var / float(warm_epochs), 0.0, 1.0)
                else:
                    t = 1.0
                ce_w = ce_weight_start * (1.0 - t) + ce_weight_end * t
                afl_w = 1.0 - ce_w
                return ce_w * ce_loss(y_true, y_pred) + afl_w * focal_loss(y_true, y_pred)
        
        mixed_loss = MixedLoss(name='mixed_ce_afl')
        
        class EpochTracker(tf.keras.callbacks.Callback):
            def on_epoch_begin(self, epoch, logs=None):
                # Передаем текущую эпоху в лосс, чтобы он плавно менял веса
                try:
                    if hasattr(self.model, 'loss') and hasattr(self.model.loss, 'set_epoch'):
                        self.model.loss.set_epoch(int(epoch))
                except Exception as e:
                    print(f"[MixedLoss] skip epoch assign: {e}")
        
        self._mixed_loss_callback = EpochTracker()
        
        self.actor_model.compile(
            optimizer=self.supervised_optimizer,
            loss=mixed_loss,
            metrics=['accuracy'],
            run_eagerly=False,
            jit_compile=True  # Включаем XLA для ускорения
        )
        print("✅ Модель скомпилирована: смешанная потеря CE→AFL с линейным спадом CE в первые эпохи")

    def compile_for_reward_modeling(self):
        """Компилирует модель для этапа 2: Reward Model Training"""
        optimizer = tf.keras.optimizers.Adam(
            clipnorm=self.gradient_clip_norm  # 🔥 ДОБАВЛЕНО: Gradient clipping
        )
        optimizer.learning_rate = tf.Variable(0.001)
        self.critic_model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae'],
            jit_compile=True  # Включаем XLA для ускорения
        )
        print("✅ Модель скомпилирована для reward modeling")

    def get_training_callbacks(self, total_epochs=50, patience=10):
        """
        Возвращает улучшенные callbacks для обучения
        """
        callbacks = [
            # WarmUp + Cosine Decay Learning Rate (lower peak, shorter warmup)
            WarmUpCosineDecayScheduler(
                warmup_epochs=getattr(config, 'LR_WARMUP_EPOCHS', 2),
                total_epochs=total_epochs,
                base_lr=getattr(config, 'LR_BASE', 6e-4),
                min_lr=getattr(config, 'LR_MIN', 1e-6)
            ),
            
            # EMA весов: стабильная валидация за счет EMA-параметров
            EMAWeightsCallback(
                decay=getattr(config, 'EMA_DECAY', 0.999),
                use_for_validation=getattr(config, 'USE_EMA_VALIDATION', True)
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
        
        # Динамическое переназначение весов классов по эпохам (мягкий режим)
        if getattr(config, 'DYNAMIC_CLASS_WEIGHTS', False):
            class DynamicWeightsCallback(tf.keras.callbacks.Callback):
                def __init__(self, step=0.05, target_ratios=None):
                    super().__init__()
                    self.step = step
                    self.target = np.array(target_ratios or [0.3, 0.3, 0.4])
                def on_epoch_end(self, epoch, logs=None):
                    try:
                        # Получаем распределение предсказаний на валидации
                        vd = getattr(self.model, 'validation_data', None)
                        if vd is None:
                            return
                        X_val, y_val = vd[:2]
                        preds = self.model.predict(X_val, verbose=0)
                        pred_labels = np.argmax(preds, axis=1)
                        hist = np.bincount(pred_labels, minlength=3)
                        total = max(1, len(pred_labels))
                        dist = hist / total
                        print(f"[DynWeights] pred BUY/HOLD/SELL: {dist[2]:.2%}/{dist[1]:.2%}/{dist[0]:.2%}")
                        # 🔥 УЛУЧШЕНО: Агрессивная корректировка при сильном дисбалансе
                        loss_obj = getattr(self.model, 'loss', None)
                        if hasattr(loss_obj, 'per_class_alpha') and loss_obj.per_class_alpha is not None:
                            alpha = np.array(loss_obj.per_class_alpha, dtype=np.float32)
                            delta = dist - self.target
                            
                            # Увеличиваем шаг корректировки при сильном дисбалансе
                            max_deviation = np.max(np.abs(delta))
                            adaptive_step = self.step * (1.0 + 2.0 * max_deviation)  # До 3x больше при сильном дисбалансе
                            
                            # Корректируем веса с адаптивным шагом
                            alpha[2] = float(np.clip(alpha[2] - adaptive_step * np.sign(delta[2]), 0.1, 3.0))
                            alpha[0] = float(np.clip(alpha[0] - adaptive_step * np.sign(delta[0]), 0.1, 3.0))
                            alpha[1] = float(np.clip(alpha[1] - adaptive_step * np.sign(delta[1]), 0.1, 3.0))
                            loss_obj.per_class_alpha = alpha.tolist()
                            print(f"[DynWeights] deviation={max_deviation:.3f}, step={adaptive_step:.3f}")
                            print(f"[DynWeights] per_class_alpha -> {loss_obj.per_class_alpha}")
                    except Exception as e:
                        print(f"[DynWeights] skip adjust: {e}")
            callbacks.append(DynamicWeightsCallback(
                step=getattr(config, 'DYNAMIC_WEIGHT_STEP', 0.05),
                target_ratios=getattr(config, 'TARGET_CLASS_RATIOS', [0.3, 0.3, 0.4])
            ))
        
        # Добавляем трекер эпох для смешанного лосса CE→AFL, если он инициализирован при compile
        try:
            if hasattr(self, '_mixed_loss_callback') and self._mixed_loss_callback is not None:
                callbacks.append(self._mixed_loss_callback)
        except Exception as e:
            print(f"[Callbacks] skip adding MixedLoss epoch tracker: {e}")
        
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