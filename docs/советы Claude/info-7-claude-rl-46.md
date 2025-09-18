Подробная инструкция по созданию торгового бота xLSTM с RL
Создадим файлы с нуля для торгового бота на основе xLSTM с reinforcement learning. Бот будет обучаться предсказывать действия BUY, SELL, HOLD на основе данных биржи без использования индикаторов.
1. models/xlstm_memory_cell.py
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

class XLSTMMemoryCell(layers.Layer):
    """
    Расширенная ячейка памяти xLSTM с улучшенной структурой памяти
    """
    def __init__(self, units, memory_size=64, **kwargs):
        super(XLSTMMemoryCell, self).__init__(**kwargs)
        self.units = units
        self.memory_size = memory_size
        self.state_size = [tf.TensorShape([units]), tf.TensorShape([memory_size, units])]
        self.output_size = units

    def build(self, input_shape):
        input_dim = input_shape[-1]
        
        # Веса для входного гейта
        self.Wi = self.add_weight(shape=(input_dim, self.units),
                                  name='Wi', initializer='glorot_uniform')
        self.Ui = self.add_weight(shape=(self.units, self.units),
                                 name='Ui', initializer='glorot_uniform')
        self.bi = self.add_weight(shape=(self.units,),
                                 name='bi', initializer='zeros')
        
        # Веса для гейта забывания
        self.Wf = self.add_weight(shape=(input_dim, self.units),
                                  name='Wf', initializer='glorot_uniform')
        self.Uf = self.add_weight(shape=(self.units, self.units),
                                 name='Uf', initializer='glorot_uniform')
        self.bf = self.add_weight(shape=(self.units,),
                                 name='bf', initializer='ones')
        
        # Веса для выходного гейта
        self.Wo = self.add_weight(shape=(input_dim, self.units),
                                  name='Wo', initializer='glorot_uniform')
        self.Uo = self.add_weight(shape=(self.units, self.units),
                                 name='Uo', initializer='glorot_uniform')
        self.bo = self.add_weight(shape=(self.units,),
                                 name='bo', initializer='zeros')
        
        # Веса для кандидата на обновление ячейки
        self.Wc = self.add_weight(shape=(input_dim, self.units),
                                  name='Wc', initializer='glorot_uniform')
        self.Uc = self.add_weight(shape=(self.units, self.units),
                                 name='Uc', initializer='glorot_uniform')
        self.bc = self.add_weight(shape=(self.units,),
                                 name='bc', initializer='zeros')
        
        # Веса для расширенной памяти
        self.Wm = self.add_weight(shape=(self.memory_size, self.units),
                                 name='Wm', initializer='glorot_uniform')
        
        # Экспоненциальные гейты для улучшенной памяти
        self.We = self.add_weight(shape=(input_dim, self.memory_size),
                                 name='We', initializer='glorot_uniform')
        self.Ue = self.add_weight(shape=(self.units, self.memory_size),
                                 name='Ue', initializer='glorot_uniform')
        self.be = self.add_weight(shape=(self.memory_size,),
                                 name='be', initializer='zeros')
        
        self.built = True

    def call(self, inputs, states):
        # Предыдущие состояния
        h_prev, memory_prev = states
        
        # Вычисляем гейты с экспоненциальной активацией для стабильности
        i = tf.nn.sigmoid(tf.matmul(inputs, self.Wi) + tf.matmul(h_prev, self.Ui) + self.bi)
        f = tf.nn.sigmoid(tf.matmul(inputs, self.Wf) + tf.matmul(h_prev, self.Uf) + self.bf)
        o = tf.nn.sigmoid(tf.matmul(inputs, self.Wo) + tf.matmul(h_prev, self.Uo) + self.bo)
        
        # Кандидат на новое значение ячейки
        c_tilde = tf.nn.tanh(tf.matmul(inputs, self.Wc) + tf.matmul(h_prev, self.Uc) + self.bc)
        
        # Экспоненциальный гейт для расширенной памяти
        e = tf.nn.softmax(tf.matmul(inputs, self.We) + tf.matmul(h_prev, self.Ue) + self.be)
        e = tf.reshape(e, [-1, self.memory_size, 1])
        
        # Обновление расширенной памяти с использованием механизма внимания
        memory_contribution = tf.reduce_sum(memory_prev * e, axis=1)
        
        # Обновление состояния ячейки с учетом расширенной памяти
        c = f * memory_contribution + i * c_tilde
        
        # Обновление выхода
        h = o * tf.nn.tanh(c)
        
        # Обновление памяти - сдвигаем старые значения и добавляем новое
        new_memory_item = tf.expand_dims(c, axis=1)
        memory_new = tf.concat([new_memory_item, memory_prev[:, :-1, :]], axis=1)
        
        return h, [h, memory_new]

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        if inputs is not None:
            batch_size = tf.shape(inputs)[0]
            dtype = inputs.dtype
        
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

2. models/xlstm_rl_model.py
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from models.xlstm_memory_cell import XLSTMMemoryCell
import os

class XLSTMRLModel:
    """
    Модель xLSTM с RL для торговли
    """
    def __init__(self, input_shape, memory_size=64, memory_units=128):
        self.input_shape = input_shape
        self.memory_size = memory_size
        self.memory_units = memory_units
        
        # Создаем модель актора (принимает решения о действиях)
        self.actor_model = self._build_actor_model()
        
        # Создаем модель критика (оценивает действия актора)
        self.critic_model = self._build_critic_model()
        
        # Оптимизаторы
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)

    def _build_actor_model(self):
        """Создает модель актора для принятия решений"""
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
        
        # Полносвязные слои
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dense(32, activation='relu')(x)
        
        # Выходной слой для действий (BUY, HOLD, SELL)
        outputs = layers.Dense(3, activation='softmax')(x)
        
        model = models.Model(inputs=inputs, outputs=outputs)
        return model

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
        
        # Полносвязные слои
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dense(32, activation='relu')(x)
        
        # Выходной слой для значения состояния (скаляр)
        outputs = layers.Dense(1)(x)
        
        model = models.Model(inputs=inputs, outputs=outputs)
        return model

    def save(self, path='models'):
        """Сохраняет модель и скейлер"""
        if not os.path.exists(path):
            os.makedirs(path)
        
        # Сохранение моделей
        self.actor_model.save(os.path.join(path, 'xlstm_rl_actor.keras'))
        self.critic_model.save(os.path.join(path, 'xlstm_rl_critic.keras'))
        
        print(f"Модели сохранены в {path}")

    def load(self, path='models'):
        """Загружает модель и скейлер"""
        actor_path = os.path.join(path, 'xlstm_rl_actor.keras')
        critic_path = os.path.join(path, 'xlstm_rl_critic.keras')
        
        if os.path.exists(actor_path) and os.path.exists(critic_path):
            self.actor_model = tf.keras.models.load_model(
                actor_path, 
                custom_objects={'XLSTMMemoryCell': XLSTMMemoryCell}
            )
            self.critic_model = tf.keras.models.load_model(
                critic_path,
                custom_objects={'XLSTMMemoryCell': XLSTMMemoryCell}
            )
            print("Модели успешно загружены")
        else:
            print("Не удалось найти сохраненные модели")

    def predict_action(self, state):
        """Предсказывает действие на основе состояния"""
        state = np.expand_dims(state, axis=0)
        action_probs = self.actor_model.predict(state, verbose=0)[0]
        return action_probs

    def predict_value(self, state):
        """Предсказывает значение состояния"""
        state = np.expand_dims(state, axis=0)
        value = self.critic_model.predict(state, verbose=0)[0]
        return value

3. feature_engineering.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle
import os

class FeatureEngineering:
    """
    Класс для обработки и подготовки признаков для модели
    """
    def __init__(self, sequence_length=60):
        self.sequence_length = sequence_length
        self.scaler = StandardScaler()
        self.feature_columns = ['open', 'high', 'low', 'close', 'volume', 'turnover']
        
    def prepare_data(self, df):
        """
        Подготовка данных: нормализация и создание последовательностей
        """
        # Убедимся, что данные отсортированы по времени
        df = df.sort_values('timestamp')
        
        # Преобразуем признаки в числовой формат
        for col in self.feature_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Получаем только нужные колонки
        data = df[self.feature_columns].values
        
        # Обучаем скейлер на всех данных
        scaled_data = self.scaler.fit_transform(data)
        
        # Создаем последовательности
        X, y_close = self._create_sequences(scaled_data)
        
        return X, y_close, df
    
    def prepare_test_data(self, df):
        """
        Подготовка тестовых данных с использованием уже обученного скейлера
        """
        # Убедимся, что данные отсортированы по времени
        df = df.sort_values('timestamp')
        
        # Преобразуем признаки в числовой формат
        for col in self.feature_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Получаем только нужные колонки
        data = df[self.feature_columns].values
        
        # Применяем уже обученный скейлер
        scaled_data = self.scaler.transform(data)
        
        # Создаем последовательности
        X, y_close = self._create_sequences(scaled_data)
        
        return X, y_close, df
    
    def _create_sequences(self, data):
        """
        Создает последовательности для обучения
        """
        X = []
        y_close = []
        
        for i in range(len(data) - self.sequence_length):
            # Последовательность признаков
            X.append(data[i:i+self.sequence_length])
            # Целевая цена закрытия (для мониторинга)
            y_close.append(data[i+self.sequence_length, 3])  # индекс 3 - это 'close'
        
        return np.array(X), np.array(y_close)
    
    def save_scaler(self, path='models'):
        """
        Сохраняет обученный скейлер
        """
        if not os.path.exists(path):
            os.makedirs(path)
        
        with open(os.path.join(path, 'scaler.pkl'), 'wb') as f:
            pickle.dump(self.scaler, f)
        
        print(f"Скейлер сохранен в {path}/scaler.pkl")
    
    def load_scaler(self, path='models'):
        """
        Загружает обученный скейлер
        """
        scaler_path = os.path.join(path, 'scaler.pkl')
        
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            print("Скейлер успешно загружен")
            return True
        else:
            print("Не удалось найти сохраненный скейлер")
            return False

4. trading_env.py
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

class TradingEnvironment(gym.Env):
    """
    Среда для обучения торгового агента с помощью RL
    """
    def __init__(self, data, sequence_length=60, initial_balance=10000, transaction_fee=0.001):
        super(TradingEnvironment, self).__init__()
        
        self.data = data  # Нормализованные данные
        self.sequence_length = sequence_length
        self.initial_balance = initial_balance
        self.transaction_fee = transaction_fee
        
        # Определяем пространство действий: 0 - BUY, 1 - HOLD, 2 - SELL
        self.action_space = spaces.Discrete(3)
        
        # Пространство наблюдений: последовательность цен и объемов + текущая позиция
        # [sequence_length, features] + [position]
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(self.sequence_length, data.shape[2] + 1)
        )
        
        # Сбрасываем среду
        self.reset()
    
    def reset(self, seed=None):
        """Сбрасывает среду в начальное состояние"""
        super().reset(seed=seed)
        
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0  # 0 - нет позиции, 1 - длинная позиция, -1 - короткая позиция
        self.shares_held = 0
        self.cost_basis = 0
        self.total_trades = 0
        self.total_profit = 0
        
        # Добавляем информацию о позиции к наблюдению
        observation = self._get_observation()
        
        return observation, {}
    
    def step(self, action):
        """Выполняет шаг в среде"""
        # Получаем текущую цену закрытия
        current_price = self._get_current_price()
        
        # Выполняем действие
        reward = 0
        if action == 0:  # BUY
            if self.position != 1:  # Если у нас нет длинной позиции
                # Закрываем короткую позицию, если есть
                if self.position == -1:
                    reward += self._close_position(current_price)
                
                # Открываем длинную позицию
                shares_to_buy = self.balance / current_price
                cost = shares_to_buy * current_price * (1 + self.transaction_fee)
                
                if cost <= self.balance:
                    self.balance -= cost
                    self.shares_held = shares_to_buy
                    self.cost_basis = current_price
                    self.position = 1
                    self.total_trades += 1
        
        elif action == 2:  # SELL
            if self.position != -1:  # Если у нас нет короткой позиции
                # Закрываем длинную позицию, если есть
                if self.position == 1:
                    reward += self._close_position(current_price)
                
                # Открываем короткую позицию
                shares_to_sell = self.balance / current_price
                self.shares_held = -shares_to_sell
                self.cost_basis = current_price
                self.position = -1
                self.total_trades += 1
        
        # Переходим к следующему шагу
        self.current_step += 1
        
        # Проверяем, закончился ли эпизод
        done = self.current_step >= len(self.data) - 1
        
        # Если эпизод закончился, закрываем все открытые позиции
        if done and self.position != 0:
            reward += self._close_position(current_price)
        
        # Получаем новое наблюдение
        observation = self._get_observation()
        
        # Рассчитываем стоимость портфеля
        portfolio_value = self.balance
        if self.position == 1:
            portfolio_value += self.shares_held * current_price
        elif self.position == -1:
            portfolio_value -= self.shares_held * current_price
        
        # Дополнительная информация
        info = {
            'balance': self.balance,
            'position': self.position,
            'shares_held': self.shares_held,
            'total_trades': self.total_trades,
            'total_profit': self.total_profit,
            'portfolio_value': portfolio_value
        }
        
        return observation, reward, done, False, info
    
    def _get_observation(self):
        """Возвращает текущее наблюдение"""
        # Получаем последовательность данных
        obs = self.data[self.current_step].copy()
        
        # Добавляем информацию о текущей позиции
        position_channel = np.ones((self.sequence_length, 1)) * self.position
        
        # Объединяем данные и информацию о позиции
        observation = np.concatenate([obs, position_channel], axis=1)
        
        return observation
    
    def _get_current_price(self):
        """Возвращает текущую цену закрытия"""
        return self.data[self.current_step][-1, 3]  # индекс 3 - это 'close'
    
    def _close_position(self, current_price):
        """Закрывает текущую позицию и возвращает полученную награду"""
        reward = 0
        
        if self.position == 1:  # Закрываем длинную позицию
            profit = self.shares_held * (current_price - self.cost_basis)
            fee = self.shares_held * current_price * self.transaction_fee
            self.balance += self.shares_held * current_price - fee
            reward = profit / self.initial_balance  # Нормализуем награду
            self.total_profit += profit
        
        elif self.position == -1:  # Закрываем короткую позицию
            profit = -self.shares_held * (current_price - self.cost_basis)
            fee = -self.shares_held * current_price * self.transaction_fee
            self.balance -= self.shares_held * current_price + fee
            reward = profit / self.initial_balance  # Нормализуем награду
            self.total_profit += profit
        
        self.shares_held = 0
        self.cost_basis = 0
        self.position = 0
        
        return reward

5. rl_agent.py
import numpy as np
import tensorflow as tf
from models.xlstm_rl_model import XLSTMRLModel
import os
import logging

class RLAgent:
    """
    Агент Reinforcement Learning для торговли
    """
    def __init__(self, state_shape, memory_size=64, memory_units=128, gamma=0.99, epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.995, batch_size=64):
        self.state_shape = state_shape
        self.gamma = gamma  # Коэффициент дисконтирования
        self.epsilon = epsilon  # Вероятность случайного действия
        self.epsilon_min = epsilon_min  # Минимальная вероятность случайного действия
        self.epsilon_decay = epsilon_decay  # Скорость уменьшения epsilon
        self.batch_size = batch_size
        
        # Инициализация модели
        self.model = XLSTMRLModel(input_shape=state_shape, 
                                 memory_size=memory_size, 
                                 memory_units=memory_units)
        
        # Буфер опыта
        self.memory = []
        self.max_memory_size = 10000
        
        # Логирование
        self.logger = logging.getLogger('rl_agent')
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def act(self, state, training=True):
        """
        Выбирает действие на основе текущего состояния
        """
        if training and np.random.rand() < self.epsilon:
            # Случайное действие во время обучения
            return np.random.randint(0, 3)
        
        # Получаем вероятности действий от модели актора
        action_probs = self.model.predict_action(state)
        
        # Выбираем действие с наибольшей вероятностью
        return np.argmax(action_probs)
    
    def remember(self, state, action, reward, next_state, done):
        """
        Сохраняет опыт в буфер
        """
        self.memory.append((state, action, reward, next_state, done))
        
        # Ограничиваем размер буфера
        if len(self.memory) > self.max_memory_size:
            self.memory.pop(0)
    
    def update_epsilon(self):
        """
        Обновляет значение epsilon для исследования
        """
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def train(self):
        """
        Обучает модель на основе сохраненного опыта
        """
        if len(self.memory) < self.batch_size:
            return
        
        # Выбираем случайные примеры из буфера
        indices = np.random.choice(len(self.memory), self.batch_size, replace=False)
        batch = [self.memory[i] for i in indices]
        
        states = np.array([experience[0] for experience in batch])
        actions = np.array([experience[1] for experience in batch])
        rewards = np.array([experience[2] for experience in batch])
        next_states = np.array([experience[3] for experience in batch])
        dones = np.array([experience[4] for experience in batch])
        
        # Обучение модели критика
        with tf.GradientTape() as tape:
            # Предсказываем значения текущих состояний
            values = self.model.critic_model(states, training=True)
            
            # Предсказываем значения следующих состояний
            next_values = self.model.critic_model(next_states, training=True)
            
            # Вычисляем целевые значения
            targets = rewards + self.gamma * tf.squeeze(next_values) * (1 - dones)
            targets = tf.expand_dims(targets, axis=1)
            
            # Вычисляем функцию потерь
            critic_loss = tf.reduce_mean(tf.square(targets - values))
        
        # Применяем градиенты для критика
        critic_grads = tape.gradient(critic_loss, self.model.critic_model.trainable_variables)
        self.model.critic_optimizer.apply_gradients(zip(critic_grads, self.model.critic_model.trainable_variables))
        
        # Обучение модели актора
        with tf.GradientTape() as tape:
            # Предсказываем вероятности действий
            action_probs = self.model.actor_model(states, training=True)
            
            # Создаем one-hot вектор для выбранных действий
            action_masks = tf.one_hot(actions, 3)
            
            # Предсказываем значения состояний
            values = self.model.critic_model(states, training=True)
            
            # Вычисляем преимущество (advantage)
            advantages = targets - values
            
            # Вычисляем функцию потерь актора
            selected_action_probs = tf.reduce_sum(action_probs * action_masks, axis=1)
            log_probs = tf.math.log(selected_action_probs + 1e-10)
            actor_loss = -tf.reduce_mean(log_probs * tf.squeeze(advantages))
            
            # Добавляем регуляризацию энтропии
            entropy = -tf.reduce_sum(action_probs * tf.math.log(action_probs + 1e-10), axis=1)
            actor_loss -= 0.01 * tf.reduce_mean(entropy)
        
        # Применяем градиенты для актора
        actor_grads = tape.gradient(actor_loss, self.model.actor_model.trainable_variables)
        self.model.actor_optimizer.apply_gradients(zip(actor_grads, self.model.actor_model.trainable_variables))
        
        return {
            'critic_loss': float(critic_loss),
            'actor_loss': float(actor_loss),
            'mean_value': float(tf.reduce_mean(values)),
            'mean_reward': float(np.mean(rewards))
        }
    
    def save(self, path='models'):
        """
        Сохраняет модель
        """
        self.model.save(path)
    
    def load(self, path='models'):
        """
        Загружает модель
        """
        self.model.load(path)
    
    def log_action_distribution(self, states):
        """
        Логирует распределение действий для набора состояний
        """
        if len(states) == 0:
            return
        
        actions = []
        for state in states:
            action_probs = self.model.predict_action(state)
            actions.append(np.argmax(action_probs))
        
        actions = np.array(actions)
        buy_count = np.sum(actions == 0)
        hold_count = np.sum(actions == 1)
        sell_count = np.sum(actions == 2)
        
        Продолжаю код для файла rl_agent.py, где остановился:
        total = len(actions)
        self.logger.info(f"Распределение действий: BUY: {buy_count/total:.2%}, HOLD: {hold_count/total:.2%}, SELL: {sell_count/total:.2%}")
        
        return {
            'buy_count': int(buy_count),
            'hold_count': int(hold_count),
            'sell_count': int(sell_count),
            'total': total
        }

6. hybrid_decision_maker.py
import numpy as np
import logging

class HybridDecisionMaker:
    """
    Гибридный механизм принятия решений, объединяющий RL-агента
    """
    def __init__(self, rl_agent):
        self.rl_agent = rl_agent
        
        # Логирование
        self.logger = logging.getLogger('hybrid_decision_maker')
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def make_decision(self, state, training=False):
        """
        Принимает торговое решение на основе текущего состояния рынка
        
        Возвращает:
        - action: 0 (BUY), 1 (HOLD), или 2 (SELL)
        - confidence: уверенность в решении (0-1)
        """
        # Получаем вероятности действий от RL-агента
        action_probs = self.rl_agent.model.predict_action(state)
        
        # В режиме обучения может использоваться epsilon-greedy
        if training and np.random.rand() < self.rl_agent.epsilon:
            action = np.random.randint(0, 3)
            confidence = 1.0 / 3.0  # Равномерное распределение
        else:
            # Выбираем действие с наибольшей вероятностью
            action = np.argmax(action_probs)
            confidence = action_probs[action]
        
        # Логирование
        action_names = {0: "BUY", 1: "HOLD", 2: "SELL"}
        self.logger.debug(f"Принято решение: {action_names[action]} с уверенностью {confidence:.4f}")
        self.logger.debug(f"Распределение вероятностей: BUY: {action_probs[0]:.4f}, HOLD: {action_probs[1]:.4f}, SELL: {action_probs[2]:.4f}")
        
        return action, confidence
    
    def explain_decision(self, state):
        """
        Объясняет принятое решение
        """
        # Получаем вероятности действий
        action_probs = self.rl_agent.model.predict_action(state)
        action = np.argmax(action_probs)
        
        # Получаем значение состояния от критика
        value = float(self.rl_agent.model.predict_value(state)[0])
        
        # Формируем объяснение
        action_names = {0: "BUY", 1: "HOLD", 2: "SELL"}
        explanation = {
            'action': action_names[action],
            'confidence': float(action_probs[action]),
            'all_probs': {
                'BUY': float(action_probs[0]),
                'HOLD': float(action_probs[1]),
                'SELL': float(action_probs[2])
            },
            'state_value': value
        }
        
        return explanation

7. simulation_engine.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os
import logging

class SimulationEngine:
    """
    Движок для симуляции торговли
    """
    def __init__(self, environment, decision_maker, initial_balance=10000):
        self.env = environment
        self.decision_maker = decision_maker
        self.initial_balance = initial_balance
        
        # Логирование
        self.logger = logging.getLogger('simulation_engine')
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            
            # Также добавим логирование в файл
            file_handler = logging.FileHandler('simulation.log')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def run_simulation(self, episodes=1, training=False, render=False):
        """
        Запускает симуляцию торговли
        """
        all_episode_info = []
        
        for episode in range(episodes):
            self.logger.info(f"Запуск эпизода {episode+1}/{episodes}")
            
            # Сбрасываем среду
            state, _ = self.env.reset()
            
            done = False
            total_reward = 0
            step = 0
            
            episode_data = {
                'steps': [],
                'rewards': [],
                'balances': [],
                'positions': [],
                'actions': [],
                'confidences': []
            }
            
            while not done:
                # Принимаем решение
                action, confidence = self.decision_maker.make_decision(state, training=training)
                
                # Выполняем действие
                next_state, reward, done, _, info = self.env.step(action)
                
                # Если в режиме обучения, сохраняем опыт
                if training:
                    self.decision_maker.rl_agent.remember(state, action, reward, next_state, done)
                
                # Обновляем состояние
                state = next_state
                total_reward += reward
                step += 1
                
                # Сохраняем данные шага
                episode_data['steps'].append(step)
                episode_data['rewards'].append(reward)
                episode_data['balances'].append(info['balance'])
                episode_data['positions'].append(info['position'])
                episode_data['actions'].append(action)
                episode_data['confidences'].append(confidence)
                
                # Логируем каждые 100 шагов
                if step % 100 == 0:
                    self.logger.info(f"Шаг {step}, Баланс: {info['balance']:.2f}, Награда: {reward:.4f}, "
                                    f"Позиция: {info['position']}, Действие: {action}")
            
            # Логируем результаты эпизода
            final_balance = episode_data['balances'][-1]
            profit_percentage = (final_balance - self.initial_balance) / self.initial_balance * 100
            
            self.logger.info(f"Эпизод {episode+1} завершен. Итоговый баланс: {final_balance:.2f} "
                            f"(Прибыль: {profit_percentage:.2f}%), Всего шагов: {step}")
            
            # Если в режиме обучения, обновляем модель и epsilon
            if training:
                for _ in range(10):  # Несколько итераций обучения на каждый эпизод
                    training_info = self.decision_maker.rl_agent.train()
                    if training_info:
                        self.logger.info(f"Обучение: critic_loss: {training_info['critic_loss']:.4f}, "
                                        f"actor_loss: {training_info['actor_loss']:.4f}, "
                                        f"mean_value: {training_info['mean_value']:.4f}")
                
                self.decision_maker.rl_agent.update_epsilon()
                self.logger.info(f"Epsilon обновлен до {self.decision_maker.rl_agent.epsilon:.4f}")
            
            # Сохраняем данные эпизода
            all_episode_info.append({
                'episode': episode + 1,
                'total_reward': total_reward,
                'final_balance': final_balance,
                'profit_percentage': profit_percentage,
                'steps': step,
                'data': episode_data
            })
            
            # Визуализируем результаты, если требуется
            if render:
                self._render_episode(episode_data, episode + 1)
        
        return all_episode_info
    
    def _render_episode(self, episode_data, episode_num):
        """
        Визуализирует результаты эпизода
        """
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        
        # График баланса
        ax1.plot(episode_data['steps'], episode_data['balances'], 'b-')
        ax1.set_title(f'Эпизод {episode_num} - Баланс')
        ax1.set_ylabel('Баланс')
        ax1.grid(True)
        
        # График наград
        ax2.plot(episode_data['steps'], episode_data['rewards'], 'g-')
        ax2.set_title('Награды')
        ax2.set_ylabel('Награда')
        ax2.grid(True)
        
        # График позиций и действий
        ax3.plot(episode_data['steps'], episode_data['positions'], 'r-', label='Позиция')
        ax3.scatter(episode_data['steps'], episode_data['actions'], c='purple', alpha=0.5, label='Действие')
        ax3.set_title('Позиции и действия')
        ax3.set_ylabel('Значение')
        ax3.set_xlabel('Шаг')
        ax3.grid(True)
        ax3.legend()
        
        plt.tight_layout()
        
        # Создаем директорию для графиков, если её нет
        os.makedirs('plots', exist_ok=True)
        
        # Сохраняем график
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f'plots/episode_{episode_num}_{timestamp}.png')
        plt.close()

8. trade_manager.py
import numpy as np
import pandas as pd
from pybit.unified_trading import HTTP
import time
import logging
import json
from datetime import datetime
import os

class TradeManager:
    """
    Менеджер для управления торговлей на бирже Bybit
    """
    def __init__(self, api_key, api_secret, api_url, order_amount, symbol, leverage="2"):
        self.api_key = api_key
        self.api_secret = api_secret
        self.api_url = api_url
        self.order_amount = order_amount
        self.symbol = symbol
        self.leverage = leverage
        
        # Инициализация API
        self.session = HTTP(
            testnet=(api_url == "https://api-testnet.bybit.com"),
            api_key=api_key,
            api_secret=api_secret
        )
        
        # Логирование
        self.logger = logging.getLogger('trade_manager')
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            
            # Также добавим логирование в файл
            file_handler = logging.FileHandler('trading.log')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        
        # Инициализация торгового журнала
        self.trade_log = []
        self.position = 0  # 0 - нет позиции, 1 - длинная позиция, -1 - короткая позиция
        
        # Устанавливаем плечо
        self._set_leverage()
    
    def _set_leverage(self):
        """
        Устанавливает плечо для торговли
        """
        try:
            response = self.session.set_leverage(
                category="linear",
                symbol=self.symbol,
                buyLeverage=self.leverage,
                sellLeverage=self.leverage
            )
            
            if response['retCode'] == 0:
                self.logger.info(f"Установлено плечо {self.leverage} для {self.symbol}")
            else:
                self.logger.warning(f"Не удалось установить плечо: {response['retMsg']}")
        
        except Exception as e:
            self.logger.error(f"Ошибка при установке плеча: {e}")
    
    def get_current_price(self):
        """
        Получает текущую цену инструмента
        """
        try:
            response = self.session.get_tickers(
                category="linear",
                symbol=self.symbol
            )
            
            if response['retCode'] == 0:
                price = float(response['result']['list'][0]['lastPrice'])
                self.logger.debug(f"Текущая цена {self.symbol}: {price}")
                return price
            else:
                self.logger.warning(f"Не удалось получить текущую цену: {response['retMsg']}")
                return None
        
        except Exception as e:
            self.logger.error(f"Ошибка при получении текущей цены: {e}")
            return None
    
    def place_order(self, action):
        """
        Размещает ордер на бирже
        
        action: 0 - BUY, 1 - HOLD, 2 - SELL
        """
        if action == 1:  # HOLD - ничего не делаем
            return True
        
        try:
            current_price = self.get_current_price()
            
            if current_price is None:
                self.logger.error("Не удалось получить текущую цену для размещения ордера")
                return False
            
            # Определяем тип ордера и сторону
            if action == 0:  # BUY
                side = "Buy"
                if self.position == -1:  # Если у нас короткая позиция, закрываем её
                    self.logger.info("Закрываем короткую позицию")
                    self._close_position()
            elif action == 2:  # SELL
                side = "Sell"
                if self.position == 1:  # Если у нас длинная позиция, закрываем её
                    self.logger.info("Закрываем длинную позицию")
                    self._close_position()
            else:
                self.logger.error(f"Неизвестное действие: {action}")
                return False
            
            # Вычисляем количество контрактов
            qty = self.order_amount / current_price
            
            # Размещаем рыночный ордер
            response = self.session.place_order(
                category="linear",
                symbol=self.symbol,
                side=side,
                orderType="Market",
                qty=str(round(qty, 4)),
                timeInForce="GTC"
            )
            
            if response['retCode'] == 0:
                order_id = response['result']['orderId']
                self.logger.info(f"Размещен {side} ордер на {qty} {self.symbol} по рыночной цене. ID: {order_id}")
                
                # Обновляем позицию
                if action == 0:  # BUY
                    self.position = 1
                elif action == 2:  # SELL
                    self.position = -1
                
                # Записываем в журнал
                self.trade_log.append({
                    'timestamp': datetime.now().isoformat(),
                    'action': 'BUY' if action == 0 else 'SELL',
                    'price': current_price,
                    'qty': qty,
                    'order_id': order_id
                })
                
                # Сохраняем журнал
                self._save_trade_log()
                
                return True
            else:
                self.logger.error(f"Ошибка при размещении ордера: {response['retMsg']}")
                return False
        
        except Exception as e:
            self.logger.error(f"Ошибка при размещении ордера: {e}")
            return False
    
    def _close_position(self):
        """
        Закрывает текущую позицию
        """
        try:
            # Получаем информацию о текущей позиции
            response = self.session.get_positions(
                category="linear",
                symbol=self.symbol
            )
            
            if response['retCode'] != 0:
                self.logger.error(f"Ошибка при получении информации о позиции: {response['retMsg']}")
                return False
            
            position_info = response['result']['list'][0]
            size = float(position_info['size'])
            
            if size == 0:
                self.logger.info("Нет открытой позиции для закрытия")
                self.position = 0
                return True
            
            # Определяем сторону для закрытия позиции
            side = "Sell" if position_info['side'] == "Buy" else "Buy"
            
            # Размещаем рыночный ордер для закрытия позиции
            response = self.session.place_order(
                category="linear",
                symbol=self.symbol,
                side=side,
                orderType="Market",
                qty=str(size),
                timeInForce="GTC",
                reduceOnly=True
            )
            
            if response['retCode'] == 0:
                order_id = response['result']['orderId']
                self.logger.info(f"Закрыта позиция {self.symbol}. ID ордера: {order_id}")
                self.position = 0
                
                # Записываем в журнал
                self.trade_log.append({
                    'timestamp': datetime.now().isoformat(),
                    'action': 'CLOSE',
                    'price': self.get_current_price(),
                    'qty': size,
                    'order_id': order_id
                })
                
                # Сохраняем журнал
                self._save_trade_log()
                
                return True
            else:
                self.logger.error(f"Ошибка при закрытии позиции: {response['retMsg']}")
                return False
        
        except Exception as e:
            self.logger.error(f"Ошибка при закрытии позиции: {e}")
            return False
    
    def get_position_info(self):
        """
        Получает информацию о текущей позиции
        """
        try:
            response = self.session.get_positions(
                category="linear",
                symbol=self.symbol
            )
            
            if response['retCode'] == 0:
                position_info = response['result']['list'][0]
                return {
                    'symbol': position_info['symbol'],
                    'side': position_info['side'],
                    'size': float(position_info['size']),
                    'entry_price': float(position_info['entryPrice']),
                    'leverage': float(position_info['leverage']),
                    'unrealised_pnl': float(position_info['unrealisedPnl']),
                    'position_value': float(position_info['positionValue'])
                }
            else:
                self.logger.warning(f"Не удалось получить информацию о позиции: {response['retMsg']}")
                return None
        
        except Exception as e:
            self.logger.error(f"Ошибка при получении информации о позиции: {e}")
            return None
    
    def _save_trade_log(self):
        """
        Сохраняет журнал торговли в файл
        """
        try:
            with open('trade_log.json', 'w') as f:
                json.dump(self.trade_log, f, indent=2)
        except Exception as e:
            self.logger.error(f"Ошибка при сохранении журнала торговли: {e}")

9. run_live_trading.py
import os
import sys
import time
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tensorflow as tf
from pybit.unified_trading import HTTP

# Импорт наших модулей
from feature_engineering import FeatureEngineering
from models.xlstm_rl_model import XLSTMRLModel
from hybrid_decision_maker import HybridDecisionMaker
from trade_manager import TradeManager
from rl_agent import RLAgent
import config

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('live_trading.log')
    ]
)

logger = logging.getLogger('live_trading')

def fetch_latest_data(session, symbol, timeframe, limit=100):
    """
    Получает последние свечи с биржи
    """
    try:
        response = session.get_kline(
            category="linear",
            symbol=symbol,
            interval=timeframe,
            limit=limit
        )
        
        if response['retCode'] == 0:
            data = response['result']['list']
            
            # Преобразуем данные в DataFrame
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
            ])
            
            # Преобразуем типы данных
            numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'turnover']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df['timestamp'] = pd.to_datetime(pd.to_numeric(df['timestamp']), unit='ms')
            df['symbol'] = symbol
            
            # Сортируем по времени
            df.sort_values('timestamp', inplace=True)
            
            return df
        else:
            logger.error(f"Ошибка при получении данных: {response['retMsg']}")
            return None
    
    except Exception as e:
        logger.error(f"Ошибка при получении данных: {e}")
        return None

def main():
    """
    Основная функция для запуска живой торговли
    """
    logger.info("Запуск системы живой торговли...")
    
    # Загружаем конфигурацию
    api_key = config.BYBIT_API_KEY
    api_secret = config.BYBIT_API_SECRET
    api_url = config.API_URL
    symbol = config.SYMBOLS[0]  # Берем первый символ из списка
    timeframe = config.TIMEFRAME
    order_amount = config.ORDER_USDT_AMOUNT
    leverage = config.LEVERAGE
    sequence_length = config.SEQUENCE_LENGTH
    required_candles = config.REQUIRED_CANDLES
    
    # Инициализация API
    session = HTTP(
        testnet=(api_url == "https://api-demo.bybit.com"),
        api_key=api_key,
        api_secret=api_secret
    )
    ---
    # Инициализация компонентов системы
    feature_engineering = FeatureEngineering(sequence_length=sequence_length)
    
    # Загружаем скейлер, если он существует
    if not feature_engineering.load_scaler():
        logger.error("Не удалось загрузить скейлер. Убедитесь, что модель обучена.")
        return
    
    # Инициализация модели
    input_shape = (sequence_length, len(feature_engineering.feature_columns))
    rl_model = XLSTMRLModel(input_shape=input_shape, 
                          memory_size=config.XLSTM_MEMORY_SIZE, 
                          memory_units=config.XLSTM_MEMORY_UNITS)
    
    # Загружаем модель, если она существует
    try:
        rl_model.load()
        logger.info("Модель успешно загружена")
    except Exception as e:
        logger.error(f"Не удалось загрузить модель: {e}")
        return
    
    # Инициализация RL-агента
    rl_agent = RLAgent(state_shape=input_shape, 
                      memory_size=config.XLSTM_MEMORY_SIZE, 
                      memory_units=config.XLSTM_MEMORY_UNITS)
    rl_agent.model = rl_model
    
    # Инициализация механизма принятия решений
    decision_maker = HybridDecisionMaker(rl_agent)
    
    # Инициализация менеджера торговли
    trade_manager = TradeManager(
        api_key=api_key,
        api_secret=api_secret,
        api_url=api_url,
        order_amount=order_amount,
        symbol=symbol,
        leverage=leverage
    )
    
    logger.info("Система инициализирована, начинаем торговлю...")
    
    # Основной цикл торговли
    while True:
        try:
            # Получаем текущее время
            current_time = datetime.now()
            
            # Получаем последние данные
            df = fetch_latest_data(session, symbol, timeframe, limit=required_candles)
            
            if df is None or len(df) < sequence_length:
                logger.error(f"Недостаточно данных для анализа. Получено: {len(df) if df is not None else 0} строк")
                time.sleep(10)
                continue
            
            # Подготавливаем данные
            X, _, _ = feature_engineering.prepare_test_data(df)
            
            if len(X) == 0:
                logger.error("Не удалось подготовить данные для анализа")
                time.sleep(10)
                continue
            
            # Получаем последнее состояние рынка
            current_state = X[-1]
            
            # Принимаем решение
            action, confidence = decision_maker.make_decision(current_state)
            
            # Логируем решение
            action_names = {0: "BUY", 1: "HOLD", 2: "SELL"}
            logger.info(f"Принято решение: {action_names[action]} с уверенностью {confidence:.4f}")
            
            # Получаем объяснение решения
            explanation = decision_maker.explain_decision(current_state)
            logger.info(f"Объяснение решения: {explanation}")
            
            # Выполняем действие
            if trade_manager.place_order(action):
                logger.info(f"Ордер успешно размещен: {action_names[action]}")
            else:
                logger.error(f"Не удалось разместить ордер: {action_names[action]}")
            
            # Получаем информацию о позиции
            position_info = trade_manager.get_position_info()
            if position_info:
                logger.info(f"Текущая позиция: {position_info}")
            
            # Ждем некоторое время перед следующей итерацией
            # Для минутного таймфрейма ждем 30 секунд
            time.sleep(30)
            
        except KeyboardInterrupt:
            logger.info("Торговля остановлена пользователем")
            break
        except Exception as e:
            logger.error(f"Ошибка в процессе торговли: {e}")
            time.sleep(10)

if __name__ == "__main__":
    main()

10. train_model.py
import os
import sys
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from datetime import datetime

# Импорт наших модулей
from feature_engineering import FeatureEngineering
from trading_env import TradingEnvironment
from rl_agent import RLAgent
from hybrid_decision_maker import HybridDecisionMaker
from simulation_engine import SimulationEngine
from models.xlstm_rl_model import XLSTMRLModel
import config

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log')
    ]
)

logger = logging.getLogger('train_model')

def load_data(file_path):
    """
    Загружает данные из CSV файла
    """
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Данные загружены из {file_path}, строк: {len(df)}")
        return df
    except Exception as e:
        logger.error(f"Ошибка при загрузке данных: {e}")
        return None

def train_model(data_path, epochs=30, batch_size=64, validation_split=0.2, test_split=0.1):
    """
    Обучает модель xLSTM RL для торговли
    """
    logger.info(f"Начало обучения модели с {epochs} эпохами")
    
    # Загружаем данные
    df = load_data(data_path)
    if df is None:
        return
    
    # Фильтруем только один символ для упрощения
    symbol = config.SYMBOLS[0]
    df_symbol = df[df['symbol'] == symbol].copy()
    
    if len(df_symbol) == 0:
        logger.error(f"Нет данных для символа {symbol}")
        return
    
    logger.info(f"Использую данные для символа {symbol}, строк: {len(df_symbol)}")
    
    # Подготовка данных
    feature_eng = FeatureEng
Продолжаю код для файла train_model.py, где остановился:
    # Подготовка данных
    feature_eng = FeatureEngineering(sequence_length=config.SEQUENCE_LENGTH)
    X, y_close, df_processed = feature_eng.prepare_data(df_symbol)
    
    logger.info(f"Подготовлено последовательностей: {len(X)}")
    
    # Разделение на обучающую, валидационную и тестовую выборки
    # Сначала отделяем тестовую выборку
    X_temp, X_test, y_temp, y_test = train_test_split(X, y_close, test_size=test_split, shuffle=False)
    
    # Затем разделяем оставшиеся данные на обучающую и валидационную выборки
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=validation_split, shuffle=False)
    
    logger.info(f"Размеры выборок: Обучающая: {len(X_train)}, Валидационная: {len(X_val)}, Тестовая: {len(X_test)}")
    
    # Сохраняем скейлер для последующего использования
    feature_eng.save_scaler()
    
    # Создаем среды для обучения, валидации и тестирования
    train_env = TradingEnvironment(X_train, sequence_length=config.SEQUENCE_LENGTH)
    val_env = TradingEnvironment(X_val, sequence_length=config.SEQUENCE_LENGTH)
    test_env = TradingEnvironment(X_test, sequence_length=config.SEQUENCE_LENGTH)
    
    # Инициализация RL-агента
    input_shape = (config.SEQUENCE_LENGTH, X.shape[2])
    rl_agent = RLAgent(
        state_shape=input_shape,
        memory_size=config.XLSTM_MEMORY_SIZE,
        memory_units=config.XLSTM_MEMORY_UNITS,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.1,
        epsilon_decay=0.995,
        batch_size=batch_size
    )
    
    # Инициализация механизма принятия решений
    decision_maker = HybridDecisionMaker(rl_agent)
    
    # Инициализация движка симуляции
    train_sim = SimulationEngine(train_env, decision_maker)
    val_sim = SimulationEngine(val_env, decision_maker)
    test_sim = SimulationEngine(test_env, decision_maker)
    
    # Создаем директорию для сохранения результатов
    os.makedirs('results', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Метрики для отслеживания
    train_rewards = []
    val_rewards = []
    train_profits = []
    val_profits = []
    best_val_profit = -float('inf')
    
    # Обучение модели
    logger.info("Начало процесса обучения...")
    
    for epoch in range(epochs):
        logger.info(f"Эпоха {epoch+1}/{epochs}")
        
        # Обучение на тренировочной выборке
        logger.info("Тренировка на обучающей выборке...")
        train_results = train_sim.run_simulation(episodes=1, training=True)
        train_reward = train_results[0]['total_reward']
        train_profit = train_results[0]['profit_percentage']
        train_rewards.append(train_reward)
        train_profits.append(train_profit)
        
        # Логирование распределения действий на обучающей выборке
        train_actions = rl_agent.log_action_distribution(X_train[:1000])  # Используем только часть для скорости
        
        # Валидация
        logger.info("Валидация модели...")
        val_results = val_sim.run_simulation(episodes=1, training=False)
        val_reward = val_results[0]['total_reward']
        val_profit = val_results[0]['profit_percentage']
        val_rewards.append(val_reward)
        val_profits.append(val_profit)
        
        # Логирование распределения действий на валидационной выборке
        val_actions = rl_agent.log_action_distribution(X_val[:1000])  # Используем только часть для скорости
        
        # Логирование метрик
        logger.info(f"Эпоха {epoch+1} завершена:")
        logger.info(f"  Тренировка - Награда: {train_reward:.4f}, Прибыль: {train_profit:.2f}%")
        logger.info(f"  Валидация - Награда: {val_reward:.4f}, Прибыль: {val_profit:.2f}%")
        
        # Логирование распределения действий
        logger.info(f"  Распределение действий (Тренировка) - BUY: {train_actions['buy_count']}, HOLD: {train_actions['hold_count']}, SELL: {train_actions['sell_count']}")
        logger.info(f"  Распределение действий (Валидация) - BUY: {val_actions['buy_count']}, HOLD: {val_actions['hold_count']}, SELL: {val_actions['sell_count']}")
        
        # Сохранение лучшей модели по прибыли на валидационной выборке
        if val_profit > best_val_profit:
            logger.info(f"  Улучшение на валидации! Сохраняем модель. Прибыль: {val_profit:.2f}% (предыдущая лучшая: {best_val_profit:.2f}%)")
            rl_agent.save()
            best_val_profit = val_profit
        
        # Каждые 5 эпох делаем подробный анализ модели
        if (epoch + 1) % 5 == 0:
            logger.info(f"Детальный анализ модели после эпохи {epoch+1}:")
            
            # Анализ признаков, влияющих на решения
            logger.info("Анализ признаков, влияющих на решения:")
            
            # Проверяем предсказания модели на валидационной выборке
            val_actions_full = []
            for i in range(len(X_val)):
                action_probs = rl_agent.model.predict_action(X_val[i])
                val_actions_full.append(np.argmax(action_probs))
            
            val_actions_full = np.array(val_actions_full)
            buy_indices = np.where(val_actions_full == 0)[0]
            hold_indices = np.where(val_actions_full == 1)[0]
            sell_indices = np.where(val_actions_full == 2)[0]
            
            # Анализ паттернов для каждого типа действия
            if len(buy_indices) > 0:
                buy_patterns = X_val[buy_indices]
                logger.info(f"Паттерны BUY (среднее значение признаков):")
                for i in range(X_val.shape[2]):
                    logger.info(f"  Признак {i}: {np.mean(buy_patterns[:, -1, i]):.4f}")
            
            if len(hold_indices) > 0:
                hold_patterns = X_val[hold_indices]
                logger.info(f"Паттерны HOLD (среднее значение признаков):")
                for i in range(X_val.shape[2]):
                    logger.info(f"  Признак {i}: {np.mean(hold_patterns[:, -1, i]):.4f}")
            
            if len(sell_indices) > 0:
                sell_patterns = X_val[sell_indices]
                logger.info(f"Паттерны SELL (среднее значение признаков):")
                for i in range(X_val.shape[2]):
                    logger.info(f"  Признак {i}: {np.mean(sell_patterns[:, -1, i]):.4f}")
            
            # Визуализация метрик обучения
            plt.figure(figsize=(12, 10))
            
            plt.subplot(2, 1, 1)
            plt.plot(range(1, epoch+2), train_rewards, 'b-', label='Обучение')
            plt.plot(range(1, epoch+2), val_rewards, 'r-', label='Валидация')
            plt.title('Награды по эпохам')
            plt.xlabel('Эпоха')
            plt.ylabel('Награда')
            plt.legend()
            plt.grid(True)
            
            plt.subplot(2, 1, 2)
            plt.plot(range(1, epoch+2), train_profits, 'b-', label='Обучение')
            plt.plot(range(1, epoch+2), val_profits, 'r-', label='Валидация')
            plt.title('Прибыль по эпохам')
            plt.xlabel('Эпоха')
            plt.ylabel('Прибыль (%)')
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(f'results/training_metrics_epoch_{epoch+1}.png')
            plt.close()
    
    # Загружаем лучшую модель для финального тестирования
    logger.info("Загрузка лучшей модели для тестирования...")
    rl_agent.load()
    
    # Тестирование на тестовой выборке
    logger.info("Тестирование модели на тестовой выборке...")
    test_results = test_sim.run_simulation(episodes=1, training=False, render=True)
    test_reward = test_results[0]['total_reward']
    test_profit = test_results[0]['profit_percentage']
    
    # Логирование распределения действий на тестовой выборке
    test_actions = rl_agent.log_action_distribution(X_test)
    
    logger.info("Результаты тестирования:")
    logger.info(f"  Награда: {test_reward:.4f}")
    logger.info(f"  Прибыль: {test_profit:.2f}%")
    logger.info(f"  Распределение действий - BUY: {test_actions['buy_count']}, HOLD: {test_actions['hold_count']}, SELL: {test_actions['sell_count']}")
    logger.info(f"  Процент действий - BUY: {test_actions['buy_count']/test_actions['total']:.2%}, HOLD: {test_actions['hold_count']/test_actions['total']:.2%}, SELL: {test_actions['sell_count']/test_actions['total']:.2%}")
    
    # Финальная оценка модели
    logger.info("\nФинальная оценка модели:")
    logger.info(f"Лучшая прибыль на валидации: {best_val_profit:.2f}%")
    logger.info(f"Прибыль на тестовой выборке: {test_profit:.2f}%")
    
    return {
        'train_rewards': train_rewards,
        'val_rewards': val_rewards,
        'train_profits': train_profits,
        'val_profits': val_profits,
        'test_reward': test_reward,
        'test_profit': test_profit,
        'test_actions': test_actions
    }

if __name__ == "__main__":
    # Путь к файлу с данными
    data_path = "historical_data.csv"
    
    # Количество эпох обучения
    epochs = 30  # Это можно изменить на 50 позже
    
    # Запуск обучения
    results = train_model(data_path, epochs=epochs)

Теперь добавим последний файл:
11. screener.py (пустой, так как указано не менять его)
Итак, мы создали все необходимые файлы для торгового бота на основе xLSTM с RL. Вот краткое описание каждого файла:

models/xlstm_memory_cell.py - Реализация ячейки памяти xLSTM с расширенной структурой.
models/xlstm_rl_model.py - Модель xLSTM для RL с архитектурой актор-критик.
feature_engineering.py - Подготовка и обработка данных для модели.
trading_env.py - Среда для обучения RL-агента торговле.
rl_agent.py - Агент reinforcement learning для торговли.
hybrid_decision_maker.py - Механизм принятия торговых решений.
simulation_engine.py - Движок для симуляции торговли.
trade_manager.py - Управление торговлей на бирже Bybit.
run_live_trading.py - Запуск живой торговли с использованием обученной модели.
train_model.py - Обучение модели xLSTM RL для торговли.

Модель использует только стандартные данные с биржи (timestamp, open, high, low, close, volume, turnover, symbol) без дополнительных индикаторов, как и требовалось. Обучение проходит в течение 30 эпох (с возможностью увеличения до 50), с подробным логированием каждые 5 эпох.
Система сохраняет лучшую модель на основе результатов валидации и проводит финальное тестирование на отдельной тестовой выборке. Все логи, графики и результаты сохраняются для последующего анализа.