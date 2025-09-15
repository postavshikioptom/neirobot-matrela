import numpy as np
import tensorflow as tf
from models.xlstm_rl_model import XLSTMRLModel
import os
# import logging # 🔥 УДАЛЕНО: Импорт logging

class RLAgent:
    """
    Агент Reinforcement Learning для торговли - ПОДДЕРЖКА ТРЁХЭТАПНОГО ОБУЧЕНИЯ
    """
    def __init__(self, state_shape, memory_size=64, memory_units=128, gamma=0.99, epsilon=0.3, epsilon_min=0.1, epsilon_decay=0.995, batch_size=64):
        self.state_shape = state_shape
        self.gamma = gamma  # Коэффициент дисконтирования
        # 🔥 ИЗМЕНЕНО: Начальный epsilon ниже для fine-tuning предобученной модели
        self.epsilon = epsilon  # Начинаем с меньшего epsilon для fine-tuning
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        
        # Инициализация модели
        self.model = XLSTMRLModel(input_shape=state_shape, 
                                 memory_size=memory_size, 
                                 memory_units=memory_units)
        
        # Буфер опыта
        self.memory = []
        self.max_memory_size = 10000
        
        # 🔥 УДАЛЕНО: Инициализация логгера
        # self.logger = logging.getLogger('rl_agent')
        # self.logger.setLevel(logging.INFO)
        # if not self.logger.handlers:
        #     handler = logging.StreamHandler()
        #     formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        #     handler.setFormatter(formatter)
        #     self.logger.addHandler(handler)
    
    def act(self, state, training=True):
        """Выбирает действие на основе текущего состояния"""
        if training and np.random.rand() < self.epsilon:
            # Случайное действие во время обучения
            return np.random.randint(0, 3)
        
        # Получаем вероятности действий от модели актора
        action_probs = self.model.predict_action(state)
        
        # Выбираем действие с наибольшей вероятностью
        return np.argmax(action_probs)
    
    def remember(self, state, action, reward, next_state, done):
        """Сохраняет опыт в буфер"""
        self.memory.append((state, action, reward, next_state, done))
        
        # Ограничиваем размер буфера
        if len(self.memory) > self.max_memory_size:
            self.memory.pop(0)
    
    def update_epsilon(self):
        """Обновляет значение epsilon для исследования"""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def train(self):
        """Обучает модель на основе сохраненного опыта"""
        if len(self.memory) < self.batch_size:
            return None
        
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
        """Сохраняет модель"""
        self.model.save(path, stage="_rl_final")
    
    def load(self, path='models'):
        """Загружает модель"""
        self.model.load(path, stage="_rl_finetuned")
    
    def log_action_distribution(self, states):
        """Логирует распределение действий для набора состояний"""
        if len(states) == 0:
            return {'buy_count': 0, 'hold_count': 0, 'sell_count': 0, 'total': 0}
        
        actions = []
        for state in states:
            action_probs = self.model.predict_action(state)
            actions.append(np.argmax(action_probs))
        
        actions = np.array(actions)
        buy_count = np.sum(actions == 0)
        hold_count = np.sum(actions == 1)
        sell_count = np.sum(actions == 2)
        
        total = len(actions)
        # 🔥 ИЗМЕНЕНО: logger.info -> print
        print(f"Распределение действий: BUY: {buy_count/total:.2%}, HOLD: {hold_count/total:.2%}, SELL: {sell_count/total:.2%}")
        
        return {
            'buy_count': int(buy_count),
            'hold_count': int(hold_count),
            'sell_count': int(sell_count),
            'total': total
        }