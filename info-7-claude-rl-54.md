Хорошо, понял. Значит, нужно полностью убрать использование модуля logging и заменить все вызовы logger.info(), logger.error(), logger.warning() и logger.debug() на print() во всех файлах. Также удалим все импорты logging и инициализации логгеров.
Это довольно много изменений, поэтому я буду предоставлять код по файлам.

📁 1. models/xlstm_rl_model.py
(Без изменений, там нет logger)

📁 2. config.py
(Без изменений, там нет logger)

📁 3. feature_engineering.py
(Без изменений, там нет logger)

📁 4. train_model.py
(Уже исправлено, как вы ранее предоставили)

📁 5. rl_agent.py
🔄 ИЗМЕНЯЕМ:
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
            return np.random.randint(0, 3)
        
        action_probs = self.model.predict_action(state)
        return np.argmax(action_probs)
    
    def remember(self, state, action, reward, next_state, done):
        """Сохраняет опыт в буфер"""
        self.memory.append((state, action, reward, next_state, done))
        
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
        
        indices = np.random.choice(len(self.memory), self.batch_size, replace=False)
        batch = [self.memory[i] for i in indices]
        
        states = np.array([experience[0] for experience in batch])
        actions = np.array([experience[1] for experience in batch])
        rewards = np.array([experience[2] for experience in batch])
        next_states = np.array([experience[3] for experience in batch])
        dones = np.array([experience[4] for experience in batch])
        
        with tf.GradientTape() as tape:
            values = self.model.critic_model(states, training=True)
            next_values = self.model.critic_model(next_states, training=True)
            targets = rewards + self.gamma * tf.squeeze(next_values) * (1 - dones)
            targets = tf.expand_dims(targets, axis=1)
            critic_loss = tf.reduce_mean(tf.square(targets - values))
        
        critic_grads = tape.gradient(critic_loss, self.model.critic_model.trainable_variables)
        self.model.critic_optimizer.apply_gradients(zip(critic_grads, self.model.critic_model.trainable_variables))
        
        with tf.GradientTape() as tape:
            action_probs = self.model.actor_model(states, training=True)
            action_masks = tf.one_hot(actions, 3)
            values = self.model.critic_model(states, training=True)
            advantages = targets - values
            selected_action_probs = tf.reduce_sum(action_probs * action_masks, axis=1)
            log_probs = tf.math.log(selected_action_probs + 1e-10)
            actor_loss = -tf.reduce_mean(log_probs * tf.squeeze(advantages))
            entropy = -tf.reduce_sum(action_probs * tf.math.log(action_probs + 1e-10), axis=1)
            actor_loss -= 0.01 * tf.reduce_mean(entropy)
        
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



📁 6. hybrid_decision_maker.py
🔄 ИЗМЕНЯЕМ:
import numpy as np
# import logging # 🔥 УДАЛЕНО: Импорт logging

class HybridDecisionMaker:
    """
    Гибридный механизм принятия решений, объединяющий RL-агента
    """
    def __init__(self, rl_agent):
        self.rl_agent = rl_agent
        
        # 🔥 УДАЛЕНО: Инициализация логгера
        # self.logger = logging.getLogger('hybrid_decision_maker')
        # self.logger.setLevel(logging.INFO)
        # if not self.logger.handlers:
        #     handler = logging.StreamHandler()
        #     formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        #     handler.setFormatter(formatter)
        #     self.logger.addHandler(handler)
    
    def make_decision(self, state, training=False, position=0):
        """
        Принимает торговое решение на основе текущего состояния рынка
        
        Args:
            state: состояние рынка
            training: режим обучения
            position: текущая позиция (0, 1, -1)
        
        Возвращает:
        - action: 0 (BUY), 1 (HOLD), или 2 (SELL)
        - confidence: уверенность в решении (0-1)
        """
        action_probs = self.rl_agent.model.predict_action(state)
        
        if training and np.random.rand() < self.rl_agent.epsilon:
            action = np.random.randint(0, 3)
            confidence = 1.0 / 3.0
        else:
            action = np.argmax(action_probs)
            confidence = action_probs[action]
        
        action_names = {0: "BUY", 1: "HOLD", 2: "SELL"}
        # 🔥 ИЗМЕНЕНО: logger.debug -> print
        print(f"Принято решение: {action_names[action]} с уверенностью {confidence:.4f}")
        print(f"Распределение вероятностей: BUY: {action_probs[0]:.4f}, HOLD: {action_probs[1]:.4f}, SELL: {action_probs[2]:.4f}")
        print(f"Текущая позиция: {position}")
        
        return action, confidence
    
    def explain_decision(self, state):
        """
        Объясняет принятое решение
        """
        action_probs = self.rl_agent.model.predict_action(state)
        action = np.argmax(action_probs)
        
        value = float(self.rl_agent.model.predict_value(state)[0])
        
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



📁 7. simulation_engine.py
🔄 ИЗМЕНЯЕМ:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os
# import logging # 🔥 УДАЛЕНО: Импорт logging

class SimulationEngine:
    """
    Движок для симуляции торговли
    """
    def __init__(self, environment, decision_maker, initial_balance=10000):
        self.env = environment
        self.decision_maker = decision_maker
        self.initial_balance = initial_balance
        
        # 🔥 УДАЛЕНО: Инициализация логгера и файлового обработчика
        # self.logger = logging.getLogger('simulation_engine')
        # self.logger.setLevel(logging.INFO)
        # if not self.logger.handlers:
        #     handler = logging.StreamHandler()
        #     formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        #     handler.setFormatter(formatter)
        #     self.logger.addHandler(handler)
            
        #     file_handler = logging.FileHandler('simulation.log')
        #     file_handler.setFormatter(formatter)
        #     self.logger.addHandler(file_handler)
    
    def run_simulation(self, episodes=1, training=False, render=False):
        """
        Запускает симуляцию торговли
        """
        all_episode_info = []
        
        for episode in range(episodes):
            # 🔥 ИЗМЕНЕНО: logger.info -> print
            print(f"Запуск эпизода {episode+1}/{episodes}")
            
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
                action, confidence = self.decision_maker.make_decision(
                    state,
                    training=training,
                    position=self.env.position
                )
                
                next_state, reward, done, _, info = self.env.step(action)
                
                if training:
                    self.decision_maker.rl_agent.remember(state, action, reward, next_state, done)
                
                state = next_state
                total_reward += reward
                step += 1
                
                episode_data['steps'].append(step)
                episode_data['rewards'].append(reward)
                episode_data['balances'].append(info['balance'])
                episode_data['positions'].append(info['position'])
                episode_data['actions'].append(action)
                episode_data['confidences'].append(confidence)
                
                if step % 100 == 0:
                    # 🔥 ИЗМЕНЕНО: logger.info -> print
                    print(f"Шаг {step}, Баланс: {info['balance']:.2f}, Награда: {reward:.4f}, "
                                    f"Позиция: {info['position']}, Действие: {action}")
            
            final_balance = episode_data['balances'][-1]
            profit_percentage = (final_balance - self.initial_balance) / self.initial_balance * 100
            
            # 🔥 ИЗМЕНЕНО: logger.info -> print
            print(f"Эпизод {episode+1} завершен. Итоговый баланс: {final_balance:.2f} "
                            f"(Прибыль: {profit_percentage:.2f}%), Всего шагов: {step}")
            
            if training:
                for _ in range(10):
                    training_info = self.decision_maker.rl_agent.train()
                    if training_info:
                        # 🔥 ИЗМЕНЕНО: logger.info -> print
                        print(f"Обучение: critic_loss: {training_info['critic_loss']:.4f}, "
                                        f"actor_loss: {training_info['actor_loss']:.4f}, "
                                        f"mean_value: {training_info['mean_value']:.4f}")
                
                self.decision_maker.rl_agent.update_epsilon()
                # 🔥 ИЗМЕНЕНО: logger.info -> print
                print(f"Epsilon обновлен до {self.decision_maker.rl_agent.epsilon:.4f}")
            
            all_episode_info.append({
                'episode': episode + 1,
                'total_reward': total_reward,
                'final_balance': final_balance,
                'profit_percentage': profit_percentage,
                'steps': step,
                'data': episode_data
            })
            
            if render:
                self._render_episode(episode_data, episode + 1)
        
        return all_episode_info
    
    def _render_episode(self, episode_data, episode_num):
        """
        Визуализирует результаты эпизода
        """
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        
        ax1.plot(episode_data['steps'], episode_data['balances'], 'b-')
        ax1.set_title(f'Эпизод {episode_num} - Баланс')
        ax1.set_ylabel('Баланс')
        ax1.grid(True)
        
        ax2.plot(episode_data['steps'], episode_data['rewards'], 'g-')
        ax2.set_title('Награды')
        ax2.set_ylabel('Награда')
        ax2.grid(True)
        
        ax3.plot(episode_data['steps'], episode_data['actions'], 'r-', label='Позиция')
        ax3.scatter(episode_data['steps'], episode_data['actions'], c='purple', alpha=0.5, label='Действие')
        ax3.set_title('Позиции и действия')
        ax3.set_ylabel('Значение')
        ax3.set_xlabel('Шаг')
        ax3.grid(True)
        ax3.legend()
        
        plt.tight_layout()
        
        os.makedirs('plots', exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f'plots/episode_{episode_num}_{timestamp}.png')
        plt.close()



📁 8. trading_env.py
🔄 ИЗМЕНЯЕМ:
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
# import logging # 🔥 УДАЛЕНО: Импорт logging

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
        
        # Пространство наблюдений: только последовательность цен и объемов БЕЗ позиции
        # Используем исходную форму данных
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.sequence_length, data.shape[2])
        )
        
        # 🔥 УДАЛЕНО: Инициализация логгера
        # self.logger = logging.getLogger('trading_env')
        
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
        
        observation = self._get_observation()
        
        return observation, {}
    
    def step(self, action):
        """Выполняет шаг в среде"""
        if self.current_step >= len(self.data) - 1:
            observation = self._get_observation() if self.current_step < len(self.data) else self.data[-1].copy()
            info = {
                'balance': float(self.balance),
                'position': self.position,
                'shares_held': float(self.shares_held),
                'total_trades': self.total_trades,
                'total_profit': float(self.total_profit),
                'portfolio_value': float(self.balance)
            }
            return observation, 0.0, True, False, info
        
        current_price = self._get_current_price()
        
        max_position_size = self.initial_balance * 0.95
        
        reward = 0
        if action == 0:  # BUY
            if self.position != 1 and self.balance > 0:
                if self.position == -1:
                    reward += self._close_position(current_price)
                
                max_shares = min(max_position_size / current_price, self.balance / (current_price * (1 + self.transaction_fee)))
                shares_to_buy = max_shares * 0.9
                
                if shares_to_buy > 0:
                    cost = shares_to_buy * current_price * (1 + self.transaction_fee)
                    
                    if cost <= self.balance:
                        self.balance -= cost
                        self.shares_held = shares_to_buy
                        self.cost_basis = current_price
                        self.position = 1
                        self.total_trades += 1
        
        elif action == 2:  # SELL
            if self.position != -1 and self.balance > 0:
                if self.position == 1:
                    reward += self._close_position(current_price)
                
                max_shares = min(max_position_size / current_price, self.balance / current_price)
                shares_to_sell = max_shares * 0.9
                
                if shares_to_sell > 0:
                    self.shares_held = -shares_to_sell
                    self.cost_basis = current_price
                    self.position = -1
                    self.total_trades += 1
        
        self.current_step += 1
        
        done = self.current_step >= len(self.data) - 1
        
        if done and self.position != 0:
            reward += self._close_position(current_price)
        
        observation = self._get_observation()
        
        portfolio_value = self.balance
        
        if np.isnan(self.balance) or np.isinf(self.balance):
            self.balance = self.initial_balance
            portfolio_value = self.initial_balance
            reward = -1.0
        
        if self.position == 1 and self.shares_held > 0:
            position_value = self.shares_held * current_price
            if not (np.isnan(position_value) or np.isinf(position_value)):
                portfolio_value += position_value
        elif self.position == -1 and self.shares_held < 0:
            position_value = -self.shares_held * current_price
            if not (np.isnan(position_value) or np.isinf(position_value)):
                portfolio_value -= position_value
        
        if portfolio_value > self.initial_balance * 1000:
            portfolio_value = self.initial_balance * 1000
            reward = -0.5
        elif portfolio_value < self.initial_balance * 0.001:
            portfolio_value = self.initial_balance * 0.001
            reward = -0.5
        
        info = {
            'balance': float(self.balance) if not (np.isnan(self.balance) or np.isinf(self.balance)) else float(self.initial_balance),
            'position': self.position,
            'shares_held': float(self.shares_held) if not (np.isnan(self.shares_held) or np.isinf(self.shares_held)) else 0.0,
            'total_trades': self.total_trades,
            'total_profit': float(self.total_profit) if not (np.isnan(self.total_profit) or np.isinf(self.total_profit)) else 0.0,
            'portfolio_value': float(portfolio_value)
        }
        
        return observation, reward, done, False, info

    def _get_observation(self):
        """Возвращает текущее наблюдение"""
        obs = self.data[self.current_step].copy()
        return obs
    
    def _get_current_price(self):
        """Возвращает текущую цену закрытия"""
        try:
            price = self.data[self.current_step][-1, 3]
            
            if np.isnan(price) or np.isinf(price) or price <= 0:
                if self.current_step > 0:
                    return self.data[self.current_step-1][-1, 3]
                else:
                    return 100.0
            
            return price
        except (IndexError, ValueError):
            return 100.0
    
    def _close_position(self, current_price):
        """Закрывает текущую позицию и возвращает полученную награду"""
        reward = 0
        
        try:
            if self.position == 1 and self.shares_held > 0:
                profit = self.shares_held * (current_price - self.cost_basis)
                fee = self.shares_held * current_price * self.transaction_fee
                
                if not (np.isnan(profit) or np.isinf(profit) or np.isnan(fee) or np.isinf(fee)):
                    self.balance += self.shares_held * current_price - fee
                    reward = np.clip(profit / self.initial_balance, -10.0, 10.0)
                    self.total_profit += profit
            
            elif self.position == -1 and self.shares_held < 0:
                profit = -self.shares_held * (self.cost_basis - current_price)
                fee = -self.shares_held * current_price * self.transaction_fee
                
                if not (np.isnan(profit) or np.isinf(profit) or np.isnan(fee) or np.isinf(fee)):
                    self.balance += profit - fee
                    reward = np.clip(profit / self.initial_balance, -10.0, 10.0)
                    self.total_profit += profit
            
            if np.isnan(self.balance) or np.isinf(self.balance) or self.balance <= 0:
                self.balance = self.initial_balance * 0.1
                reward = -1.0
            
        except (OverflowError, ValueError) as e:
            self.balance = self.initial_balance
            reward = -1.0
            # 🔥 ИЗМЕНЕНО: self.logger.warning -> print
            print(f"Переполнение при закрытии позиции: {e}")
        
        self.shares_held = 0
        self.cost_basis = 0
        self.position = 0
        
        return reward



📁 9. trade_manager.py
🔄 ИЗМЕНЯЕМ:
import numpy as np
import pandas as pd
from pybit.unified_trading import HTTP
import time
# import logging # 🔥 УДАЛЕНО: Импорт logging
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
        
        self.session = HTTP(
            testnet=(api_url == "https://api-testnet.bybit.com"),
            api_key=api_key,
            api_secret=api_secret
        )
        
        # 🔥 УДАЛЕНО: Инициализация логгера и файлового обработчика
        # self.logger = logging.getLogger('trade_manager')
        # self.logger.setLevel(logging.INFO)
        # if not self.logger.handlers:
        #     handler = logging.StreamHandler()
        #     formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        #     handler.setFormatter(formatter)
        #     self.logger.addHandler(handler)
            
        #     file_handler = logging.FileHandler('trading.log')
        #     file_handler.setFormatter(formatter)
        #     self.logger.addHandler(file_handler)
        
        self.trade_log = []
        self.position = 0
        
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
                # 🔥 ИЗМЕНЕНО: self.logger.info -> print
                print(f"Установлено плечо {self.leverage} для {self.symbol}")
            else:
                # 🔥 ИЗМЕНЕНО: self.logger.warning -> print
                print(f"Не удалось установить плечо: {response['retMsg']}")
        
        except Exception as e:
            # 🔥 ИЗМЕНЕНО: self.logger.error -> print
            print(f"Ошибка при установке плеча: {e}")
    
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
                # 🔥 ИЗМЕНЕНО: self.logger.debug -> print
                print(f"Текущая цена {self.symbol}: {price}")
                return price
            else:
                # 🔥 ИЗМЕНЕНО: self.logger.warning -> print
                print(f"Не удалось получить текущую цену: {response['retMsg']}")
                return None
        
        except Exception as e:
            # 🔥 ИЗМЕНЕНО: self.logger.error -> print
            print(f"Ошибка при получении текущей цены: {e}")
            return None
    
    def place_order(self, action):
        """
        Размещает ордер на бирже
        
        action: 0 - BUY, 1 - HOLD, 2 - SELL
        """
        if action == 1:
            return True
        
        try:
            current_price = self.get_current_price()
            
            if current_price is None:
                # 🔥 ИЗМЕНЕНО: self.logger.error -> print
                print("Не удалось получить текущую цену для размещения ордера")
                return False
            
            if action == 0:
                side = "Buy"
                if self.position == -1:
                    # 🔥 ИЗМЕНЕНО: self.logger.info -> print
                    print("Закрываем короткую позицию")
                    self._close_position()
            elif action == 2:
                side = "Sell"
                if self.position == 1:
                    # 🔥 ИЗМЕНЕНО: self.logger.info -> print
                    print("Закрываем длинную позицию")
                    self._close_position()
            else:
                # 🔥 ИЗМЕНЕНО: self.logger.error -> print
                print(f"Неизвестное действие: {action}")
                return False
            
            qty = self.order_amount / current_price
            
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
                # 🔥 ИЗМЕНЕНО: self.logger.info -> print
                print(f"Размещен {side} ордер на {qty} {self.symbol} по рыночной цене. ID: {order_id}")
                
                if action == 0:
                    self.position = 1
                elif action == 2:
                    self.position = -1
                
                self.trade_log.append({
                    'timestamp': datetime.now().isoformat(),
                    'action': 'BUY' if action == 0 else 'SELL',
                    'price': current_price,
                    'qty': qty,
                    'order_id': order_id
                })
                
                self._save_trade_log()
                
                return True
            else:
                # 🔥 ИЗМЕНЕНО: self.logger.error -> print
                print(f"Ошибка при размещении ордера: {response['retMsg']}")
                return False
        
        except Exception as e:
            # 🔥 ИЗМЕНЕНО: self.logger.error -> print
            print(f"Ошибка при размещении ордера: {e}")
            return False
    
    def _close_position(self):
        """
        Закрывает текущую позицию
        """
        try:
            response = self.session.get_positions(
                category="linear",
                symbol=self.symbol
            )
            
            if response['retCode'] != 0:
                # 🔥 ИЗМЕНЕНО: self.logger.error -> print
                print(f"Ошибка при получении информации о позиции: {response['retMsg']}")
                return False
            
            position_info = response['result']['list'][0]
            size = float(position_info['size'])
            
            if size == 0:
                # 🔥 ИЗМЕНЕНО: self.logger.info -> print
                print("Нет открытой позиции для закрытия")
                self.position = 0
                return True
            
            side = "Sell" if position_info['side'] == "Buy" else "Buy"
            
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
                # 🔥 ИЗМЕНЕНО: self.logger.info -> print
                print(f"Закрыта позиция {self.symbol}. ID ордера: {order_id}")
                self.position = 0
                
                self.trade_log.append({
                    'timestamp': datetime.now().isoformat(),
                    'action': 'CLOSE',
                    'price': self.get_current_price(),
                    'qty': size,
                    'order_id': order_id
                })
                
                self._save_trade_log()
                
                return True
            else:
                # 🔥 ИЗМЕНЕНО: self.logger.error -> print
                print(f"Ошибка при закрытии позиции: {response['retMsg']}")
                return False
        
        except Exception as e:
            # 🔥 ИЗМЕНЕНО: self.logger.error -> print
            print(f"Ошибка при закрытии позиции: {e}")
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
                # 🔥 ИЗМЕНЕНО: self.logger.warning -> print
                print(f"Не удалось получить информацию о позиции: {response['retMsg']}")
                return None
        
        except Exception as e:
            # 🔥 ИЗМЕНЕНО: self.logger.error -> print
            print(f"Ошибка при получении информации о позиции: {e}")
            return None
    
    def _save_trade_log(self):
        """
        Сохраняет журнал торговли в файл
        """
        try:
            with open('trade_log.json', 'w') as f:
                json.dump(self.trade_log, f, indent=2)
        except Exception as e:
            # 🔥 ИЗМЕНЕНО: self.logger.error -> print
            print(f"Ошибка при сохранении журнала торговли: {e}")



📁 10. run_live_trading.py
🔄 ИЗМЕНЯЕМ:
import os
import sys
import time
# import logging # 🔥 УДАЛЕНО: Импорт logging
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

# 🔥 УДАЛЕНО: Настройка логирования
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.StreamHandler(),
#         logging.FileHandler('live_trading.log')
#     ]
# )
# logger = logging.getLogger('live_trading')

def fetch_latest_data(session, symbol, timeframe, limit=100):
    """Получает последние свечи с биржи"""
    try:
        response = session.get_kline(
            category="linear",
            symbol=symbol,
            interval=timeframe,
            limit=limit
        )
        
        if response['retCode'] == 0:
            data = response['result']['list']
            
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
            ])
            
            numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'turnover']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df['timestamp'] = pd.to_datetime(pd.to_numeric(df['timestamp']), unit='ms')
            df['symbol'] = symbol
            
            df.sort_values('timestamp', inplace=True)
            
            return df
        else:
            # 🔥 ИЗМЕНЕНО: logger.error -> print
            print(f"Ошибка при получении данных: {response['retMsg']}")
            return None
    
    except Exception as e:
        # 🔥 ИЗМЕНЕНО: logger.error -> print
        print(f"Ошибка при получении данных: {e}")
        return None

def main():
    """Основная функция для запуска живой торговли"""
    # 🔥 ИЗМЕНЕНО: logger.info -> print
    print("🚀 ЗАПУСК СИСТЕМЫ ЖИВОЙ ТОРГОВЛИ С ТРЁХЭТАПНОЙ МОДЕЛЬЮ")
    
    api_key = config.BYBIT_API_KEY
    api_secret = config.BYBIT_API_SECRET
    api_url = config.API_URL
    symbol = config.SYMBOLS[0]
    timeframe = config.TIMEFRAME
    order_amount = config.ORDER_USDT_AMOUNT
    leverage = config.LEVERAGE
    sequence_length = config.SEQUENCE_LENGTH
    required_candles = config.REQUIRED_CANDLES
    
    session = HTTP(
        testnet=(api_url == "https://api-demo.bybit.com"),
        api_key=api_key,
        api_secret=api_secret
    )
    
    feature_engineering = FeatureEngineering(sequence_length=sequence_length)
    
    if not feature_engineering.load_scaler():
        # 🔥 ИЗМЕНЕНО: logger.error -> print
        print("❌ Не удалось загрузить скейлер. Убедитесь, что трёхэтапное обучение завершено.")
        return
    
    input_shape = (sequence_length, len(feature_engineering.feature_columns))
    rl_model = XLSTMRLModel(input_shape=input_shape, 
                          memory_size=config.XLSTM_MEMORY_SIZE, 
                          memory_units=config.XLSTM_MEMORY_UNITS)
    
    try:
        rl_model.load(stage="_rl_finetuned")
        # 🔥 ИЗМЕНЕНО: logger.info -> print
        print("✅ Финальная трёхэтапная модель успешно загружена")
    except Exception as e:
        # 🔥 ИЗМЕНЕНО: logger.error -> print
        print(f"❌ Не удалось загрузить финальную модель: {e}")
        # 🔥 ИЗМЕНЕНО: logger.info -> print
        print("Попытка загрузки supervised модели...")
        try:
            rl_model.load(stage="_supervised")
            # 🔥 ИЗМЕНЕНО: logger.info -> print
            print("✅ Supervised модель загружена как fallback")
        except Exception as e2:
            # 🔥 ИЗМЕНЕНО: logger.error -> print
            print(f"❌ Не удалось загрузить никакую модель: {e2}")
            return
    
    rl_agent = RLAgent(state_shape=input_shape, 
                      memory_size=config.XLSTM_MEMORY_SIZE, 
                      memory_units=config.XLSTM_MEMORY_UNITS)
    rl_agent.model = rl_model
    
    decision_maker = HybridDecisionMaker(rl_agent)
    
    trade_manager = TradeManager(
        api_key=api_key,
        api_secret=api_secret,
        api_url=api_url,
        order_amount=order_amount,
        symbol=symbol,
        leverage=leverage
    )
    
    # 🔥 ИЗМЕНЕНО: logger.info -> print
    print("✅ Система инициализирована, начинаем торговлю...")
    
    while True:
        try:
            current_time = datetime.now()
            
            df = fetch_latest_data(session, symbol, timeframe, limit=required_candles)
            
            if df is None or len(df) < sequence_length:
                # 🔥 ИЗМЕНЕНО: logger.error -> print
                print(f"❌ Недостаточно данных для анализа. Получено: {len(df) if df is not None else 0} строк")
                time.sleep(10)
                continue
            
            X, _, _ = feature_engineering.prepare_test_data(df)
            
            if len(X) == 0:
                # 🔥 ИЗМЕНЕНО: logger.error -> print
                print("❌ Не удалось подготовить данные для анализа")
                time.sleep(10)
                continue
            
            current_state = X[-1]
            
            action, confidence = decision_maker.make_decision(
                current_state,
                position=trade_manager.position
            )
            
            action_names = {0: "BUY", 1: "HOLD", 2: "SELL"}
            # 🔥 ИЗМЕНЕНО: logger.info -> print
            print(f"📊 Решение: {action_names[action]} (уверенность: {confidence:.4f})")
            
            explanation = decision_maker.explain_decision(current_state)
            # 🔥 ИЗМЕНЕНО: logger.info -> print
            print(f"🧠 Анализ: BUY={explanation['all_probs']['BUY']:.3f}, "
                       f"HOLD={explanation['all_probs']['HOLD']:.3f}, "
                       f"SELL={explanation['all_probs']['SELL']:.3f}, "
                       f"Value={explanation['state_value']:.4f}")
            
            if trade_manager.place_order(action):
                # 🔥 ИЗМЕНЕНО: logger.info -> print
                print(f"✅ Ордер размещен: {action_names[action]}")
            else:
                # 🔥 ИЗМЕНЕНО: logger.error -> print
                print(f"❌ Не удалось разместить ордер: {action_names[action]}")
            
            position_info = trade_manager.get_position_info()
            if position_info and position_info['size'] > 0:
                # 🔥 ИЗМЕНЕНО: logger.info -> print
                print(f"💰 Позиция: {position_info['side']} {position_info['size']}, "
                           f"PnL: {position_info['unrealised_pnl']}")
            
            time.sleep(30)
            
        except KeyboardInterrupt:
            # 🔥 ИЗМЕНЕНО: logger.info -> print
            print("⏹️ Торговля остановлена пользователем")
            break
        except Exception as e:
            # 🔥 ИЗМЕНЕНО: logger.error -> print
            print(f"❌ Ошибка в процессе торговли: {e}")
            time.sleep(10)

if __name__ == "__main__":
    main()


📁 11. validation_metrics_callback.py
🔄 ИЗМЕНЯЕМ:
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
# import logging # 🔥 УДАЛЕНО: Импорт logging

# 🔥 УДАЛЕНО: Инициализация логгера
# logger = logging.getLogger('validation_callback')

class ValidationMetricsCallback(tf.keras.callbacks.Callback):
    """
    Детальный мониторинг метрик валидации для Supervised Pre-training (Этап 1)
    """
    def __init__(self, X_val, y_val, class_names=['SELL', 'HOLD', 'BUY']):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val
        self.class_names = class_names
        
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 5 == 0:  # Каждые 5 эпох
            # 🔥 ИЗМЕНЕНО: logger.info -> print
            print(f"\n📊 Детальные метрики на эпохе {epoch+1}:")
            
            y_pred_probs = self.model.predict(self.X_val, verbose=0)
            y_pred_classes = np.argmax(y_pred_probs, axis=1)
            
            if self.y_val.ndim > 1 and self.y_val.shape[1] > 1:
                y_true_classes = np.argmax(self.y_val, axis=1)
            else:
                y_true_classes = self.y_val
            
            # Confusion Matrix
            cm = confusion_matrix(y_true_classes, y_pred_classes)
            # 🔥 ИЗМЕНЕНО: logger.info -> print
            print("Confusion Matrix:")
            
            header = "     " + " ".join([f"{name:4s}" for name in self.class_names])
            # 🔥 ИЗМЕНЕНО: logger.info -> print
            print(header)
            for i, row in enumerate(cm):
                row_str = " ".join([f"{val:4d}" for val in row])
                # 🔥 ИЗМЕНЕНО: logger.info -> print
                print(f"{self.class_names[i]:4s} {row_str}")
            
            # Classification Report
            report_dict = classification_report(
                y_true_classes, y_pred_classes, 
                target_names=self.class_names,
                output_dict=True,
                zero_division=0
            )
            
            # 🔥 ИЗМЕНЕНО: logger.info -> print
            print(f"Macro Avg F1-Score: {report_dict['macro avg']['f1-score']:.3f}")
            # 🔥 ИЗМЕНЕНО: logger.info -> print
            print(f"Weighted Avg F1-Score: {report_dict['weighted avg']['f1-score']:.3f}")
            
            # Распределение предсказаний
            pred_distribution = np.bincount(y_pred_classes, minlength=len(self.class_names)) / len(y_pred_classes)
            pred_dist_str = ", ".join([f"{name}={dist:.1%}" for name, dist in zip(self.class_names, pred_distribution)])
            # 🔥 ИЗМЕНЕНО: logger.info -> print
            print(f"Распределение предсказаний: {pred_dist_str}")

