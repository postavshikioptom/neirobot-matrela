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
            shape=(self.sequence_length, self.data.shape[2])
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
        
        # Добавляем информацию о позиции к наблюдению
        observation = self._get_observation()
        
        return observation, {}
    
    def step(self, action):
        """Выполняет шаг в среде"""
        # Проверяем корректность шага
        if self.current_step >= len(self.data) - 1:
            # Эпизод уже закончен
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
        
        # Получаем текущую цену закрытия
        current_price = self._get_current_price()
        
        # Ограничиваем размер позиции для избежания переполнения
        max_position_size = self.initial_balance * 0.95  # Максимум 95% от начального баланса
        
        reward = 0
        if action == 0:  # BUY
            if self.position != 1 and self.balance > 0:
                if self.position == -1:
                    reward += self._close_position(current_price)
                
                # Ограничиваем размер покупки
                max_shares = min(max_position_size / current_price, self.balance / (current_price * (1 + self.transaction_fee)))
                shares_to_buy = max_shares * 0.9  # Используем только 90% доступных средств
                
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
                
                # Ограничиваем размер продажи
                max_shares = min(max_position_size / current_price, self.balance / current_price)
                shares_to_sell = max_shares * 0.9  # Используем только 90% доступных средств
                
                if shares_to_sell > 0:
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
        
        # Рассчитываем стоимость портфеля с проверкой на переполнение
        portfolio_value = self.balance
        
        # Проверяем на NaN и бесконечность
        if np.isnan(self.balance) or np.isinf(self.balance):
            self.balance = self.initial_balance
            portfolio_value = self.initial_balance
            reward = -1.0  # Штраф за переполнение
        
        if self.position == 1 and self.shares_held > 0:
            position_value = self.shares_held * current_price
            if not (np.isnan(position_value) or np.isinf(position_value)):
                portfolio_value += position_value
        elif self.position == -1 and self.shares_held < 0:
            position_value = -self.shares_held * current_price
            if not (np.isnan(position_value) or np.isinf(position_value)):
                portfolio_value -= position_value
        
        # Ограничиваем портфель разумными значениями
        if portfolio_value > self.initial_balance * 1000:  # Максимум в 1000 раз больше начального
            portfolio_value = self.initial_balance * 1000
            reward = -0.5  # Штраф за слишком большой рост
        elif portfolio_value < self.initial_balance * 0.001:  # Минимум 0.1% от начального
            portfolio_value = self.initial_balance * 0.001
            reward = -0.5  # Штраф за большие потери
        
        # Дополнительная информация
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
        # Возвращаем только последовательность данных БЕЗ информации о позиции
        obs = self.data[self.current_step].copy()
        return obs
    
    def _get_current_price(self):
        """Возвращает текущую цену закрытия"""
        try:
            price = self.data[self.current_step][-1, 3]  # индекс 3 - это 'close'
            
            # Проверяем на корректность цены
            if np.isnan(price) or np.isinf(price) or price <= 0:
                # Возвращаем предыдущую цену или базовую цену
                if self.current_step > 0:
                    return self.data[self.current_step-1][-1, 3]
                else:
                    return 100.0  # Базовая цена
            
            return price
        except (IndexError, ValueError):
            return 100.0  # Базовая цена в случае ошибки
    
    def _close_position(self, current_price):
        """Закрывает текущую позицию и возвращает полученную награду"""
        reward = 0
        
        try:
            if self.position == 1 and self.shares_held > 0:  # Закрываем длинную позицию
                profit = self.shares_held * (current_price - self.cost_basis)
                fee = self.shares_held * current_price * self.transaction_fee
                
                # Проверяем на переполнение
                if not (np.isnan(profit) or np.isinf(profit) or np.isnan(fee) or np.isinf(fee)):
                    self.balance += self.shares_held * current_price - fee
                    reward = np.clip(profit / self.initial_balance, -10.0, 10.0)  # Ограничиваем награду
                    self.total_profit += profit
            
            elif self.position == -1 and self.shares_held < 0:  # Закрываем короткую позицию
                profit = -self.shares_held * (self.cost_basis - current_price)
                fee = -self.shares_held * current_price * self.transaction_fee
                
                # Проверяем на переполнение
                if not (np.isnan(profit) or np.isinf(profit) or np.isnan(fee) or np.isinf(fee)):
                    self.balance += profit - fee
                    reward = np.clip(profit / self.initial_balance, -10.0, 10.0)  # Ограничиваем награду
                    self.total_profit += profit
            
            # Проверяем баланс на разумность
            if np.isnan(self.balance) or np.isinf(self.balance) or self.balance <= 0:
                self.balance = self.initial_balance * 0.1  # Минимальный баланс
                reward = -1.0  # Штраф за банкротство
            
        except (OverflowError, ValueError) as e:
            # В случае переполнения сбрасываем к безопасным значениям
            self.balance = self.initial_balance
            reward = -1.0
            # 🔥 ИЗМЕНЕНО: self.logger.warning -> print
            print(f"Переполнение при закрытии позиции: {e}")
        
        # Сбрасываем позицию
        self.shares_held = 0
        self.cost_basis = 0
        self.position = 0
        
        return reward