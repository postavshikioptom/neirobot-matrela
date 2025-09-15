
❌ Что нужно исправить:
1. Отсутствует импорт logging
В методе _close_position используется self.logger, но логгер не инициализирован:
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import logging  # ДОБАВИТЬ ЭТУ СТРОКУ

class TradingEnvironment(gym.Env):
    def __init__(self, data, sequence_length=60, initial_balance=10000, transaction_fee=0.001):
        super(TradingEnvironment, self).__init__()
        
        # ... существующий код ...
        
        # ДОБАВИТЬ ИНИЦИАЛИЗАЦИЮ ЛОГГЕРА
        self.logger = logging.getLogger('trading_env')
        
        # Сбрасываем среду
        self.reset()

2. Исправить отступы в коде
Есть лишний отступ после if self.position == -1::
if action == 0:  # BUY
    if self.position != 1 and self.balance > 0:
        if self.position == -1:
            reward += self._close_position(current_price)
        
        # УБРАТЬ ЛИШНИЙ ОТСТУП ЗДЕСЬ
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

3. Добавить дополнительные проверки безопасности
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

4. Улучшить проверку данных в step()
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
    
    # ... остальной код остается без изменений ...

📝 Исправленная версия:
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import logging

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
        
        # Инициализация логгера
        self.logger = logging.getLogger('trading_env')
        
        # Определяем пространство действий: 0 - BUY, 1 - HOLD, 2 - SELL
        self.action_space = spaces.Discrete(3)
        
        # Пространство наблюдений: только последовательность цен и объемов БЕЗ позиции
        # Используем исходную форму данных
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.sequence_length, data.shape[2])
        )
        
        # Сбрасываем среду
        self.reset()
    
    # ... остальные методы остаются такими же, как вы написали, но с исправленными отступами ...

