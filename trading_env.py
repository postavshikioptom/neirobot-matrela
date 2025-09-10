import gymnasium as gym
import numpy as np
import pandas as pd

class TradingEnv(gym.Env):
    """
    Кастомная среда для обучения RL-агента фьючерсной торговле.
    Action Space:
        0: Sell / Short
        1: Buy / Long
        2: Hold / Do Nothing
    """
    def __init__(self, df, initial_balance=10000, commission=0.0008):
        super(TradingEnv, self).__init__()

        self.df = df.copy()
        self.initial_balance = initial_balance
        self.commission = commission
        self.trade_counter = 0

        # Actions: 0: Sell, 1: Buy, 2: Hold
        self.action_space = gym.spaces.Discrete(3)

        # Observations: индикаторы + состояние портфеля (баланс, тип позиции)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(df.columns) + 2,), dtype=np.float32
        )

        self.reset()

    def _get_observation(self):
        obs = self.df.iloc[self.current_step].values
        # Нормализуем баланс и позицию для лучшего обучения
        norm_balance = self.balance / self.initial_balance
        portfolio_status = np.array([norm_balance, self.position])
        return np.concatenate([obs, portfolio_status]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0  # -1 for short, 0 for flat, 1 for long
        self.entry_price = 0
        self.last_trade_step = 0 # Шаг, на котором была последняя сделка
        return self._get_observation(), {}

    def step(self, action):
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        current_price = self.df['close'].iloc[self.current_step]
        reward = 0

        # Action 2: Hold
        if action == 2:
            # Усиленный штраф за бездействие для мотивации к активной торговле
            if self.position == 0:
                reward = -0.5  # Увеличенный штраф за HOLD без позиции
            else:
                reward = -0.1  # Небольшой штраф за HOLD с открытой позицией

        # Action 1: Buy / Go Long
        elif action == 1:
            if self.position == -1: # Закрываем шорт
                profit_percent = ((self.entry_price - current_price) / self.entry_price) - (self.commission * 2)
                reward = profit_percent * 100 # Награда/штраф в процентах
                self.balance *= (1 + profit_percent)
                self.position = 0
                self.trade_counter += 1
            
            if self.position == 0: # Открываем лонг
                self.position = 1
                self.entry_price = current_price
                self.last_trade_step = self.current_step

        # Action 0: Sell / Go Short
        elif action == 0:
            if self.position == 1: # Закрываем лонг
                profit_percent = ((current_price - self.entry_price) / self.entry_price) - (self.commission * 2)
                reward = profit_percent * 100
                self.balance *= (1 + profit_percent)
                self.position = 0
                self.trade_counter += 1

            if self.position == 0: # Открываем шорт
                self.position = -1
                self.entry_price = current_price
                self.last_trade_step = self.current_step

        if done:
            # Принудительно закрываем позицию в конце данных
            if self.position == 1:
                profit_percent = ((current_price - self.entry_price) / self.entry_price) - (self.commission * 2)
                reward = profit_percent * 100
                self.balance *= (1 + profit_percent)
            elif self.position == -1:
                profit_percent = ((self.entry_price - current_price) / self.entry_price) - (self.commission * 2)
                reward = profit_percent * 100
                self.balance *= (1 + profit_percent)

        return self._get_observation(), reward, done, False, {}


class SimpleTradingEnv(gym.Env):
    """
    Упрощенная среда для новой модели с фиксированным набором признаков.
    Action Space:
        0: Sell / Short
        1: Buy / Long
        2: Hold / Do Nothing
    """
    def __init__(self, df):
        super(SimpleTradingEnv, self).__init__()

        self.df = df.copy()

        # Actions: 0: Sell, 1: Buy, 2: Hold
        self.action_space = gym.spaces.Discrete(3)

        # Observations: фиксированные 34 признаков (28 оригинальных + 6 новых)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(34,), dtype=np.float32
        )

        self.reset()

    def _get_observation(self):
        return self.df.iloc[self.current_step].values.astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        return self._get_observation(), {}

    def step(self, action):
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        # В упрощенной среде награда всегда 0, так как обучение происходит на реальных сделках
        reward = 0
        return self._get_observation(), reward, done, False, {}