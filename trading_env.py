import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import random
from collections import deque
import gc
# import logging # 🔥 УДАЛЕНО: Импорт logging

class TradingEnvironment(gym.Env):
    """
    Среда для обучения торгового агента с помощью RL
    Обновленная версия для работы с данными, сгруппированными по символам
    """
    def __init__(self, data_by_symbol, sequence_length=60, initial_balance=10000, transaction_fee=0.001, max_memory_size=1000):
        super(TradingEnvironment, self).__init__()
        
        # 🔥 ИЗМЕНЕНО: Теперь данные организованы по символам
        self.data_by_symbol = data_by_symbol  # Словарь: symbol -> массив последовательностей
        self.symbols = list(data_by_symbol.keys())
        self.sequence_length = sequence_length
        self.initial_balance = initial_balance
        self.transaction_fee = transaction_fee
        # 🔥 ИСПРАВЛЕНО: Используем deque вместо списка
        self.memory_buffer = deque(maxlen=max_memory_size)  # Автоматическое ограничение размера
        self.step_count = 0  # 🔥 ДОБАВЛЕНО: Счетчик шагов
        
        # Выбираем случайный символ для начала
        self.current_symbol = random.choice(self.symbols) if self.symbols else None
        self.current_data = self.data_by_symbol[self.current_symbol] if self.current_symbol else None
        
        # Определяем пространство действий: 0 - BUY, 1 - HOLD, 2 - SELL
        self.action_space = spaces.Discrete(3)
        
        # Пространство наблюдений: только последовательность цен и объемов БЕЗ позиции
        # Используем исходную форму данных (берем из первого символа)
        if self.current_data is not None and len(self.current_data) > 0:
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.sequence_length, self.current_data.shape[2])
            )
        else:
            # Запасной вариант, если данных нет
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.sequence_length, 10)  # 10 признаков по умолчанию
            )
        
        # 🔥 УДАЛЕНО: Инициализация логгера
        # self.logger = logging.getLogger('trading_env')
        
        # Сбрасываем среду
        self.reset()
    
    def reset(self, seed=None):
        """Сбрасывает среду в начальное состояние"""
        super().reset(seed=seed)
        
        # 🔥 ДОБАВЛЕНО: Проверка на пустые символы
        if not self.symbols or len(self.symbols) == 0:
            print("❌ Нет доступных символов для торговли")
            # Создаем dummy данные для предотвращения краха
            dummy_shape = (self.sequence_length, 10)  # 10 признаков по умолчанию
            observation = np.zeros(dummy_shape, dtype=np.float32)
            return observation, {}
        
        # Выбираем случайный символ при каждом сбросе
        try:
            self.current_symbol = random.choice(self.symbols)
            self.current_data = self.data_by_symbol[self.current_symbol]
        except (KeyError, IndexError) as e:
            print(f"❌ Ошибка при выборе символа: {e}")
            # Fallback к первому доступному символу
            if self.symbols:
                self.current_symbol = self.symbols[0]
                self.current_data = self.data_by_symbol.get(self.current_symbol, None)
        
        # 🔥 ДОБАВЛЕНО: Проверка на корректность данных
        if self.current_data is None or len(self.current_data) == 0:
            print(f"❌ Нет данных для символа {self.current_symbol}")
            dummy_shape = (self.sequence_length, 10)
            observation = np.zeros(dummy_shape, dtype=np.float32)
            return observation, {}
        
        # Случайный старт внутри данных символа
        if self.current_data is not None and len(self.current_data) > self.sequence_length:
            max_start = len(self.current_data) - self.sequence_length
            self.start_index = random.randint(0, max_start)
            self.current_step = self.start_index
        else:
            self.start_index = 0
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
        if self.current_data is None or self.current_step >= len(self.current_data) - 1:
            # Эпизод уже закончен
            observation = self._get_observation() if self.current_data is not None and self.current_step < len(self.current_data) else np.zeros(self.observation_space.shape)
            info = {
                'balance': float(self.balance),
                'position': self.position,
                'shares_held': float(self.shares_held),
                'total_trades': self.total_trades,
                'total_profit': float(self.total_profit),
                'portfolio_value': float(self.balance),
                'symbol': self.current_symbol  # 🔥 ДОБАВЛЕНО: Информация о текущем символе
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
        done = self.current_step >= len(self.current_data) - 1
        
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
        
        # 🔥 ИСПРАВЛЕНО: Эффективное добавление в deque
        self.memory_buffer.append({
            'state': observation,
            'action': action,
            'reward': reward,
            'done': done,
            'step': self.step_count
        })
        
        self.step_count += 1
        
        # 🔥 ИСПРАВЛЕНО: Менее частая очистка памяти
        if self.step_count % 500 == 0:  # Увеличена частота с 100 до 500
            gc.collect()
            print(f"Очистка памяти после {self.step_count} шагов, размер буфера: {len(self.memory_buffer)}")
        
        # Дополнительная информация
        info = {
            'balance': float(self.balance) if not (np.isnan(self.balance) or np.isinf(self.balance)) else float(self.initial_balance),
            'position': self.position,
            'shares_held': float(self.shares_held) if not (np.isnan(self.shares_held) or np.isinf(self.shares_held)) else 0.0,
            'total_trades': self.total_trades,
            'total_profit': float(self.total_profit) if not (np.isnan(self.total_profit) or np.isinf(self.total_profit)) else 0.0,
            'portfolio_value': float(portfolio_value),
            'symbol': self.current_symbol  # 🔥 ДОБАВЛЕНО: Информация о текущем символе
        }
        
        return observation, reward, done, False, info

    def _get_observation(self):
        """Возвращает текущее наблюдение"""
        # 🔥 ИЗМЕНЕНО: Возвращаем случайный отрезок последовательности вместо фиксированного индекса
        if self.current_data is None or len(self.current_data) == 0:
            return np.zeros(self.observation_space.shape)
        
        # Убедимся, что у нас есть достаточно данных
        if len(self.current_data) < self.sequence_length:
            # Если данных меньше, чем нужно, заполняем нулями
            padding = np.zeros((self.sequence_length - len(self.current_data),) + self.current_data.shape[1:])
            return np.concatenate([self.current_data, padding], axis=0)
        
        # Возвращаем последовательность данных БЕЗ информации о позиции
        obs = self.current_data[self.current_step].copy()
        return obs
    
    def _get_current_price(self):
        """Возвращает текущую цену закрытия"""
        try:
            if self.current_data is None or len(self.current_data) == 0:
                return 100.0  # Базовая цена
            
            price = self.current_data[self.current_step][-1, 3]  # индекс 3 - это 'close'
            
            # Проверяем на корректность цены
            if np.isnan(price) or np.isinf(price) or price <= 0:
                # Возвращаем предыдущую цену или базовую цену
                if self.current_step > 0 and self.current_data is not None:
                    return self.current_data[self.current_step-1][-1, 3]
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

    def get_memory_stats(self):
        """🔥 ДОБАВЛЕНО: Получение статистики памяти"""
        return {
            'buffer_size': len(self.memory_buffer),
            'max_size': self.memory_buffer.maxlen,
            'step_count': self.step_count,
            'memory_usage_percent': len(self.memory_buffer) / self.memory_buffer.maxlen * 100
        }
    
    def clear_old_memory(self, keep_last_n=100):
        """🔥 ДОБАВЛЕНО: Принудительная очистка старых данных"""
        if len(self.memory_buffer) > keep_last_n:
            # Сохраняем только последние N записей
            recent_data = list(self.memory_buffer)[-keep_last_n:]
            self.memory_buffer.clear()
            self.memory_buffer.extend(recent_data)
            gc.collect()
            print(f"Принудительная очистка памяти, оставлено {len(self.memory_buffer)} записей")