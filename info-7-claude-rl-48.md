Проблема в том, что модель ожидает входные данные с формой (None, 60, 6), но получает данные с формой (1, 60, 7). Это происходит потому, что в TradingEnvironment в методе _get_observation() добавляется дополнительный канал с информацией о позиции, что увеличивает количество признаков с 6 до 7.
Нужно исправить несколько файлов:
1. Исправить trading_env.py:
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
        
        # Пространство наблюдений: только последовательность цен и объемов БЕЗ позиции
        # Используем исходную форму данных
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(self.sequence_length, data.shape[2])
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
        
        # Получаем наблюдение без добавления информации о позиции
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
        # Возвращаем только последовательность данных БЕЗ информации о позиции
        obs = self.data[self.current_step].copy()
        return obs
    
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

2. Обновить hybrid_decision_maker.py для передачи информации о позиции:
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
        self.logger.debug(f"Текущая позиция: {position}")
        
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

3. Обновить simulation_engine.py:
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
                # Принимаем решение (передаем текущую позицию)
                action, confidence = self.decision_maker.make_decision(
                    state, 
                    training=training, 
                    position=self.env.position
                )
                
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
            
            Продолжаю код для simulation_engine.py, где остановился:
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

4. Также нужно обновить run_live_trading.py для корректной работы:
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
            
            # Принимаем решение (передаем текущую позицию трейд-менеджера)
            action, confidence = decision_maker.make_decision(
                current_state, 
                position=trade_manager.position
            )
            
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

Основные изменения для исправления ошибки:

Убрал добавление канала позиции в TradingEnvironment._get_observation() - теперь возвращаются только исходные данные с 6 признаками
Обновил HybridDecisionMaker - добавил параметр position для передачи информации о текущей позиции
Обновил SimulationEngine - теперь передает информацию о позиции в make_decision
Обновил run_live_trading.py - передает позицию трейд-менеджера в принятие решений

Теперь модель будет получать входные данные правильной формы (None, 60, 6), а информация о позиции будет передаваться отдельно через параметры функций.
