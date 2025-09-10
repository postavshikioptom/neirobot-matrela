import gymnasium as gym
import numpy as np
import pandas as pd
from typing import Dict, Any

class TradingEnvRL(gym.Env):
    """
    Расширенная торговая среда для RL агента с VSA и xLSTM интеграцией
    """
    
    def __init__(self, df: pd.DataFrame, xlstm_model, initial_balance=10000, commission=0.0008):
        super(TradingEnvRL, self).__init__()
        
        self.df = df.copy()
        self.xlstm_model = xlstm_model
        self.initial_balance = initial_balance
        self.commission = commission
        
        # Пространство действий: 0=SELL, 1=BUY, 2=HOLD
        self.action_space = gym.spaces.Discrete(3)
        
        # Пространство наблюдений: xLSTM выход + VSA + портфель
        # xLSTM выход (3) + VSA признаки (7) + портфель (4) = 14 признаков
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32
        )
        
        self.reset()
    
    def _get_xlstm_prediction(self):
        """Получает предсказание от xLSTM модели"""
        if self.current_step < 10:  # Нужно минимум 10 свечей для последовательности
            return np.array([0.33, 0.33, 0.34])  # Равномерное распределение
        
        # Берем последние 10 свечей для xLSTM
        sequence_data = self.df.iloc[self.current_step-10:self.current_step]
        
        # Подготавливаем данные для модели (нужно адаптировать под ваши признаки)
        features = sequence_data[self.feature_columns].values
        features_reshaped = features.reshape(1, 10, len(self.feature_columns))
        
        return self.xlstm_model.predict(features_reshaped)[0]
    
    def _get_vsa_features(self):
        """Получает текущие VSA признаки"""
        if self.current_step >= len(self.df):
            return np.zeros(7)
            
        current_row = self.df.iloc[self.current_step]
        return np.array([
            current_row['vsa_no_demand'],
            current_row['vsa_no_supply'], 
            current_row['vsa_stopping_volume'],
            current_row['vsa_climactic_volume'],
            current_row['vsa_test'],
            current_row['vsa_effort_vs_result'],
            current_row['vsa_strength']
        ])
    
    def _get_portfolio_state(self):
        """Получает состояние портфеля"""
        return np.array([
            self.balance / self.initial_balance,  # Нормализованный баланс
            self.position,  # -1, 0, 1
            self.unrealized_pnl / self.initial_balance if self.position != 0 else 0,
            self.steps_in_position / 100.0  # Нормализованное время в позиции
        ])
    
    def _get_observation(self):
        """Формирует полное наблюдение для RL агента"""
        xlstm_pred = self._get_xlstm_prediction()  # 3 элемента
        vsa_features = self._get_vsa_features()    # 7 элементов  
        portfolio_state = self._get_portfolio_state()  # 4 элемента
        
        return np.concatenate([xlstm_pred, vsa_features, portfolio_state]).astype(np.float32)
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 10  # Начинаем с 10-й свечи для xLSTM
        self.balance = self.initial_balance
        self.position = 0  # -1=short, 0=flat, 1=long
        self.entry_price = 0
        self.unrealized_pnl = 0
        self.steps_in_position = 0
        
        # Определяем колонки признаков (адаптируйте под ваши данные)
        self.feature_columns = [
            'RSI_14', 'MACD_12_26_9', 'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0', 
            'ADX_14', 'STOCHk_14_3_3', 'STOCHd_14_3_3',
            'CDLHAMMER', 'CDLENGULFING', 'CDLDOJI', 'CDLSHOOTINGSTAR', 
            'CDLHANGINGMAN', 'CDLMARUBOZU',
            'vsa_no_demand', 'vsa_no_supply', 'vsa_stopping_volume', 
            'vsa_climactic_volume', 'vsa_test', 'vsa_effort_vs_result', 'vsa_strength'
        ]
        
        return self._get_observation(), {}
    
    def step(self, action):
        if self.current_step >= len(self.df) - 1:
            return self._get_observation(), 0, True, False, {}
            
        current_price = self.df['close'].iloc[self.current_step]
        reward = 0
        
        # Обновляем нереализованный PnL
        if self.position != 0:
            if self.position == 1:  # Long
                self.unrealized_pnl = (current_price - self.entry_price) / self.entry_price
            else:  # Short
                self.unrealized_pnl = (self.entry_price - current_price) / self.entry_price
            self.steps_in_position += 1
        
        # Выполняем действие
        if action == 0:  # SELL
            if self.position == 1:  # Закрываем long
                pnl = self.unrealized_pnl - (self.commission * 2)
                vsa_features = self._get_vsa_features()
                reward = self._calculate_advanced_reward(action, pnl * 100, vsa_features)
                self.balance *= (1 + pnl)
                self.position = 0
                self.steps_in_position = 0
                    
            elif self.position == 0:  # Открываем short
                self.position = -1
                self.entry_price = current_price
                self.steps_in_position = 0
                
        elif action == 1:  # BUY
            if self.position == -1:  # Закрываем short
                pnl = self.unrealized_pnl - (self.commission * 2)
                vsa_features = self._get_vsa_features()
                reward = self._calculate_advanced_reward(action, pnl * 100, vsa_features)
                self.balance *= (1 + pnl)
                self.position = 0
                self.steps_in_position = 0
                    
            elif self.position == 0:  # Открываем long
                self.position = 1
                self.entry_price = current_price
                self.steps_in_position = 0
                
        else:  # HOLD
            if self.position == 0:
                reward = -0.1  # Небольшой штраф за бездействие
            else:
                # Штраф за слишком долгое удержание позиции
                if self.steps_in_position > 50:
                    reward = -2
                else:
                    reward = self.unrealized_pnl * 10  # Поощряем прибыльные позиции
        
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        
        return self._get_observation(), reward, done, False, {}
    def _calculate_advanced_reward(self, action, pnl_pct, vsa_features):
        """
        Расширенная система наград с учетом качества сигналов
        """
        base_reward = pnl_pct if pnl_pct != 0 else 0
        
        # Бонусы за качественные VSA сигналы
        vsa_bonus = 0
        if action in [0, 1]:  # BUY или SELL
            if action == 1:  # BUY
                if vsa_features[1] > 0 or vsa_features[2] > 0:  # no_supply или stopping_volume
                    vsa_bonus = 3
            else:  # SELL
                if vsa_features[0] > 0 or vsa_features[3] > 0:  # no_demand или climactic_volume
                    vsa_bonus = 3
        
        # Штраф за противоречащие VSA сигналы
        vsa_penalty = 0
        if action == 1 and (vsa_features[0] > 0 or vsa_features[3] > 0):  # BUY при медвежьих VSA
            vsa_penalty = -5
        elif action == 0 and (vsa_features[1] > 0 or vsa_features[2] > 0):  # SELL при бычьих VSA
            vsa_penalty = -5
        
        # Бонус за скорость закрытия прибыльных позиций
        speed_bonus = 0
        if pnl_pct > 0 and self.steps_in_position < 20:
            speed_bonus = 2
        
        # Штраф за долгое удержание убыточных позиций
        hold_penalty = 0
        if pnl_pct < 0 and self.steps_in_position > 30:
            hold_penalty = -3
        
        total_reward = base_reward + vsa_bonus + vsa_penalty + speed_bonus + hold_penalty
        
        return total_reward