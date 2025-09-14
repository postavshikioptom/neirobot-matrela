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
        
        # Пространство наблюдений: xLSTM выход + портфель
        # xLSTM выход (3) + портфель (4) = 7 признаков
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32
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
        portfolio_state = self._get_portfolio_state()  # 4 элемента
        
        return np.concatenate([xlstm_pred, portfolio_state]).astype(np.float32)
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 10  # Начинаем с 10-й свечи для xLSTM
        self.balance = self.initial_balance
        self.position = 0  # -1=short, 0=flat, 1=long
        self.entry_price = 0
        self.unrealized_pnl = 0
        self.steps_in_position = 0
        
        # 🔥 НОВЫЕ FEATURE_COLUMNS - ТОЛЬКО ИНДИКАТОРЫ (для RL среды)
        self.feature_columns = [
            # ✅ ВСЕ ТЕХНИЧЕСКИЕ ИНДИКАТОРЫ (БЕЗ БОЛЛИНДЖЕРА И ATR_14)
            'RSI_14', 'MACD_12_26_9', 'MACD_signal', 'MACD_hist',
            'ADX_14', 'STOCHk_14_3_3', 'STOCHd_14_3_3',
            'WILLR_14', # 🔥 НОВЫЙ ИНДИКАТОР
            'AO_5_34',  # 🔥 НОВЫЙ ИНДИКАТОР
            
            # ❌ ВСЕ ПАТТЕРНЫ ЗАКОММЕНТИРОВАНЫ
            # 'CDLHAMMER', 'CDLENGULFING', 'CDLDOJI', 'CDLSHOOTINGSTAR',
            # 'CDLHANGINGMAN', 'CDLMARUBOZU',
            # 'CDLINVERTEDHAMMER', 'CDLDRAGONFLYDOJI', 'CDLBELTHOLD',
            # 'hammer_f', 'hangingman_f', 'engulfing_f', 'doji_f',
            # 'shootingstar_f', 'bullish_marubozu_f',
            # 'inverted_hammer_f', 'dragonfly_doji_f', 'bullish_pin_bar_f', 'bullish_belt_hold_f',
            
            # ✅ ОСТАВЛЯЕМ EVENT SAMPLING
            'is_event'
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
                xlstm_pred_for_reward = self._get_xlstm_prediction()
                reward = self._calculate_advanced_reward(action, pnl * 100, xlstm_pred_for_reward)
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
                xlstm_pred_for_reward = self._get_xlstm_prediction()
                reward = self._calculate_advanced_reward(action, pnl * 100, xlstm_pred_for_reward)
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
    def _calculate_advanced_reward(self, action, pnl_pct, xlstm_prediction):
        """
        Advanced reward system for RL agent with VSA and xLSTM integration.
        """
        base_reward = pnl_pct if pnl_pct != 0 else 0
        
        # Бонус за скорость закрытия прибыльных позиций
        speed_bonus = 0
        if pnl_pct > 0 and self.steps_in_position < 20:
            speed_bonus = 2

        # Штраф за долгое удержание убыточных позиций
        hold_penalty = 0
        if pnl_pct < 0 and self.steps_in_position > 30:
            hold_penalty = -3

        # Бонус за уверенность xLSTM
        xlstm_conf = np.max(xlstm_prediction)
        if xlstm_conf > 0.7:
            base_reward += xlstm_conf * 2

        # Штраф за противоречие xLSTM
        predicted_action_idx = np.argmax(xlstm_prediction)
        xlstm_to_rl_map = {0: 1, 1: 0, 2: 2}  # xLSTM_BUY->RL_BUY, xLSTM_SELL->RL_SELL
        
        if action != 2 and action != xlstm_to_rl_map.get(predicted_action_idx):
            base_reward -= 1

        # Штраф за отклонение от баланса (риск-менеджмент)
        if self.balance < self.initial_balance * 0.9:
            base_reward -= 5

        # СКОРРЕКТИРОВАННЫЙ БОНУС ЗА ИССЛЕДОВАНИЕ И ЭНТРОПИЮ
        exploration_bonus = 0
        if action in [0, 1]:
            exploration_bonus = 0.2
        
        entropy_bonus = 0
        entropy = -np.sum(xlstm_prediction * np.log(xlstm_prediction + 1e-10))
        normalized_entropy = entropy / np.log(len(xlstm_prediction))
        entropy_bonus = normalized_entropy * 0.2

        # НОВЫЙ КОД - Корректируем функцию наград для RL (более сбалансированное вознаграждение, с акцентом на HOLD)
        hold_reward = 0
        overtrading_penalty = 0

        current_row = self.df.iloc[self.current_step]
        # Используем индикаторы для определения "явного сигнала"
        buy_signal_strength = (
            (current_row.get('RSI_14', 50) < 30) +
            (current_row.get('ADX_14', 0) > 25) +
            (current_row.get('MACD_hist', 0) > 0.001) +
            (current_row.get('WILLR_14', -50) < -80) + # 🔥 НОВОЕ: WILLR_14 для BUY (сильно перепродано)
            (current_row.get('AO_5_34', 0) > 0) # 🔥 НОВОЕ: AO выше нуля
        )
        sell_signal_strength = (
            (current_row.get('RSI_14', 50) > 70) +
            (current_row.get('ADX_14', 0) > 25) +
            (current_row.get('MACD_hist', 0) < -0.001) +
            (current_row.get('WILLR_14', -50) > -20) + # 🔥 НОВОЕ: WILLR_14 для SELL (сильно перекуплено)
            (current_row.get('AO_5_34', 0) < 0) # 🔥 НОВОЕ: AO ниже нуля
        )

        if action == 2: # HOLD
            # 🔥 ИЗМЕНЕНО: Использование AO_5_34 и ADX_14 для HOLD reward
            ao_value = current_row.get('AO_5_34', 0)
            adx = current_row.get('ADX_14', 0)

            # Если моментум низкий (AO близко к 0) и ADX низкий (флэт)
            if abs(ao_value) < 0.001 and adx < 20: # Пороги нужно будет подобрать
                hold_reward = 0.5
            # Если сильный моментум (большой AO) или сильный тренд (большой ADX)
            elif abs(ao_value) > 0.005 or adx > 30:
                hold_reward = -0.5
            else:
                hold_reward = 0.1
            
            if pnl_pct < 0 and self.steps_in_position > 30:
                hold_penalty = -3
            
            # Добавляем бонус за HOLD, если нет сильных сигналов
            if buy_signal_strength < 1 and sell_signal_strength < 1:
                hold_reward += 1.0
            else:
                hold_reward -= 1.0

        else: # Если действие BUY или SELL (не HOLD)
            # Штраф за overtrading (слишком частые сделки, когда нет явного сигнала)
            # Увеличиваем штраф за слабые BUY-сигналы, если RL предсказывает BUY
            if action == 1 and buy_signal_strength < 2:
                overtrading_penalty = -1.0
            # Увеличиваем штраф за слабые SELL-сигналы, если RL предсказывает SELL
            elif action == 0 and sell_signal_strength < 2:
                overtrading_penalty = -1.0

        total_reward = base_reward + speed_bonus + hold_penalty + exploration_bonus + entropy_bonus + hold_reward + overtrading_penalty
        
        return total_reward