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
            'ATR_14', # <--- ДОБАВЛЕНО
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
                xlstm_pred_for_reward = self._get_xlstm_prediction()
                reward = self._calculate_advanced_reward(action, pnl * 100, vsa_features, xlstm_pred_for_reward)
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
                xlstm_pred_for_reward = self._get_xlstm_prediction()
                reward = self._calculate_advanced_reward(action, pnl * 100, vsa_features, xlstm_pred_for_reward)
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
    def _calculate_advanced_reward(self, action, pnl_pct, vsa_features, xlstm_prediction):
        """
        Расширенная система наград с учетом качества сигналов
        """
        base_reward = pnl_pct if pnl_pct != 0 else 0
        
        # Бонусы за качественные VSA сигналы (ОСЛАБЛЕНЫ ПОРОГИ)
        vsa_bonus = 0
        if action in [0, 1]: # SELL или BUY
            # BUY (действие 1): если есть no_supply (vsa_features[1]) или stopping_volume (vsa_features[2])
            if action == 1 and (vsa_features[1] > 0 or vsa_features[2] > 0 or vsa_features[6] > 0.2): # Добавлено: vsa_strength > 0.2
                vsa_bonus = 2 # СНИЖЕНО с 3 до 2, чтобы не перевешивать PnL
            # SELL (действие 0): если есть no_demand (vsa_features[0]) или climactic_volume (vsa_features[3])
            elif action == 0 and (vsa_features[0] > 0 or vsa_features[3] > 0 or vsa_features[6] < -0.2): # Добавлено: vsa_strength < -0.2
                vsa_bonus = 2 # СНИЖЕНО с 3 до 2

        # Штраф за противоречащие VSA сигналы (ОСЛАБЛЕНЫ ПОРОГИ)
        vsa_penalty = 0
        if action == 1 and (vsa_features[0] > 0 or vsa_features[3] > 0 or vsa_features[6] < -0.5): # Усилен порог для penalization
            vsa_penalty = -3 # СНИЖЕНО с -5 до -3
        elif action == 0 and (vsa_features[1] > 0 or vsa_features[2] > 0 or vsa_features[6] > 0.5): # Усилен порог для penalization
            vsa_penalty = -3 # СНИЖЕНО с -5 до -3

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

        # =====================================================================
        # НОВЫЙ БЛОК: СКОРРЕКТИРОВАННЫЙ БОНУС ЗА ИССЛЕДОВАНИЕ И ЭНТРОПИЮ
        # =====================================================================
        exploration_bonus = 0
        # Меньший, но все еще стимулирующий бонус
        if action in [0, 1]: # Если действие - BUY или SELL
            exploration_bonus = 0.2 # <--- ИЗМЕНЕНО с 0.5 на 0.2
        
        entropy_bonus = 0
        # Ослабляем бонус за энтропию
        entropy = -np.sum(xlstm_prediction * np.log(xlstm_prediction + 1e-10))
        normalized_entropy = entropy / np.log(len(xlstm_prediction))
        entropy_bonus = normalized_entropy * 0.2 # <--- ИЗМЕНЕНО с 0.5 на 0.2
        # =====================================================================
        # КОНЕЦ НОВОГО БЛОКА
        # =====================================================================

        # =====================================================================
        # НОВЫЙ БЛОК: ЯВНОЕ ВОЗНАГРАЖДЕНИЕ ЗА HOLD И ШТРАФ ЗА OVERTRADING
        # =====================================================================
        hold_reward = 0
        overtrading_penalty = 0

        # Если действие HOLD
        if action == 2: # HOLD
            # Вознаграждаем за HOLD, если рынок действительно находится в консолидации
            # (например, низкая волатильность, нет сильного тренда)
            current_row = self.df.iloc[self.current_step]
            volatility = current_row.get('ATR_14', 0) / current_row.get('close', 1) # Нормализованная волатильность
            adx = current_row.get('ADX_14', 0)

            if volatility < 0.005 and adx < 25: # Низкая волатильность и слабый тренд
                hold_reward = 0.5 # Небольшой бонус за правильный HOLD
            elif volatility > 0.01 and adx > 30: # Высокая волатильность и сильный тренд - HOLD менее желателен
                hold_reward = -0.5 # Небольшой штраф за HOLD в тренде
            else:
                hold_reward = 0.1 # Небольшой нейтральный бонус за HOLD
            
            # Штраф за слишком долгое удержание позиции (если она убыточна)
            if pnl_pct < 0 and self.steps_in_position > 30:
                hold_penalty = -3 # Уже есть, но убедимся, что он применяется к HOLD
            
        else: # Если действие BUY или SELL (не HOLD)
            # Штраф за overtrading (слишком частые сделки, когда нет явного сигнала)
            # Используем VSA-скор для определения "явного сигнала"
            current_row = self.df.iloc[self.current_step]
            vsa_buy_score = (0.3 * (current_row.get('vsa_no_supply', 0) == 1) + 0.3 * (current_row.get('vsa_stopping_volume', 0) == 1) + 0.4 * (current_row.get('vsa_strength', 0) > 0.1))
            vsa_sell_score = (0.3 * (current_row.get('vsa_no_demand', 0) == 1) + 0.3 * (current_row.get('vsa_climactic_volume', 0) == 1) + 0.4 * (current_row.get('vsa_strength', 0) < -0.1))

            if action == 1 and vsa_buy_score < 0.4: # Если BUY, но VSA-скор низкий
                overtrading_penalty = -1.0
            elif action == 0 and vsa_sell_score < 0.4: # Если SELL, но VSA-скор низкий
                overtrading_penalty = -1.0
        # =====================================================================
        # КОНЕЦ НОВОГО БЛОКА
        # =====================================================================

        total_reward = base_reward + vsa_bonus + vsa_penalty + speed_bonus + hold_penalty + exploration_bonus + entropy_bonus + hold_reward + overtrading_penalty # <--- ДОБАВЛЕНО: hold_reward, overtrading_penalty
        
        return total_reward