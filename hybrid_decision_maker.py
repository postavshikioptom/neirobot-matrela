import numpy as np
import pandas as pd
from models.xlstm_rl_model import XLSTMRLModel
from rl_agent import IntelligentRLAgent
from market_regime_detector import MarketRegimeDetector

class HybridDecisionMaker:
    """
    Гибридный принимающий решения с адаптацией к рыночным режимам
    """
    
    def __init__(self, xlstm_model_path, rl_agent_path, feature_columns, sequence_length):
        self.sequence_length = sequence_length
        self.xlstm_model = XLSTMRLModel(input_shape=(self.sequence_length, len(feature_columns)))
        self.xlstm_model.load_model(xlstm_model_path, 'models/xlstm_rl_scaler.pkl')
        
        self.rl_agent = IntelligentRLAgent()
        if rl_agent_path: # <--- ДОБАВЛЕНО: Проверяем, существует ли путь
            self.rl_agent.load_agent(rl_agent_path)
        else:
            print("⚠️ RL агент не загружен, так как путь не указан (возможно, еще не обучен).")
        
        self.feature_columns = feature_columns
        self.decision_history = []
        
        # Для отслеживания состояния
        self.current_position = 0
        self.current_balance = 10000
        self.steps_in_position = 0
        
        # Добавляем детектор режимов
        self.regime_detector = MarketRegimeDetector()
        self.current_regime = 'UNKNOWN'
        self.regime_confidence = 0.0
        
    def fit_regime_detector(self, historical_df, xlstm_model, xlstm_feature_columns):
        """Обучает детектор режимов на исторических данных"""
        try:
            self.regime_detector.set_xlstm_context(xlstm_model, xlstm_feature_columns)
            self.regime_detector.fit(historical_df)
            print("✅ Детектор рыночных режимов обучен")
        except Exception as e:
            print(f"❌ Ошибка обучения детектора режимов: {e}")
    
    def get_decision(self, df_sequence, confidence_threshold=0.6):
        """
        Принимает решение с адаптивным порогом
        """
        if len(df_sequence) < self.sequence_length:
            return 'HOLD'
            
        try:
            # Вычисляем адаптивный порог
            adaptive_threshold = self._calculate_adaptive_threshold(df_sequence)
            print(f"🎯 Адаптивный порог: {adaptive_threshold:.3f} (базовый: {confidence_threshold:.3f})")
            
            # Используем адаптивный порог вместо статичного
            final_threshold = max(adaptive_threshold, confidence_threshold)
            
            # === ШАГ 0: ОПРЕДЕЛЕНИЕ РЫНОЧНОГО РЕЖИМА ===
            if self.regime_detector.is_fitted:
                self.current_regime, self.regime_confidence = self.regime_detector.predict_regime(df_sequence)
                regime_params = self.regime_detector.get_regime_trading_params(self.current_regime)
                regime_threshold = regime_params['confidence_threshold']
                
                # Комбинируем все пороги
                final_threshold = max(final_threshold, regime_threshold)
                print(f"📊 Финальный порог (адаптивный + режим): {final_threshold:.3f}")
            
            # === ШАГ 1: xLSTM АНАЛИЗ ===
            sequence_data = df_sequence.tail(self.sequence_length)[self.feature_columns].values
            sequence_reshaped = sequence_data.reshape(1, self.sequence_length, len(self.feature_columns))
            
            xlstm_prediction = self.xlstm_model.predict(sequence_reshaped)[0]
            xlstm_decision_idx = np.argmax(xlstm_prediction)
            xlstm_confidence = np.max(xlstm_prediction)
            
            print(f"xLSTM: BUY={xlstm_prediction[0]:.3f}, SELL={xlstm_prediction[1]:.3f}, HOLD={xlstm_prediction[2]:.3f}")
            
            # === ШАГ 2: VSA АНАЛИЗ ===
            latest_row = df_sequence.iloc[-1]
            vsa_signals = self._analyze_vsa_context(latest_row)
            
            print(f"VSA Сигналы: {vsa_signals}")
            
            # === ШАГ 3: RL ПРИНЯТИЕ РЕШЕНИЯ ===
            rl_observation = self._create_rl_observation(xlstm_prediction, latest_row)
            rl_action = self.rl_agent.predict(rl_observation, deterministic=True)
            
            # Конвертируем RL действие в решение
            rl_decision = ['SELL', 'BUY', 'HOLD'][rl_action]
            
            print(f"RL Решение: {rl_decision}")

            # === ШАГ 4: ФИНАЛЬНОЕ РЕШЕНИЕ С УЧЕТОМ VSA ===
            final_decision = self._make_final_decision(
                xlstm_prediction, xlstm_confidence,
                vsa_signals, rl_decision, final_threshold
            )
            
            # Обновляем историю
            self.decision_history.append({
                'xlstm_prediction': xlstm_prediction,
                'xlstm_confidence': xlstm_confidence,
                'vsa_signals': vsa_signals,
                'rl_decision': rl_decision,
                'final_decision': final_decision,
                'market_regime': self.current_regime,
                'regime_confidence': self.regime_confidence
            })
            
            # Обновляем состояние для следующего решения
            self._update_state(final_decision)
            
            return final_decision
            
        except Exception as e:
            print(f"Ошибка в принятии решения: {e}")
            return 'HOLD'
    
    def _analyze_vsa_context(self, row):
        """VSA отключен, возвращаем пустые сигналы."""
        return {
            'bullish_strength': 0,
            'bearish_strength': 0,
            'uncertainty': 0,
            'volume_confirmation': False
        }
    
    def _create_rl_observation(self, xlstm_prediction, latest_row):
        """Создает наблюдение для RL агента (без VSA)"""
        # VSA признаки удалены, поэтому размер наблюдения уменьшится.
        # Убедитесь, что TradingEnvRL также отражает это изменение.
        
        portfolio_state = np.array([
            self.current_balance / 10000,  # Нормализованный баланс
            self.current_position,  # -1, 0, 1
            0,  # Нереализованный PnL (упрощенно)
            self.steps_in_position / 100.0
        ])
        
        return np.concatenate([xlstm_prediction, portfolio_state]) # ИЗМЕНЕНО: Без vsa_features
    
    def _make_final_decision(self, xlstm_pred, xlstm_conf, vsa_signals, rl_decision, threshold):
        """Принимает финальное решение с учетом всех факторов"""
        
        # Если уверенность xLSTM низкая, полагаемся на RL
        if xlstm_conf < threshold:
            print(f"Низкая уверенность xLSTM ({xlstm_conf:.3f}), используем RL: {rl_decision}")
            return rl_decision
        
        # Определяем xLSTM решение
        xlstm_decision = ['BUY', 'SELL', 'HOLD'][np.argmax(xlstm_pred)]
        
        # Если xLSTM и RL согласны, принимаем решение
        if xlstm_decision == rl_decision:
            print(f"xLSTM и RL согласны: {xlstm_decision}")
            return xlstm_decision
        
        # Если не согласны, и xLSTM уверенность высокая, доверяем xLSTM
        if xlstm_conf >= threshold + 0.1: # ИЗМЕНЕНО: Добавляем небольшой запас
            print(f"xLSTM уверенность ({xlstm_conf:.3f}) выше RL, решение: {xlstm_decision}")
            return xlstm_decision
        
        # В противном случае, по умолчанию возвращаем RL решение (или HOLD, если RL не уверен)
        print(f"xLSTM и RL не согласны, доверяем RL: {rl_decision}")
        return rl_decision
    
    def _update_state(self, decision):
        """Обновляет внутреннее состояние для следующих решений"""
        if decision in ['BUY', 'SELL'] and self.current_position == 0:
            self.current_position = 1 if decision == 'BUY' else -1
            self.steps_in_position = 0
        elif decision in ['BUY', 'SELL'] and self.current_position != 0:
            self.current_position = 0
            self.steps_in_position = 0
        else:
            self.steps_in_position += 1
    
    def get_decision_explanation(self):
        """Возвращает объяснение последнего решения"""
        if not self.decision_history:
            return "Нет истории решений"
        
        last_decision = self.decision_history[-1]
        
        explanation = f"""
        === ОБЪЯСНЕНИЕ РЕШЕНИЯ ===
        Финальное решение: {last_decision['final_decision']}
        
        xLSTM предсказание:
        - BUY: {last_decision['xlstm_prediction'][0]:.3f}
        - SELL: {last_decision['xlstm_prediction'][1]:.3f} 
        - HOLD: {last_decision['xlstm_prediction'][2]:.3f}
        - Уверенность: {last_decision['xlstm_confidence']:.3f}
        
        - Анализ паттернов и индикаторов: Активирован
        
        RL решение: {last_decision['rl_decision']}
        """
        
        return explanation
    def _calculate_adaptive_threshold(self, df_sequence):
        """
        Вычисляет адаптивный порог уверенности на основе волатильности
        """
        if len(df_sequence) < self.sequence_length:
            return 0.6
        
        # Вычисляем волатильность
        returns = df_sequence['close'].pct_change().dropna()
        volatility = returns.std()
        
        # Адаптируем порог: высокая волатильность = выше порог
        base_threshold = 0.6
        volatility_adjustment = min(volatility * 10, 0.2)  # Максимум +0.2
        
        adaptive_threshold = base_threshold + volatility_adjustment
        return min(adaptive_threshold, 0.85)  # Максимум 0.85