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