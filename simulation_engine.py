import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os
# import logging # 🔥 УДАЛЕНО: Импорт logging

# Безопасные импорты для анализа признаков
try:
    import numpy as np
except Exception:
    np = None

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠️ SHAP не установлен. Анализ важности признаков будет недоступен.")

class SimulationEngine:
    """
    Движок для симуляции торговли
    """
    def __init__(self, environment, decision_maker, initial_balance=10000):
        self.env = environment
        self.decision_maker = decision_maker
        self.initial_balance = initial_balance
        
        # 🔥 УДАЛЕНО: Инициализация логгера и файлового обработчика
        # self.logger = logging.getLogger('simulation_engine')
        # self.logger.setLevel(logging.INFO)
        # if not self.logger.handlers:
        #     handler = logging.StreamHandler()
        #     formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        #     handler.setFormatter(formatter)
        #     self.logger.addHandler(handler)
            
        #     file_handler = logging.FileHandler('simulation.log')
        #     file_handler.setFormatter(formatter)
        #     self.logger.addHandler(file_handler)
    
    def run_simulation(self, episodes=1, training=False, render=False):
        """
        Запускает симуляцию торговли
        """
        all_episode_info = []
        
        for episode in range(episodes):
            # 🔥 ИЗМЕНЕНО: logger.info -> print
            print(f"Запуск эпизода {episode+1}/{episodes}")
            
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
                # Принимаем решение
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
                    # 🔥 ИЗМЕНЕНО: logger.info -> print
                    print(f"Шаг {step}, Баланс: {info['balance']:.2f}, Награда: {reward:.4f}, "
                                    f"Позиция: {info['position']}, Действие: {action}")
            
            # Логируем результаты эпизода
            final_balance = episode_data['balances'][-1]
            profit_percentage = (final_balance - self.initial_balance) / self.initial_balance * 100
            
            # 🔥 ИЗМЕНЕНО: logger.info -> print
            print(f"Эпизод {episode+1} завершен. Итоговый баланс: {final_balance:.2f} "
                            f"(Прибыль: {profit_percentage:.2f}%), Всего шагов: {step}")
            
            # Если в режиме обучения, обновляем модель и epsilon
            if training:
                for _ in range(10):  # Несколько итераций обучения на каждый эпизод
                    training_info = self.decision_maker.rl_agent.train()
                    if training_info:
                        # 🔥 ИЗМЕНЕНО: logger.info -> print
                        print(f"Обучение: critic_loss: {training_info['critic_loss']:.4f}, "
                                        f"actor_loss: {training_info['actor_loss']:.4f}, "
                                        f"mean_value: {training_info['mean_value']:.4f}")
                
                self.decision_maker.rl_agent.update_epsilon()
                # 🔥 ИЗМЕНЕНО: logger.info -> print
                print(f"Epsilon обновлен до {self.decision_maker.rl_agent.epsilon:.4f}")
            
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

    def analyze_feature_importance_shap(self, sample_data, feature_names=None, max_samples=100):
        """
        Анализирует важность признаков с помощью SHAP

        Args:
            sample_data: Образцы данных для анализа (X_test или X_val)
            feature_names: Названия признаков
            max_samples: Максимальное количество образцов для анализа
        """
        if not SHAP_AVAILABLE:
            print("❌ SHAP не доступен. Установите: pip install shap")
            return None

        try:
            # Получаем модель из decision_maker
            if hasattr(self.decision_maker, 'model') and hasattr(self.decision_maker.model, 'actor_model'):
                model = self.decision_maker.model.actor_model
            else:
                print("❌ Не удалось получить модель для SHAP анализа")
                return None

            # Ограничиваем количество образцов для анализа
            if len(sample_data) > max_samples:
                indices = np.random.choice(len(sample_data), max_samples, replace=False)
                sample_data = sample_data[indices]

            print(f"🔍 Начинаем SHAP анализ на {len(sample_data)} образцах...")

            # Создаем background dataset (случайные образцы)
            background_size = min(50, len(sample_data))
            background_indices = np.random.choice(len(sample_data), background_size, replace=False)
            background = sample_data[background_indices]

            # Создаем explainer
            explainer = shap.DeepExplainer(model, background)

            # Вычисляем SHAP значения
            shap_values = explainer.shap_values(sample_data)

            # Если модель возвращает несколько выходов, берем первый (для классификации)
            if isinstance(shap_values, list):
                shap_values = shap_values[0]

            # Создаем названия признаков, если не предоставлены
            if feature_names is None:
                feature_names = [f"Feature_{i}" for i in range(sample_data.shape[-1])]

            # Визуализируем summary plot
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values, sample_data.reshape(sample_data.shape[0], -1),
                            feature_names=feature_names, show=False)

            # Сохраняем график
            os.makedirs('plots', exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plt.savefig(f'plots/shap_summary_{timestamp}.png', bbox_inches='tight')
            plt.close()

            # Вычисляем среднюю важность признаков
            mean_shap = np.mean(np.abs(shap_values), axis=0)

            # Если данные 3D (sequence_length, features), усредняем по временным шагам
            if len(mean_shap.shape) > 1:
                mean_shap = np.mean(mean_shap, axis=0)

            # Сортируем признаки по важности
            feature_importance = sorted(zip(feature_names, mean_shap),
                                      key=lambda x: x[1], reverse=True)

            print("📊 Топ-10 наиболее важных признаков по SHAP:")
            for i, (feature, importance) in enumerate(feature_importance[:10]):
                print("2d")

            # Сохраняем детальную информацию
            shap_results = {
                'feature_importance': feature_importance,
                'shap_values': shap_values,
                'sample_data': sample_data,
                'feature_names': feature_names,
                'timestamp': timestamp
            }

            return shap_results

        except Exception as e:
            print(f"❌ Ошибка при SHAP анализе: {e}")
            return None

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