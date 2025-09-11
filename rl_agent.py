from stable_baselines3 import PPO, SAC, A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np
import pandas as pd
from trading_env import TradingEnvRL
import torch

class IntelligentRLAgent:
    """
    Интеллектуальный RL агент с адаптивным обучением
    """
    
    def __init__(self, algorithm='PPO'):
        self.algorithm = algorithm
        self.model = None
        self.training_env = None
        self.eval_env = None
        
    def create_training_environment(self, train_df, xlstm_model):
        """Создает среду для обучения"""
        self.training_env = TradingEnvRL(train_df, xlstm_model)
        return DummyVecEnv([lambda: self.training_env])
        
    def create_evaluation_environment(self, eval_df, xlstm_model):
        """Создает среду для оценки"""
        self.eval_env = TradingEnvRL(eval_df, xlstm_model)
        return self.eval_env
    
    def build_agent(self, vec_env):
        """Строит RL агента с оптимизированными гиперпараметрами"""
        
        if self.algorithm == 'PPO':
            # Оптимизированные гиперпараметры для торговли
            self.model = PPO(
                'MlpPolicy',
                vec_env,
                learning_rate=0.0003,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.03, # <--- ИЗМЕНЕНО с 0.01 на 0.03 (увеличиваем энтропию)
                vf_coef=0.5,
                max_grad_norm=0.5,
                policy_kwargs=dict(
                    net_arch=dict(pi=[256, 128, 64], vf=[256, 128, 64]),
                    activation_fn=torch.nn.ReLU
                ),
                verbose=0, # <-- ИЗМЕНЕНО: 0 для отключения детального логирования
                tensorboard_log="./tensorboard_logs/",
                progress_bar=False # <-- ДОБАВЛЕНО: отключаем прогресс-бар
            )
            
        elif self.algorithm == 'SAC':
            self.model = SAC(
                'MlpPolicy',
                vec_env,
                learning_rate=0.0003,
                buffer_size=100000,
                learning_starts=1000,
                batch_size=256,
                tau=0.005,
                gamma=0.99,
                train_freq=1,
                gradient_steps=1,
                ent_coef='auto', # Можно оставить 'auto' или задать конкретное значение, например, 0.03
                policy_kwargs=dict(net_arch=[256, 256]),
                verbose=0, # <-- ИЗМЕНЕНО: 0 для отключения детального логирования
                tensorboard_log="./tensorboard_logs/",
                progress_bar=False # <-- ДОБАВЛЕНО: отключаем прогресс-бар
            )
            
        return self.model
    
    def train_with_callbacks(self, total_timesteps=100000, eval_freq=5000):
        """Обучение с колбэками для раннего останова"""
        
        # Колбэк для остановки при достижении целевой награды
        stop_callback = StopTrainingOnRewardThreshold(
            reward_threshold=200,  # Остановка при средней награде 200
            verbose=1
        )
        
        # Колбэк для периодической оценки
        eval_callback = EvalCallback(
            self.eval_env,
            best_model_save_path='./models/rl_best_model',
            log_path='./logs/rl_evaluation',
            eval_freq=eval_freq,
            deterministic=True,
            render=False,
            callback_on_new_best=stop_callback
        )
        
        # Обучение
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=eval_callback,
            progress_bar=False # <-- ИЗМЕНЕНО
        )
        
        return self.model
    
    def save_agent(self, path='models/rl_agent'):
        """Сохранение агента"""
        self.model.save(path)
        print(f"RL агент сохранен: {path}")
        
    def load_agent(self, path='models/rl_agent'):
        """Загрузка агента"""
        if self.algorithm == 'PPO':
            self.model = PPO.load(path)
        elif self.algorithm == 'SAC':
            self.model = SAC.load(path)
        print(f"RL агент загружен: {path}")
        
    def predict(self, observation, deterministic=True):
        """Предсказание действия"""
        action, _states = self.model.predict(observation, deterministic=deterministic)
        return action