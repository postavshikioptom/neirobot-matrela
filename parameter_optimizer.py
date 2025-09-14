import numpy as np
import pandas as pd
from scipy.optimize import minimize
import json

class ParameterOptimizer:
    """
    Автоматическая оптимизация параметров бота на основе производительности
    """
    
    def __init__(self):
        self.performance_history = []
        self.current_params = {
            'confidence_threshold': 0.65,
            'take_profit_pct': 1.5,
            'stop_loss_pct': -1.0,
            'xlstm_weight': 0.6
        }
        
    def record_performance(self, trades_data):
        """Записывает производительность для последующей оптимизации"""
        if not trades_data:
            return
            
        metrics = {
            'total_return': sum([t['pnl_pct'] for t in trades_data]),
            'win_rate': len([t for t in trades_data if t['pnl_pct'] > 0]) / len(trades_data),
            'max_drawdown': self._calculate_max_drawdown(trades_data),
            'sharpe_ratio': self._calculate_sharpe_ratio(trades_data),
            'total_trades': len(trades_data),
            'timestamp': pd.Timestamp.now(),
            'parameters': self.current_params.copy()
        }
        
        self.performance_history.append(metrics)
        
        # Сохраняем историю
        self._save_performance_history()
        
        # Запускаем оптимизацию каждые 50 записей
        if len(self.performance_history) % 50 == 0:
            self.optimize_parameters()
    
    def optimize_parameters(self):
        """Оптимизирует параметры на основе исторической производительности"""
        if len(self.performance_history) < 20:
            return
            
        print("\n🔧 Запуск автоматической оптимизации параметров...")
        
        # Определяем границы параметров
        bounds = [
            (0.5, 0.9),   # confidence_threshold
            (0.8, 3.0),   # take_profit_pct
            (-3.0, -0.5), # stop_loss_pct
            (0.2, 0.8)    # xlstm_weight
        ]
        
        # Начальные значения
        x0 = [
            self.current_params['confidence_threshold'],
            self.current_params['take_profit_pct'],
            abs(self.current_params['stop_loss_pct']),  # Делаем положительным для оптимизации
            self.current_params['xlstm_weight']
        ]
        
        # Оптимизация
        result = minimize(
            self._objective_function,
            x0,
            bounds=bounds,
            method='L-BFGS-B'
        )
        
        if result.success:
            # Обновляем параметры
            old_params = self.current_params.copy()
            
            self.current_params = {
                'confidence_threshold': result.x[0],
                'take_profit_pct': result.x[1],
                'stop_loss_pct': -result.x[2],  # Возвращаем отрицательное значение
                'xlstm_weight': result.x[3]
            }
            
            print(f"✅ Параметры оптимизированы!")
            print(f"📊 Старые параметры: {old_params}")
            print(f"🔥 Новые параметры: {self.current_params}")
            
            # Сохраняем оптимизированные параметры
            self._save_optimized_parameters()
        else:
            print("❌ Оптимизация не удалась")
    
    def _objective_function(self, params):
        """Целевая функция для оптимизации (минимизируем отрицательный Sharpe ratio)"""
        # Симулируем производительность с новыми параметрами
        # (упрощенная версия - в реальности нужна более сложная симуляция)
        
        confidence_threshold = params[0]
        take_profit_pct = params[1]
        stop_loss_pct = -params[2]
        
        # Фильтруем историю по похожим параметрам
        similar_periods = []
        for period in self.performance_history[-100:]:  # Последние 100 периодов
            param_diff = abs(period['parameters']['confidence_threshold'] - confidence_threshold)
            if param_diff < 0.1:  # Похожие параметры
                similar_periods.append(period)
        
        if len(similar_periods) < 5:
            return 0  # Недостаточно данных
        
        # Вычисляем средний Sharpe ratio для похожих параметров
        sharpe_ratios = [p['sharpe_ratio'] for p in similar_periods]
        avg_sharpe = np.mean(sharpe_ratios)
        
        # Возвращаем отрицательное значение для минимизации
        return -avg_sharpe

    def _calculate_max_drawdown(self, trades_data):
        """Вычисляет максимальную просадку"""
        returns = pd.Series([t['pnl_pct'] for t in trades_data])
        cumulative_returns = (1 + returns / 100).cumprod()
        peak = cumulative_returns.expanding(min_periods=1).max()
        drawdown = (cumulative_returns - peak) / peak
        return drawdown.min()

    def _calculate_sharpe_ratio(self, trades_data, risk_free_rate=0):
        """Вычисляет коэффициент Шарпа"""
        returns = pd.Series([t['pnl_pct'] for t in trades_data])
        if returns.std() == 0:
            return 0
        return (returns.mean() - risk_free_rate) / returns.std()

    def _save_performance_history(self):
        """Сохраняет историю производительности в файл"""
        with open('performance_history.json', 'w') as f:
            json.dump(self.performance_history, f, indent=2, default=str)

    def _save_optimized_parameters(self):
        """Сохраняет оптимизированные параметры в файл"""
        with open('optimized_params.json', 'w') as f:
            json.dump(self.current_params, f, indent=2)