import json
import time
import pandas as pd
from datetime import datetime, timedelta

class PerformanceMonitor:
    """
    Мониторинг производительности бота в реальном времени
    """
    
    def __init__(self):
        self.stats_file = 'real_time_performance.json'
        self.reset_daily_stats()
    
    def reset_daily_stats(self):
        """Сбрасывает дневную статистику"""
        self.daily_stats = {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'trades_opened': 0,
            'trades_closed': 0,
            'total_pnl': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'model_accuracy': [],
            'start_time': time.time()
        }
    
    def log_trade_opened(self, symbol, decision): # ИЗМЕНЕНО: Удален vsa_confirmed
        """Логирует открытие сделки (без VSA подтверждения)"""
        self.daily_stats['trades_opened'] += 1
        # if vsa_confirmed: # <--- УДАЛЕНО
        #     self.daily_stats['vsa_confirmed_trades'] += 1 # <--- УДАЛЕНО
        
        self.save_stats()
    
    def log_trade_closed(self, symbol, pnl_pct, was_correct_prediction=None):
        """Логирует закрытие сделки"""
        self.daily_stats['trades_closed'] += 1
        self.daily_stats['total_pnl'] += pnl_pct
        
        if pnl_pct > 0:
            self.daily_stats['winning_trades'] += 1
        else:
            self.daily_stats['losing_trades'] += 1
            
        if was_correct_prediction is not None:
            self.daily_stats['model_accuracy'].append(was_correct_prediction)
        
        self.save_stats()
        self.print_current_stats()
    
    def print_current_stats(self):
        """Выводит текущую статистику"""
        stats = self.daily_stats
        win_rate = (stats['winning_trades'] / max(stats['trades_closed'], 1)) * 100
        
        print(f"\n📊 === ДНЕВНАЯ СТАТИСТИКА ===")
        print(f"🕐 Время работы: {(time.time() - stats['start_time']) / 3600:.1f} часов")
        print(f"📈 Открыто сделок: {stats['trades_opened']}")
        print(f"📉 Закрыто сделок: {stats['trades_closed']}")
        print(f"💰 Общий PnL: {stats['total_pnl']:.2f}%")
        print(f"🎯 Win Rate: {win_rate:.1f}%")
        # print(f"✅ VSA подтвержденных: {stats['vsa_confirmed_trades']}") # <--- УДАЛЕНО
        
        if stats['model_accuracy']:
            accuracy = sum(stats['model_accuracy']) / len(stats['model_accuracy']) * 100
            print(f"🧠 Точность модели: {accuracy:.1f}%")
    
    def save_stats(self):
        """Сохраняет статистику в файл"""
        with open(self.stats_file, 'w') as f:
            json.dump(self.daily_stats, f, indent=2)