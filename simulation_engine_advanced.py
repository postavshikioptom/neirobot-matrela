import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from feature_engineering import calculate_features, detect_candlestick_patterns
from hybrid_decision_maker import HybridDecisionMaker
import config
import warnings
warnings.filterwarnings('ignore')

class AdvancedSimulationEngine:
    """
    Продвинутый движок симуляции для тестирования xLSTM + VSA + RL системы
    """
    
    def __init__(self, data_path, xlstm_model_path, rl_agent_path):
        print("Инициализация продвинутого движка симуляции...")
        
        self.full_df = pd.read_csv(data_path)
        self.results = {}
        
        # Признаки для новой системы
        self.feature_columns = [
            'RSI_14', 'MACD_12_26_9', 'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0',
            'ADX_14', 'STOCHk_14_3_3', 'STOCHd_14_3_3',
            'ATR_14', # <--- ДОБАВЛЕНО
            'CDLHAMMER', 'CDLENGULFING', 'CDLDOJI', 'CDLSHOOTINGSTAR',
            'CDLHANGINGMAN', 'CDLMARUBOZU',
        ]
        
        # Загружаем гибридную систему
        try:
            self.decision_maker = HybridDecisionMaker(
                xlstm_model_path=xlstm_model_path,
                rl_agent_path=rl_agent_path,
                feature_columns=self.feature_columns,
                sequence_length=config.SEQUENCE_LENGTH
            )
            print("✅ Гибридная система загружена для симуляции")
        except Exception as e:
            print(f"❌ Ошибка загрузки системы: {e}")
            raise
    
    def prepare_symbol_data(self, symbol):
        """Подготавливает данные для одного символа с полной обработкой"""
        print(f"Подготовка данных для {symbol}...")
        
        df_symbol = self.full_df[self.full_df['symbol'] == symbol].copy()
        if df_symbol.empty:
            return None
            
        # Полная обработка с VSA
        df_symbol = calculate_features(df_symbol)
        df_symbol = detect_candlestick_patterns(df_symbol)
        # df_symbol = calculate_vsa_features(df_symbol) # <--- ЗАКОММЕНТИРОВАНО
        
        # Убираем NaN
        df_symbol.dropna(inplace=True)
        df_symbol.reset_index(drop=True, inplace=True)
        
        # Проверяем наличие всех признаков
        for col in self.feature_columns:
            if col not in df_symbol.columns:
                df_symbol[col] = 0
        
        print(f"Данные подготовлены для {symbol}. Строк: {len(df_symbol)}")
        return df_symbol
    
    def run_comprehensive_simulation(self, symbols=None, initial_balance=10000, commission=0.0008):
        """Запускает комплексную симуляцию по всем символам"""
        
        if symbols is None:
            symbols = self.full_df['symbol'].unique()[:5]  # Тестируем на первых 5 символах
        
        print(f"Запуск симуляции на {len(symbols)} символах...")
        
        all_results = {}
        
        for symbol in symbols:
            print(f"\n{'='*50}")
            print(f"СИМУЛЯЦИЯ: {symbol}")
            print(f"{'='*50}")
            
            df = self.prepare_symbol_data(symbol)
            if df is None or len(df) < 50:
                print(f"❌ Недостаточно данных для {symbol}")
                continue
            
            # Разные стратегии симуляции
            strategies = {
                'hybrid_conservative': {'confidence_threshold': 0.8, 'tp': 1.5, 'sl': -1.0},
                'hybrid_balanced': {'confidence_threshold': 0.6, 'tp': 1.2, 'sl': -1.2},
                'hybrid_aggressive': {'confidence_threshold': 0.4, 'tp': 1.0, 'sl': -1.5},
            }
            
            symbol_results = {}
            
            for strategy_name, params in strategies.items():
                print(f"\n--- Стратегия: {strategy_name} ---")
                
                result = self.simulate_strategy(df, symbol, strategy_name, params, initial_balance, commission)
                symbol_results[strategy_name] = result
                
                # Краткий отчет
                print(f"Финальный баланс: ${result['final_balance']:.2f}")
                print(f"Доходность: {result['total_return']:.2f}%")
                print(f"Сделок: {result['total_trades']}")
                print(f"Винрейт: {result['win_rate']:.1f}%")
                print(f"Максимальная просадка: {result['max_drawdown']:.2f}%")
                
            all_results[symbol] = symbol_results
        
        self.results = all_results
        self.generate_comprehensive_report()
        return all_results
    
    def simulate_strategy(self, df, symbol, strategy_name, params, initial_balance, commission):
        """Симулирует одну стратегию на данных"""
        
        balance = initial_balance
        position = 0  # 0=flat, 1=long, -1=short
        entry_price = 0
        trades = []
        balance_history = [balance]
        
        tp_pct = params.get('tp', 1.2)
        sl_pct = params.get('sl', -1.2)
        confidence_threshold = params.get('confidence_threshold', 0.6)
        vsa_only = params.get('vsa_only', False)
        
        for i in range(config.SEQUENCE_LENGTH, len(df)):  # Начинаем с 15-й свечи для истории
            current_price = df.iloc[i]['close']
            sequence_df = df.iloc[i-config.SEQUENCE_LENGTH:i+1]
            
            # Получаем решение
            try:
                decision = self.decision_maker.get_decision(sequence_df, confidence_threshold)
            except:
                decision = 'HOLD'
            
            # Рассчитываем текущий PnL если в позиции
            current_pnl_pct = 0
            if position != 0:
                if position == 1:  # Long
                    current_pnl_pct = ((current_price - entry_price) / entry_price) * 100
                else:  # Short
                    current_pnl_pct = ((entry_price - current_price) / entry_price) * 100
            
            # Логика торговли
            trade_executed = False
            
            # Проверяем условия закрытия позиции
            if position != 0:
                should_close = False
                close_reason = ""
                
                # TP/SL
                if current_pnl_pct >= tp_pct:
                    should_close = True
                    close_reason = "TP"
                elif current_pnl_pct <= sl_pct:
                    should_close = True
                    close_reason = "SL"
                # Сигнал модели на закрытие
                elif (position == 1 and decision == 'SELL') or (position == -1 and decision == 'BUY'):
                    should_close = True
                    close_reason = "MODEL"
                
                if should_close:
                    pnl_pct = current_pnl_pct - (commission * 2 * 100)  # Учитываем комиссию
                    balance *= (1 + pnl_pct / 100)
                    
                    trades.append({
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'side': 'LONG' if position == 1 else 'SHORT',
                        'pnl_pct': pnl_pct,
                        'close_reason': close_reason,
                        'bars_held': i - entry_bar,
                        'timestamp': i
                    })
                    
                    position = 0
                    trade_executed = True
            
            # Открытие новой позиции
            if position == 0 and not trade_executed:
                if decision == 'BUY':
                    position = 1
                    entry_price = current_price
                    entry_bar = i
                elif decision == 'SELL':
                    position = -1
                    entry_price = current_price
                    entry_bar = i
            
            balance_history.append(balance)
        
        # Закрываем позицию в конце если открыта
        if position != 0:
            if position == 1:
                pnl_pct = ((df.iloc[-1]['close'] - entry_price) / entry_price) * 100
            else:
                pnl_pct = ((entry_price - df.iloc[-1]['close']) / entry_price) * 100
            
            pnl_pct -= (commission * 2 * 100)
            balance *= (1 + pnl_pct / 100)
            
            trades.append({
                'entry_price': entry_price,
                'exit_price': df.iloc[-1]['close'],
                'side': 'LONG' if position == 1 else 'SHORT',
                'pnl_pct': pnl_pct,
                'close_reason': 'END',
                'bars_held': len(df) - 1 - entry_bar,
                'timestamp': len(df) - 1
            })
        
        # Вычисляем метрики
        return self.calculate_performance_metrics(trades, balance_history, initial_balance)
    
    
    def calculate_performance_metrics(self, trades, balance_history, initial_balance):
        """Вычисляет детальные метрики производительности"""
        
        if not trades:
            return {
                'final_balance': balance_history[-1],
                'total_return': (balance_history[-1] - initial_balance) / initial_balance * 100,
                'total_trades': 0,
                'win_rate': 0,
                'avg_trade_pnl': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                'profit_factor': 0,
                'avg_bars_held': 0
            }
        
        trades_df = pd.DataFrame(trades)
        
        # Базовые метрики
        final_balance = balance_history[-1]
        total_return = (final_balance - initial_balance) / initial_balance * 100
        total_trades = len(trades)
        winning_trades = len(trades_df[trades_df['pnl_pct'] > 0])
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        avg_trade_pnl = trades_df['pnl_pct'].mean()
        
        # Максимальная просадка
        balance_series = pd.Series(balance_history)
        running_max = balance_series.expanding().max()
        drawdown = (balance_series - running_max) / running_max * 100
        max_drawdown = drawdown.min()
        
        # Sharpe Ratio (упрощенный)
        if len(trades_df) > 1:
            returns = trades_df['pnl_pct'].values
            sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
        else:
            sharpe_ratio = 0
        
        # Profit Factor
        gross_profit = trades_df[trades_df['pnl_pct'] > 0]['pnl_pct'].sum()
        gross_loss = abs(trades_df[trades_df['pnl_pct'] < 0]['pnl_pct'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0
        
        # Среднее время удержания позиции
        avg_bars_held = trades_df['bars_held'].mean()
        
        return {
            'final_balance': final_balance,
            'total_return': total_return,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_trade_pnl': avg_trade_pnl,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'profit_factor': profit_factor,
            'avg_bars_held': avg_bars_held,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'winning_trades': winning_trades,
            'losing_trades': total_trades - winning_trades
        }
    
    def generate_comprehensive_report(self):
        """Генерирует детальный отчет по всем симуляциям"""
        
        if not self.results:
            print("Нет результатов для отчета")
            return
        
        print("\n" + "="*80)
        print("КОМПЛЕКСНЫЙ ОТЧЕТ ПО СИМУЛЯЦИЯМ xLSTM + RL")
        print("="*80)
        
        # Собираем данные для сравнения
        comparison_data = []
        
        for symbol, strategies in self.results.items():
            for strategy, metrics in strategies.items():
                comparison_data.append({
                    'Symbol': symbol,
                    'Strategy': strategy,
                    'Return (%)': metrics['total_return'],
                    'Trades': metrics['total_trades'],
                    'Win Rate (%)': metrics['win_rate'],
                    'Avg Trade (%)': metrics['avg_trade_pnl'],
                    'Max DD (%)': metrics['max_drawdown'],
                    'Sharpe': metrics['sharpe_ratio'],
                    'Profit Factor': metrics['profit_factor']
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Отчет по символам
        print("\n📊 РЕЗУЛЬТАТЫ ПО СИМВОЛАМ:")
        print("-" * 80)
        
        for symbol in comparison_df['Symbol'].unique():
            symbol_data = comparison_df[comparison_df['Symbol'] == symbol]
            print(f"\n{symbol}:")
            print(symbol_data.to_string(index=False))
        
        # Сравнение стратегий
        print("\n🏆 СРАВНЕНИЕ СТРАТЕГИЙ (СРЕДНИЕ ЗНАЧЕНИЯ):")
        print("-" * 80)
        
        strategy_comparison = comparison_df.groupby('Strategy').agg({
            'Return (%)': 'mean',
            'Trades': 'mean',
            'Win Rate (%)': 'mean',
            'Max DD (%)': 'mean',
            'Sharpe': 'mean',
            'Profit Factor': 'mean'
        }).round(2)
        
        print(strategy_comparison)
        
        # Лучшие результаты
        print("\n🥇 ТОП-3 ЛУЧШИХ РЕЗУЛЬТАТА ПО ДОХОДНОСТИ:")
        print("-" * 80)
        
        top_results = comparison_df.nlargest(3, 'Return (%)')
        print(top_results[['Symbol', 'Strategy', 'Return (%)', 'Win Rate (%)', 'Max DD (%)']].to_string(index=False))
        
        # Анализ рисков
        print("\n⚠️  АНАЛИЗ РИСКОВ:")
        print("-" * 80)
        
        risk_analysis = comparison_df.groupby('Strategy').agg({
            'Max DD (%)': ['mean', 'max'],
            'Return (%)': 'std'
        }).round(2)
        
        print("Максимальные просадки и волатильность доходности:")
        print(risk_analysis)
        
        # Рекомендации
        print("\n💡 РЕКОМЕНДАЦИИ:")
        print("-" * 80)
        
        best_strategy = strategy_comparison['Return (%)'].idxmax()
        safest_strategy = strategy_comparison['Max DD (%)'].idxmax()  # Наименьшая просадка (ближе к 0)
        
        print(f"• Лучшая стратегия по доходности: {best_strategy}")
        print(f"• Наиболее консервативная стратегия: {safest_strategy}")
        
        if strategy_comparison.loc[best_strategy, 'Sharpe'] > 1.0:
            print(f"• {best_strategy} показывает отличное соотношение риск/доходность (Sharpe > 1.0)")
        
        # Сохраняем результаты
        comparison_df.to_csv('simulation_results.csv', index=False)
        print(f"\n💾 Результаты сохранены в simulation_results.csv")
        
        return comparison_df
    
    def plot_results(self):
        """Создает визуализацию результатов"""
        
        if not self.results:
            print("Нет данных для визуализации")
            return
        
        # Подготавливаем данные
        plot_data = []
        for symbol, strategies in self.results.items():
            for strategy, metrics in strategies.items():
                plot_data.append({
                    'Symbol': symbol,
                    'Strategy': strategy,
                    'Return': metrics['total_return'],
                    'Sharpe': metrics['sharpe_ratio'],
                    'Max_DD': metrics['max_drawdown'],
                    'Win_Rate': metrics['win_rate']
                })
        
        plot_df = pd.DataFrame(plot_data)
        
        # Создаем графики
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # График доходности по стратегиям
        sns.boxplot(data=plot_df, x='Strategy', y='Return', ax=ax1)
        ax1.set_title('Доходность по стратегиям')
        ax1.tick_params(axis='x', rotation=45)
        
        # Sharpe Ratio по стратегиям
        sns.boxplot(data=plot_df, x='Strategy', y='Sharpe', ax=ax2)
        ax2.set_title('Sharpe Ratio по стратегиям')
        ax2.tick_params(axis='x', rotation=45)
        
        # Максимальная просадка
        sns.boxplot(data=plot_df, x='Strategy', y='Max_DD', ax=ax3)
        ax3.set_title('Максимальная просадка по стратегиям')
        ax3.tick_params(axis='x', rotation=45)
        
        # Винрейт
        sns.boxplot(data=plot_df, x='Strategy', y='Win_Rate', ax=ax4)
        ax4.set_title('Win Rate по стратегиям')
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('simulation_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("📈 Графики сохранены в simulation_analysis.png")

# Пример использования
if __name__ == '__main__':
    # Запуск продвинутой симуляции
    sim_engine = AdvancedSimulationEngine(
        data_path='historical_data.csv',
        xlstm_model_path='models/xlstm_rl_model.keras',
        rl_agent_path='models/rl_agent_BTCUSDT'
    )
    
    # Тестируем на нескольких символах
    test_symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT']
    results = sim_engine.run_comprehensive_simulation(symbols=test_symbols)
    
    # Создаем визуализацию
    sim_engine.plot_results()
    
    print("\n✅ Продвинутая симуляция завершена!")