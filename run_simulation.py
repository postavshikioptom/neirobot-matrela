import argparse
from simulation_engine import SimulationEngine

def run_single_simulation(mode, data_path, symbol):
    """Запускает одну симуляцию в заданном режиме для указанного символа."""
    print(f"\n--- Запуск симуляции: {mode} для символа {symbol} ---")
    
    sim = SimulationEngine(
        data_path=data_path,
        lstm_model_path='models/lstm_pattern_model.keras',
        xlstm_model_path='models/xlstm_indicator_model.keras',
        lstm_scaler_path='models/lstm_pattern_scaler.pkl',
        xlstm_scaler_path='models/xlstm_indicator_scaler.pkl'
    )
    
    report = sim.run_simulation(symbol=symbol, mode=mode, sequence_length=30)
    
    print(f"\n--- Результаты симуляции ({mode} для {symbol}) ---")
    if report:
        for key, value in report.items():
            # Форматируем вывод для лучшей читаемости
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
    else:
        print("  Симуляция не дала результатов (возможно, недостаточно данных).")
    
    return report

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Запуск торговой симуляции на исторических данных.')
    
    parser.add_argument(
        '--data', 
        type=str, 
        default='historical_data.csv', 
        help='Путь к файлу с историческими данными.'
    )
    parser.add_argument(
        '--symbol', 
        type=str, 
        default='BTCUSDT', 
        help='Символ для симуляции (например, BTCUSDT).'
    )
    parser.add_argument(
        '--mode', 
        type=str, 
        default='Consensus', 
        choices=['LSTM_only', 'xLSTM_only', 'Consensus'],
        help="Режим симуляции: 'LSTM_only', 'xLSTM_only', 'Consensus'."
    )
    
    args = parser.parse_args()
    
    run_single_simulation(mode=args.mode, data_path=args.data, symbol=args.symbol)