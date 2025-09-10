import argparse
from multi_symbol_simulation import MultiSymbolSimulation

def run_multi_simulation(mode, data_path):
    with open("hotlist.txt", 'r') as f:
        symbols = [line.strip() for line in f.readlines()]
    
    sim = MultiSymbolSimulation(
        data_path=data_path,
        lstm_model_path='models/lstm_pattern_model.keras',
        xlstm_model_path='models/xlstm_indicator_model.keras',
        lstm_scaler_path='models/lstm_pattern_scaler.pkl',
        xlstm_scaler_path='models/xlstm_indicator_scaler.pkl'
    )
    
    report = sim.run_simulation(symbols=symbols, mode=mode)
    
    print(f"\n--- Результаты симуляции ({mode}) ---")
    if report:
        for key, value in report.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
    else:
        print("  Симуляция не дала результатов.")
    
    return report

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Запуск торговой симуляции на нескольких монетах.')
    
    parser.add_argument(
        '--data', 
        type=str, 
        default='historical_data.csv', 
        help='Путь к файлу с историческими данными.'
    )
    parser.add_argument(
        '--mode', 
        type=str, 
        default='Consensus', 
        choices=['LSTM_only', 'xLSTM_only', 'Consensus'],
        help="Режим симуляции: 'LSTM_only', 'xLSTM_only', 'Consensus'."
    )
    
    args = parser.parse_args()
    
    run_multi_simulation(mode=args.mode, data_path=args.data)