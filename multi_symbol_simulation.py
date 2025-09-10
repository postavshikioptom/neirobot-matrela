import pandas as pd
import numpy as np
from trade_manager import ConsensusDecisionMaker
from feature_engineering import calculate_features, detect_candlestick_patterns

class MultiSymbolSimulation:
    def __init__(self, data_path, xlstm_pattern_model_path, xlstm_indicator_model_path, xlstm_pattern_scaler_path, xlstm_indicator_scaler_path, sequence_length=10):
        self.full_df = pd.read_csv(data_path)
        self.pattern_cols = [
            'CDLHAMMER', 'CDLENGULFING', 'CDLDOJI', 'CDLSHOOTINGSTAR',
            'CDLHANGINGMAN', 'CDL3BLACKCROWS',
            'hammer_f', 'hangingman_f', 'engulfing_f', 'doji_f',
            'shootingstar_f', '3blackcrows_f'
        ]
        self.indicator_cols = ['RSI_14', 'MACD_12_26_9', 'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0', 'ADX_14', 'STOCHk_14_3_3', 'STOCHd_14_3_3']
        self.decision_maker = ConsensusDecisionMaker(
            xlstm_pattern_model_path,
            xlstm_indicator_model_path,
            xlstm_pattern_scaler_path,
            xlstm_indicator_scaler_path,
            sequence_length=sequence_length,
            pattern_feature_count=len(self.pattern_cols),
            indicator_feature_count=len(self.indicator_cols)
        )

    def _prepare_data_for_symbol(self, symbol):
        df_symbol = self.full_df[self.full_df['symbol'] == symbol].copy()
        if df_symbol.empty:
            return None
        df_symbol = calculate_features(df_symbol)
        df_symbol = detect_candlestick_patterns(df_symbol)
        df_symbol.dropna(inplace=True)
        df_symbol.reset_index(drop=True, inplace=True)
        for col in self.pattern_cols + self.indicator_cols:
            if col not in df_symbol.columns:
                df_symbol[col] = 0
        return df_symbol

    def run_simulation(self, symbols, mode='Consensus', sequence_length=30, initial_balance=10000):
        balance = initial_balance
        open_positions = {}
        all_trades = []

        for i in range(sequence_length, len(self.full_df)):
            for symbol in symbols:
                df_symbol = self._prepare_data_for_symbol(symbol)
                if df_symbol is None or len(df_symbol) <= i:
                    continue

                sequence_df = df_symbol.iloc[i - sequence_length : i]
                pattern_features = np.reshape(sequence_df[self.pattern_cols].values, (1, sequence_length, len(self.pattern_cols)))
                indicator_features = np.reshape(sequence_df[self.indicator_cols].values, (1, sequence_length, len(self.indicator_cols)))
                decision = self.decision_maker.get_decision(pattern_features, indicator_features, mode=mode)
                current_price = df_symbol.iloc[i]['close']

                if symbol not in open_positions:
                    if decision == 'BUY':
                        open_positions[symbol] = {'type': 'LONG', 'entry_price': current_price}
                    elif decision == 'SELL':
                        open_positions[symbol] = {'type': 'SHORT', 'entry_price': current_price}
                else:
                    if open_positions[symbol]['type'] == 'LONG' and decision == 'SELL':
                        pnl = (current_price - open_positions[symbol]['entry_price']) / open_positions[symbol]['entry_price']
                        balance *= (1 + pnl)
                        all_trades.append(pnl)
                        del open_positions[symbol]
                    elif open_positions[symbol]['type'] == 'SHORT' and decision == 'BUY':
                        pnl = (open_positions[symbol]['entry_price'] - current_price) / open_positions[symbol]['entry_price']
                        balance *= (1 + pnl)
                        all_trades.append(pnl)
                        del open_positions[symbol]

        return self._generate_report(initial_balance, balance, all_trades)

    def _generate_report(self, initial_balance, final_balance, trades):
        if not trades:
            return {
                "Initial Balance": initial_balance,
                "Final Balance": final_balance,
                "Net PnL (%)": (final_balance - initial_balance) / initial_balance * 100,
                "Total Trades": 0,
                "Win Rate": 0,
            }
        
        win_rate = len([t for t in trades if t > 0]) / len(trades)
        return {
            "Initial Balance": initial_balance,
            "Final Balance": final_balance,
            "Net PnL (%)": (final_balance - initial_balance) / initial_balance * 100,
            "Total Trades": len(trades),
            "Win Rate": win_rate,
        }