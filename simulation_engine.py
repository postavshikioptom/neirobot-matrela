import pandas as pd
import numpy as np
from trade_manager import ConsensusDecisionMaker
from feature_engineering import calculate_features, detect_candlestick_patterns

class SimulationEngine:
    """
    A class to simulate trading strategies on historical data using sequences.
    """

    def __init__(self, data_path, xlstm_pattern_model_path, xlstm_indicator_model_path, xlstm_pattern_scaler_path, xlstm_indicator_scaler_path, sequence_length=10):
        """
        Initializes the SimulationEngine.
        """
        try:
            self.full_df = pd.read_csv(data_path)
            # Define feature columns based on what the training script uses
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
        except FileNotFoundError:
            print(f"Error: Data file not found at {data_path}")
            raise
        except Exception as e:
            print(f"Error during SimulationEngine initialization: {e}")
            raise

    def _prepare_data_for_symbol(self, symbol):
        """
        Prepares the data for a single symbol for simulation.
        """
        print(f"Preparing data for {symbol}...")
        df_symbol = self.full_df[self.full_df['symbol'] == symbol].copy()
        if df_symbol.empty:
            print(f"No data found for symbol {symbol}")
            return None

        df_symbol = calculate_features(df_symbol)
        df_symbol = detect_candlestick_patterns(df_symbol)
        df_symbol.dropna(inplace=True)
        df_symbol.reset_index(drop=True, inplace=True)
        
        # Ensure all feature columns exist
        for col in self.pattern_cols + self.indicator_cols:
            if col not in df_symbol.columns:
                df_symbol[col] = 0
        
        print(f"Data prepared for {symbol}. Total rows: {len(df_symbol)}")
        return df_symbol

    def run_simulation(self, symbol, mode='Consensus', sequence_length=30, initial_balance=10000):
        """
        Runs the trading simulation for a single symbol.
        """
        df = self._prepare_data_for_symbol(symbol)
        if df is None or len(df) < sequence_length:
            print("Not enough data to run simulation.")
            return None

        balance = initial_balance
        position = None  # None, 'LONG', 'SHORT'
        entry_price = 0
        trades = []

        print(f"Starting simulation for {symbol} with initial balance ${balance:.2f}")
        # We start the loop from `sequence_length` because we need `sequence_length` previous candles to make the first prediction.
        for i in range(sequence_length, len(df)):
            
            # --- Get the sequence of the last `sequence_length` data points ---
            sequence_df = df.iloc[i - sequence_length : i]
            
            # --- Prepare features for models ---
            pattern_features = sequence_df[self.pattern_cols].values
            indicator_features = sequence_df[self.indicator_cols].values

            # Reshape for the models: (1, timesteps, features)
            pattern_features = np.reshape(pattern_features, (1, sequence_length, len(self.pattern_cols)))
            indicator_features = np.reshape(indicator_features, (1, sequence_length, len(self.indicator_cols)))

            # --- Make a decision ---
            decision = "HOLD"
            decision = self.decision_maker.get_decision(pattern_features, indicator_features, mode=mode)

            # --- Trading Logic ---
            current_price = df.iloc[i]['close']
            
            if position is None: # No open position
                if decision == 'BUY':
                    position = 'LONG'
                    entry_price = current_price
                    # print(f"  {df.index[i]}: OPEN LONG at {current_price:.4f}")
                elif decision == 'SELL':
                    position = 'SHORT'
                    entry_price = current_price
                    # print(f"  {df.index[i]}: OPEN SHORT at {current_price:.4f}")
            elif position == 'LONG' and decision == 'SELL':
                pnl = (current_price - entry_price) / entry_price
                balance *= (1 + pnl)
                trades.append(pnl)
                position = None
            elif position == 'SHORT' and decision == 'BUY':
                pnl = (entry_price - current_price) / entry_price
                balance *= (1 + pnl)
                trades.append(pnl)
                position = None

        print(f"Simulation for {symbol} finished. Final balance: ${balance:.2f}")
        return self._generate_report(initial_balance, balance, trades)

    def _generate_report(self, initial_balance, final_balance, trades):
        """
        Generates a report of the simulation results.
        """
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
            "Average Trade PnL (%)": np.mean(trades) * 100,
        }