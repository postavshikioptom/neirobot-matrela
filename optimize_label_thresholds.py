import numpy as np
import pandas as pd
import json
import argparse
import importlib
import copy

import config as cfg
from feature_engineering import FeatureEngineering

"""
Offline Grid/BO sweep for labeling thresholds:
- Sweeps PRICE_CHANGE_THRESHOLD and ADAPTIVE_* within reasonable ranges
- Evaluates class distribution and a simple proxy score (balance + entropy)
- Writes best params back to config.py (print patch instructions) or to a JSON file
"""

def compute_labels(fe: FeatureEngineering, df: pd.DataFrame, pc_thresh, adapt_min, adapt_max, adapt_mult):
    # Temporarily patch config values inside the FeatureEngineering instance
    old_vals = (cfg.PRICE_CHANGE_THRESHOLD, cfg.ADAPTIVE_THRESHOLD_MIN, cfg.ADAPTIVE_THRESHOLD_MAX, cfg.ADAPTIVE_THRESHOLD_MULTIPLIER)
    try:
        cfg.PRICE_CHANGE_THRESHOLD = pc_thresh
        cfg.ADAPTIVE_THRESHOLD_MIN = adapt_min
        cfg.ADAPTIVE_THRESHOLD_MAX = adapt_max
        cfg.ADAPTIVE_THRESHOLD_MULTIPLIER = adapt_mult
        labels = fe.create_trading_labels(df)
        return labels
    finally:
        (
            cfg.PRICE_CHANGE_THRESHOLD,
            cfg.ADAPTIVE_THRESHOLD_MIN,
            cfg.ADAPTIVE_THRESHOLD_MAX,
            cfg.ADAPTIVE_THRESHOLD_MULTIPLIER,
        ) = old_vals


def evaluate_distribution(y: np.ndarray):
    if y is None or len(y) == 0:
        return None, 0.0
    u, c = np.unique(y, return_counts=True)
    total = len(y)
    dist = {int(k): c[list(u).index(k)]/total for k in [0,1,2]}
    # Proxy score: encourage closeness to target ratios and entropy of distribution
    target = getattr(cfg, 'TARGET_CLASS_RATIOS', [0.3,0.3,0.4])
    mse = sum((dist.get(i,0.0) - target[i])**2 for i in range(3))
    p = np.array([dist.get(i,1e-9) for i in range(3)])
    entropy = -np.sum(p * np.log(p + 1e-9))
    # Higher entropy and lower mse is better
    score = (entropy) - 5.0 * mse
    return dist, score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='Path to CSV with columns including timestamp, open, high, low, close, volume, turnover, symbol')
    parser.add_argument('--out', type=str, default='threshold_sweep_results.json')
    parser.add_argument('--target', type=str, default=None, help="Target ratios 'SELL,HOLD,BUY', e.g. '0.3,0.4,0.3'")
    parser.add_argument('--max-symbols', type=int, default=None, help='Limit number of top symbols to include (default ~30)')
    args = parser.parse_args()

    # Optional target override for scoring
    if args.target:
        try:
            parts = [float(x.strip()) for x in args.target.split(',')]
            if len(parts) == 3 and abs(sum(parts) - 1.0) < 1e-6:
                cfg.TARGET_CLASS_RATIOS = parts
                print(f"Using target class ratios override: {parts} (SELL,HOLD,BUY)")
            else:
                print(f"Ignoring --target: must be three ratios summing to 1. Got: {args.target}")
        except Exception as e:
            print(f"Ignoring --target due to parse error: {e}")

    df = pd.read_csv(args.data)
    symbols = df['symbol'].value_counts().index.tolist()
    # Use top-N symbols to limit runtime; allow override via --max-symbols
    if args.max_symbols is not None and args.max_symbols > 0:
        symbols = symbols[:min(args.max_symbols, len(symbols))]
    else:
        symbols = symbols[:max(10, min(30, len(symbols)))]

    fe = FeatureEngineering(sequence_length=cfg.SEQUENCE_LENGTH)

    # Ranges (grid). You can adjust or later replace by Bayesian optimization
    # Expanded search space to better explore BUY≈SELL and HOLD>others (target ~ 30:40:30)
    pc_range = [round(x, 3) for x in np.linspace(0.004, 0.022, 19)]  # 0.004..0.022 step ~0.001
    adapt_min_range = [round(x, 4) for x in np.arange(0.0015, 0.0065, 0.0005)]  # 0.0015..0.006 step 0.0005
    adapt_max_range = [0.012, 0.015, 0.018, 0.02, 0.022, 0.025, 0.028, 0.03]
    adapt_mult_range = [0.9, 1.0, 1.1, 1.2, 1.3, 1.4]

    best = None
    results = []

    for pc in pc_range:
        for amin in adapt_min_range:
            for amax in adapt_max_range:
                if amax <= amin:
                    continue
                for amul in adapt_mult_range:
                    try:
                        all_labels = []
                        for i, sym in enumerate(symbols):
                            sdf = df[df['symbol'] == sym].copy()
                            if len(sdf) < cfg.SEQUENCE_LENGTH + cfg.FUTURE_WINDOW + 5:
                                continue
                            # Ensure indicators and scaler path is set up but we only need labels based on prices
                            labels = compute_labels(fe, sdf, pc, amin, amax, amul)
                            if labels is not None and len(labels) > 0:
                                all_labels.append(labels)
                        if not all_labels:
                            continue
                        y = np.concatenate(all_labels)
                        dist, score = evaluate_distribution(y)
                        if dist is None:
                            continue
                        rec = dict(pc=pc, amin=amin, amax=amax, amul=amul, dist=dist, score=float(score))
                        results.append(rec)
                        if best is None or score > best['score']:
                            best = rec
                        print(f"pc={pc} amin={amin} amax={amax} amul={amul} => dist={dist}, score={score:.4f}")
                    except Exception as e:
                        print(f"Skip combo pc={pc}, amin={amin}, amax={amax}, amul={amul} due to error: {e}")
                        continue

    payload = dict(best=best, results=results)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    if best:
        print("=== BEST LABELING PARAMS ===")
        print(best)
        print("Patch config.py with:")
        print(f"PRICE_CHANGE_THRESHOLD = {best['pc']}")
        print(f"ADAPTIVE_THRESHOLD_MIN = {best['amin']}")
        print(f"ADAPTIVE_THRESHOLD_MAX = {best['amax']}")
        print(f"ADAPTIVE_THRESHOLD_MULTIPLIER = {best['amul']}")
    else:
        print("No valid combination produced labels.")

if __name__ == '__main__':
    main()