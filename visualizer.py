import matplotlib.pyplot as plt
import pandas as pd

def plot_equity_curve(report):
    if not report or not report.get('trades'):
        print("No trades to plot.")
        return
        
    trades = report['trades']
    equity_curve = [10000]
    for pnl in trades:
        equity_curve.append(equity_curve[-1] * (1 + pnl))
        
    plt.figure(figsize=(10, 6))
    plt.plot(equity_curve)
    plt.title('Equity Curve')
    plt.xlabel('Trade Number')
    plt.ylabel('Equity')
    plt.grid(True)
    plt.show()

def plot_pnl_distribution(report):
    if not report or not report.get('trades'):
        print("No trades to plot.")
        return
        
    trades = report['trades']
    plt.figure(figsize=(10, 6))
    plt.hist(trades, bins=50, alpha=0.75)
    plt.title('PnL Distribution')
    plt.xlabel('PnL (%)')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()