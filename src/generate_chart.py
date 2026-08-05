import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def generate_performance_chart():
    """Generate performance chart from predictions log."""
    
    log_path = Path('data/logs/nhl_predictions_log.csv')
    
    if not log_path.exists():
        print("No predictions log yet")
        return
    
    # Load predictions
    df = pd.read_csv(log_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Calculate rolling win rate (last 50 bets)
    df['result_binary'] = (df['result'] == 'WIN').astype(int)
    df['rolling_wr'] = df['result_binary'].rolling(50).mean() * 100
    
    # Create chart
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Win rate over time
    ax1.plot(df.index, df['rolling_wr'], linewidth=2, color='#2E7D32', label='Win Rate (50-game rolling)')
    ax1.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50% (no edge)')
    ax1.axhline(y=55, color='green', linestyle='--', alpha=0.5, label='55% (target)')
    ax1.set_ylabel('Win Rate (%)', fontsize=12)
    ax1.set_title('NHL Model Performance Over Time', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax1.set_ylim([40, 65])
    
    # Cumulative profit
    df['profit'] = df['result_binary'] * 100 - (1 - df['result_binary']) * 100
    df['cum_profit'] = df['profit'].cumsum()
    
    ax2.plot(df.index, df['cum_profit'], linewidth=2, color='#1565C0', label='Cumulative Profit')
    ax2.fill_between(df.index, df['cum_profit'], alpha=0.3, color='#1565C0')
    ax2.set_ylabel('Profit ($)', fontsize=12)
    ax2.set_xlabel('Predictions', fontsize=12)
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('performance_chart.png', dpi=150, bbox_inches='tight')
    print("✅ Chart saved to performance_chart.png")
    plt.close()

if __name__ == "__main__":
    generate_performance_chart()