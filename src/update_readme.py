import pandas as pd
from pathlib import Path

def update_readme_stats():
    """Update README with latest performance stats."""
    
    log_path = Path('data/logs/nhl_predictions_log.csv')
    
    if not log_path.exists():
        print("No predictions log yet")
        return
    
    # Load and calculate stats
    df = pd.read_csv(log_path)
    total_bets = len(df)
    wins = len(df[df['result'] == 'WIN'])
    losses = total_bets - wins
    win_rate = (wins / total_bets * 100) if total_bets > 0 else 0
    
    profit = (wins * 100) - (losses * 100)
    roi = (profit / (total_bets * 100) * 100) if total_bets > 0 else 0
    
    # Read README
    readme_path = Path('README.md')
    with open(readme_path, 'r') as f:
        content = f.read()
    
    # Update the Key Results table
    new_table = f"""| Metric | Value |
|--------|-------|
| Test Accuracy | {win_rate:.1f}% |
| Games Backtested | {total_bets} |
| Profit (per 100 bets) | ${profit:,} |
| ROI | {roi:.2f}% |
| Edge vs Random | {win_rate - 50:.1f}% |"""
    
    # Replace old table (between "### Key Results" and "## Architecture")
    import re
    pattern = r'(### Key Results\n\n)\| Metric \| Value \|.*?(?=\n\n##)'
    content = re.sub(pattern, r'\1' + new_table + '\n', content, flags=re.DOTALL)
    
    # Write back
    with open(readme_path, 'w') as f:
        f.write(content)
    
    print(f"✅ README updated: {win_rate:.1f}% WR, {roi:.2f}% ROI")

if __name__ == "__main__":
    update_readme_stats()