import pandas as pd

def prepare_nhl_data():
    """Convert NHL history CSV to correct format and clean bad data."""
    
    input_path = "data/history/nhl_history.csv"
    output_path = "data/history/nhl_history.csv"
    
    # Load with tab separator
    df = pd.read_csv(input_path, sep='\t')
    
    # Convert date format MM/DD/YYYY to YYYY-MM-DD
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y').dt.strftime('%Y-%m-%d')
    
    # Remove teams with very few games (data quality issues)
    all_teams = set(df['Home'].unique()) | set(df['Away'].unique())
    teams_to_keep = []
    
    for team in all_teams:
        team_games = len(df[(df['Home'] == team) | (df['Away'] == team)])
        if team_games >= 10:  # Keep teams with 10+ games
            teams_to_keep.append(team)
    
    # Filter
    df = df[(df['Home'].isin(teams_to_keep)) & (df['Away'].isin(teams_to_keep))]
    
    # Save as comma-separated
    df.to_csv(output_path, index=False)
    print(f"✅ Prepared {len(df)} games")
    print(f"   Date range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"   Teams: {len(teams_to_keep)}")

if __name__ == "__main__":
    prepare_nhl_data()