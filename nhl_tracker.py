import os
import requests
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt  # <--- NEW: Graphing Library

# --- CONFIGURATION ---
LOG_FILE = "nhl_predictions_log.csv"
HISTORY_FILE = "nhl_history.csv"

# --- STEP 1: API HELPERS ---
def get_latest_stats():
    """Fetches LIVE standings for today's prediction inputs."""
    url = "https://api-web.nhle.com/v1/standings/now"
    try:
        response = requests.get(url)
        data = response.json()
    except Exception as e:
        print(f"⚠️ Error fetching standings: {e}")
        return pd.DataFrame()
    
    teams_data = []
    for team in data['standings']:
        teams_data.append({
            'Team': team['teamAbbrev']['default'],
            'Points': team['points'],
            'GoalDiff': team['goalDifferential'],
            'WinPct': team['winPctg']
        })
    return pd.DataFrame(teams_data)

def get_schedule(date_str):
    """Fetches games for a specific date."""
    url = f"https://api-web.nhle.com/v1/schedule/{date_str}"
    try:
        response = requests.get(url)
        data = response.json()
        for day in data['gameWeek']:
            if day['date'] == date_str:
                return day['games']
        return []
    except Exception as e:
        print(f"⚠️ Could not fetch schedule: {e}")
        return []

# --- STEP 2: THE BRAIN (REAL TRAINING) ---
def train_smart_model():
    """Trains on History + Finished Games from your Log."""
    # 1. Load the massive history file
    if os.path.exists(HISTORY_FILE):
        history_df = pd.read_csv(HISTORY_FILE)
    else:
        history_df = pd.DataFrame()

    # 2. Load the new finished games from your daily log
    if os.path.exists(LOG_FILE):
        log_df = pd.read_csv(LOG_FILE)
        # Filter for only finished games
        new_games = log_df[log_df['Status'].isin(['Correct', 'Wrong'])].copy()
        
        if not new_games.empty:
            print(f"🧠 Incorporating {len(new_games)} recent games from your log...")
            formatted_new_games = []
            for _, row in new_games.iterrows():
                if pd.isna(row['ActualWinner']): continue
                
                formatted_new_games.append({
                    "Date": row['Date'],
                    "Home": row['Home'],
                    "Away": row['Away'],
                    "HomeScore": 5 if row['ActualWinner'] == row['Home'] else 2, 
                    "AwayScore": 5 if row['ActualWinner'] == row['Away'] else 2,
                    "Winner": row['ActualWinner']
                })
            
            history_df = pd.concat([history_df, pd.DataFrame(formatted_new_games)], ignore_index=True)

    if history_df.empty:
        print("⚠️ No data to train on!")
        return None

    # Handle various date formats (Excel vs System)
    history_df['Date'] = pd.to_datetime(history_df['Date'], format='mixed', errors='coerce')
    history_df = history_df.dropna(subset=['Date'])
    history_df = history_df.sort_values('Date')
    
    # --- TRAINING LOOP ---
    print(f"🎓 Training on {len(history_df)} total games...")
    
    team_stats = {} 
    X = []
    y = []
    
    for _, row in history_df.iterrows():
        home = row['Home']
        away = row['Away']
        
        if home not in team_stats: team_stats[home] = {'pts': 0, 'gd': 0, 'games': 0}
        if away not in team_stats: team_stats[away] = {'pts': 0, 'gd': 0, 'games': 0}
        
        # 1. Capture stats BEFORE the game
        h_games = max(1, team_stats[home]['games'])
        a_games = max(1, team_stats[away]['games'])
        
        h_ppg = team_stats[home]['pts'] / h_games
        a_ppg = team_stats[away]['pts'] / a_games
        h_gd_pg = team_stats[home]['gd'] / h_games
        a_gd_pg = team_stats[away]['gd'] / a_games
        
        X.append([h_ppg - a_ppg, h_gd_pg - a_gd_pg])
        
        # 2. Who actually won?
        if 'Winner' in row and pd.notna(row['Winner']):
            home_won = 1 if row['Winner'] == home else 0
        else:
            home_won = 1 if row['HomeScore'] > row['AwayScore'] else 0
        y.append(home_won)
        
        # 3. Update stats for next time
        team_stats[home]['games'] += 1
        team_stats[away]['games'] += 1
        
        h_score = row['HomeScore'] if pd.notna(row['HomeScore']) else (3 if home_won else 1)
        a_score = row['AwayScore'] if pd.notna(row['AwayScore']) else (3 if not home_won else 1)
        
        team_stats[home]['gd'] += (h_score - a_score)
        team_stats[away]['gd'] += (a_score - h_score)
        
        if home_won:
            team_stats[home]['pts'] += 2
        else:
            team_stats[away]['pts'] += 2

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

def make_prediction(home_team, away_team, df, model):
    """Uses LIVE stats to predict today's game."""
    try:
        home_stats = df[df['Team'] == home_team].iloc[0]
        away_stats = df[df['Team'] == away_team].iloc[0]
    except IndexError:
        return "Unknown", 0.0

    # Calculate same features as training
    h_games = 50 
    a_games = 50
    
    h_ppg = home_stats['Points'] / h_games
    a_ppg = away_stats['Points'] / a_games
    h_gd_pg = home_stats['GoalDiff'] / h_games
    a_gd_pg = away_stats['GoalDiff'] / a_games
    
    features = [h_ppg - a_ppg, h_gd_pg - a_gd_pg]
    prob_home = model.predict_proba([features])[0][1]
    
    if prob_home > 0.5:
        return home_team, prob_home
    else:
        return away_team, (1 - prob_home)

# --- STEP 3: TRACKER SYSTEM ---
def load_log():
    if os.path.exists(LOG_FILE):
        df = pd.read_csv(LOG_FILE)
        df['ActualWinner'] = df['ActualWinner'].astype(object)
        df['Status'] = df['Status'].astype(object)
        return df
    else:
        return pd.DataFrame(columns=["Date", "Home", "Away", "PredictedWinner", "Confidence", "ActualWinner", "Status"])

def update_past_results(log_df):
    updated = False
    pending_indices = log_df[log_df['Status'] == 'Pending'].index
    
    if len(pending_indices) > 0:
        print(f"🔄 Checking results for {len(pending_indices)} pending games...")
        for idx in pending_indices:
            row = log_df.loc[idx]
            games = get_schedule(row['Date'])
            for game in games:
                if game['homeTeam']['abbrev'] == row['Home'] and game['awayTeam']['abbrev'] == row['Away']:
                    if 'score' in game['homeTeam']:
                        winner = game['homeTeam']['abbrev'] if game['homeTeam']['score'] > game['awayTeam']['score'] else game['awayTeam']['abbrev']
                        log_df.at[idx, 'ActualWinner'] = winner
                        log_df.at[idx, 'Status'] = 'Correct' if winner == row['PredictedWinner'] else 'Wrong'
                        updated = True
                        print(f"   ✅ {row['Date']}: {winner} Won (Prediction: {log_df.at[idx, 'Status']})")
    if updated: log_df.to_csv(LOG_FILE, index=False)
    return log_df

def predict_todays_games(log_df, stats_df, model):
    today = datetime.now().strftime("%Y-%m-%d")
    games = get_schedule(today)
    
    if not games:
        print(f"📅 No games scheduled for today ({today}).")
        return

    new_picks = []
    print(f"\n🔮 Generating picks for today ({today}):")
    
    for game in games:
        h_team = game['homeTeam']['abbrev']
        a_team = game['awayTeam']['abbrev']
        
        # Skip duplicates
        if not log_df[(log_df['Date'] == today) & (log_df['Home'] == h_team) & (log_df['Away'] == a_team)].empty:
            continue

        winner, conf = make_prediction(h_team, a_team, stats_df, model)
        print(f"   ⚔️ {a_team} @ {h_team} -> Pick: {winner} ({conf*100:.1f}%)")
        
        new_picks.append({
            "Date": today, "Home": h_team, "Away": a_team,
            "PredictedWinner": winner, "Confidence": round(conf, 4),
            "ActualWinner": None, "Status": "Pending"
        })
        
    if new_picks:
        log_df = pd.concat([log_df, pd.DataFrame(new_picks)], ignore_index=True)
        log_df.to_csv(LOG_FILE, index=False)

def show_stats(log_df):
    """Calculates accuracy and generates a trend graph."""
    completed = log_df[log_df['Status'].isin(['Correct', 'Wrong'])].copy()
    
    if len(completed) == 0:
        print("\n📊 No completed games tracked yet.")
        return

    correct = len(completed[completed['Status'] == 'Correct'])
    total = len(completed)
    accuracy = (correct / total) * 100
    print(f"\n📊 LIFETIME ACCURACY: {accuracy:.1f}% ({correct}/{total})")

    # --- NEW: GENERATE CHART ---
    try:
        completed['IsCorrect'] = completed['Status'].apply(lambda x: 1 if x == 'Correct' else 0)
        # Create rolling average
        completed['RollingAcc'] = completed['IsCorrect'].rolling(window=10, min_periods=1).mean() * 100
        
        plt.figure(figsize=(10, 5))
        plt.plot(completed['Date'], completed['RollingAcc'], marker='o', linestyle='-', color='#00a2e8', label='Rolling Accuracy')
        plt.title(f'NHL Model Performance (Lifetime: {accuracy:.1f}%)')
        plt.xlabel('Date')
        plt.ylabel('Accuracy (%)')
        plt.axhline(y=50, color='r', linestyle='--', alpha=0.5, label='Coin Flip (50%)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        plt.savefig('accuracy_plot.png')
        print("📈 Graph saved to 'accuracy_plot.png'")
    except Exception as e:
        print(f"⚠️ Could not generate graph: {e}")

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    print("\n--- 🏒 NHL AI PREDICTOR (TRAINED ON HISTORY) 🏒 ---")
    log_df = load_log()
    stats_df = get_latest_stats()
    model = train_smart_model()
    
    if model:
        log_df = update_past_results(log_df)
        predict_todays_games(log_df, stats_df, model)
        show_stats(log_df)
    
    print("\nDone.")