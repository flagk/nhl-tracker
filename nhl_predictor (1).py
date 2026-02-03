import os
import requests
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from datetime import datetime, timedelta

# --- THE FIX: specific to your computer's security settings ---
if 'SSLKEYLOGFILE' in os.environ:
    del os.environ['SSLKEYLOGFILE']

# --- CONFIGURATION ---
LOG_FILE = "nhl_predictions_log.csv"

# --- STEP 1: API HELPERS ---

def get_latest_standings():
    """Fetches current team stats to feed the predictor."""
    url = "https://api-web.nhle.com/v1/standings/now"
    response = requests.get(url)
    data = response.json()
    
    teams_data = []
    for team in data['standings']:
        teams_data.append({
            'Team': team['teamAbbrev']['default'],
            'Wins': team['wins'],
            'Losses': team['losses'],
            'Points': team['points'],
            'GoalDiff': team['goalDifferential'],
            'HomeWins': team['homeWins'],
            'RoadWins': team['roadWins'],
            'GamesPlayed': team['gamesPlayed'],
            'GoalsFor': team['goalFor'],
            'GoalsAgainst': team['goalAgainst'],
            'L10Points': team['l10Points']
        })
    return pd.DataFrame(teams_data)

def get_schedule(date_str):
    """Fetches games for a specific date (YYYY-MM-DD)."""
    url = f"https://api-web.nhle.com/v1/schedule/{date_str}"
    try:
        response = requests.get(url)
        data = response.json()
        
        # The API returns a week; we filter for the specific day requested
        for day in data['gameWeek']:
            if day['date'] == date_str:
                return day['games']
        return []
    except Exception as e:
        print(f"⚠️ Could not fetch schedule: {e}")
        return []

# --- STEP 2: YOUR PREDICTION ENGINE ---

def train_model():
    """Same synthetic logic model from your original script."""
    X = []
    y = []
    # Synthetic training loop (Simplified for example)
    for _ in range(1000):
        points_diff = np.random.randint(-50, 50)
        goal_diff_diff = np.random.randint(-30, 30)
        wins_diff = np.random.randint(-20, 20)
        l10_points_diff = np.random.randint(-10, 10)
        
        score = (points_diff * 0.1) + (goal_diff_diff * 0.05) + (wins_diff * 0.05) + (l10_points_diff * 0.1)
        win_prob = 1 / (1 + np.exp(-score))
        result = 1 if np.random.random() < win_prob else 0
        
        X.append([points_diff, goal_diff_diff, wins_diff, l10_points_diff])
        y.append(result)
        
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X, y)
    return model

def make_prediction(home_team, away_team, df, model):
    """Returns (Predicted_Winner, Confidence_Score)"""
    try:
        home_stats = df[df['Team'] == home_team].iloc[0]
        away_stats = df[df['Team'] == away_team].iloc[0]
    except IndexError:
        return "Unknown", 0.0

    features = [
        home_stats['Points'] - away_stats['Points'],
        home_stats['GoalDiff'] - away_stats['GoalDiff'],
        home_stats['Wins'] - away_stats['Wins'],
        home_stats['L10Points'] - away_stats['L10Points']
    ]
    
    prob_home = model.predict_proba([features])[0][1]
    
    if prob_home > 0.5:
        return home_team, prob_home
    else:
        return away_team, (1 - prob_home)

# --- STEP 3: THE TRACKER SYSTEM (FILE I/O) ---

def load_log():
    """Loads the CSV log or creates a new one if it doesn't exist."""
    if os.path.exists(LOG_FILE):
        return pd.read_csv(LOG_FILE)
    else:
        return pd.DataFrame(columns=[
            "Date", "Home", "Away", "PredictedWinner", "Confidence", "ActualWinner", "Status"
        ])

def update_past_results(log_df):
    """Checks the API for final scores of 'Pending' games."""
    updated = False
    
    # Filter for rows that are 'Pending'
    pending_indices = log_df[log_df['Status'] == 'Pending'].index
    
    if len(pending_indices) > 0:
        print(f"🔄 Checking results for {len(pending_indices)} pending games...")
        
        for idx in pending_indices:
            row = log_df.loc[idx]
            date = row['Date']
            
            # Fetch games for that date
            games = get_schedule(date)
            
            for game in games:
                h_team = game['homeTeam']['abbrev']
                a_team = game['awayTeam']['abbrev']
                
                # Match found
                if h_team == row['Home'] and a_team == row['Away']:
                    # Check if game is Final (gameState 6 or 7 usually means final/final OT)
                    # simpler check: do we have a score?
                    if 'score' in game['homeTeam']:
                        winner = h_team if game['homeTeam']['score'] > game['awayTeam']['score'] else a_team
                        
                        # Update the DataFrame
                        log_df.at[idx, 'ActualWinner'] = winner
                        log_df.at[idx, 'Status'] = 'Correct' if winner == row['PredictedWinner'] else 'Wrong'
                        updated = True
                        print(f"   ✅ {date}: {h_team} vs {a_team} -> Winner: {winner} (Prediction: {row['Status']})")
    
    if updated:
        log_df.to_csv(LOG_FILE, index=False)
        print("💾 Log file updated with new results.")
    else:
        print("ℹ️ No new results found for pending games.")
        
    return log_df

def predict_todays_games(log_df, stats_df, model):
    """Fetches today's games and appends new predictions to the log."""
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
        
        # Check if we already predicted this game
        duplicate = log_df[
            (log_df['Date'] == today) & 
            (log_df['Home'] == h_team) & 
            (log_df['Away'] == a_team)
        ]
        
        if not duplicate.empty:
            continue # Skip if already in log

        winner, conf = make_prediction(h_team, a_team, stats_df, model)
        
        print(f"   ⚔️ {a_team} @ {h_team} -> Pick: {winner} ({conf*100:.1f}%)")
        
        new_picks.append({
            "Date": today,
            "Home": h_team,
            "Away": a_team,
            "PredictedWinner": winner,
            "Confidence": round(conf, 4),
            "ActualWinner": None,
            "Status": "Pending"
        })
        
    if new_picks:
        new_df = pd.DataFrame(new_picks)
        # Combine and save
        log_df = pd.concat([log_df, new_df], ignore_index=True)
        log_df.to_csv(LOG_FILE, index=False)
        print(f"💾 Added {len(new_picks)} new picks to {LOG_FILE}")
    else:
        print("✅ All of today's games are already tracked.")

def show_stats(log_df):
    """Calculates accuracy."""
    completed = log_df[log_df['Status'].isin(['Correct', 'Wrong'])]
    if len(completed) == 0:
        print("\n📊 No completed games tracked yet.")
        return

    correct = len(completed[completed['Status'] == 'Correct'])
    total = len(completed)
    accuracy = (correct / total) * 100
    print(f"\n📊 MODEL ACCURACY: {accuracy:.1f}% ({correct}/{total})")

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    print("\n--- 🏒 NHL AUTO-TRACKER 🏒 ---")
    
    # 1. Load Data
    log_df = load_log()
    stats_df = get_latest_standings()
    model = train_model()
    
    # 2. Check Yesterday's Results
    log_df = update_past_results(log_df)
    
    # 3. Predict Today's Games
    predict_todays_games(log_df, stats_df, model)
    
    # 4. Show Stats
    show_stats(log_df)
    
    print("\nDone. Press Enter to exit.")
    input()