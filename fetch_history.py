import requests
import pandas as pd
from datetime import datetime, timedelta
import time

# --- CONFIGURATION ---
START_DATE = "2023-10-10"  # Start of 2023-24 Season
END_DATE = datetime.now().strftime("%Y-%m-%d")
OUTPUT_FILE = "nhl_history.csv"

def fetch_season_history():
    print(f"📚 Starting history download from {START_DATE} to {END_DATE}...")
    
    all_games = []
    current_date = datetime.strptime(START_DATE, "%Y-%m-%d")
    end_date_obj = datetime.strptime(END_DATE, "%Y-%m-%d")
    
    # Loop through weeks (The API returns 1 week of data at a time)
    while current_date < end_date_obj:
        date_str = current_date.strftime("%Y-%m-%d")
        print(f"   Fetching week of {date_str}...")
        
        url = f"https://api-web.nhle.com/v1/schedule/{date_str}"
        
        try:
            response = requests.get(url)
            data = response.json()
            
            for day in data.get('gameWeek', []):
                for game in day.get('games', []):
                    # Only grab games that are finished
                    if 'score' in game['homeTeam']:
                        home_score = game['homeTeam']['score']
                        away_score = game['awayTeam']['score']
                        
                        # Determine winner
                        winner = game['homeTeam']['abbrev'] if home_score > away_score else game['awayTeam']['abbrev']
                        
                        all_games.append({
                            "Date": day['date'],
                            "Home": game['homeTeam']['abbrev'],
                            "Away": game['awayTeam']['abbrev'],
                            "HomeScore": home_score,
                            "AwayScore": away_score,
                            "Winner": winner
                        })
                        
        except Exception as e:
            print(f"⚠️ Error on {date_str}: {e}")

        # Jump forward 7 days (API gives a week at a time)
        current_date += timedelta(days=7)
        time.sleep(0.5) # Be polite to the API (don't spam it)

    # Save to CSV
    df = pd.DataFrame(all_games)
    # Remove duplicates just in case
    df = df.drop_duplicates(subset=['Date', 'Home', 'Away'])
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✅ Success! Downloaded {len(df)} historical games to {OUTPUT_FILE}")

if __name__ == "__main__":
    fetch_season_history()