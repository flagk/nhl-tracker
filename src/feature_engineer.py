import pandas as pd
from typing import Dict, List, Tuple
from datetime import datetime, timedelta

class FeatureEngineer:
    """Create features that actually predict wins."""
    
    def __init__(self, recent_games_window: int = 5):
        """Initialize with window for recent form."""
        self.window = recent_games_window
    
    def calculate_team_features(self, team: str, history: pd.DataFrame, reference_date: str) -> Dict:
        """
        Calculate all features for a team up to (but not including) reference_date.
        This ensures we only use data available BEFORE the game.
        """
        # Filter games before reference date
        cutoff = pd.to_datetime(reference_date)
        team_games = history[
            (history['Date'] < cutoff) & 
            ((history['Home'] == team) | (history['Away'] == team))
        ].sort_values('Date')
        
        if len(team_games) == 0:
            return self._zero_features()
        
        # --- FEATURE 1-2: Points Per Game (PPG) and Goal Differential ---
        total_games = len(team_games)
        total_points = team_games['Points'].sum()
        total_goal_diff = team_games['GoalDiff'].sum()
        
        ppg = total_points / max(1, total_games)
        goal_diff_per_game = total_goal_diff / max(1, total_games)
        
        # --- FEATURE 3-4: Recent Form (Last 5 games) ---
        recent = team_games.tail(self.window)
        recent_ppg = recent['Points'].sum() / max(1, len(recent))
        recent_gd = recent['GoalDiff'].sum() / max(1, len(recent))
        
        # --- FEATURE 5: Win Streak ---
        win_streak = self._calculate_win_streak(team_games)
        
        # --- FEATURE 6: Home/Away Split ---
        home_games = team_games[team_games['Home'] == team]
        home_ppg = home_games['Points'].sum() / max(1, len(home_games)) if len(home_games) > 0 else ppg
        
        away_games = team_games[team_games['Away'] == team]
        away_ppg = away_games['Points'].sum() / max(1, len(away_games)) if len(away_games) > 0 else ppg
        
        home_away_split = home_ppg - away_ppg
        
        # --- FEATURE 7: Rest Days ---
        rest_days = self._calculate_rest_days(team_games, reference_date)
        
        # --- FEATURE 8: Win Percentage ---
        wins = len(team_games[team_games['Winner'] == team])
        win_pct = wins / max(1, total_games)
        
        # --- FEATURE 9: Goals For Per Game ---
        gf_per_game = team_games['GoalsFor'].sum() / max(1, total_games)
        
        # --- FEATURE 10: Goals Against Per Game ---
        ga_per_game = team_games['GoalsAgainst'].sum() / max(1, total_games)
        
        return {
            'ppg': ppg,
            'goal_diff_pg': goal_diff_per_game,
            'recent_ppg': recent_ppg,
            'recent_gd': recent_gd,
            'win_streak': win_streak,
            'home_away_split': home_away_split,
            'rest_days': rest_days,
            'win_pct': win_pct,
            'gf_pg': gf_per_game,
            'ga_pg': ga_per_game,
            'games_played': total_games
        }
    
    def create_training_features(self, home: str, away: str, history: pd.DataFrame, game_date: str) -> List[float]:
        """Create feature vector for a specific matchup."""
        home_features = self.calculate_team_features(home, history, game_date)
        away_features = self.calculate_team_features(away, history, game_date)
        
        # Create feature vector: differences are often more predictive than absolutes
        features = [
            home_features['ppg'] - away_features['ppg'],
            home_features['goal_diff_pg'] - away_features['goal_diff_pg'],
            home_features['recent_ppg'] - away_features['recent_ppg'],
            home_features['recent_gd'] - away_features['recent_gd'],
            home_features['win_streak'] - away_features['win_streak'],
            home_features['home_away_split'],  # Home team advantage
            away_features['home_away_split'],  # Away team disadvantage
            home_features['rest_days'] - away_features['rest_days'],  # Rest advantage
            home_features['win_pct'] - away_features['win_pct'],
            home_features['gf_pg'] - away_features['gf_pg'],
            home_features['ga_pg'] - away_features['ga_pg'],
        ]
        
        return features
    
    def _calculate_win_streak(self, team_games: pd.DataFrame) -> int:
        """Calculate current win streak (negative for loss streak)."""
        if len(team_games) == 0:
            return 0
        
        streak = 0
        for _, game in team_games.iloc[::-1].iterrows():
            if game['Winner'] == team_games.iloc[0]['Home'] or game['Winner'] == team_games.iloc[0]['Away']:
                # Determine if this game is a win
                is_win = game['Winner'] == (game['Home'] if game['Home'] in team_games.index else game['Away'])
                
                if streak == 0:
                    streak = 1 if is_win else -1
                elif (streak > 0 and is_win) or (streak < 0 and not is_win):
                    streak += 1 if is_win else -1
                else:
                    break
        
        return streak
    
    def _calculate_rest_days(self, team_games: pd.DataFrame, reference_date: str) -> int:
        """Calculate days of rest before reference_date."""
        if len(team_games) == 0:
            return 0
        
        last_game_date = pd.to_datetime(team_games.iloc[-1]['Date'])
        reference = pd.to_datetime(reference_date)
        
        rest = (reference - last_game_date).days
        return min(rest, 5)  # Cap at 5 days (diminishing returns)
    
    def _zero_features(self) -> Dict:
        """Return zero feature vector for teams with no history."""
        return {
            'ppg': 0,
            'goal_diff_pg': 0,
            'recent_ppg': 0,
            'recent_gd': 0,
            'win_streak': 0,
            'home_away_split': 0,
            'rest_days': 0,
            'win_pct': 0,
            'gf_pg': 0,
            'ga_pg': 0,
            'games_played': 0
        }
    
    def get_feature_names(self) -> List[str]:
        """Return feature names for model interpretation."""
        return [
            'ppg_diff',
            'goal_diff_diff',
            'recent_ppg_diff',
            'recent_gd_diff',
            'win_streak_diff',
            'home_advantage',
            'away_disadvantage',
            'rest_diff',
            'win_pct_diff',
            'gf_diff',
            'ga_diff'
        ]