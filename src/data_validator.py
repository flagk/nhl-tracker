import pandas as pd
from typing import Dict, List, Tuple

class DataValidator:
    """Validate NHL game data quality before training."""
    
    def __init__(self, min_games: int = 10):
        """Initialize with minimum games required per team."""
        self.min_games = min_games
    
    def validate_game_record(self, record: Dict) -> Tuple[bool, str]:
        """
        Validate a single game record.
        Returns: (is_valid, error_message)
        """
        # Check required fields
        required = ['Date', 'Home', 'Away', 'HomeScore', 'AwayScore', 'Winner']
        for field in required:
            if field not in record or pd.isna(record[field]):
                return False, f"Missing field: {field}"
        
        home = record['Home']
        away = record['Away']
        home_score = record['HomeScore']
        away_score = record['AwayScore']
        winner = record['Winner']
        
        # Validate scores are non-negative
        if home_score < 0 or away_score < 0:
            return False, f"Negative score: {home} {home_score} vs {away} {away_score}"
        
        # Validate scores are reasonable (max ~10 goals in NHL)
        if home_score > 15 or away_score > 15:
            return False, f"Unrealistic score: {home_score} vs {away_score}"
        
        # Validate winner matches scores
        if home_score > away_score:
            if winner != home:
                return False, f"Winner mismatch: {winner} but {home} scored more"
        elif away_score > home_score:
            if winner != away:
                return False, f"Winner mismatch: {winner} but {away} scored more"
        else:
            return False, f"Tie game {home} {home_score} vs {away} {away_score} (NHL has OT/SO)"
        
        # Validate teams aren't same
        if home == away:
            return False, f"Team playing itself: {home}"
        
        return True, "OK"
    
    def validate_dataset(self, history: pd.DataFrame) -> Dict:
        """
        Validate entire dataset. Return quality report.
        """
        if history.empty:
            return {
                'status': 'FAILED',
                'total_records': 0,
                'valid_records': 0,
                'invalid_records': 0,
                'quality_score': 0.0,
                'errors': ['Empty dataset']
            }
        
        valid_count = 0
        invalid_count = 0
        errors = []
        
        for idx, row in history.iterrows():
            is_valid, error_msg = self.validate_game_record(row.to_dict())
            if is_valid:
                valid_count += 1
            else:
                invalid_count += 1
                if len(errors) < 5:  # Keep first 5 errors
                    errors.append(f"Row {idx}: {error_msg}")
        
        total = len(history)
        quality_score = (valid_count / total * 100) if total > 0 else 0
        
        # Determine status
        if valid_count == 0:
            status = "FAILED"
        elif valid_count < total * 0.9:
            status = "DEGRADED"
        else:
            status = "GOOD"
        
        # Check team coverage
        teams = set(history['Home'].unique()) | set(history['Away'].unique())
        
        report = {
            'status': status,
            'total_records': total,
            'valid_records': valid_count,
            'invalid_records': invalid_count,
            'quality_score': round(quality_score, 2),
            'unique_teams': len(teams),
            'errors': errors
        }
        
        return report
    
    def validate_sufficient_history(self, history: pd.DataFrame) -> Tuple[bool, str]:
        """
        Check if we have enough data to train on.
        """
        if len(history) < 100:
            return False, f"Only {len(history)} games. Need at least 100 for training."
        
        teams = set(history['Home'].unique()) | set(history['Away'].unique())
        if len(teams) < 20:
            return False, f"Only {len(teams)} unique teams. Need at least 20."
        
        # Check each team has minimum games
        for team in teams:
            team_games = len(history[(history['Home'] == team) | (history['Away'] == team)])
            if team_games < self.min_games:
                return False, f"Team {team} has only {team_games} games (need {self.min_games})"
        
        return True, "Sufficient history for training"
    
    def clean_dataset(self, history: pd.DataFrame) -> pd.DataFrame:
        """
        Remove invalid records from dataset.
        """
        valid_rows = []
        
        for idx, row in history.iterrows():
            is_valid, _ = self.validate_game_record(row.to_dict())
            if is_valid:
                valid_rows.append(idx)
        
        cleaned = history.loc[valid_rows].copy()
        removed = len(history) - len(cleaned)
        
        if removed > 0:
            print(f"🧹 Removed {removed} invalid records")
        
        return cleaned