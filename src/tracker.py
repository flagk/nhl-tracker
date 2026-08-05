import pandas as pd
import os
from datetime import datetime
from typing import List, Dict

class PredictionTracker:
    """Track predictions and actual results for performance monitoring."""
    
    def __init__(self):
        """Initialize tracker."""
        pass
    
    def log_predictions(self, predictions: List[Dict], log_path: str):
        """
        Log predictions to CSV for tracking.
        
        Args:
            predictions: List of prediction dicts from predictor
            log_path: Path to predictions log CSV
        """
        # Prepare data for logging
        log_entries = []
        
        for pred in predictions:
            if 'error' in pred:
                continue
            
            entry = {
                'timestamp': datetime.now().isoformat(),
                'date': pred['game_date'],
                'home': pred['home'],
                'away': pred['away'],
                'prediction': pred['predicted_winner'],
                'confidence': pred['confidence'],
                'edge_pct': pred['edge_pct'],
                'should_bet': pred['should_bet'],
                'result': None  # Will be filled in later
            }
            log_entries.append(entry)
        
        if not log_entries:
            print("⚠️  No predictions to log")
            return
        
        # Load existing log or create new
        if os.path.exists(log_path):
            existing_log = pd.read_csv(log_path)
            new_log = pd.DataFrame(log_entries)
            combined_log = pd.concat([existing_log, new_log], ignore_index=True)
        else:
            combined_log = pd.DataFrame(log_entries)
        
        # Save
        combined_log.to_csv(log_path, index=False)
        print(f"✅ Logged {len(log_entries)} predictions")
    
    def update_results(self, log_path: str, updates: List[Dict]):
        """
        Update predictions with actual game results.
        
        Args:
            log_path: Path to predictions log
            updates: List of dicts with 'prediction_id' and 'result' (WIN/LOSS)
        """
        if not os.path.exists(log_path):
            print(f"❌ Log file not found: {log_path}")
            return
        
        log_df = pd.read_csv(log_path)
        
        updated_count = 0
        for update in updates:
            # Find matching prediction
            mask = (
                (log_df['date'] == update['date']) &
                (log_df['home'] == update['home']) &
                (log_df['away'] == update['away']) &
                (log_df['result'].isna())
            )
            
            if mask.any():
                log_df.loc[mask, 'result'] = update['result']
                updated_count += 1
        
        # Save updated log
        log_df.to_csv(log_path, index=False)
        print(f"✅ Updated {updated_count} results")
    
    def calculate_performance(self, log_path: str) -> Dict:
        """
        Calculate prediction performance metrics.
        
        Returns dict with accuracy, ROI, win rate, etc.
        """
        if not os.path.exists(log_path):
            print(f"❌ Log file not found: {log_path}")
            return {}
        
        log_df = pd.read_csv(log_path)
        
        # Filter to predictions with results
        completed = log_df[log_df['result'].notna()].copy()
        
        if len(completed) == 0:
            return {
                'total_predictions': len(log_df),
                'completed': 0,
                'message': 'No completed predictions yet'
            }
        
        # Overall stats
        total_preds = len(completed)
        wins = len(completed[completed['result'] == 'WIN'])
        losses = len(completed[completed['result'] == 'LOSS'])
        win_rate = wins / total_preds if total_preds > 0 else 0
        
        # Bet recommendations
        bet_preds = completed[completed['should_bet'] == True]
        bet_wins = len(bet_preds[bet_preds['result'] == 'WIN'])
        bet_losses = len(bet_preds[bet_preds['result'] == 'LOSS'])
        bet_win_rate = bet_wins / len(bet_preds) if len(bet_preds) > 0 else 0
        
        # ROI (assuming -110 odds, 1 unit per bet)
        # Win = +1 unit, Loss = -1 unit
        bet_roi = (bet_wins - bet_losses) / len(bet_preds) * 100 if len(bet_preds) > 0 else 0
        
        # Confidence analysis
        avg_confidence = completed['confidence'].mean()
        high_confidence_preds = completed[completed['confidence'] >= 0.60]
        high_conf_win_rate = len(high_confidence_preds[high_confidence_preds['result'] == 'WIN']) / len(high_confidence_preds) if len(high_confidence_preds) > 0 else 0
        
        return {
            'total_predictions': len(log_df),
            'completed_predictions': total_preds,
            'total_wins': wins,
            'total_losses': losses,
            'win_rate': round(win_rate, 4),
            'bet_recommendations': len(bet_preds),
            'bet_wins': bet_wins,
            'bet_losses': bet_losses,
            'bet_win_rate': round(bet_win_rate, 4),
            'bet_roi_pct': round(bet_roi, 2),
            'avg_confidence': round(avg_confidence, 4),
            'high_confidence_win_rate': round(high_conf_win_rate, 4)
        }
    
    def print_performance(self, log_path: str):
        """Print formatted performance report."""
        perf = self.calculate_performance(log_path)
        
        if not perf or 'message' in perf:
            print(f"ℹ️  {perf.get('message', 'No performance data')}")
            return
        
        print(f"\n{'='*60}")
        print(f"PREDICTION PERFORMANCE REPORT")
        print(f"{'='*60}")
        
        print(f"\n📊 Overall Accuracy:")
        print(f"   Total predictions: {perf['total_predictions']}")
        print(f"   Completed: {perf['completed_predictions']}")
        print(f"   Wins: {perf['total_wins']} | Losses: {perf['total_losses']}")
        print(f"   Win rate: {perf['win_rate']*100:.1f}%")
        
        print(f"\n💰 Bet Recommendations:")
        print(f"   Total bets: {perf['bet_recommendations']}")
        print(f"   Bet wins: {perf['bet_wins']} | Bet losses: {perf['bet_losses']}")
        print(f"   Bet win rate: {perf['bet_win_rate']*100:.1f}%")
        print(f"   ROI on bets: {perf['bet_roi_pct']:.2f}%")
        
        print(f"\n⚡ Confidence Analysis:")
        print(f"   Average confidence: {perf['avg_confidence']*100:.1f}%")
        print(f"   High confidence (≥60%) win rate: {perf['high_confidence_win_rate']*100:.1f}%")
        
        print(f"\n{'='*60}\n")
    
    def get_streak(self, log_path: str) -> Dict:
        """
        Calculate current win/loss streak.
        """
        if not os.path.exists(log_path):
            return {'streak': 0, 'type': 'N/A'}
        
        log_df = pd.read_csv(log_path)
        completed = log_df[log_df['result'].notna()].copy()
        
        if len(completed) == 0:
            return {'streak': 0, 'type': 'N/A'}
        
        # Recent results (last 20)
        recent = completed.tail(20)
        
        streak = 0
        streak_type = None
        
        for _, row in recent.iloc[::-1].iterrows():
            is_win = row['result'] == 'WIN'
            
            if streak == 0:
                streak = 1
                streak_type = 'W' if is_win else 'L'
            elif (is_win and streak_type == 'W') or (not is_win and streak_type == 'L'):
                streak += 1
            else:
                break
        
        return {
            'streak': streak,
            'type': streak_type,
            'display': f"{streak_type}{streak}" if streak_type else "N/A"
        }