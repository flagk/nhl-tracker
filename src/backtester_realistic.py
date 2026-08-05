import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime

from src.trainer import NHLModelTrainer
from src.feature_engineer import FeatureEngineer

class RealisticBacktester:
    """Walk-forward backtesting (no look-ahead bias)."""
    
    def __init__(self, trainer: NHLModelTrainer, min_confidence: float = 0.55):
        """Initialize."""
        self.trainer = trainer
        self.min_confidence = min_confidence
        self.feature_engineer = FeatureEngineer()
    
    def walk_forward_backtest(self, history: pd.DataFrame, train_size: int = 500, test_size: int = 100) -> Dict:
        """
        Walk-forward backtest: train on N games, test on next M games, repeat.
        
        This prevents look-ahead bias by only using data available at prediction time.
        """
        history['Date'] = pd.to_datetime(history['Date'])
        history = history.sort_values('Date').reset_index(drop=True)
        
        print(f"\n🎲 Walk-Forward Backtest")
        print(f"   Train window: {train_size} games")
        print(f"   Test window: {test_size} games")
        print(f"   Total games: {len(history)}")
        
        total_bets = 0
        total_wins = 0
        total_losses = 0
        all_predictions = []
        
        # Walk forward through time
        for start_idx in range(0, len(history) - train_size - test_size, test_size):
            train_end = start_idx + train_size
            test_end = train_end + test_size
            
            if test_end > len(history):
                break
            
            # Split
            train_data = history[:train_end]
            test_data = history[train_end:test_end]
            
            # Train on this window
            try:
                X, y = self.trainer.prepare_training_data(train_data)
                if len(X) < 50:
                    continue
                self.trainer.train_with_cv(X, y, cv_folds=3)
            except Exception as e:
                print(f"   ⚠️ Training failed: {e}")
                continue
            
            # Test on held-out data
            for idx, row in test_data.iterrows():
                home = row['Home']
                away = row['Away']
                game_date = row['Date'].strftime("%Y-%m-%d")
                actual_winner = row['Winner']
                
                try:
                    predicted_winner, confidence = self.trainer.predict_game(
                        home, away, train_data, game_date
                    )
                    
                    if confidence >= self.min_confidence:
                        total_bets += 1
                        is_correct = (predicted_winner == actual_winner)
                        
                        if is_correct:
                            total_wins += 1
                        else:
                            total_losses += 1
                        
                        all_predictions.append({
                            'date': game_date,
                            'home': home,
                            'away': away,
                            'prediction': predicted_winner,
                            'confidence': confidence,
                            'actual': actual_winner,
                            'result': 'WIN' if is_correct else 'LOSS'
                        })
                except Exception as e:
                    continue
        
        if total_bets == 0:
            return {'status': 'FAILED', 'message': 'No bets placed'}
        
        win_rate = total_wins / total_bets
        profit = (total_wins * 100) - (total_losses * 100)
        roi = (profit / (total_bets * 100)) * 100
        
        return {
            'status': 'SUCCESS',
            'total_bets': total_bets,
            'wins': total_wins,
            'losses': total_losses,
            'win_rate': round(win_rate, 4),
            'profit': profit,
            'roi_pct': round(roi, 2),
            'predictions': all_predictions
        }
    
    def print_results(self, results: Dict):
        """Print formatted results."""
        if results['status'] == 'FAILED':
            print(f"❌ {results['message']}")
            return
        
        print(f"\n{'='*60}")
        print(f"WALK-FORWARD BACKTEST RESULTS")
        print(f"{'='*60}")
        print(f"\n📊 Performance:")
        print(f"   Total bets: {results['total_bets']}")
        print(f"   Wins: {results['wins']} | Losses: {results['losses']}")
        print(f"   Win rate: {results['win_rate']*100:.1f}%")
        print(f"\n💰 Financial:")
        print(f"   Profit: ${results['profit']:,.0f}")
        print(f"   ROI: {results['roi_pct']:.2f}%")
        
        if results['win_rate'] > 0.55:
            print(f"\n✅ Model has positive edge!")
        else:
            print(f"\n⚠️ Model does not beat 50% (no edge)")
        
        print(f"\n{'='*60}\n")